# 引擎侧 ms/step 探针（门 #3 能力）+ 批大小伸缩实测（2026-10-06）

> 目的：目标的第 3 道验收门是「**`[bneck] hp`（引擎侧 ms/step，不用聚合 tok/s）**」。
> 实测发现这条门在 tiny 上**根本取不到数**（见 §1），先把它修好，再用它量批大小伸缩。
> 全部为【实测】。

## 1. 问题：`[bneck] hp` 在 tiny 上是 0 行

```bash
grep -ac "\[bneck\]" ~/tmp/ab_mkc/run_ab_mkc_1001_215615/serve_prof.log   # ⇒ 0
```

根因：`_BneckState.tick()` **只在 `prepare_engram_inputs()` 里被调用**
（`models/deepseek_v41/model.py:1598`），而 tiny 跑 `ENGRAM=0` ⇒ 探针永不执行。

## 2. 修法：探针改挂 `NPUModelRunner.execute_model`

第一版按直觉插在 `DeepseekV41Model.forward` 开头 ⇒ **失败**：
那一段**被 torch.compile/dynamo 追踪**，`perf_counter()` 直接让 engine 起不来
（实测 EngineCore 报错 + 空 `engine core initialization failed`）。已回退。

改用 **`NPUModelRunner.execute_model`** —— 它是每步唯一的**普通 Python** 入口
（图 replay 在它内部发生），不会被追踪，相邻两次调用的墙钟差就是引擎侧 ms/step。

```python
# worker/model_runner_v1.py 模块级（[V41-STEP-PROBE]）
def _sp_tick(n_tokens):
    now = time.perf_counter(); last = _SP["last"]; _SP["last"] = now
    if last is None or n_tokens > _SP_DEC_TOK:   # 只统计 decode 步
        return
    _SP["n"] += 1; _SP["ntok"] += n_tokens; _SP["acc"] += (now - last) * 1000.0
    if _SP["n"] % _SP_EVERY == 0:
        print("[step] dec_steps=%d ms/step=%.3f tok/step=%.2f" % (...), flush=True)
```

开关：`V41_STEP_PROBE=1`（默认关 ⇒ 零影响）、`_EVERY`（默认 20）、`_DECODE_TOKENS`（默认 64）。
产物：`tools/dbo_fix_step_probe_execute_model.py`（一键打补丁）。

实测输出（tiny，TP2，DCP=1，graph，多流开）：

```
[step] dec_steps=20 ms/step=38.9 tok/step=8.00
[step] dec_steps=20 ms/step=48.6 tok/step=32.00
[step] dec_steps=20 ms/step=66.9 tok/step=64.00
```

## 3. ★ 批大小伸缩：**固定开销主导**（这是本轮最有价值的数字）

| conc | tok/step | **ms/step** | 每 token 摊销 |
|---:|---:|---:|---:|
| 1 | 8 | **38.9** | **4.86 ms** |
| 4 | 32 | **48.6** | **1.52 ms** |
| 8 | 64 | **66.9** | **1.05 ms** |

读法：

* **token 数涨 8×，步长只涨 1.72×**（38.9 → 66.9）；
* 边际成本：`(66.9 − 38.9) / (64 − 8) = **0.50 ms/token**`；
  而固定的那部分 ≈ **35 ms/步**；
* 每 token 摊销从 **4.86 ms 降到 1.05 ms（4.6×）**。

⇒ 单流 decode 的瓶颈**不是"每个 token 要算什么"，而是"每步的固定开销"**。
这与暴露度分析完全一致（MoE 4.44 / HcPre 1.99 / 通信 2.50，都是每步固定发生、与 token 数弱相关）。

## 4. ⚠️ 方法论警告：tiny 的「聚合 tok/s」不能当判据

同一时刻：

```
探针：ms/step = 38.9，tok/step = 8.0  ⇒ 8.0 / 0.0389 = 206「调度 token/s」
基准：单流 = 25.8 tok/s
```

**差 8 倍**。原因是 **tiny 用 `--load-format dummy`（随机权重）**：

| | 说明 |
|---|---|
| `tok/step = 8` | 1 个 target token + 7 个 draft token（SP_TOKENS=7） |
| 每步**真正产出** | dummy 权重下 draft 全错 ⇒ **接受长度 ≈ 1** ⇒ 每步只吐 1 个 token |
| 于是 | `1 / 0.0389 s = 25.7 tok/s` ——**与实测 25.8 完全吻合** |

⇒ **tiny 上"聚合 tok/s"测的是接受率（随机权重 ⇒ 恒为 ~1），不是引擎效率。**
这正是目标把验收门定为 `[bneck] hp` 而不是 tok/s 的原因，也说明：

* 之前所有基于 tiny「tok/s」的 DBO A/B（0.52× 等）**在方向上仍成立**
  （同一 dummy 接受率下比较，Engine 时间是主变量），但**幅度不能外推到生产**；
* 今后 tiny 上的一切 A/B **必须同时报 `[step] ms/step`**。

## 5. 结论与下一步

1. 门 #3 现在**可用**（tiny 与 tp8 都能取 `ms/step`）；
2. 实测确认**每步固定开销 ≈ 35 ms** 主导单流延迟，边际 token 成本只有 0.5 ms；
3. ⇒ 提升单流要靠**削减每步的固定开销**，而固定开销的三大块已定位：
   MoE（4.44 ms，带宽利用率 58%）、HcPre（1.99 ms，43 µs/次 × 101 次）、通信（2.50 ms）；
4. ⇒ 提升吞吐要靠**把批做大**（本表已给出摊销曲线），而不是拆批（线 B 已证净亏）。

## 6. 复现

```bash
ssh a3-21 'docker cp ~/tmp/fix_step_probe2.py dsv41-tinyspark:/tmp/ && \
           docker exec dsv41-tinyspark python3 /tmp/fix_step_probe2.py'
ssh a3-21 'bash ~/tmp/launch_tiny_stepprobe.sh'          # tiny + 探针
ssh a3-21 'grep -a "\[step\]" ~/tmp/ab_mkc/run_ab_mkc_1001_215615/serve_prof.log | tail -6'
```
