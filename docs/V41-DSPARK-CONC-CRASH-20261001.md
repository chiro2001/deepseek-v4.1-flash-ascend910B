# ★★ DSpark × DCP8 并发打崩引擎（2026-10-01）—— 第 6 个缺陷，并发专属

> 用户问「DSpark 多流请求时性能掉得很快，测一下」。
> **实测结果比"性能掉"严重得多：并发 16 直接把引擎打崩（AI Core Error）。**
> 本文记录触发条件、根因链、以及一个被忽略的配置变量。

---

## 1. 实测：并发测试打崩引擎【实测】

run `dcpcap_1001_1145_s1prof`（SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1 DCP=8 ENGRAM=0 MAX_SEQS=16）

| 时刻 | 现象 |
|---|---|
| 04:24:24 | `Running: 13 reqs` |
| 04:24:34 | `Running: 16 reqs` |
| 04:24:44 | `Avg generation throughput: 0.0 tokens/s` ← **卡住** |
| 04:24:59 | `WorkerProc hit an exception.` 8 个 worker 全挂 |
| 之后 | health 000、容器不退出、`/metrics` 无响应 |

worker 报错（8 rank 一致）：

```
File ".../vllm_ascend/worker/dcp_utils.py", line 331, in rebuild_async_spec_decode_inputs
File ".../torch_npu/npu/streams.py", line 192, in synchronize
RuntimeError: synchronize:.../NPUEvent.cpp:215 NPU function error:
              aclrtSynchronizeEvent(event_), error code is 507011
[Error]: Model execution failed.
For details, see ... Search for the keyword "AI Core Error".
rtMemcpy execution failed, reason=driver error:internal error
```

**`AI Core Error` = 设备侧算子崩了**，不是 host 侧 Python 异常。

### 1.1 ★ 第 331 行不是凶手，是**报丧点**

`dcp_utils.py:331` = `valid_sampled_token_count_event.synchronize()` ——
它是 `can_rebuild_on_device = False` 分支里的**同步点**。Ascend 的 kernel 异步执行，
**错误在下一个同步点才暴露** ⇒ 真正的越界发生在**之前入队的某个 kernel**。

---

## 2. 触发条件：admission gate 一次性释放 120 个 deferred decode

```
[admission_gate] enabled: each step is either one prefill request or decode requests
[admission_gate] WARNING max_concurrent_batches=2 (async scheduling/PP):
                 scheduler outputs are pure, but batches may still overlap on workers.
[admission_gate] prefill-only step #10 (step=23): prefill_reqs=1 decode_reqs=0
                 total_tokens=1429 deferred_decode_reqs=8 (cumulative=36)
[admission_gate] prefill-only episode ended: steps=16 deferred_decode_reqs=120
                 (cumulative_deferred=120, cumulative_prefill_steps=17)
```

16 并发下，gate 连做 **16 步 prefill-only**、期间**攒了 120 个 deferred decode**，
然后一次性放出来。崩溃就发生在释放之后的那一步。

---

## 3. 根因链（候选，算式已从代码读出）

`vllm_ascend/worker/dcp_utils.py::rebuild_async_spec_decode_inputs`：

```python
self.decode_threshold = 1 + num_speculative_tokens   # = 1 + 7 = 8
extra_tokens = self.decode_threshold - 2             # = 6

mtp_lens = query_lens + extra_tokens
num_tokens_mtp = self.async_rebuild_num_tokens + num_reqs * extra_tokens   # ★
req_indices_mtp = torch.repeat_interleave(
    self.req_offsets[:num_reqs], mtp_lens, output_size=num_tokens_mtp,
)
```

**★ 这个算式把两个不同 step 的量混用了**：

| 量 | 来源 | 时序 |
|---|---|---|
| `self.async_rebuild_num_tokens` | `generate_dcp_mtp_input` 里 `int(cumulative[-1])` | **上一步**写入 |
| `num_reqs` | 函数入参 | **本步** |

只有 `sum(query_lens) == async_rebuild_num_tokens` 时 `sum(mtp_lens) == num_tokens_mtp`
才成立。一旦不成立：

* `sum(mtp_lens) < num_tokens_mtp` ⇒ `repeat_interleave` 末尾**留未初始化垃圾**
  ⇒ `num_computed_tokens[req_indices_mtp]` / `mtp_start_loc[req_indices_mtp]`
  **越界读** ⇒ AI Core Error（device 越界是静默的，这正是它会崩成设备错误的原因）
* `sum(mtp_lens) > num_tokens_mtp` ⇒ 直接抛 RuntimeError

**为什么 16 并发 + admission gate 会打破这个等式**：`async_rebuild_num_tokens`
是 gate 攒的那 16 步里**最后一步**留下的（那时是 prefill-only），而放行后
`num_reqs` 变成 16 个 decode 请求 ⇒ 两者来自完全不同的调度形态。
**标记【推断】**：算式与时序都对得上，但**尚未**在崩溃现场打出实际数值
（诊断已注入，见 §5，默认 env 关闭以免 `.item()` 污染性能）。

---

## 4. ★ 被忽略的配置变量：`--no-async-scheduling`

对比两个 run 的 `serve_cmd.txt`：

| run | `KV_ARGS_EXTRA` |
|---|---|
| `dcpcap_1001_102816`（线 A 的 32.58 基线） | `--decode-context-parallel-size 8 --no-async-scheduling` |
| `dcpcap_1001_1145_s1prof`（本轮，**崩了**） | `--decode-context-parallel-size 8` |

`dcp_stage_capacity.sh:178` 是
`export KV_ARGS_EXTRA="--decode-context-parallel-size $DCP${EXTRA_KV_ARGS:+ $EXTRA_KV_ARGS}"`
⇒ **不传 `EXTRA_KV_ARGS` 就没有 `--no-async-scheduling`**。

**后果**：`use_async_spec_decode = True` ⇒ 走 `rebuild_async_spec_decode_inputs`
的 device 重建路径。而我们此前对 DSpark×DCP 的 **5 个修复验证全部在
tiny（TP2/DCP2）上做的**，那次夹具恰好也没传 `--no-async-scheduling`
但**只有单流**，没触发这条路径。

⇒ **结论**：`DSpark × DCP × async-scheduling` 是一个**从未被验证过的组合**。
本轮之前"打通"的结论只在 `--no-async-scheduling` 下成立。

---

## 5. 已做的诊断注入（默认关闭，不影响性能）

`~/dcpw/vllm_ascend/worker/dcp_utils.py`（新加入 overlay）：

```python
import os as _os
if _os.environ.get("V41_DSPARK_DCP_DIAG", "0") == "1":
    _mtp_lens_sum = int(mtp_lens.sum().item())
    _ql_sum = int(query_lens.sum().item())
    if _mtp_lens_sum != num_tokens_mtp or _ql_sum != self.async_rebuild_num_tokens:
        logger.warning("[V41-DSPARK-DCP-DIAG] MISMATCH num_reqs=%s num_tokens_mtp=%s "
                       "sum(mtp_lens)=%s sum(query_lens)=%s async_rebuild_num_tokens=%s ...")
        num_tokens_mtp = _mtp_lens_sum   # 用真实和，避免 output_size 说谎
```

* 默认 `V41_DSPARK_DCP_DIAG=0` ⇒ **零开销**（`.item()` 会插 device 同步，热路径上必须关）。
* 排查时开 `=1`，能同时验证"是否 mismatch"与"用真实和是否能避开崩溃"。
* **注意诊断里那句 `num_tokens_mtp = _mtp_lens_sum`** —— 它不只是打日志，
  也是**候选修复**（用真实和覆盖错误算式）。如果开启后并发不再崩且精度正常，
  根因即确认。

---

## 6. 状态与下一步

| 项 | 状态 |
|---|---|
| `DSpark × DCP8`（单流，`--no-async-scheduling`） | ✅ 通，A=2.25，ms/step 37.61 |
| `DSpark × DCP8`（并发 16，**无** `--no-async-scheduling`） | ❌ **崩**（本文） |
| 并发曲线（用户诉求） | 待测，必须先固定 `--no-async-scheduling` |
| `DSpark × DCP × async-scheduling` | ❌ 未验证组合，已知崩溃 |

**下一步**：
1. 用 `--no-async-scheduling` 跑并发曲线（对齐线 A 的 32.58 基线口径），
   回答用户"DSpark 多流性能掉多快"。
2. 单独开一轮：开 `V41_DSPARK_DCP_DIAG=1` 复现崩溃，打出实际数值，确认根因。
3. 若确认，修法是把 `num_tokens_mtp` 改成 `sum(mtp_lens)`（device 侧算，不 sync），
   或把 `async_rebuild_num_tokens` 与本步 `num_reqs` 一起刷新。
