# ★ TP=8 decode 非确定性：三段定位，排除投机解码（2026-10-06）


> ✅ **最终实践结论见 `docs/ANSWER-STABILITY-VIA-CHAT-API-20261007.md`**：
> 用服务实际接口（`/v1/chat/completions`）实测 —— **结构化任务（抽取/事实/算术）逐字一致且 100% 正确**，
> 开放生成仅措辞差异（前 8 字一致）。本文的数值层观测仍成立，但**不构成使用层面的风险**。
> 另：本文若出现 `/v1/completions` 裸 prompt 的结论，属**接口误用（OOD）**，以该文为准。


> ⚠️ **本文 §3 的「非确定性是 TP=8 特有」被后续发现更正**：tp8k5 与 tiny 跑的是**两套不同的
> attention 代码**（tp8k5 = 镜像烘焙版 1092 行 / 0 探针；tiny = 开发版 4636 行 / 71 探针）。
> 因此该对比**同时**被「TP 规模」与「代码版本」两个变量污染。另：本文 §4 引用的
> `det_reduce` / `_v41_ordered_allreduce` 等记录**只存在于 tiny 的开发版**，
> 不在 tp8k5 的运行代码里。详见 **`docs/CODE-FIDELITY-tp8k5-vs-tiny-20261007.md`**。

> 上一轮发现 tp8k5（TP=8）在 temperature=0 下生成结果非确定，而 tiny（TP=2）逐位为 0 差。
> 本轮用**两个零重启实验 + 一个判别性重启**把它定位到具体区段。
> 全部为【实测】，推断处已标注。

## 0. 结论

| 判据 | 结果 |
|---|---|
| **prefill 是否确定** | ✅ **完全确定**（1240 个位置 × 8 轮，max\|Δlogprob\| = **0.000**，argmax token 全同） |
| **decode 是否确定** | ❌ 不确定，**从第 0/1 个 token 起就分叉** |
| **是否由投机解码引起** | ❌ **不是**：`SPEC=0`（`speculative_config=None`）下**抖动照旧** |
| tiny（TP=2）对照 | ✅ 逐位 0 差 |
| 【推断】根因 | **TP=8 的 decode 段集合通信（HCCL allreduce）不保证定序**；prefill 的规约形状不同，恰好稳定 |

## 1. 先排除 prefill（零重启）

`tools/gate_prefill_det.py`：短 prompt（1240 token）、`prompt_logprobs=1`（每位置只取 top-1 ⇒ 输出有界），
同 prompt 连跑 8 轮，逐位置比较 **(argmax token_id, logprob)**：

```
r01 vs r00: max|Δlogprob|=0.000e+00  首个偏离=None  偏离位置数=0  argmax token 不同数=0
r02 vs r00: max|Δlogprob|=0.000e+00  ...          0
…（7 轮全部 0）
⇒ prefill 完全确定
```

（这一点很关键：它证明**框架、算子、KV 写入、cache 布局在 prefill 路径上都是确定的**。）

## 2. 定位 decode 的分叉点（零重启）

`tools/gate_decode_diverge.py`：同 prompt、`max_tokens=48`、`logprobs=1`，8 轮，逐 token 比较。

**热轮内部**（r01..r07，全部命中同一 prefix cache）：

| token# | max\|Δlogprob\| | token 身份不同的轮数 |
|---:|---:|---:|
| 0 | 2.00e-01 | **0**（token 相同，logprob 已不同） |
| 1 | 2.55e-01 | **6 / 7** |
| 2 | 7.13e-02 | 0 |
| 3 | 3.78e-01 | 6 / 7 |
| 4 | **1.94e+00** | 6 / 7 |
| … | … | 6~7 / 7 |

内部两两：最大 \|Δlogprob\| = **1.97**（r1/r5 @token11），token 不同的 (轮对×位置) 数 = **491**。

⇒ **decode 从第 0~1 个 token 就开始分叉**，且**第 0 个 token 的 logprob 就已经差 0.2**
（而 prefill 同位置的 logits 被证明是确定到 0.000 的）。

## 3. 判别性重启：`SPEC=0` 下抖动照旧（决定性）

为区分"主干抖动"与"投机解码放大"，用 `SPEC=0` 重启（`DRAFT_GRAPH=0`）后跑同一个门：

```
speculative_config=None                                    ← 确认已关闭
热轮内部两两：最大 |Δlogprob| = 1.7903e+00（r2/r5 @token4）
             token 不同的 (轮对×位置) 数 = 584
```

对照（`SPEC=1`，同一台机器、同一门）：

```
热轮内部两两：最大 |Δlogprob| = 1.9692e+00（r1/r5 @token11）
             token 不同的 (轮对×位置) 数 = 491
```

⇒ **两种配置下抖动同量级** ⇒ **投机解码不是根因**；
抖动属于 **decode 主干**。

### 3.1 为什么排除了"投机放大"

若根因是投机解码（draft 不确定 ⇒ 接受集变化），`SPEC=0` 应当让抖动**归零**。
实测不仅没有归零，量级几乎相同 ⇒ 抖动源在**每一步都必须执行的集合通信**上。

## 4. 【推断】根因与为什么 prefill 不受影响

**本仓代码里已有同类记录**（`dsa_v41.py`）：

> "分叉只能来自**归约的求和顺序**（HCCL all_reduce 在该形状/T 上顺序可变）"

这正是我们为 DCP 路径实现 `_v41_ordered_allreduce`（定序求和）的原因。
tp8k5 的 O-Projection、MoE（shared/routed/final）用的是普通
`tensor_model_parallel_all_reduce`，**没有定序保证**。

【推断】**为什么 prefill 确定而 decode 不确定**：

| | prefill | decode |
|---|---|---|
| 每步 shape | T 大（数百~数万行） | **T 小（≈1~2 行/请求 × 并发）** |
| allreduce 次数 | 每次 prefill 少 | **每步 91.5 次**，且跨 step 累积 |
| 算法/切分 | 大张量 ⇒ 稳定路径 | 小张量 ⇒ 可能走不同规约分支/树形，且顺序随时序抖动 |

⇒ 小张量的规约更容易表现出非定序，而 decode 每步要做 91 次、还要在 40 层间累积。

## 5. 可立即验证的两条修法（尚未执行）

| 修法 | 成本 | 说明 |
|---|---|---|
| **把 O-Projection / MoE 的 allreduce 换成定序实现** | 中 | 本仓已有 `_v41_ordered_allreduce`（DCP 路径在用），可复用同一模式 |
| **临时用 `--enforce-eager` 或单卡路径复测** | 低 | 用于确认"抖动确实来自集合通信"（eager 不改规约顺序，但可排除图捕获因素） |

【未确认】是否所有 91.5 次 allreduce 都需定序，或只有其中若干（如 O-Projection）贡献主要抖动 ——
需要分项开关做归因。

## 6. 对已有结论的影响

1. **门①的定义必须修正**：TP=8 下"逐位 max\|Δ\|=0"不是当前实现的自然属性。
   建议改成：
   * **强判据**（定序改造 + 单卡/tiny）：max\|Δ\|=0；
   * **弱判据**（现行交付）：同 prompt 多轮**文本一致率** + 长文针准确率。
2. **吞吐类结论不受影响**（`ms/step`、`tokens/step` 是统计量）。
3. **`MAX_SEQS=64` 的 +32~37% 仍成立**，但**在非确定性定性之前不宜写进交付默认**。
4. 这解释了此前多次"同配置两次结果不同"（冷/热 prefill 差异、接受长度 4.97 vs 1.95）。

## 7. 复现

```bash
# 1) prefill 确定性（零重启，有界输出）
ssh a3-21 'python3 ~/tmp/gate_prefill_det.py http://127.0.0.1:19210 256 8'
# 2) decode 分叉点（零重启）
ssh a3-21 'python3 ~/tmp/gate_decode_diverge.py http://127.0.0.1:19210 48 8'
# 3) 判别性重启：关投机解码
SPEC=0 DRAFT_GRAPH=0 MAX_SEQS=32 bash scripts/serve_a2.sh      # 注意 KV32 守卫需显式压池
ssh a3-21 'python3 ~/tmp/gate_decode_diverge.py http://127.0.0.1:19210 48 8'
# 4) tiny 对照（TP=2）
ssh a3-21 'python3 ~/tmp/gate_bit_exact2.py http://127.0.0.1:19310 8000 6 32'
```

> 附：`SPEC=0` 会让可用 KV 变大（少了 draft 模型占用）⇒ 触发 `[KV32]` 池越界守卫，
> 需按脚本提示显式 `KV_CACHE_MEMORY_BYTES=15728022528` 压池后重跑。

## 8. 环境状态（按约束恢复并核验）

本轮共 3 次起服（SPEC=0 首轮被 KV32 守卫拒绝 → 压池重试成功 → 恢复交付配置）：

| 项 | **恢复后** | 原始交付基线（restore11） | 一致 |
|---|---|---|---|
| tp8k5 health | **200** | 200 | ✅ |
| **GPU KV cache size** | **2,987,509 token** | **2,987,509 token** | ✅ **逐位相同** |
| max_num_batched_tokens | 8192 | 8192 | ✅ |
| max_num_seqs | 32 | 32 | ✅ |
| capture_sizes | `…96,192` | 同 | ✅ |
| SP_TOKENS | 5 | 5 | ✅ |
| speculative_config | `dspark`（SPEC=1） | 同 | ✅ |
| 1M 上下文并发 | 2.85× | 同 | ✅ |
| 功能抽查 | `'水的化学式是H2O。…'` 正常生成 | — | ✅ |

| 其他 | 状态 |
|---|---|
| tiny（a3-21 chip2/3） | health=200，**全程未动** |
| a3-21 chips 0–3 | 仍为他人负载（`hlz-dsv4-dp2`）+ 我们的 tiny，未受影响 |
