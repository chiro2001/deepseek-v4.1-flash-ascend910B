# 补充：prefill 在 **T=32 / 1024 / 1240** 下都逐位确定 ⇒ 排除「小 T 规约」假设（2026-10-07）

> 承接 `TP8-DECODE-NONDETERMINISM-ROOTCAUSE`。该文把我此前的假设收敛到
> 「TP=8 decode 的小张量 allreduce 不定序」。**本轮实测把它否掉了。**
> 全部为【实测】，运行在**交付配置**（graph 模式、TP=8、镜像版 attention）。

## 1. 结果

同一门（`tools/gate_prefill_determinism.py`：同 prompt、`temperature=0`、
`prompt_logprobs` 逐位置比 **(argmax token_id, logprob)**、6 轮）：

| prompt T | max\|Δlogprob\| | argmax token 不同数 | 判定 |
|---:|---:|---:|---|
| **32** | **0.000** | 0 | ✅ 完全确定 |
| **1024** | **0.000** | 0 | ✅ 完全确定 |
| 1240（上一轮） | **0.000** | 0 | ✅ 完全确定 |

而 decode（每请求 T≈1、每步总 T≈6~100）**不确定**（热轮内部两两 max\|Δ\|=1.97、token 差 491 处）。

## 2. 这否掉了什么

我上一轮的假设是：
> "decode 每步 shape 小（≈1~2 行/请求）⇒ HCCL allreduce 在小张量上走不同规约分支 ⇒ 不定序"

**T=32 的 prefill 同样是小张量，却逐位确定** ⇒ 单纯的"shape 小"不足以解释。
而且 prefill 与 decode 走的是**同一个 TP 组、同一批 RowParallelLinear allreduce**
（O-Projection、MoE shared/routed/final）⇒ **TP allreduce 本身不是抖动的充分原因**。

## 3. 这留下什么（候选收窄到"decode 独有"的东西）

| 候选 | 是否只影响 decode | 依据 |
|---|---|---|
| **capture 图 replay 语义** | ✅ | decode 走 `FULL_DECODE_ONLY` 捕获图；prefill 不走图（eager/piecewise） |
| **SWA / KV 的"同 forward 内先写后读"竞态** | ✅ | decode 每步写入当前 token 的 KV 并立刻被本层注意力读；本仓开发版里已有 [V41-SYNCATTN] 记录同类竞态 |
| **decode 专属算子**（`npu_sparse_flash_mla` 的 decode 布局、QLI top-k 并列 tie-break） | ✅ | decode 与 prefill 的 metadata/算子选择不同 |
| ~~TP allreduce 小张量不定序~~ | ❌ | **被 T=32 prefill 的确定性否掉** |
| ~~投机解码~~ | ❌ | 已被 `SPEC=0` 判别实验否掉（抖动同量级） |

### 3.1 两条实测的支持性证据

1. **graph 放大**：同一门在 **eager** 下（TP=8、SPEC=0、STATIC_KERNEL=0、NPUGRAPH_EX=0）
   decode 的 token 身份**0 处不同**，logprob 抖动 0.405；
   在 **graph** 下 token 差 491~584 处、抖动 1.97。
   ⇒ 抖动源在 eager 下已存在（小），**图模式把它放大**。
2. **prefill 在任何 T 都不抖** ⇒ 抖动不是"每步都会发生"的全局属性，
   而是 **decode 这一条路径特有**。

## 4. 下一步（可执行的判别）

| 步骤 | 目的 | 成本 |
|---|---|---|
| 在 **CED 模式**起服（会挂载 `experimental/ced/dsa_v41.py`，1455 行）再跑同一门 | 若 CED 版也抖 ⇒ 与 attention 实现版本无关；若 CED 版不抖 ⇒ 差异在实现里 | 一次重启 |
| 在**交付配置**下用 `PROBE=1` 起服，用 `sparse_capture.py` 抓 decode 的 KV/索引快照 | 直接看"写进去的 KV"与"读出来的"是否一致 ⇒ 判竞态 | 一次重启 |
| 把 dev 版的 `_perf_flags` 探针**移植到镜像版/`experimental/ced` 版** | 让 tp8k5 具备与 tiny 同等的诊断能力 | 改代码 |

【建议】优先第 1 条（只需一次重启，且能同时回答"版本差异 vs TP 规模"这个被污染的问题）。

## 5. 环境状态

本轮结束时 tp8k5 已恢复到**交付配置**并核验：health=200、KV 2,987,618 token、
BAT 8192、MAX_SEQS 32、SP_TOKENS 5、`capture_sizes=…,96,192`、dspark 投机解码开启、
1M 上下文并发 2.85×。tiny（TP=2）health=200、全程未动。

## 6. 复现

```bash
ssh a3-21 'python3 ~/tmp/gate_prefill_det.py http://127.0.0.1:19210 32 6'
ssh a3-21 'python3 ~/tmp/gate_prefill_det.py http://127.0.0.1:19210 1024 6'
ssh a3-21 'python3 ~/tmp/gate_decode_diverge.py http://127.0.0.1:19210 48 8'
```
