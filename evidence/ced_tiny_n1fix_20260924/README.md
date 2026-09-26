# A3-22 CED 单 token 边界修复验证

日期：2026-09-24。修复仅在独立 worktree/远端临时包验证；源码提交为
`fix/ced-single-token-boundary@bfe00c1`，基于 `2df46ec`。scheduler 原件来自 A3-22
P 容器，SHA-256 为
`533eed493cb307e6d4423ff550910278f6434d71f00581737ce420d60298e8bc`。
新 `core_scheduler_replay.patch` 由完整原件和修改后 scheduler 的
`git diff --no-index --unified=0` 生成，并在原件副本上重新 apply、grep
新旧关键分支、`py_compile`；A3 镜像内先应用 admission gate 后，CED patch
检查与实际应用也通过。patch SHA-256：
`ab84b5e2ff4ee5951ec662a30be29b38df4b7b5a386991028906fb0776a4572c`。

验证使用 A3-22 Phy-ID 6 的既有 P 和 Phy-ID 7 顺序运行 D/全 40 层 baseline，
dummy `model-tiny`、BF16 KV、seed 0、单条顺序请求；chip0/1 未使用。D 在
`prompt=loaded=declared=1` 时将 `num_computed_tokens` 退回0、设
`ced_replay_end=1`，重算唯一 prompt token。`max_query_len=1` 不进入有界
replay fast path，因此层20执行正常全局源写入。

| prompt 长度 | PD 与 baseline | top-20 数值 | D 结果 |
|---|---|---|---|
| 1 | 均 HTTP 200，选中 token `Apart`、选中 logprob 完全相同 | 20/20 相同，最大共同 logprob 差 0 | `prompt_tokens=1`、`completion_tokens=1`、`total_tokens=2`，只生成一个 token；finish reason=`length` |
| 2 | 均 HTTP 200，选中 token 与 logprob 相同 | 20/20 相同，最大共同 logprob 差 `9.54e-7` | replay 日志 `0..0`，正常完成 |
| 127 | 均 HTTP 200，选中 token 与 logprob 相同 | 20/20 相同，最大共同 logprob 差 `9.54e-7` | replay 日志 `0..125`，正常完成 |
| 128 | 均 HTTP 200，选中 token 与 logprob 相同 | 20/20 相同，最大共同 logprob 差 0 | replay 日志 `0..126`，正常完成 |
| 129 | 均 HTTP 200，选中 token 与 logprob 相同 | 20/20 相同，最大共同 logprob 差 0 | replay 日志 `0..127`，正常完成 |

D 日志有一条 N=1 `single-token prompt: recompute position 0 for logits`，随后
四条旧边界 replay 均正常；未见 prefix mismatch、Traceback 或多生成 token。
完整 API 响应、top-20 对照、容器 inspect 和 D/baseline 日志在本目录归档。

本次证明该 scheduler 边界修复能通过 A3-22 tiny dummy 的单 token API 和
相邻边界对照；真实权重仍需单独验收。
