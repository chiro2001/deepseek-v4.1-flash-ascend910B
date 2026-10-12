# DSpark 贪心路径 vs 1 行 decode 的 gsm8k-12 分叉：完整归因（2026-10-13 v51）

结论先行：**分叉由"量化网格上的近并列步"造成，不是 accept/rollback 语义错误**。
DSpark 侧的逐轮收据显示每一步的发出 token 都是其自身上报分布的 top-1，
接受的 draft 等于 verifier 的 argmax；分叉点是 verifier 的 bonus token（第 19 个答案 token），
该步 DSpark 上报分布 top-1/top-2 **精确并列**，而 A1（1 行 decode）在同一上下文里 top-1 领先 0.25。

## 1. 三个臂的答案

| 臂 | 路径 | gsm8k-12 | 全量 |
|---|---|---|---|
| DSpark async+mls (`formal_dspark_quality_async_v2`) | 6 行 verify（K=5） | **12** | 99/100 |
| DSpark 冻结 sync (`formal_dspark_quality_sync_ref_v1`) | 6 行 verify（K=5） | **12**（文本与上一行逐字节相同） | 99/100 |
| A1 诊断 (`formal_dspark_a1diag_v1`，`speculative_config=None`) | 1 行 decode | **13** | **100/100**（271.7s） |
| 历史 A1（无 DSpark，`formal_best_async_service_v5/v4`） | 1 行 decode | 13 | 100/100 |

预测层面 100 题中只有 gsm8k-12 不同；原始文本 55/100 存在**细微分叉**
（首个分叉 token 位置 1..152，中位 31；≤5 有 5 题），无系统性位移/错位特征。

## 2. 分叉点（精确重建 prompt：官方编码器 chat、1206 token、rendered_chars 4113）

单 prompt 同步 DSpark 诊断（`formal_dspark_gsmdiag_v1`，64 token、K=5、draft-graph、
逐轮收据 + top-8 logprobs）：

- 前 18 个答案 token 与 A1 逐位一致；
- 第 19 个 token：DSpark 发出 **671("The")**，A1 发出 **58559("Carl")**；
- 该 token 是第 5 轮的 **bonus token**（verifier 自己的 argmax；轮次流相对输出流少第一个
  prefill 后 token，属于既有边界约定）。

**同一步上报分布（top-2）**：

| 路径 | top-1 | top-2 | 间距 |
|---|---|---|---|
| DSpark（6 行 verify） | 671 "The" **−0.918505** | 58559 "Carl" **−0.918505** | **0.0（并列）** |
| A1（1 行 decode） | "Carl" **−0.77259** | "The" **−1.02259** | 0.25 |

- 两条路径的 top-k logprob 差都落在**同一量化网格**上（0.125 的整数倍，含共同偏移），
  即该 stack 的 logits/logprob 以约 1/8 步长呈现（与 W4A8/qrot 量化输出一致）；
- DSpark 侧顶部两 token 恰好同为 0 步 → 以最低索引 671 打破并列；
  A1 侧 "Carl" 领先 2 步 → 选 "Carl"。后续推理走向不同结论（12 vs 13）。

## 3. accept/rollback 语义核查（诊断收据）

- 第 i 步发出 token 均为其 logprobs[i] 的 top-1（对 i=16..25 全查，且整流如此）；
- 逐轮：`scheduled_draft_token_ids` 与 `accepted_draft_tokens_raw` 一致地解释每个发出 token
  （accepted 位置= draft 值；bonus 位置=回退采样值）；
- 轮次流与输出流的唯一差异是**首个生成 token 不在轮内**（prefill seed 边界），与既有约定一致；
- 未发现"接受未验证 draft"或"跳过位置"的迹象。

⇒ 这是**数值层**问题：两条计算路径（1 行 decode vs 6 行 verify）在同一量化网格上
给出至多一步（0.125）的差别，恰好在该题把并列从"2 步"变成"0 步"。

## 4. 证据（本地）

`evidence/formal_a321/divergence_audit_20261013/`

- `divergence_summary.json`：三臂原文/预测、首个分叉字符（61）；
- `token_divergence_distribution.json`：100 题 token 级分叉分布；
- `dspark_diag_raw_request.json`（64 token + top-8 logprobs）、`dspark_diag_result.json`（18 轮）；
- `a1_logprob_probe.json`（A1 同 prompt 24 token 的 top-8 logprobs，含第 19 步）、`gsm12_prompt.json`
  （重建 prompt：1206 token、rendered sha）；
`evidence/formal_a321/a1_control_20261013/gsm100.json`：A1 100/100 原始输出；
`evidence/formal_a321/quality_gate_runs/`：两 DSpark 臂的质量收据。

远端大件：`results/formal_dspark_gsmdiag_v1/gsm12_counters.json`（全 8 rank 收据）、
`results/formal_dspark_a1diag_v1/`（服务选择与 gsm100 原始）。

## 5. 对目标的影响与后续选项

- 目标要求"最优正式服务（含 DSpark）通过 GSM8K100/100"。当前证据表明该 1/100 差异来自
  **量化网格上的近并列被两条路径分别打破**，并非优化引入（冻结 DSpark 基线同样 99/100）。
- 可选后续（按代价）：
  1. 继续性能主线（≤20ms 仍差 6.6ms，全部在设备侧）；
  2. 与用户确认质量门口径：接受"99/100 + 完整根因证据"，或要求模型级修改
     （如提高 LM head/logits 精度，属权重/格式层面，需重新验证全部数值门与性能）；
  3. 评估"仅对判定步做 1 行复算"的保语义修正（仅在近并列时触发）——实现代价高、需重新审计；
  4. 复核该 stack 的 logits 量化开关是否存在且可关（本轮未发现显式配置）。
- 资源：三个诊断服务均已退出；physical 8–15 无我方残留（另有他租户 NPU0–2 进程，未触碰）。
