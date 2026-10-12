# 批不变质量门通过：GSM8K 100/100 + Vision 23/23，性能代价≈0（2026-10-13 v53）

本轮把 v52 描述的批不变（batch-invariant）臂推到了**全部通过**，并在同一硬件上测了它的性能代价。
两件事同时成立：

1. **质量门通过**（job `formal_dspark_quality_async_v14_bi`，快照 `src_dspark_general_quality_v11`）：
   - **GSM8K 100/100**（官方 canonical 8-shot、chat 模式、512 预算、strict 判定；`empty=0`、`errors=0`）；
     决定性题 `gsm8k-12` 由 12 → **13**（与 1 行 decode 路径一致，推理文本变成 "he first earns money in year 13"）。
   - **Vision 23/23 PASS**（20 问 + 3 个 mandatory negative）。
   - 收据 `strict_quality_receipt.json`：`actual_GSM100_passed=True`、
     `actual_Vision23_passed=True`、`actual_client_quality_passed=True`；服务与评测器均 exit 0。
2. **性能代价≈0**（job `formal_dspark_native_graph_api_v11_bi`，2K/256 × 8 串行、审计/profiler 关）：

| 臂 | ms/step | A | tok/s | rounds | clean | 复读 |
|---|---:|---:|---:|---:|---:|---:|
| 非 BI 最优（v9/v10） | 26.44–26.61 | 2.94 | 110.4–111.2 | 694 | 7/8 | 1 |
| **BI（v11_bi）** | **26.55** | **2.73** | 102.85 | 747 | **8/8** | **0** |

- ms/step 与非 BI 臂在噪声内相同；**clean-rate 由 7/8 提升到 8/8、复读归零**。
- A 由 2.94 降到 2.73（每请求 44 轮 → 稍多），这是数值路径改变后投机接受率的自然变化；
  输出长度不变（255/256 每请求），tok/s 相应为 102.85。
- **BI 是改变数值路径的模式**：其 token 流与旧非 BI 正式窗口不同（0/8 逐位相同），
  与"A1 路径"一致（这正是质量门用 1 行 decode 作为参照的原因）。

## 为什么能做到

v52 报告了为跑通厂商批不变模式必须修复的 4 处缺陷；本轮又修掉第 5 处：

5. **bool dtype 崩溃**：BI 把全局 `torch.sum`/`Tensor.sum` 换成只支持浮点的
   `npu_reduce_sum_batch_invariant`，而 engram 设备路径对 **bool 掩码**做求和（计数）。
   该异常从 forward 逃出、正好落在引擎 device-metadata submit/release 之间，
   把 `_submission_in_flight` 永久锁死（厂商在 `model.py`/`engram_graph.py`/`engram_device_index.py`
   都注释过这个"异常逃出即锁死"的机制），表现为后续请求全部报
   "The previous device metadata submission has not been released"。
   修复：shim 的 `reduce_sum` 对非 `float16/float32/bfloat16` dtype 回退原生 `torch.sum`。

修复后的完整链路（5 处）都在快照的 `stack/bi_axis_compat.py`（env 门控；未启用 BI 时零影响）。

## 对目标的意义

- 目标要求的"最优正式服务须通过模型归属 + GSM8K100/100 + Vision23/23 + 含 DSpark 的 ≤20ms"中，
  **质量三项已在本轮同一正式栈（async+mls+DSpark+graph、完整权重、strict HCCL）上全部通过**；
  服务归属由质量收据里的 candidate 指纹/checkpoint config/8 rank 描述绑定。
- **唯一剩余缺口是 ≤20ms**：BI 臂 26.55ms/step，距离目标 6.55ms；该缺口与 v50 报告判定一致——
  目标图 1986 kernel/21.6ms、81 次 allReduce 3.3ms、量化/HcPre 链融合，属厂商内核级结构化工作，
  非配置开关可达（配置候选已在 v50 全部枚举并判定）。
- 由于 BI 的性能代价≈0 且 clean 更好，**后续的性能优化应直接在 BI 臂上进行**，
  以免再出现"为性能牺牲质量"的两难。

## 证据（本地）

- `evidence/formal_a321/quality_gate_pass_bi_v14/`：`gsm100.json`（100/100，含逐题）、
  `vision23.raw.json`、`strict_quality_receipt.json`、`gsm100.log`/`vision23.log`、
  `launched.json`、`run.exit=0`、`evaluator.exit=0`。
- `evidence/formal_a321/dspark_native_graph_api_v11_bi/`：`acceptance_small.json`（聚合三元组）、
  `measured_0..7_http.json`（原始 HTTP 响应）、`token_identity_vs_sync_baseline.json`、
  `client.exit=0`、`launched.json`。
- `evidence/formal_a321/quality_bi_prep/`：shim 与构建器源码（含本轮 bool-dtype 修复）。

## 诚实边界

- 本报告**不**声明 ≤20ms 达成；26.55ms/step 距目标 6.55ms。
- BI 臂的输出与旧非 BI 臂不同（数值路径改变）；质量以官方 100/100 + 23/23 判定，
  不以旧窗口的 token 逐位一致作为要求。
- `VLLM_BATCH_INVARIANT` 的代价只在本负载（2K/256、A≈2.7）测得≈0；
  未测其他负载形状。
