# GMM2/M1/K5 与并行 tiny 算子候选历史审计

日期：2026-10-13。此文只汇总既存冻结报告与回执；本次没有重跑测试、访问 NPU/服务或更改证据。当前 `field_gate/` 与 [近期 a3-21 算子复核](history_recent.md) 保持原样。

## 结论

- **不建议重复 M1 数值门、M1 event 窗口或 K5 数值/最小 timing 门。** M1 v5 已有固定 M1 几何的逐位成功门，K5 已有 target/draft 两种正式几何的设备数值门；它们都不是完整模型消费者证明。旧 timing 已明确显示没有一致收益（M1）或 candidate 整体变慢（K5），复跑相同几何不能补上这两个边界。
- **没有可由历史结果直接准入的 GMM2 稀疏 MSD candidate。** 真实 route 投影中，M1 event 唯一稳定有利的单 e47、R∈{1,3,6} 模式占 13,440 个观察的 32 个（0.238%）；route 投影不是设备 GMM2 counts 或 consumer proof，draft route coverage 还是 0。K5 的 target/draft 容量分别为 M36/E48 与 M15/E16，与 M1 的 6 槽几何不同。
- 若以后重新授权并确有决策需要，唯一不重复的 GMM2 工作是**审查并观测当前 full43 native consumer/ABI/lifecycle 接缝**：selected overload/schema、真实 role/count、原始 ND `x` 的 pack 前后字节与恢复、借用 stream/workspace、真实 copy-before-pack 顺序、capture/replay consumer 覆盖。当前没有窄化的现成 launcher；full43 诊断入口至少要跑 32 generations，必须先审查冻结源与资源计划，不能称作轻量单算子测试。
- 本审计没有检查或占用 physical 8–15，不能据此判断其实时可用性。用户已另行授权本轮使用这八个 chip；现场状态以 `field_gate/` 的宿主 NPU 快照和执行前复核为准，不能沿用暂停前的外占判断。

## GMM2 M1：数值成功不等于性能或消费者通过

| 阶段 | 已冻结的观察 | 可支持的结论 / 边界 |
|---|---|---|
| v1–v3 | v1 在 format29 guard 前终止；v2 `ACL_MEM_MALLOC_NORMAL_ONLY` 申请 OOM，SDK/raw/candidate/planner 数学调用均为 0；v3 SDK R1/e0 成功且 GM 参数字节验证通过，但 workspace base `%512=128`，raw kernel 未启动。 | 三次是不同的准备/布局阻断，不能写成 candidate 数学失败，也不能计作精度或性能样本。见 [`m1_numeric_v1/README.md`](../../parallel_tiny/gmm_lane/sparse_msd/m1_numeric_v1/README.md)、[`m1_numeric_v2/README.md`](../../parallel_tiny/gmm_lane/sparse_msd/m1_numeric_v2/README.md)、[`m1_numeric_v3/README.md`](../../parallel_tiny/gmm_lane/sparse_msd/m1_numeric_v3/README.md)。 |
| v4 | 真实 E48 layer0/rank0 checkpoint W2；SDK 与 raw native 各执行一个 R1/e0 case，但原 1 ULP 门失败：5119/5120 项不同、5112 项超过 1 ULP、最大 32,253 ULP。随后停止，candidate/planner 为 0。 | 保留为不可覆盖的负结果；失败不是 K5 结论。见 [`m1_numeric_v4/README.md`](../../parallel_tiny/gmm_lane/sparse_msd/m1_numeric_v4/README.md) 与 [`first_numeric_differences.json`](../../parallel_tiny/gmm_lane/sparse_msd/m1_numeric_v4/evidence/runtime_v4_terminal/first_numeric_differences.json)。 |
| v5 | 原始 `x` 在每次 SDK/raw/candidate 调用前都恢复到同一 GM 指针并验证字节；9 个 R1/R3/R6、e0/e47、混合 count、六槽 case 的 SDK/raw/candidate 均 max 0 ULP，planner frame、canary、unused rows 与 36 组参数 seal 通过。使用真实 W2，activation/counts 为合成。 | M1 固定 capacity6 的设备数值门通过；未覆盖真实 SwigluQuant activation、route/cache/capture consumer、K5 verifier、正式 TP8 或性能。见 [`m1_numeric_v5/README.md`](../../parallel_tiny/gmm_lane/sparse_msd/m1_numeric_v5/README.md) 与 [`numeric_gate_status.json`](../../parallel_tiny/gmm_lane/sparse_msd/m1_numeric_v5/evidence/runtime_v5_success/numeric_gate_status.json)。 |
| event v2 | 同一 physical6 实例、synthetic activation、candidate 含 planner；9 case、每臂 3 warmup + 12 轮交错，共 324 个计时样本。raw 对比只有 3/9 case 的 operator-event 中位更快；fresh-input + event 只有 4/9 case 更快，终态 `MIXED_M1_EVENT_RESULTS_NO_UNIFORM_GAIN`。 | 有利点仅落在单 active local e47 的 R1/R3/R6 三例；事件包含队列/提交 idle，不能当纯 math 核时间、入图收益或整网预算。见 [`RESULT.md`](../../parallel_tiny/gmm_lane/sparse_msd/m1_event_perf_v2/RESULT.md)、[`event_statistics.json`](../../parallel_tiny/gmm_lane/sparse_msd/m1_event_perf_v2/event_statistics.json)。 |

## K5：几何真实，实际 counts/consumer 未证

- CPU 线为纯 CPU：3/3 编译单元、228/228 检查通过，`NPU_calls=0`，没有设备数值、性能、admission 或 formal receipt。主/draft 几何来自真实配置和 profile：target M36/E48/top6，draft M15/E16/top3；几何并不表示 observation 中实际发生了这些 counts。见 [`k5_capacity_numeric_cpu/REPORT_20261012.md`](../../parallel_tiny/gmm_lane/sparse_msd/k5_capacity_numeric_cpu/REPORT_20261012.md) 与 [`actual_GMM2_geometry_all_ranks.json`](../../parallel_tiny/gmm_lane/sparse_msd/k5_capacity_numeric_cpu/evidence/actual_GMM2_geometry_all_ranks.json)。
- 后续 physical4 设备门对 target/draft 各 5 个合法 case 与每几何 2 个非法 count case 均通过：frame 对冻结 oracle 逐字节相同，8 个非空正例的 SDK/raw/candidate 为 0 ULP。target 使用真实 checkpoint W2，设备 NZ SHA 与 M1 冻结物理权重一致；activation/counts 是固定合成输入，不是实际 MoE consumer。
- K5 最小 operator-window 每 case 3 warmup + 10 samples，candidate 含 planner；四个非空 case 的 candidate/raw 中位比分别为 **1.269、1.324、1.284、1.216**，即均慢于 raw。即使 `timing_passed=true`，receipt 仍为 false；窗口还包含队列/host-submit idle，不是纯 core 或 E2E 结果。见 [`REPORT_DEVICE_20261012.md`](../../parallel_tiny/gmm_lane/sparse_msd/k5_capacity_numeric_cpu/REPORT_DEVICE_20261012.md)、[`k5_capacity_device_numeric_20261012.json`](../../parallel_tiny/gmm_lane/sparse_msd/k5_capacity_numeric_cpu/evidence/k5_capacity_device_numeric_20261012.json)、[`k5_capacity_device_timing_20261012.json`](../../parallel_tiny/gmm_lane/sparse_msd/k5_capacity_numeric_cpu/evidence/k5_capacity_device_timing_20261012.json)。
- route feasibility 是对已接受 full-model route receipts 做的 CPU 投影：48 个名义 rounds 中 42 个完整、14 个唯一 route slices、13,440 个 target layer×rank observations；R 中位/均值/最大 4/4.5/19，active 中位/均值/最大 3/2.7213/10。M1 有利模式只占 32/13,440（0.238%），按唯一 slices 去重是 10/4,480（0.223%）。**这些不是设备 GMM2 group counts/prepack activation，也不是 candidate 选择收益；draft coverage 为 0。** 见 [`k5_route_feasibility_cpu/README.md`](../../parallel_tiny/gmm_lane/sparse_msd/k5_route_feasibility_cpu/README.md)、[`route_distribution_summary.json`](../../parallel_tiny/gmm_lane/sparse_msd/k5_route_feasibility_cpu/evidence/route_distribution_summary.json)。

## 其他 parallel_tiny 候选：通过的正确性门不代表值得采用

| 候选 | 历史结果 | 当前判定 |
|---|---|---|
| QKV merge（正式 V4.1） | 真实 TP8 consumers 和 token/route/logprob 审计通过；同实例 12 组配对 qkv 19.555371 ms vs mask 19.324495 ms，0/12 更快。 | 拒绝部署，无新源码/几何时不重复。独立投影微基准不能替代整网。见 [`v35 报告`](../../reports/a321-v41-qkv-audit-pass-20261011-v35.md)。 |
| early-KV-tail QKV | 正式 audit 通过；qkv 19.307216 ms vs mask 19.208722 ms，0/12 更快。 | 继续拒绝部署；不要把调度假设当性能结果。见 [`v36 报告`](../../reports/a321-qkv-tail-and-sdk-followup-20261011-v36.md)。 |
| IDS64 hoist | 路由/token/Top5/logprob 一致，真实每步 Cast 从 162 降到 123（少 39）；但 TP1 BF16 dummy tiny 的 12/12 配对均慢，配对中位 −0.108590 ms。 | 保持默认 off；Cast 计数下降不能外推正式 TP8。见 [`ids_lane/README.md`](../../parallel_tiny/ids_lane/README.md)。 |
| Engram 8-shard 恢复 | 真实 lookup 与消费者驻留重放；全消费者 1 ULP 门仅 39/48 组通过、9/48 失败，最大 96 ULP。精度失败后未计时（pairs=0）。 | 拒绝，不重跑相同分片算法。完整 MatMul 预计算是不同设计，只有新生命周期/消费者证据后才值得评估。见 [`engram_lane/README.md`](../../parallel_tiny/engram_lane/README.md) 与 [`precision_result.json`](../../parallel_tiny/engram_lane/evidence/precision_result.json)。 |
| BF16 QKV toy lane | 完整 tiny 被设备故障阻断；projection-only 24 输入合并输出最大 8 ULP，超过 ≤1 ULP 门，且请求 NZ 时发生 base-format 回退。 | 不将 toy 结果与正式 W4A8 QKV 合并；已拒绝，未做 candidate timing。见 [`qkv_lane/README.md`](../../parallel_tiny/qkv_lane/README.md)。 |

已有 full-model metastack / exact-mask 路线及 async/BI 当前最佳配置已在 [history_recent.md](history_recent.md) 单独审计，不列为新测候选。新 mask 的 A=1 收益不是含 DSpark 的证明；当前最佳 full40 DSpark+BI 的环境仍为 base arm，slot batching 与 exact mask 关闭。full43 私有诊断虽然完成 8×37 个 generation，但 `full_consumer_audit_passed=false`，不构成准入；CPU prepack 组件也明确尚未证明真实 selected overload、ND `x` 生命周期、workspace/stream 地址或 copy-before-pack 顺序。参见 [`full43 诊断报告`](../../reports/a321-full43-diagnostic-pass-20261013.md)、[`full43 gate result`](../formal_a321/full43_diag_v17/gate/result.json)、[`prepack readiness`](../../parallel_tiny/dspark_prepack_audit_cpu/readiness.json)。

## 再测门槛

不安排 M1/K5 的相同几何复测，也不复跑上述已拒绝的 tiny/正式负收益方向。若资源重新授权且要回答“GMM2 MSD 是否可以进入 full43”，应先通过 CPU/source review 冻结真实 consumer 接线及失败关闭契约，再以**独立 fresh full43 作业**证明 selected overload/schema、实际 counts 与 role、raw ND x 的前后内容和恢复时序、真实 stream/workspace/consumer copy-before-pack、完整 capture/replay 与数值输出；之后才决定是否值得单独测 timing。历史 M1/K5 的数值或 timing 不能填补这些门，也不能绕过 full43 的质量、admission 与 HTTP 门。
