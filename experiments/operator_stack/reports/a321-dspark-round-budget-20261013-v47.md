# DSpark 正式轮预算：宿主/设备分解与结构性差距（2026-10-13 v47）

本轮目标仍是「包含 DSpark 全部开销 ≤20ms/step」，**未达成**。本轮完成三件事：
宿主相位探针（新的可复验诊断）、逐轮预算分解、两个候选的实测结论（一个中性、一个被合同阻塞）。

## 1. 宿主相位探针（新增诊断设施，env 门控、零数学改动）

新快照 `src_dspark_native_hostprobe_v1`（= draft-graph 快照 + 探针）：

- `stack/host_span_probe.py`：在 counter 安装后**外层**包裹 `execute_model / sample_tokens /
  _prepare_inputs / _calc_spec_decode_metadata / _bookkeeping_sync / propose_draft_token_ids /
  _sample / drafter._propose / block_table.compute_slot_mapping`，只做 `time.monotonic_ns()`
  计时与调用序列记录；**不读张量、不同步设备、不改数学**。
- 门控 `DSPARK_HOST_SPANS=1`；调用序列（含起止纳秒）随 counter 收据一起返回，可与每轮
  `execute_start_ns..sample_end_ns` 窗口离线对齐（`align_round_spans.py`）。
- 冻结：freeze 17 文件（新 counters `3dcc924d…`、probe `c1ee90f0…`）。

AA 审计屏在带探针下仍然通过（3/3 对、tokens/routes/logprobs 全等），证明探针不改变语义。

## 2. 逐轮分解（AA 审计屏 rank0，8 个解码轮，A=5.75，探针开启）

| 项 | 每轮 ms | 说明 |
|---|---:|---|
| 轮窗口 | 40.298 | execute_start→sample_end |
| `execute_model` 宿主跨度 | 11.579 | 内含下面两项 |
| ├ `_prepare_inputs` | 2.240 | 槽位映射 12 组、位置、块表等 |
| ├ `_calc_spec_decode_metadata` | 0.418 | |
| └ 其余（目标图提交/前段设备） | 8.921 | |
| `sample_tokens` 宿主跨度 | 28.565 | |
| ├ `_bookkeeping_sync` | 1.106 | D2H（已采样 token 回读） |
| ├ `propose_draft_token_ids` | 3.676 | 其中 `drafter._propose` 2.963 |
| └ 其余（`_sample` 等待与拒绝采样、终端 fence） | 23.783 | |
| 步间 gap | 0.154 | 很小 |

补充：`_sample`（采样+拒绝采样入口）本身平均 18.5ms/次——它包含**等待目标图 logits 的设备时间**，
不是纯宿主。结合设备侧剖析（上轮）：目标图 21.6ms、草稿 6.8ms、验证 0.4ms、准备核 2.8ms。

**结论**：每轮 ≈ 设备 30–33ms + 宿主可串行段 ≈ 6.5–7.5ms + 设备等待间隙。轮窗口以设备为主，
宿主暴露量 ~7ms（其中 `_prepare_inputs` 2.2、`_propose`/`propose_draft` 3.7、其余 ~1）。

## 3. 与无 DSpark 基线的对照（量级标尺）

- 无 DSpark mask 服务：`[bneck] hp=19.226`、客户端 `(19.289956, A=1, 51.840451)`（n=1 行图）。
- 本 DSpark 候选：`[bneck] hp=41.006`、客户端 `(46.48740792151163, A=2.9651162790697674, 63.783213812996586)`（n=6 行图）。
- **每 token**：DSpark 15.68ms/token < mask 19.29ms/token（已更优）；
  但 ≤20ms/step 且 A≈2.97 ⇒ **6.74ms/token**，比现状再快 2.3×。
- 结构性门槛：即使把宿主暴露全部消除（−7ms），轮窗口仍 ≈33ms；≤20ms 必须让设备侧
  （目标图 21.6 + 草稿 6.8 + 验证/准备 3.2）整体减半。这是一项结构化工程
  （通信次数、MC2/融合算子、草稿图瘦身），不是单开关可达。

## 4. 候选实测账

| 候选 | 结果 | 处置 |
|---|---|---|
| `V41_SLOT_MAP_FUSED=verify` | 融合/逐组双跑逐元素一致；8×256 token 与基线逐位一致 | 数值门通过 |
| `V41_SLOT_MAP_FUSED=1` | 设备 `hp` 41.006→39.797（−1.21ms），客户端 46.504 vs 46.487（**噪声级无收益**） | **不部署**（新增宿主侧融合准备抵消设备收益） |
| `V41_MOE_COMM_ALLGATHER=0`（走 MC2/FUSED_MC2） | 在 `route_capture_patch.capture` 断言 `(255, 2044)` 失败：MC2 把 prefill 2044 token 按 floor 分片（255/rank），而正式路由审计合同要求 `ceil(total/tp)` 的完整路由分片 | **被审计合同阻塞**；要测需先重做路由导出合同（新快照+重新验证），本轮未做 |

证据：`evidence/formal_a321/hostsplit_hspan/`（探针脚本、aa_v5/v6 回执与请求）、
`evidence/formal_a321/dspark_native_graph_aa_v7_moecomm_rejected/`（MC2 失败原文）。

## 5. 下一步（按期望值排序，均需设备验证）

1. **目标图通信收缩**（最大单块 6.12ms/160 次 allReduce）：评估把 4 次/层收敛为 2 次（
   共享专家与路由专家合并归约）或改走融合 MC2；需先解决路由审计合同的自适应（接受 floor/ceil 分片）。
2. **草稿图瘦身**（6.8ms/轮）：草稿 3 层 × 4 collectives；评估更少通信或 K 自适应。
3. **宿主 7ms**：`.item()`/阻塞 H2D 清理（storm 归因，0.3–1ms）与 `_propose` 宿主路径简化。
4. 保持：draft-graph 候选（46.49）为当前最优；slot-map 融合默认关。
