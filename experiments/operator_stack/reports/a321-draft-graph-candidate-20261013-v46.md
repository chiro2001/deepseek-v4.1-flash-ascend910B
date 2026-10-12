# DSpark draft-graph 候选：AA 通过 + 正式客户端 46.49ms/step + 逐相位设备归因

2026-10-13 · a3-21 physical8–15 · 完整40层/hidden5120/384专家top6/W4A8_DYNAMIC/Engram int8/完整视觉/MTP3 · HCCL strict

当前 goal 仍是「包含 DSpark 全部开销 ≤20ms/step」，**尚未达成**。本轮把 tiny 线验证过的
draft 入图机制接进正式栈，拿到第一批正式数字与设备归因。

## 1. 接线方式（长驻容器不可加挂载 ⇒ 快照内 overlay + 门控 import hook）

- 交付件沿用已发布四件套 `patches/files/draft/{dspark_proposer,llm_base_proposer,dsa_v1}.py`
  （A2 线同实例 36.9→24.9ms/step、Vision23/23、GSM8K 99.0% 的那一版）。
- 新快照 `src_dspark_native_graph_v1`：`draft_graph_files/` 放三件套字节、`draft_graph_overlay.py`
  用 meta-path finder 在 `DSPARK_DRAFT_USE_CUDAGRAPH=1/DRAFT_GRAPH=1` 时才把这三个模块解析到
  overlay 路径；门控关闭时 finder 回答 None、eager 路径字节不变；门控开启但缺件时 sitecustomize
  直接 exit2（fail-closed）。freeze 记录 16 文件 SHA。
- 排障两处（都留了证据）：
  1. `stock_graph_observer.py` 的源守卫增加 overlay 版 `llm_base_proposer.py` SHA（d5003b77→36d7c887）；
  2. **wrapper 追踪对象由 `runner.drafter.model` 改为 `runner.drafter._runnable`**——
     本栈 draft 提案真实调用点是 `_runnable`（`ACLGraphWrapper(_run_merged_draft)`），
     graph_aa_v1/v3 的 “used eager” 拒绝根因就是 observer 从未拦到真实 wrapper（events=[]）。

## 2. AA 审计屏（原生同构，3 对）

`formal_dspark_native_graph_aa_v4` exit0、3/3 对 passed：tokens 相等、routes(2094×40×6) 相等、
Top5 keys 相等、logprob 最大差 **0.0**、`actual_round_streams_equal=true`。
同几何三元组：eager `59.62ms/step,A=5.75,96.44tok/s` → graph `44.89ms/step,A=5.75,128.10tok/s`（−24.7%）。

## 3. 正式 API 客户端窗口（draft-graph ON，8×2K/256，串行，审计/profiler 关）

`formal_dspark_native_graph_api_v1`：688 实际 verifier rounds、2040 decoded tokens、
decode 合计 31.983336650s ⇒ **`(46.48740792151163 ms/step, A=2.9651162790697674, 63.783213812996586 tok/s)`**、
clean 7/8、复读 1、空答 0。逐请求 rank 收据中 `runtime_graph=true`、`actual_graph_submission=true`
（图提交被逐轮计数，不是“仅配置 ON”）。

对照 eager 基线 `(57.5816614535, A=2.9651162791, 51.4941077458)`：**−11.09ms/step（−19.3%）、tok/s +23.9%、A 不变**。
设备侧 `[bneck] hp` 53.02→41.01 ms/step（−12.0）。距离 ≤20ms 仍需 −26.5ms。

Host 边界分解（688 轮精确重建，与 eager 同法）：initial 0.195、**execute→bookkeeping 34.319**、
bookkeeping→sample 3.677（eager 16.850，正是图化省下的部分）、间轮 3.061、terminal 5.235 ms/round。

## 4. 逐相位设备归因（profiler 开启，10 个稳态 decode 步，8 rank 一致）

`formal_dspark_native_graph_profile_v1`（exit0，8rank×15步；离线 CSV 解析，无新 NPU 初始化）：

| 相位 | span(ms) | kernel 并集(ms) | 最大项 |
|---|---:|---:|---|
| pre_target | 15.35 | 2.84 | 431 个小算子（Cast/Fill/SelectV2/ViewCopy/BroadcastTo/FloorMod…）、4×allReduce |
| target_graph（6行） | 22.69 | 21.65 | **160× hcom_allReduce 6.12ms**、40× GroupedMatmulSwigluQuant 3.13、80× HcPre 2.30、97× MatMulV2 2.15、208× QuantBatchMatmulV3 2.00、40× GroupedMatmul 1.46 |
| verify_and_prepare | 1.62 | 0.43 | 采样/拒绝索引 |
| draft_context_projection_cache | 4.54 | 1.71 | 2×allReduce 0.61、2×allGather 0.42 |
| draft_body_logits | 2.31 | 2.44 | 12×MatMulV2、12×allReduce、3×SparseAttnSharedkv |

与 eager profile 对照：draft_body 14.15→2.31ms、draft_ctx 7.88→4.54ms（图化收益），
target_graph 与 pre_target **几乎不变**。pre_target 窗口 `wait_union` 仅 8.4µs ⇒ 该窗口设备几乎空闲，
是**主机入队受限**（launch-bound），不是等事件。

这些是 profiler 开启的区间/并集，不能当关闭 profiler 的报价或可回收预算。

## 5. 结论与下一步

1. draft 入图在本栈实测有效且数值等价（AA 全 0 差）；正式客户端 −11.09ms。
2. 剩余最大三块：target 6 行图 22.7ms（其中 allReduce 6.1ms/160 次）、pre_target 15.4ms 的
   小算子入队风暴（kernel 仅 2.8ms）、draft_ctx 4.5ms（kernel 1.7ms）。
3. 因此 ≤20ms/step 必须同时压缩 target 图本身与两处入队风暴，属结构化工程；
   所有 actual/formal 标志保持 false，质量/归属/交付门未过。

证据：`evidence/formal_a321/dspark_native_graph_aa_v4/`、
`evidence/formal_a321/dspark_native_graph_api_v1/{launched,acceptance_small}.json`、
`evidence/formal_a321/dspark_native_graph_profile_v1/{launched,result,timeline_compact}.json`。
原始 34.4MB acceptance、profile CSV/trace 留远端并记 SHA。

## 6. 附：storm 归因与 slot-map 融合同实例配对（同日追加）

- 子代理离线归因（只读 profile CSV）：pre_target 窗口设备 kernel 并集仅 2.11ms、aclnn API 653 次/2.64ms，
  其余 ~10.6ms 为未追踪主机胶水 ⇒ **主机 launch-bound**；四大来源：12×slot-mapping 启动、
  Engram 查表图 replay 87 kernel、DSA 元数据 37 算子、~250 个输入准备小算子（含 26×`.item()`、13×阻塞 H2D）。
  与 eager profile 对照逐键零差异 ⇒ draft 入图没有改变 pre_target。证据 `storm_probe_20261013.json`。
- `V41_SLOT_MAP_FUSED` 同实例配对：`=verify` 模式 8×256 token 与基线逐位一致、无回落告警（数值门通过）；
  `=1` 客户端 `46.5040ms/step` vs 基线 `46.4874ms/step` ⇒ **客户端无稳定收益**（噪声级），
  但设备 `[bneck] hp` 41.006→39.797（−1.21ms）——设备省时不传导到客户端，与 host-bound 结论一致。
  按纪律记为无收益、不部署、开关保持默认 off。证据 `dspark_native_graph_api_{v2_verify,v3_fused}/`。
