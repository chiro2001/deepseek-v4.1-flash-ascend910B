# base2 draft 图：捕获、派发与计数证据

来源为实际 a3-21 D 容器，SID `3cbd844d9f684c43867b4b85c9c78ebe`。
只读现有源码/日志，没有增加生成、worker RPC 或 profiler；唯一生成窗口仍为 1 warmup + 8 measured。

| 层级 | 当前证据 | 结论与边界 |
|---|---|---|
| 草稿包装 | 八 rank 逐一打印 `Wrapping draft model with ACLGraphWrapper: runtime_mode=FULL, use_eagle=True` | 实际包装，不是仅 CPU 计划 |
| 带注意力捕获 | 八 rank 各有 `dspark-graph-capture capture #0`，descriptor `(6,1,uniform=True)`、`layers=3,num_query_total=5,num_input_tokens=5`，随后捕获完成 | 草稿实际执行带 draft attention metadata 的捕获，不是 LEGACY metadata-less 路径 |
| 请求派发 | 八 rank 各 808 个 `runtime_graph=true` proposal、8 个 eager proposal，与 actual rounds 对齐 | 实际 `_propose` 调用选择图入口，不是由 output tokens 反推 |
| 底层 replay | 同一服务日志有 `acl_graph.py:252 Replaying aclgraph`，实际源码随后调用 `entry.aclgraph.replay()` | 服务确实重放 ACL graph；一次性通用日志不能逐次归属 target/draft |

因此当前 draft 包装与带注意力捕获已实际完成，测量请求选择其图派发入口。
**不把 808 个 selector 计数称为独立采集的 808 个底层 replay 事件**；没有新增逐调用低层 trace，
也不据此授予 full43 数值消费者或 performance admission 收据。

主代理已读以下实际叶源码：

- [llm_base_proposer.py](frozen_sources/llm_base_proposer.py)：164 的 selector 检查实际 `use_cuda_graph` 与热切换；
  951–965 构造 `ACLGraphWrapper(self._run_merged_draft,FULL)`；1373–1415 用 dispatcher 定最终 mode；
  1758–1785 在 force-eager 时用原函数，否则用 `_runnable`。
- [dspark_proposer.py](frozen_sources/dspark_proposer.py)：807–843 构造/记录 draft metadata；863–915 把它传入 draft context 并执行 `_runnable`。
- [acl_graph.py](frozen_sources/acl_graph.py)：134–146 在 mode NONE/不匹配时可回退 eager；匹配时创建/捕获 entry，252–267 重放。
  所以 selector 本身不能替代逐调用最终 mode/replay 计数。
- [主代理原始收据重评分](../baseline2_client/primary_raw_audit.json) 核对八 rank selector 差分、SID、request tag、terminal fence/TP join。

source220 在实际 D 内重新核验 220/220 OK。冻结副本逐件小于 1 MB，远端/本地双 SHA 相同，原样保留：

| D 内原件 | SHA-256 |
|---|---|
| `/work/src_dspark_general_quality_v11/stack/draft_graph_files/llm_base_proposer.py` | `36d7c8870538e10e310863f9b628b90b946b534861563542cf7fba4cd88d7c39` |
| 同目录 `dspark_proposer.py` | `56943080f99da2f2dee516aa3e76605c5150408df9a26691cc6529f327ae29b4` |
| `/vllm-workspace/vllm-ascend/vllm_ascend/compilation/acl_graph.py` | `c8ab2fe6aaae034b03e9e34b9c07fd445046564acd5d45e2c24b5ea5232c5a74` |

源日志 `jobs/base2/run.log` 快照为 249,772 bytes，读取前/后 SHA 均为
`bd5324cfba948bd8f58c8b1c11be3dfaa88c90206ad6a7bffab7451193f08b70`。
[带原行号摘录](base2_draft_graph_log_excerpt.txt) SHA 为
`fac9e96dfcbaafbef8d50253230b89dad3ec2bc87c53e78b57e50bacba0e8844`；
长行明确标截断，摘录不是原始完整日志。原行 681–737 为八 rank draft 包装，1072–1093 为八 rank metadata 捕获，
1098 为捕获完成，1216 为通用 replay。旧 slot/mask 候选仍关闭，不能借此声称其 DSpark 六行收益。
