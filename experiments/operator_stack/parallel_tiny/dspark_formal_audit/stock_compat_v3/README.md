# 正式 DSpark stock v3 计数快照

本副本修正正式 eager A/A 的实际计数边界，不改推理、权重、SDK 或已冻结 v1/v2。正式目标仍是 TP8 包含 DSpark 全部开销 ≤20ms/step；本目录的 CPU 检查不授予设备消费者或性能通过。

`formal_dspark_native_eager_aa_v1` 加载了完整主模型及三层 MTP，完成两个 warmup。首条计数请求结束后，stock `execute_model` 收到 `total_num_scheduled_tokens=0`、空调度表和 `finished_req_ids=['2-ae372658']`。该路径只清理请求状态并返回 `EMPTY_MODEL_RUNNER_OUTPUT`，v2 的单请求断言错误地拒绝了它。主线程核对 argv、源码根、芯片环境及实际 fatal 日志后只终止本任务 driver；exit143，计数请求闭环数为0，不能报价。

v3 对这个真实空调度 ABI 单列 `zero_token_executions`，校验总数、空 new/cached/spec/encoder 输入及实际 empty return，禁止它贡献 target/verifier rounds、sampler 或草稿。客户端墙钟仍包住实际请求和已提交 terminal draft 完成，不减去清理时间。

`mapped_generation` 从实际 `LLMEngine.add_request` 返回值和 `OutputProcessor.external_req_ids` 保存 external/core ID 对应；最终输出的 external ID 必须一致，8 rank 原始计数必须全部绑定该 core ID。相同 token 来自其他请求或其他请求的结束清理均拒绝。稳态计数没有新增 D2H/设备同步，末轮仍使用一次真实 device fence，保证 terminal 已提交草稿被计入。

保留 v2 的 stock graph observer：配置字段不能证明图执行，实际已有 entry 的 `NPUGraph.replay` 返回才记录提交；本次新 A/A 请求真实 draft eager。v3 只覆盖新快照 `/work/src_dspark_native_eager_v2` 的两个顶层文件，并更新 prep freeze；`formal_dspark_native_eager_aa_v2` 由主线程通过8–15 fresh资源门启动。

26项 CPU 合约/回归检查通过，包括原16项计数门、空调度不计轮次、非空总数/提议假空调度、假 sampler、相同 token 的跨请求假阳性、错误外部/核心映射及未完成清理。原冻结文件 SHA 保持，正式 NPU A/A、43层/cache/rollback消费者审计和20ms性能仍待取得。

```bash
python3 experiments/operator_stack/parallel_tiny/dspark_formal_audit/prepare_stock_v3.py
python3 experiments/operator_stack/parallel_tiny/dspark_formal_audit/stock_compat_v3/test_stock_v3.py
```
