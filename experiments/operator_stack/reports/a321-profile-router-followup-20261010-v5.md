# 正式TP8：路由覆盖修正与无审计微架构采集

2026-10-10 · a3-21 chip8–15 · `feat/operator-stack-tp8-20261010`

本轮完成了第二次原生算子重复诊断。HC在8rank均捕获80份真实输入，各重复5次，重复输出和实际forward输出全部一致；路由投影实际覆盖仍为0，覆盖门明确拒绝通过。原生整网的decode路由/logprob不稳定仍未解决，19ms目标未验收。

## 为什么路由投影探针仍未覆盖

生产代码在 `experts.is_internal_router` 为true时，把输入交给 `FusedMoEFactory`，外层gate的forward没有被调用。内部实现使用 `F.linear(hidden_states_fp32, gate.weight_fp32)`，或在另一版本分支使用临时转换到FP32的权重。

原探针以外层gate.weight的地址匹配，不能保证对应内部FP32权重或临时Tensor的地址。因此已在内部专家模块进入/退出时跟踪归属，再对实际F.linear调用记录输入与输出。原生callable保持原样，重复测试仍调用它，不引入新的路由算法。每rank的80次HC、40次router覆盖门槛保持不变。这项修正尚未完成上板验证，不能据此宣布路由稳定。

第二轮记录和失败退出码位于 `evidence/formal_a321/results/formal_native_eager_probe_v2/`。HC固定输入稳定仅说明这些输入的原生计算稳定，不证明不同请求中HC输入相同。

## 为什么现在采集原生profiling

继续诊断精度时，也需要正式W4A8整网的真实热点证据。新增 `profile_formal_native.py` 使用正式权重、标准NPU worker、完整视觉模块与Engram；不安装候选算子包装，不导出路由，不请求logprobs，不做cache快照。该采集不会放宽或替代精度门槛，结果明确标为diagnostic，`precision_validated=false`、`performance_claim=null`。

每个指标独立发请求，先执行9步，再做5步profiler warmup与10步active；8rank各用独立目录。采集PipeUtilization、ArithmeticUtilization、Memory、MemoryL0、MemoryUB、L2Cache、ResourceConflictRatio以及CPU/NPU时间线。结果要核实每rank实际Device ID、窗口步数、非空CSV、字段和缺失值，再统计原生HC、路由、GMM、attention、Engram、HCCL与图间隙。

`formal_native_profile_v1` 已在启动前8chip空闲时实际启动，正在加载正式模型；尚无本轮可验收的计数CSV。累计kernel时间不能当成端到端step；路径带宽不能当作UB容量或整芯片HBM利用率；NA和工具失败不写为0。UB双缓冲仍需由多tile工作集和关键路径上的MTE证据决定。

## 后续验收

先分析原生计数，再运行修正归属后的路由固定输入测试与跨请求首个偏离点诊断。只有原门槛通过，才采纳两条优化线的叠加收益、发布有效 `(ms/step,A,token/s)`，部署最佳正式TP8服务。目标保持active。
