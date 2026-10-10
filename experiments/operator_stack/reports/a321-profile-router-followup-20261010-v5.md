# 正式TP8：路由覆盖修正与无审计微架构采集

2026-10-10 · a3-21 chip8–15 · `feat/operator-stack-tp8-20261010`

本轮完成了第二次原生算子重复诊断。HC在8rank均捕获80份真实输入，各重复5次，重复输出和实际forward输出全部一致；路由投影实际覆盖仍为0，覆盖门明确拒绝通过。原生整网的decode路由/logprob不稳定仍未解决，19ms目标未验收。

## 为什么路由投影探针仍未覆盖

生产代码在 `experts.is_internal_router` 为true时，把输入交给 `FusedMoEFactory`，外层gate的forward没有被调用。内部实现使用 `F.linear(hidden_states_fp32, gate.weight_fp32)`，或在另一版本分支使用临时转换到FP32的权重。

原探针以外层gate.weight的地址匹配，不能保证对应内部FP32权重或临时Tensor的地址。因此已在内部专家模块进入/退出时跟踪归属，再对实际F.linear调用记录输入与输出。原生callable保持原样，重复测试仍调用它，不引入新的路由算法。每rank的80次HC、40次router覆盖门槛保持不变。

第三轮 `formal_native_eager_probe_v3` 已完成上板验证：8rank全部达到80次HC、40次router覆盖，共960份真实输入，每份重复5次。全部重复输出与彼此及实际forward输出一致，最大重复绝对差为0。整网原A/A仍失败；该结果说明这些固定输入的原生HC和路由投影稳定，不证明跨请求输入、TopK选择、通信或cache状态稳定。

第二轮记录和失败退出码位于 `evidence/formal_a321/results/formal_native_eager_probe_v2/`。HC固定输入稳定仅说明这些输入的原生计算稳定，不证明不同请求中HC输入相同。

## 为什么现在采集原生profiling

继续诊断精度时，也需要正式W4A8整网的真实热点证据。新增 `profile_formal_native.py` 使用正式权重、标准NPU worker、完整视觉模块与Engram；不安装候选算子包装，不导出路由，不请求logprobs，不做cache快照。该采集不会放宽或替代精度门槛，结果明确标为diagnostic，`precision_validated=false`、`performance_claim=null`。

每个指标独立发请求，先执行9步，再做5步profiler warmup与10步active；8rank各用独立目录。采集PipeUtilization、ArithmeticUtilization、Memory、MemoryL0、MemoryUB、L2Cache、ResourceConflictRatio以及CPU/NPU时间线。结果要核实每rank实际Device ID、窗口步数、非空CSV、字段和缺失值，再统计原生HC、路由、GMM、attention、Engram、HCCL与图间隙。

`formal_native_profile_v1` 已完成采集。Torch worker是daemon进程，在线解析被工具拒绝；原始数据仍完整保存。首次对多层父目录调用analyse未发现数据，改为逐个 `_ascend_pt` 目录在独立非daemon子进程中解析，限制并发4。56个CSV全部验证通过：7组×8rank，每文件10个有效步，实际Device ID为8–15，时长有限且非负。PipeUtilization中每rank均为2356 task/step。

实际字段包含MTE1/MTE2/MTE3与Scalar/Vector/Cube占比、L1/主存/L2/UB路径带宽、L0A/B/C带宽、L2命中/未命中计数和Vector资源冲突。完整跨rank和逐算子计数解读仍在继续；本轮没有UB容量occupancy。

首个rank的采集态kernel累计热点如下，**允许重叠，不等于端到端时间**：AllReduce为164次、5874.002μs/step；MatMulV2为98次、2273.656μs；GroupedMatmulSwigluQuantV2为40次、2087.794μs；HcPre为80次、2046.924μs；QuantBatchMatmulV3为208次、1967.117μs；GroupedMatmul为40次、1394.558μs。通信等待与rank到达偏差需进一步区分，不能单凭AllReduce时长判为通信带宽瓶颈。

Cast为193次、278.634μs/step，主要包含79次单元素INT32→INT64和40次 `[1,6]` FLOAT→BF16；未发现384×5120路由权重Cast，不能据此宣布缓存该权重会有收益。实际路由MatMulV3为 `[1,5120]×[384,5120]` 的FP32输入，每步40次，后接40次MoeGatingTopKHash。

累计kernel时间不能当成端到端step；Profiler控制RPC影响CPU/图间隙。路径带宽不能当作UB容量或整芯片HBM利用率；NA和工具失败不写为0。UB双缓冲仍需由多tile工作集和关键路径上的MTE证据决定。

## 后续验收

先分析原生计数，再运行修正归属后的路由固定输入测试与跨请求首个偏离点诊断。只有原门槛通过，才采纳两条优化线的叠加收益、发布有效 `(ms/step,A,token/s)`，部署最佳正式TP8服务。目标保持active。
