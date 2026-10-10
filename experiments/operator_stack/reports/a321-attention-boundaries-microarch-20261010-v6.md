# 正式 TP8：精度首个偏离点与微架构统计修正

2026-10-10 · a3-21 physical chip8–15 · `feat/operator-stack-tp8-20261010`

正式权重的原生 A/A 尚未通过。最新证据将第一个 decode 步的首个偏离点收敛到 layer0 注意力分支：进入 HC post 之前，注意力输出已不同；HC post 的 residual、post、comb 全部一致。固定输入的原生 HC pre、HC post 和 router 重复计算全部稳定。下一轮已启动 Q/KV、SMLA、O 投影及 AllReduce 边界追踪。

同时完成 56 份已采集 CSV 的 shape 分组与通信重复记录核验。此前 AllReduce 的 164 次/step 包含逻辑事件和 AIV 执行两份记录；实际为 82 次。chip8 的采集态累计执行时间由重复相加的 5874.002μs 修正为 2937.001μs。没有新增有效端到端性能结论，≤约19ms/step 目标仍未验收。

## 为什么称为精度验证未通过

对照双方是同一组 worker、同一份正式权重、同一 prompt 的原生实现两次请求，不是两个不同模型。正式 checkpoint 为 `v41-w4a8-engram-dr-vision-qrot-mtpq`，40 层、hidden5120、384 专家/top6、W4A8_DYNAMIC、Engram int8；正常 auto/lazy 加载，完整视觉模块保留。

`formal_attention_boundary_trace_v1` 的三组 A/A 中，prefill 路由逐位一致，decode 分别有 1372、1359、1347 个 token-layer 路由记录不同。其中 896、859、850 处专家集合不同，不能只解释成专家顺序交换。47 个生成 token 一致，但 Top5 候选集合不同，共同候选的最大 logprob 差分别为 1.124987、0.749951、0.812432。集合不同的情况下，完整 Top5 差标为 null，不编造单一差值。

这些结果没有通过既定的完整路由、Top5 集合及 logprob 差 `<1e-3` 门槛。这证明当前原生请求重复性不足以充当严格优化对照；尚未证明它相对于独立可信数学参考的错误程度，也没有证据将其归因于候选优化、权重加载或设备 Alarm。未安装候选 bank 的 graph、eager 与仅关闭 Engram 子图的控制均曾复现差异。不能以最终 token 相同替代原门槛。

## 第一个 decode 步的实际边界

关闭整网 graph 和 Engram 自身子图，仅保留原生计算、实际表与原权重。forward 内只做 NPU clone，D2H 与比较在请求结束后执行。每个 rank 捕获 HC pre80、HC post80、router40，共200个记录；8rank全部达到覆盖要求。

|layer0边界|跨请求比较|意义|
|---|---|---|
|注意力前 HC pre 输入、全部输出、pre_mix|逐位一致|首个偏离点在其下游|
|注意力输出，即 HC post 的 x `[1,5120]` BF16|2830–3020个元素不同，最大绝对差0.015625–0.0234375|进入 HC post 前已出现差异|
|HC post residual `[1,4,5120]`|逐位一致|残差输入稳定|
|HC post post/comb FP32系数|逐位一致|混合系数稳定|
|HC post 输出，即 FFN前HC输入|2312–2508个元素不同，最大绝对差0.0009765625–0.001220703125|差异经混合传到FFN|
|FFN前HC传入的pre_mix|逐位一致|不能用该字段稳定代替完整输入稳定|

每份真实输入再重复调用原生 callable 5次，共1600份输入、8000次重复。HC pre、HC post、router均与实际forward及彼此逐位一致，最大重复绝对差0。结论只覆盖固定实参；不替代跨请求注意力、cache、通信或整网验收。

## 新一轮如何继续区分根因

`formal_attention_internal_trace_v1` 已在启动前核实8–15无占用，逐chip复核仍为已授权的同一80C98001；保留其他chip租户。没有reset或修改模型表。

诊断记录 layer0 输入归一化、Q/qr/KV、注意力核心的Q/可见KV/序列元数据/输出、反向RoPE后的O投影输入、wo_b输入、本rank wo_b局部乘积及AllReduce后的输出。可见KV按逻辑token顺序读取128行，避免把请求分配到不同物理page误报成cache数值差异；不比较未使用cache内容。算子保留原实现。

此外，8rank按相同顺序，用各自保存的wo_b局部向量重复原生TP AllReduce 10次，与首轮及实际forward输出分别比较。它能检验固定rank输入的通信归约是否稳定；不能仅由边界差异就宣称HCCL是根因。该作业结果仍待完成。

## 微架构计数：按执行与shape重算

原始采集为7组×8rank×10步：PipeUtilization、ArithmeticUtilization、Memory、MemoryL0、MemoryUB、L2Cache、ResourceConflictRatio。56CSV均有有效时长、正确Device ID8–15。没有重新采集。

新分析器只在 Step、Device、Type、Start Time、Duration 全部精确对应，且存在 `AivKernel` 执行时，排除 block0 的 `hcom_` 逻辑记录。每份Pipe CSV移除820条AllReduce和10条AllGather重复事件，23560原始行变为22730执行行，即2273执行记录/step。原来的2356是原始行数，不是去重后的执行数。

8rank AllReduce均为82次/step，累计执行时间为2.422–3.700ms/step；chip8为2.937ms。所有累计值可能与别的任务重叠，不能当作端到端step时间，也不能直接视为可回收时间。前两个collective的跨rank启动时间跨度均值分别约1299和590μs，结束跨度约13和12μs，提示入口不齐；启动跨度包含host/RPC、到达及时间标定影响，尚未隔离为纯网络等待。

chip8采集态，按实际shape拆分：

|算子/shape|次数/step|累计执行μs/step|核内计数线索|
|---|---:|---:|---|
|O投影 wo_a，BF16 `[1,4096]×[4096,1024]`|40|945.201|AIC MTE2 86.66%，MAC5.21%|
|O投影 wo_b，BF16 `[1,1024]×[5120,1024]`|40|543.634|AIC MTE2 75.55%，MAC9.09%|
|Router FP32 `[1,5120]×[384,5120]`|40|526.987|AIC MTE2 74.17%，MAC17.56%|
|融合GMM，48本地expert、6行输入|40|2087.794|AIC Scalar63.27%，MAC1.34%，MTE2 8.74%|
|GMM down|40|1394.558|AIC Scalar47.82%，AIV Scalar53.82%|
|HC pre|80|2046.924|AIC Scalar34.00%，AIV VEC1.01%|
|HC post|80|509.168|1个Vector block，VEC66.48%；UB Vector读/写29.69/18.50GB/s|

这些ratio与带宽使用工具的task/core归一化口径，不能当成全芯片利用率；Block Num是启动配置，不是实测活跃核容量。分组采集不是同一次执行，计数不能按timestamp逐任务拼接。

L2读计数的 hit/(hit+miss_allocate) 在wo_a约87.7%、wo_b约0.8%、router约7.0%；融合GMM的AIC约94.1%。这包含算子内的重复访问，不代表跨层权重全常驻L2。Memory中的L2带宽原始字段曾为0但命中计数非零，不能解释为没有L2流量。MemoryUB提供读写带宽，没有采得UB容量occupancy。

## 对后续优化的实际约束

小M的O投影/router首先需要研究GM→L1→L0的数据路径、tile/core切分和重读。这里不能把Cube的L1/L0双缓冲等同于Vector的UB双缓冲；应先证明多个tile确实有搬运与计算可重叠，并检查容量与数值顺序。MTE2高支持优先研究搬运路径，不自动证明增加buffer就有收益。

GMM的Scalar占比高、Cube计算占比低，且launch覆盖24个block，优先检查稀疏路由下非空expert数、空group控制、调度与核间等待。需要实际group_list/tiling和活跃核证据；不能凭较低HBM带宽把它解释为单一带宽瓶颈。TP1的8expert/top2 selected-GMM未适配正式384/top6，保持禁用。

HC static、顺序求和、Div-RN及HC post候选在正式入口继续保持原生；已有数值拒绝结果不重复计入收益。Indexer融合已有正式独立与消费者cache逐位通过证据，可继续组合，但不能替代整网门槛。正式W4A8激活候选实际覆盖0，没有收益可计。

下一步以首个实测偏离点修复原生重复性，再在同一组正式worker中对兼容优化配对；关闭审计/profiler后测端到端，并报告自洽的 `(ms/step,A,tok/s)`。通过质量、归属与时延验收后才部署最优服务。本轮没有达成19ms目标。

## 可复查证据

- `evidence/formal_a321/results/formal_attention_boundary_trace_v1/`：跨请求前5边界摘要、原生1600输入重复、3组A/A、启动环境和真实退出码。
- `evidence/formal_a321/results/formal_native_profile_v1/shape_microarch_summary.json`：56CSV源sha、去重数量、shape与计数；完整1.36MB分析报告保留远端。
- `evidence/formal_a321/results/formal_native_profile_v1/communication_timeline.json`：82次collective的跨rank启动/结束跨度。
- 新代码：`attention_boundary_probe.py`、`analyse_formal_microarch.py`；所有精度门槛保持原样。
