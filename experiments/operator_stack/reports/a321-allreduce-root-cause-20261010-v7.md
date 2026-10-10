# 正式 TP8：首个数值偏离定位到 O 投影 AllReduce

2026-10-10 · a3-21 chip8–15 · 正式 W4A8 权重 · TP8

`formal_attention_internal_trace_v1` 已完成。8rank 的首个 decode 步中，注意力及 O 投影本地乘积在两次相同请求间全部逐位一致，首次差异出现在 wo_b 的 AllReduce 输出。用完全相同的 rank 局部输入重复原生 AllReduce，也直接复现了输出不稳定。当前定位是首个偏离点，不宣称它是整网所有差异的唯一来源。

## 两边到底是什么没有对齐

比较双方是同一组worker、同一份正式权重、同一prompt的两次请求。基线保留仓库生产补丁，不安装本轮候选算子bank；不是另一个checkpoint，也不是无补丁上游。

之前三组原生A/A的生成token相同，但decode路由、专家集合、Top5候选和logprob未通过原门槛。具体数据和通信去重修正见[v6报告](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-attention-boundaries-microarch-20261010-v6.html)。本轮进一步比较了layer0的16个边界。

|边界|全部8rank的跨请求结果|
|---|---|
|HC之后的输入归一化输入/输出|逐位一致|
|Q、qr、KV投影|逐位一致|
|注意力核心Q、128行可见KV、序列长度、query_start及原生metadata|逐位一致|
|SparseFlashMla输出|逐位一致|
|反向RoPE后的O投影输入|逐位一致|
|wo_b输入，即wo_a局部输出|逐位一致|
|wo_b本rank局部矩阵乘法结果|逐位一致|
|wo_b AllReduce输出|2847–3029个BF16元素不同，最大绝对差0.015625|

可见cache按逻辑token顺序比较，不把物理page分配变化当成数值错误。forward期间只做NPU快照，结束后比较，采集会改变时序，因此不报告性能收益。

固定输入的原生AllReduce独立复验：每rank使用自己保存的同一份wo_b局部向量，8rank按相同顺序共同执行10次。每rank后续9次均与第一次不同，10次均未逐位复现原forward结果；最大重复绝对差为0.0234375–0.03125。该结果把不稳定定位到归约路径，无需假设上游cache、Q/KV或HC先发生变化。

浮点求和不满足结合律，BF16归约中不同执行/累加顺序可能产生不同舍入结果。当前数据符合这一机制，但尚未用HCCL内部流水/源码证明具体采用了哪种顺序，也不能据此把设备Alarm认定为原因。可确认的是：当前原生归约对固定rank输入没有保持相同数值输出。

## 正在验证的修复与原理

已启动 `formal_fp32_decode_reduction_v1`。实验只选择world_size=8、5120个元素且末维5120的BF16隐状态归约：

1. 保留原本各rank局部乘积的BF16舍入边界，再将该结果无损转为FP32。
2. 仍调用同一个原生AllReduce，但以FP32累加。
3. 归约后转回BF16，保留后续模型接口。

它增加中间精度，不通过降低累加精度换确定性，也不全局开启`HCCL_DETERMINISTIC`。仓库历史中全局`true`曾使GSM8K-100从100/100降至91/100；那是历史模型的结果，不直接外推当前V4.1，但足以说明不能仅凭确定性改善就作为交付配置。

新验证继续使用正式权重和原路由/Top5集合/logprob `<1e-3`门槛，跑三组同worker A/A。此外，捕获首decode真实局部向量，AllGather到各rank后在CPU以FP64求和，并转为BF16作为独立数值参考。实验结果必须与该参考逐位一致，不能仅看重复稳定。

首decode预期覆盖81个单token隐状态归约，包括embedding、40次attention和40次MoE；整个47-token请求应有81×46次选择，计数不符就拒绝覆盖。此计数待真实运行核实，不能当作已经通过。旧profiling共有82次实际AllReduce，另外的归约不满足上述5120元素守卫。

该修复尚未通过整网精度、独立数学参考或性能验收。FP32会增加传输字节与转换任务，需要在通过精度后关闭探针，以图模式测端到端 `(ms/step,A,tok/s)`。没有宣布达到19ms或部署最优服务。

## 后续优化顺序

先完成固定归约输入的独立参考和正式整网A/A；若FP32仍不能保持严格结果，继续测试固定顺序、高精度等价归约，保留相同精度门槛。随后重新审计两条算子线的兼容叠加方案。

已采profiling显示O投影小M的MTE2占比高，GMM的Scalar占比高；它们分别需要数据搬运/tile与稀疏expert调度证据。仍未采得UB容量occupancy，UB双缓冲没有被认定为已验证收益。HC/HcPost正式入口保持原生，TP1 selected-GMM/router不适配正式384/top6，W4A8激活候选覆盖0，Indexer融合通过独立/cache门但不替代整网。

新增代码与紧凑证据保存在本分支：`attention_boundary_probe.py`、`decode_reduction_probe.py`、`evidence/formal_a321/results/formal_attention_internal_trace_v1/`。目标保持active。

## 官方资料与后续备选

[CANN HCCL_DETERMINISTIC文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/hcclug/hcclenvref_07_0010.html)说明：A3以AI CPU展开时，归约类算子是确定性计算；以Vector Core展开时，AllReduce/ReduceScatter涉及非确定性计算。文档同时说明确定性可能降低性能，且某些场景下会覆盖AIV展开设置。当前实测环境为`HCCL_OP_EXPANSION_MODE=AIV`，与该说明一致；文档没有承诺改为FP32就必然完全确定，因此新方案仍须实测。

[HcclCommConfig官方API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/hcclapiref/hcclcpp_07_0047.html)支持通信域粒度的确定性与展开模式，优先级高于全局环境变量。后续可测试专用通信域的AI CPU/确定性方案，前提是核实当前CANN与torch_npu的接口支持、严格数学参考、整网质量与图模式性能。它们尚未试验，不能计作已有优化成果。
