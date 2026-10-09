# 两条算子线的组合判断

2026-10-10，独立集成分支`feat/operator-stack-tp8-20261010`。目标仍active，TP8精度/性能与服务尚未全部验收。

上游950迁移线的最终有效增量只有Indexer INT8后处理。它融合K源层的RMSNorm、尾64维RoPE、INT8量化与K/FP16-scale cache写入，保持两个BF16舍入边界及原生Div(127,max)→Mul→RINT。其他五方向已有拒绝证据，本轮不重复筛选。

我们的主线已包括HC/router、激活、Q/KV多流、GMM1 epilogue、metadata与多group slot准备；Index K后处理属于不同位置，可组合但不能累加历史百分比。

## TP1组合实测

三组非恒定gate/HC/MLP/attention权重审计通过，实际路由、47/48输出和logprobs一致，delta0。
Indexer的functional INT8/FP32 scale及消费者点INT8/FP16 cache逐位通过；末步C2 valid rows交替4/1，完整覆盖完成/未完成状态。

同进程12组，profiler/审计关闭，2K输入/48输出、A=1：

|模式|ms/step|A|token/s|
|---|---:|---:|---:|
|当前mdfull主线|19.338175|1|51.711188|
|加Indexer后处理|19.101980|1|52.350594|

配对加速中位1.002596，10/12组更快。额外收益约0.26%，目前没有大的额外提升；过程有漂移，不能将整体中位差当作稳定收益，也不能引用另一条线26ms与本线19ms跨进程相减。

## TP8适配状态

用户授权a3-21 chip8–15。8个chip虽均有80C98001 RAS Alarm，但Vector、Cube和HCCL AllReduce均精确通过；未重置设备。
隔离TP8容器已运行完整模型，并成功捕获native/core/stack的8rank图bank。

实际覆盖显示：每rank HC80/router40、core HcPost80、stack Indexer4；原TP1激活shape守卫在TP8均未选择，TP1 selected-GMM/路由特化显式禁用。因此不能把TP1的全部收益直接移到TP8。
后续需要按实际shared/routed形状适配激活，并分别验证；目前尚未声称这一项成功。

首次TP8审计请求的路由回传只有256/2048个prefill token有有效top2（10240/81920个token-layer对），decode有效；安装态single-DP capturer没有复原TP8/SP的全路由。
这是审计数据完整性问题，尚不能判定模型精度通过。正在使用真实跨rank ID gather修正观测，所有route范围/唯一性/token/logprob/cache门槛保留，正式性能不开此观测路径。

下一步：完成TP8原生/核心/Indexer叠加的真实路由与cache审计，适配缺失的有效激活形状，关闭审计做同worker配对和客户端端到端验收，达到≤约19ms/step后交付最优TP8服务。
