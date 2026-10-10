# 两条算子线的组合判断

2026-10-10，独立集成分支`feat/operator-stack-tp8-20261010`。目标仍active，正式权重TP8精度/性能与服务尚未验收。用户已再次授权a3-21 chip8–15，启动前拒绝他人占用、复核80C98001，不reset。

最新正式A/A并未对齐：原生实现的同prompt请求在decode路由/Top5 logprob上不同。8rank首个decode偏离在layer0注意力输出、HC post之前；固定输入HC pre/HC post/router全部稳定。Q/KV、SMLA、O投影与AllReduce追踪已启动。两条线的正式叠加收益仍未验收；下方TP1/dummy与设备迁移叙述为历史证据。

新报告`reports/a321-attention-boundaries-microarch-20261010-v6.md`还修正profiling的通信重复事件：82次实际AllReduce/step，chip8累计2.937ms，而非164次/5.874ms。

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

历史dummy prototype使用a3-21 chip8–15。它们有80C98001 RAS Alarm，基础冒烟曾通过；用户最新要求避开问题设备，因此这8个chip全部排除，旧容器已停止。历史prototype捕获图的结果只保留为tiny证据，不计正式模型验收。

实际覆盖显示：每rank HC80/router40、core HcPost80、stack Indexer4；原TP1激活shape守卫在TP8均未选择，TP1 selected-GMM/路由特化显式禁用。因此不能把TP1的全部收益直接移到TP8。
后续需要按实际shared/routed形状适配激活，并分别验证；目前尚未声称这一项成功。

首次TP8审计请求的路由回传只有256/2048个prefill token有有效top2（10240/81920个token-layer对），decode有效；安装态single-DP capturer没有复原TP8/SP的全路由。
这是审计数据完整性问题，尚不能判定模型精度通过。正在使用真实跨rank ID gather修正观测，所有route范围/唯一性/token/logprob/cache门槛保留，正式性能不开此观测路径。

正式checkpoint为`v41-w4a8-engram-dr-vision-qrot-mtpq`：40层/5120 hidden/384专家top6、W4A8_DYNAMIC，Engram层1/14。a3-22正式权重90个分片与辅助软链闭包均通过预检查；Indexer wk分配为BF16、k_norm checkpoint为F32但按原模型BF16参数加载。

a3-22的chip14/15已通过Vector、Cube与2rank HCCL，私有容器只透传这两个节点，ACL逻辑编号为0/1。正在用正式HC/Indexer参数做独立数值核验，输入为合成数据，不计整网精度通过。当前没有同机健康空闲8chip组合。

下一步：获得健康空闲8chip后，使用正式权重、Engram开启，完成native/core/stack同worker图bank的真实top6路由、cache、token/logprobs审计；关闭审计做配对墙钟和客户端端到端验收，达到≤约19ms/step后交付最优TP8服务。

最新核验：原Indexer融合用真实参数在两个cache page累计128例逐位通过；单系数变体也通过128例，但相对原融合配对中位0.999512，未采用。HC static、顺序求和和Div-RN均在同一个BF16舍入边界用例失败，正式入口保留原生HC。正式激活形状96例通过原1ULP门槛，实际整网覆盖仍待验证。

a3-22最新空闲chip为8/9/10/11/14/15，共6个，仍不足单机TP8。详细数据与优化原理见`reports/formal-weights-device-migration-20261010-v1.md`。
