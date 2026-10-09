# 本轮工作与优化原理：专家激活融合及 Q/KV 多流

日期：2026-10-09 · 环境：a3-21 / chip4 · 分支：`feat/tiny-operator-opt-20261009`

**本轮在上一轮已经优化HC/router的tiny基线上，进一步完成routed/shared专家激活融合，
并验证、启用了框架已有的Q/KV多流前处理。同进程六组对照的decode时延从26.456降到
24.446 ms/步，A=1，吞吐从37.799升到40.907 token/s，提升约8.2%。六组组合方案全部更快。**

优化的核心是减少小算子的固定成本，并把互相独立的Cube/Vector工作安排到同一时间段。
激活公式、专家数量和Sinkhorn迭代次数保持原有定义，dtype仍为BF16。

## 1. 本轮完成了什么

|工作|具体产出|验收依据|
|---|---|---|
|建立独立开发分支|从最新`origin/main@998c47c`创建分支与独立worktree，归档旧HC/router基线|实现与证据已提交并推送；实验实现提交`ab989b0`，清单提交`16ed351`|
|定位下一批低效位置|重新聚合优化后20步trace，核对MoE激活及Q/KV源码|ViewCopy、clamp链、各流水和实际shape相互对应|
|实现routed激活融合|将两次视图clamp和SwiGLU合为一个Vector kernel|保留原地修改输入的副作用；每层7个task变为1个|
|实现shared激活融合|合并Clip、Muls、Sigmoid、Mul和Add|保留真实limit/alpha/beta及BF16中间舍入|
|验证Q/KV多流|使用已有前处理实现，恢复每套图的事件、workspace和handles|40层均切换；主辅流存在实际task区间重叠|
|完成数值与模型审计|139个算子用例；编译前随机化gate、HC和专家权重|实际路由、输出token、top-5 logprobs一致|
|完成可比性能和profiling|四模式、三模式同进程正反交错；单独采20步窗口与逐核数据|计时关闭profiler和审计；CSV、设备、调用数、trace与哈希验收|
|保留负结果并恢复服务|保留慢的flat v1，比较wo_a分派；启动最终优化tiny|服务归属正确，32输入token成功生成16输出token|

代码采用进程内custom op和独立worker接入，选定的shape与dtype之外继续使用原路径。
当前验证范围为TP1 tiny、40层、8专家/top2、intermediate=256、dummy BF16；Engram/DSpark关闭。

## 2. 原理一：激活融合减少了什么成本

### 2.1 原生routed路径为什么慢

GMM1输出为`[2,512]`，前256列是gate，后256列是up。源码先取得两半的视图，对它们
分别做原地clamp，再调用SwiGLU。已有trace确认，原地修改视图触发了以下串行链：

```text
GMM1
  → Slice → Clip → ViewCopy
  → Slice → Clip → ViewCopy
  → SwiGlu
  → GMM2
```

每层只处理1024个BF16元素，数据约2 KiB；其中一次ViewCopy仍启动48个Vector block，
平均约9.71 μs，Vector busy约0.27%。40层每步有80次ViewCopy，累计约0.776 ms。
加上Slice、Clip和SwiGLU，该routed激活链累计约1.255 ms/步。

这些数值来自采集态。它们提示小数据量下，视图写回、通用控制、task启动以及流水首发/收尾
占了较多成本；更高的持续HBM带宽难以直接消除这条链。

### 2.2 融合后的数据流

候选直接读取原始gate/up，在一个kernel内完成clamp和激活，再写输出：

```text
GMM1
  → 一个Vector kernel：读取两半 → clamp → SwiGLU → 写输入clamp结果及输出
  → GMM2
```

其数学定义为：

```text
g = clamp(gate, max=L)
u = clamp(up, min=-L, max=L)
y = g × sigmoid(g) × u
```

节省来自四个方面：减少task边界；直接寻址避免中间Slice/ViewCopy；减少重复的GM↔UB往返；
让clamp和激活共用一次加载。原生路径会修改GMM1输出，候选也写回clamp后的gate/up，
因此保留了这个可观察副作用。

实现按行划分，每个program连续访问一行的两半。目标shape的routed使用2个Vector program，
shared使用1个；每个tile为256元素。对这个数据量，少量核足以完成工作。

### 2.3 shared融合为什么要保留中间舍入

shared通用定义是`g × sigmoid(alpha × g) × (u + beta)`，前面同样有clamp。
原生BF16逐操作路径会在若干步骤写回BF16，产生中间舍入。直接改成全FP32计算、最后只cast
一次，可能得到不同的BF16结果。

本轮候选显式保留必要的转换：alpha乘法、sigmoid输出、gate乘法和up加法的BF16边界。
139个用例中shared逐位一致；routed最大差为1 BF16 ULP。在实际模型的已验证目标形状上，
两条激活路径均与native逐位一致。

### 2.4 profiling确认了结构变化

|匹配20步采集，按每步|原生激活|融合激活|
|---|---:|---:|
|全部task数|2278|1798|
|Slice / ViewCopy / SwiGlu|80 / 80 / 40|0 / 0 / 0|
|新融合kernel次数|0|80|
|新routed / shared累计时间|—|85.606 / 84.803 μs|
|全部kernel时间合计|18.097 ms|16.353 ms|

两条激活链合计每步减少480个task。新kernel均为AI_VECTOR_CORE，激活覆盖成本降到约
0.170 ms/步。这个采集态时间用于归因，端到端收益由关闭profiler的对照确认。

## 3. 原理二：Q/KV多流隐藏了哪部分延迟

Q分支和KV分支共享输入，却存在可以并行的工作。原单流把这些工作依次放在同一条队列；
已有多流实现将独立工作放到辅助流，并用事件保持依赖关系。

以下是调度结构示意，不是指令级实测时间线：

|阶段|主流|辅助流|需要保留的依赖|
|---|---|---|---|
|1|Q_a投影|独立KV输入准备；BF16场景工作较轻|Q_a结果产生后进入后续阶段|
|2|Q norm及Q_b输入准备（Vector）|KV投影（Cube）|KV投影结束后才启动下一次Cube投影|
|3|Q_b投影（Cube）及Q后处理|KV norm、RoPE、cache store（Vector/MTE）|进入attention前等待KV写入完成|

Cube矩阵计算仍被有意串行安排，收益来自Cube工作与另一分支的Vector/MTE工作重叠。
这需要完整的事件、workspace与task handles；只切换图对象可能留下错误的依赖或参数。

匹配trace显示，辅助流每步实际执行40次KV MatMul、40次RMSNorm、40次RoPE、40次Scatter，
共160个task。主辅流执行区间的交集平均为 **780.644 μs/步**。
这是task区间重叠，不能换算成逐指令MTE/Vector重叠率。

在最终同进程对照中，多流比同组“激活融合、Q/KV单流”六组全部更快；两个模式的中位
时延相差约0.58 ms/步。这个差值小于task区间交集很正常：新增事件、资源争用及其余串行
部分都会影响端到端关键路径。

## 4. 结果与如何保证比较有效

最终三模式均在同一模型进程中，使用独立图bank；2048输入、48输出token，seed轮换0/1/2，
prefix cache关闭、KV=4 GiB、MAX_SEQS=1、FULL_DECODE_ONLY `[1]`，CPU绑定相同。
每臂预热两轮，六组正反交错；计时关闭profiler、logprobs及路由回传。

|模式|decode ms/步|A|token/s|
|---|---:|---:|---:|
|已有HC/router，激活原生、Q/KV单流|26.456|1|37.799|
|激活融合、Q/KV单流|25.021|1|39.966|
|激活融合、Q/KV多流|24.446|1|40.907|

组合方案吞吐提升约8.2%、时延下降约7.6%；六组全部优于基线。
按本会话的模式中位时延看，激活融合约减少1.43 ms，多流再减少约0.58 ms。
这些是模式比较的统计结果，不能当成每条指令可回收时间的精确分解。

此前另一次四模式进程验证了routed/shared分别和组合的收益。它与最终三模式属于不同进程，
因此不将两个会话的绝对数字相减，也不把本轮8.2%直接加到前轮12%。

## 5. 数值、路由与微架构验证

算子级139个用例覆盖有符号输入、尺度0.001/1/100、limit=1/7/7.9、alpha/beta组合、
非连续视图、边界值、空batch及NaN/Inf。测试前固定精度门槛，并保存对CPU FP64的误差。

模型审计在编译前随机化40个gate、80个HC参数组，以及80个routed和80个shared MLP权重。
四种激活模式与两种多流模式分别完成三组审计，避免dummy恒定权重造成空洞验证。
每请求实际记录`[2095,40,2]`路由数组、83800个top2对；ID合法且不退化为全零。
实际路由、48个输出token和top-5 logprobs全部一致，最大logprob差为0。

每套图检查router=40、HC=80、两种激活各40次，并检查候选确实被选择。
完整模型20步profiling核验设备编号4、调用数、trace、统计文件及CSV哈希。

新routed算子的msprof op Default/MemoryDetail均成功采到2个Vector核，频率1800 MHz。
Default task约2.900 μs；block0的Vector时间0.291 μs、MTE2 0.307 μs、MTE3 0.397 μs、
Scalar 0.509 μs、wait_ib 1.103 μs。计数有重叠，不能相加。
MemoryDetail显示该核GM→UB为1 KiB、UB→GM为1.5 KiB，符合读取两半行、写回输入和输出。
路径带宽利用率约0.215%/0.381%，属于工具逐核口径，不能解释为整芯片HBM速率。

对这种单tile、小数据量的激活，追加UB双缓冲缺少足够的独立tile来隐藏搬运。
本轮主要收益来自融合和调度；未取得有效容量Occupancy或TimelineDetail，因此不声称已知
UB容量占用率或完整指令级关键路径。

## 6. 没有采用哪些方案

|尝试|实测结果|处理与启示|
|---|---|---|
|flat v1激活|精度通过，但routed独立图224.365 μs，native约25.013 μs|拒绝并保留源码；v2改为连续行/tile访问，BLOCK 1024→256、目标grid 1→2|
|v2独立图|routed 26.771→14.112 μs，约1.897×；shared只有约0.3%变化|独立图作为筛选；采用shared还依据完整模型的稳定收益|
|wo_a普通BMM分派|hot约2.16%改善；8份权重旋转、268.4 MB工作集时仅约0.455%|未接入模型；按40次估算仅约7.4 μs/步，下一轮研究tiling/分组GEMV|
|直接打开wo_a二维开关|源码要求本地组数为1，当前tiny为8组|当前shape不适用，不能据开关状态宣称优化生效|

flat v1的具体编译降低原因仍是推断，没有指令trace证明。不能把“通过精度”当作性能通过，
也不能把L2热缓存的独立成绩直接外推到完整模型。

## 7. 交付状态与边界

实现、复现脚本、失败源码、逐核CSV、结果JSON及校验清单已落到独立分支。
仓库包自检通过；最终优化tiny服务已恢复，health=200、模型归属正确，
32输入token成功生成16输出token。运行仅使用chip4，全程排除chip14–15。

本轮证明了独立tiny的数值、路由和性能改善。生产TP8/W4A8、真实checkpoint、DSpark、
CED-PD长上下文仍需要各自的验证。下一步的主要机会是8组wo_a/GMM的小M分派与tiling、
SparseFlashMla逐核等待，以及剩余短task的融合。

## 8. 实现与证据

以下链接固定在实现与清单提交，方便核对报告的数值和代码：

- [技术报告与复现方法](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/16ed351/experiments/operator_optimization/REPORT.md)
- [融合kernel](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/16ed351/experiments/operator_optimization/clamped_swiglu.py)
- [接入与图bank](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/16ed351/experiments/operator_optimization/activation_patches.py)
- [最终六组性能JSON](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/16ed351/experiments/operator_optimization/evidence/combined_performance/result.json)
- [task重叠区间](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/16ed351/experiments/operator_optimization/evidence/combined_performance/overlap_intervals.json)
- [数值与逐核证据目录](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/tree/16ed351/experiments/operator_optimization/evidence)

本总结只整理已经完成的实验，没有为写作重新测量性能。原始大体积请求与trace继续保留在
`a3-21:/home/l00886679/projects/dsv41-tiny-prof-20261009/operator_opt/results/`。
