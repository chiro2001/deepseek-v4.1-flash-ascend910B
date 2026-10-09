# 向20 ms/step推进：算子筛选、路由与HC优化、静态核和注意力逐核诊断

日期：2026-10-10 · a3-21 / chip4 · `feat/tiny-operator-opt-20261009`

**当前最佳正式结果为22.674 ms/step，A=1，44.103 token/s。20 ms/step目标尚未达到，goal继续active。**
本阶段在已有HC/router、专家激活融合和Q/KV多流基线上，增加单token路由、HcPost分块、
HC静态HF32权重预处理和GMM1 Vector候选，并验证了static kernel配置。

组合方案在两套独立的同进程六组对照中均全部更快；两套会话的绝对数值不相减、百分比不累加。
本报告是持续优化的阶段报告，不是20 ms目标完成报告。

## 1. 本阶段完成的工作

|工作|产出与结果|
|---|---|
|环境核验|确认chip4占用属于自己的tiny；核对容器label、模型id和API进程，只停止自己的API|
|重测已有基线|六组交错：激活融合+多流24.689 ms/step，A=1，40.503 token/s；单流/多流匹配20步profiling验收|
|更新热点|按真实shape重新聚合1798 tasks/step，区分wo_a、GMM1/GMM2、C128/C2/SWA注意力|
|HC筛选|静态HF32权重及K/output tile共9配置；保留UB溢出失败，未放宽精度门槛|
|投影筛选|wo_a的Vector/Cube，GMM、shared和宽投影多个布局/tile；保留慢候选及编译器断言|
|路由与HcPost|全部64种expert对；路由初始化/汇合各4种block，HcPost八种block/FMA组合|
|模型接入与审计|独立图bank、覆盖守卫；非恒定gate/HC/MLP权重、真实路由及输出审计|
|正式性能|关闭profiler和审计，六组正反交错；之后单独采20步profiling|
|静态核|确认LOCAL_WORLD_SIZE=1与static shape kernel路径，完成六组计时和三组非恒定审计|
|注意力诊断|C128/C2的逐核Default及MemoryDetail；独立复现必须先准备vendor路径再初始化设备|

实验仍为TP1 tiny dummy BF16、40层、8专家/top2、intermediate=256、MAX_SEQS=1，
Engram和DSpark关闭。未验证真实checkpoint、TP8/W4A8、CED-PD长上下文或生产吞吐。

## 2. 正式性能与比较口径

请求为2048输入、48输出token，seed轮换0/1/2，prefix cache关闭、KV=4 GiB、
FULL_DECODE_ONLY `[1]`、相同CPU绑定。每臂预热两次；六组正反交错，计时关闭profiler、
logprobs及路由回传。所有臂使用同一模型进程、同一权重，各自保存完整图及attention参数。

未启用static kernel的同一会话：

|模式|ms/step|A|token/s|六组全部更快|
|---|---:|---:|---:|---|
|已有激活融合+Q/KV多流|24.270|1|41.203|基线|
|仅HC静态预处理|24.206|1|41.312|否|
|仅HcPost候选|24.046|1|41.587|否|
|仅单token路由候选|23.874|1|41.887|是|
|仅GMM1候选|24.132|1|41.439|否|
|四项组合，GMM2保留原生|**23.541**|**1**|**42.479**|**是**|

小收益的单项结果存在波动；不能把五个中位数的差值相加，也不能据此宣称各项都稳定有效。
路由与组合的六组结果更稳定，后续仍需更紧邻的配对及组合消融，判断其余三项的净贡献。

另一次启用static kernel的同进程会话：

|模式|ms/step|A|token/s|六组全部更快|
|---|---:|---:|---:|---|
|已有激活融合+多流，static kernel启用|23.461|1|42.624|基线|
|四项组合，static kernel启用|**22.674**|**1**|**44.103**|**是**|

该会话内组合的配对加速比中位约1.0345。static开关的OFF/ON属于不同进程，
因此本报告只列出各自绝对结果，不计算跨进程的开关收益。
22.674仍高于目标；A=1时20 ms/step对应50 token/s。

## 3. 优化原理与语义保持

### 3.1 单token路由准备

原生InitRouting面向通用token/expert规模，当前一个token/top2仍启动48个Vector block。
候选仅覆盖TP1、8个本地expert、top2、BF16及未量化输入；其余情况保留原路径。

一个kernel完成两份输入复制、两个slot的排序/反向映射和8个expert计数。
直接输出int64计数，避免后续40次计数Cast。汇合按原始slot对应的概率加权，保留概率的BF16转换。
没有替换上游TopK、bias、hash/token-ID策略或权重归一化。

全部64个ID对（含重复ID）与native比较通过。实际模型40层的初始化及汇合输出也一致。

### 3.2 HcPost分块

HcPost计算每个输出通道的四路残差组合及主分支：

```text
y[h,d] = x[d] × post[h] + Σ residual[r,d] × comb[r,h]
```

候选按D维切成512元素tile，让多个Vector核共同处理，保留DAV_2201原生路径的FP32运算顺序：
先x×post，再依次加入残差行0、1、2、3，最后转BF16。
单独图的收益很小，实际模型中有改善趋势；组合审计的80次输出逐位一致。
FMA和多种block的筛选记录均保留，正式组合采用未改变原舍入顺序的版本。

### 3.3 HC静态HF32权重

原project_hf32每个K tile重复对静态权重进行位级HF32转换。
候选在捕获前按既有校准规则转换并缓存，缓存键包含权重对象、地址、版本、shape和stride；
捕获中遇到未准备权重会拒绝执行，避免偷偷把转换放进decode图。

BF16有限输入本已能被HF32精确表示，因此投影内使用BF16→FP32转换后的x，省去冗余位操作。
20次Sinkhorn、eps、pre_mix及输出定义保留。默认配置的独立图约1.056×，
实际模型的单项改善小，仍需消融确认组合中的净作用。

### 3.4 GMM1小M Vector路径

原GMM1为`[2,5120]×[8,5120,512]`，只使用4个Cube block。
候选在设备端从group list确定两个slot的expert，读取对应权重做Vector GEMV。
权重在捕获前转换到明确的ND、N-major布局；不把逻辑contiguous当作物理ND。

独立筛选的较优配置为BN=8、BK=512，约1.253×；模型中改善较小。
实际图40次输出与native最大绝对差约3.815e-6，真实路由及token保持一致。
GMM2采用同一Vector思路明显更慢，已经排除，不能由GMM1结果推断两者都适用。

### 3.5 static kernel配置

配置启用框架已有静态形状核路径，限定decode capture size为1。
日志确认LOCAL_WORLD_SIZE=1以及“static shape kernel will be used”消息。
不以compile-start次数判断是否生效，因为编译缓存可使该次数为0。

当前仅证明该配置下组合稳定、更快且审计通过；还没有同一模型中切换static开关的严格对照。

## 4. 数值、路由与图覆盖

模型审计在编译前随机化40个gate、80组HC参数以及80个routed/80个shared MLP权重，
避免dummy恒定权重造成空洞验证。三组组合审计、三组static组合审计均通过。

每请求记录`[2095,40,2]`实际路由，共83800个top2对；ID合法，非退化全零。
原生与组合的实际路由及48个输出token一致，最大top-5 logprob差约9.537e-7。
激活目标形状输入标准差非零，输出仍与既有native参照一致。

每套图检查HC/router/激活覆盖及新候选选择数：HcPost=80、路由初始化/汇合各40、
GMM1/GMM2各40。第一次接入因HcPost未覆盖而被守卫拒绝；把动态batch分派移入opaque op后重跑通过。

这是tiny覆盖范围内的验证，尚不构成生产质量或真实checkpoint验收。

## 5. 匹配profiling与注意力逐核数据

未启用static kernel的baseline/combo各采20步，设备均为4，trace、统计文件及CSV哈希通过验收。

|每步结构|基线|组合|
|---|---:|---:|
|全部task|1798|1758|
|project_hf32 / project_static|80 / 0|0 / 80|
|HcPost / 新hc_post_kernel|80 / 0|0 / 80|
|MoeInitRoutingV3 / 新route_init_kernel|40 / 0|0 / 40|
|GroupedMatmul / 新Vector GMM1|80 / 0|40 / 40|
|全部kernel累计时间|16.853 ms|16.284 ms|

新kernel均为AI_VECTOR_CORE；减少的40个task主要来自路由计数Cast。
kernel累计时间是采集态归因数据，不是无profiler的可回收墙钟。

注意力40次/步实际分为C128 20次、C2 18次、纯SWA 2次，不能把主形状误归为SWA-only。
独立复现使用相同outer cache shape、head数、分页格式、窗口与512个稀疏index slot。
复现需要在设备初始化前bootstrap自定义vendor；系统算子与V4.1 vendor的ratio/mask ABI不同。
失败的系统模板及HCA采集全部排除，最终使用模型对应的CSA C128/C2路径。

C128 Default已采到24个Cube、48个Vector，1800 MHz，task约31.019 μs。
最慢Cube核约25.546 μs：Cube 1.998、MTE2 5.098、FixPipe 6.670、Scalar 20.912 μs；
wait_id8约2.631 μs，wait_id6约0.342 μs。Vector0的wait_id3约8.639 μs，
最慢Vector约28.266 μs，其Vector工作约3.543、MTE2约7.251 μs。

**逐核分布还发现：两个独立复现均只有一个Cube核有实质矩阵计算、两个Vector核有实质Vector工作；
24/48是启动数量，不代表24/48核都在计算。** C128的Cube时间中位为0、最大1.998 μs，
C2的Cube时间中位为0、最大6.543 μs。源码的CSA `mBaseSize=groupSize`使单query的64个头作为一个工作单元；
这解释了小batch下的工作集中，是下一步核分工实验的依据，尚未证明拆分一定更快。

|Default逐核位置|C128|C2|
|---|---:|---:|
|task μs|31.019|44.279|
|最慢Cube执行 μs|25.546|38.903|
|最慢Vector执行 μs|28.266|41.670|
|Cube wait_id6最大 μs|0.342|10.807|
|Cube wait_id8最大 μs|2.631|3.254|
|Vector wait_id3最大 μs|8.639|2.290|
|Vector wait_id9最大 μs|3.772|6.028|

arch22源码映射为：flag6是V0输入准备→Cube BMM1，flag7是BMM1→Vector，
flag8是Vector softmax→Cube BMM2，flag9是BMM2→Vector；flag3另有队列/同步用途。
C2的flag6长等待与Vector MTE2搬运值得优先研究，不能机械删除依赖。

MemoryDetail中，C128核0 GM→L1为466 KiB；两个Vector核GM→UB约228.625/212.625 KiB。
C2核0 GM→L1为1520 KiB；两个Vector核各GM→UB约528.625 KiB、UB→GM约392 KiB，
工具路径利用率约5.30%/4.63%。这些是逐核口径，不是全芯片HBM。

这些计数重叠，不能相加；它们表明工作集中、搬运和CV依赖之间仍有等待，不能只根据低带宽追加UB buffer。
warm独立kernel的ICache miss明显低于模型采集，需进一步用混合回放核验指令工作集的作用，
目前不把它直接判为模型主因。MemoryDetail逐核口径也不能换算为整芯片HBM利用率。

## 6. 负结果及尚未采纳的方案

|尝试|结论|
|---|---|
|HC增加parts、改变BK/BY|多种配置更慢；BY=8192因UB溢出拒绝|
|wo_a Vector六种布局/tile|均更慢，未采用|
|wo_a Cube五种tile|可运行的较优配置仍比native慢，未采用|
|GMM2 Vector|独立图和模型都更慢；已从组合移除|
|shared下投影、q_b、wo_b Vector|当前已测配置未找到适合接入的收益|
|shared上投影|出现独立候选，真实模型的原始N-major布局还需匹配筛选|
|SWA单kernel Cube FA十二配置|Triton/BiSheng代码生成失败；没有当作性能数据或接入模型|
|首次HcPost模型接入|覆盖不足被守卫拒绝，修正后重新审计|
|错误vendor/模板的注意力采集|应用或tiling失败、无CSV，不计入微架构证据|

初次七模式审计中的combo包含慢GMM2，对应现在的`all_candidates`；正式combo已移除GMM2。
该差异在证据provenance中明确记录，不能混用两个组合的结果。

## 7. 接下来推进到20 ms的工作

1. 对路由与组合做紧邻配对、leave-one-out消融，确定HC/GMM1小收益是否值得保留。
2. 在C128/C2逐核数据和源码标志映射基础上调整分工、tiling和同步路径；保留所有KV、稀疏indices和sink语义。
3. 尝试GMM1与激活epilogue融合、GMM2支持的Cube小M配置以及shared真实布局；不复用慢Vector下投影。
4. 研究HC finish与全维RMS/RmsNormCast融合，以及norm/RoPE/cache store等邻接短task。
5. 核验图回放与metadata下发开销；不能把采集态空档全部视为host可回收时间。

UB双缓冲的前提仍是多独立tile和足够容量；已有自动multibuffer负结果不重复作为新收益。
Cube投影优先检查L1/L0流水和分派。任何接入继续经过同进程配对、非恒定数值/真实路由、图覆盖及服务验收。

## 8. 证据与复现

实现与小型JSON/CSV在`experiments/operator_optimization/`，原始请求、trace和编译错误留在
`a3-21:/home/l00886679/projects/dsv41-tiny-prof-20261009/operator_opt/results/goal20_*`。

```bash
bash /work/operator_opt/run_in_container.sh bench_goal20_model.py --pairs=6 --profile --output=/work/operator_opt/results/NEW_PERF
bash /work/operator_opt/run_in_container.sh bench_goal20_model.py --pairs=3 --audit --arms=baseline,combo --output=/work/operator_opt/results/NEW_AUDIT
bash /work/operator_opt/run_in_container.sh bench_goal20_model.py --pairs=6 --static-kernel --arms=baseline,combo --output=/work/operator_opt/results/NEW_STATIC_PERF
bash /work/operator_opt/run_in_container.sh validate_goal20_profile.py --root=/work/operator_opt/results/NEW_PERF
```

执行前核验chip4及进程归属，仅停止自己的服务，排除14–15。
`serve_goal20_tiny.sh`提供组合+static的独立服务入口；本阶段持续测量期间API暂时停止，避免同芯片干扰。
此报告的发布与仓库提交不表示goal完成。
