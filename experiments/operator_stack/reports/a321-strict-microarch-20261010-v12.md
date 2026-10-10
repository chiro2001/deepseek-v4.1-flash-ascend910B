# strict规约复验与正式TP8微架构定位

2026-10-10 · a3-21 physical chip8–15 · 正式40层/hidden5120/W4A8/384专家top6

用户提醒的历史环境变量是`HCCL_DETERMINISTIC`，启动脚本通过`HCCL_DET`传入。本轮已在新worker和通信域初始化前设置`strict`，原生图三组重复请求、base/core/stack六组配对的完整路由、Top5集合和logprob全一致，最大差0。当前精度问题的定位和这一修复互相支持：8rank的注意力局部计算原先一致，首个差异在layer0 wo_b AllReduce输出；固定rank输入的原生规约也不能重复。它不是“优化版输出和一个稳定基线不一致”的证据，因为原来的基线自身先不稳定。

本轮没有改变数学门槛。正式关闭审计/profiler的最好结果仍是core：`(26.622860ms/step, A=1, 37.561704tok/s)`，目标≤约19ms尚未达到。新增profiling只增加定位证据，没有新增可计的端到端收益。

## 环境变量的历史边界

合法值为`false/true/strict`，不能写`1`。历史`true`虽然降低了重复请求的logprob spread，但GSM8K为91/100；`strict`曾达到GSM8K100/100、Vision23/23。后续复测更正了“某个上下文阈值以内必然确定”的说法，因此不能把历史通过当作当前模型的普遍保证。出处为`CORRECTNESS_STATUS.md` §2/§6.3及正确性线原始记录。

本轮实际通过范围：正式checkpoint、2K输入、batch1、47/48输出、FULL_DECODE_ONLY capture `[1]`；三次原生控制、六次候选配对均delta0，72份rank消费者Indexer/cache审计逐位一致。当前模型客户端GSM8K/视觉验证仍待完成。

## 本轮完成的采集与分析

`formal_strict_profile_v2`正常退出0。base/core各采七组指标、八个rank，每份十个稳态decode step，共112份目录。指标为PipeUtilization、ArithmeticUtilization、Memory、MemoryL0、MemoryUB、L2Cache、ResourceConflictRatio，CPU/NPU trace和HCCL时间线一并保存。

v1遇到EI0020，NPU socket16666被占用且没有有效采集；v2使用官方支持的`HCCL_NPU_SOCKET_PORT_RANGE=auto`，没有停止其他租户、reset设备或清page cache。112份CSV均已由非daemon、四并发离线解析完成。两臂各56份微架构分析验证了Device8–15对应关系和每rank十步。

torch_npu同时记录HCCL逻辑包络和AivKernel。分析器只在Step/Device/Type/起始时间/时长完全对应时去掉重复包络；每步保留82次AllReduce执行。**strict下实际仍可见AivKernel，不能宣称已切换为AI CPU展开。**

新增`analyse_formal_timeline.py`使用跨stream区间并集，span末端取`max(end)`；保留MEMCPY和EVENT等其他任务对kernel间隙的覆盖信息。非单调结束时间、重叠区间和phase分区一致性验证通过；本批原临时脚本末端短缺实测为0，但新算法避免以后遇到其他流晚结束时低估。离线解析日志名也改为包含完整相对路径，防止不同rank重名覆盖，不必重解析已经有效的112份CSV。

原始CSV/trace及大于1MB的完整分析保留远端；本地只拉取小于1MB的紧凑摘要及SHA。没有用SSH搬运大文件。

## 延迟集中在哪里

下表来自chip8的PipeUtilization采集，用80个HcPre的Model ID确定主图边界。kernel union包括实际HCCL执行，kernel gap是这些记录之间的空档；MEMCPY、EVENT等任务可能填充空档。**这不是端到端计时，也不是已证明可以回收的CPU时间。**

|区段|base span / union / gap (ms)|core span / union / gap (ms)|判断|
|---|---:|---:|---|
|主图启动前|14.596 / 2.914 / 11.682|10.711 / 3.428 / 7.283|长间隙的主要集中位置|
|主图|20.192 / 18.888 / 1.304|19.329 / 18.167 / 1.162|图内执行仍是主要计算负载|
|主图之后|0.254 / 0.198 / 0.056|0.266 / 0.192 / 0.074|本采集下占比较小|
|整个kernel窗口|35.041 / 22.000 / 13.042|30.307 / 21.787 / 8.520|含profiler、RPC/到达及发射影响|

core多个最长间隙位于图外的`Sub → slot_mapping_kernel`之间，约1.0–1.15ms；期间记录到CPU `aten::index/sum/copy_`等调用，且存在少量MEMCPY。base对应位置是`Sub → Cast`，还有较多分散的小算子。CPU调用的时间重叠本身不能证明因果，下一步须结合调用源码和关闭采集的对照。

HCCL序号0在图外、序号1进入主图，采集态前两个AllReduce的8rank start spread分别平均2163.825µs、567.15µs；结束spread约19µs。它们明显含rank到达、发射和时钟校准影响。按执行累计，chip8每步AllReduce为base4899.161µs、core5378.365µs；不能据此断言core通信更慢或宣称能回收5ms，累计允许重叠，且两者的端到端结论必须来自关闭profiler的配对。

## 算子内部的不同瓶颈

以下为core chip8的原始任务/核心归一化计数均值。流水ratio可重叠；不能把各流水ratio相加为利用率，也不能将这些GB/s直接当成整chip HBM实测带宽。

|算子与真实shape|次数/step|执行累计µs/step|主要计数|优化方向|
|---|---:|---:|---|---|
|wo_a：`[1,4096]×[4096,1024]`|40|945.490|AIC MTE2 0.872、MAC0.052|权重搬运、L1/L0 tiling与预取|
|wo_b：`[1,1024]×[5120,1024]`|40|547.615|MTE2 0.756、MAC0.089|小M搬运/布局；通信须另看|
|router：`[1,5120]×[384,5120]`|40|537.228|MTE2 0.713、Scalar0.283|正式384/top6契约的小M路径|
|GMM up＋SwigluQuant|40|2051.231|AIC Scalar0.600、MAC0.0137|专家遍历、地址/tiling控制、空expert开销|
|GMM down|40|1394.285|AIC Scalar0.477、AIV Scalar0.486|同上，保留量化与舍入顺序|
|HcPre|80|2046.187|原生混合核；正式候选数值门尚未通过|当前保持原生|
|SparseFlashMla|40|1250.986|部分shape Scalar高、MAC低|切分/控制成本与cache访问分开分析|

Memory提供了HBM/L1/UB带宽，MemoryL0提供L0A/B/C读写，MemoryUB拆分Vector/Scalar访问，L2Cache提供命中/分配miss计数，ResourceConflictRatio提供冲突计数。这些数据已保存，**但没有测到UB容量occupancy**。Block Num只是启动配置，不能当作活跃核占用率。部分零计数不能证明“不访问该buffer”。

## UnifiedBuffer双缓冲是否合适

CANNBot双缓冲设计要求不仅`InitBuffer num=2`，还要有跨tile预取/计算/写回流水、正确的事件同步与UB预算。只有增加buffer数，不能证明隐藏了延迟。其MatMul/MC²部分参考明确面向DAV_3510，不能直接假定适用于当前910C的DAV_2201路径。

* wo_a/wo_b主要是AIC的MTE2，优先查L1/L0权重tile与预取；UB双缓冲不是可直接替换的全局开关。
* GMM的Scalar高且MAC低，当前更支持查控制循环和稀疏专家工作分配；双缓冲未必能解除Scalar发射瓶颈。
* batch1的短Vector输入可能一次就能放入UB，增加双缓冲同步可能更慢。必须有至少多个有效tile、容量预算和真实重叠证据后再对照。

因此本轮未宣称UB双缓冲有收益，也未把技能文档的示例百分比套在模型上。

## 尚未尝试、可继续验证的方案

1. **图外metadata准备融合或进入可复用图**：在已有core上继续减少slot/ring/索引/类型转换的发射链，动态位置、实际token数、Query和group-local buffer每步仍更新；先逐位消费者验证，再整网路由/Top5/logprob，最后关闭审计配对。先定位原生调用栈，避免把profiler额外开销算为收益。
2. **固定顺序通信与展开策略**：六种AllGather求和配置已独立数学参考和三次图重放通过，但未接正式整网；与strict基线比较完整精度后才测端到端。独立事件约284–290µs不用于推算整网通信成本。
3. **正式W4A8稀疏专家和小M tiling**：针对每rank48个专家、top6实际输入研究控制循环、空expert、权重tile和预取；需要当前910C可编译的实现与真实shape，不能启用tiny的8expert/top2候选。
4. **短Vector融合/核数与缓冲策略**：按kernel数量及依赖序列选择，而不是根据低带宽泛化。UB双缓冲、单缓冲常驻和核数裁剪都保留为需要独立证据的候选。

已失败的正式HC static/HcPost保留原生；BF16激活候选在正式W4A8覆盖为0；Indexer融合已过整网门，但当前没有明显稳定额外性能收益，不重复计收益。

## 服务与客户端验收进度

`formal_core_strict_service_v1`已在复查chip8–15无占用及授权的80C98001 Alarm后启动，使用正式权重、strict、auto端口、loopback18761和独立模型名`dsv41-a321-formal-core-strict-20261010`。选择器核验已完成的精度与12组性能证据后选取core；服务正在加载/初始化，尚不能称为通过客户端验收的服务。

新增串行验收runner：先核对`/v1/models`和唯一API进程argv，再依次执行8个不同2K prompt的客户端性能、23例视觉和GSM8K-100。正式不启用推测解码，A=1；客户端工具原始无spec计数为0时须依据这一配置解释，不能报A=0。

GSM8K新增可选官方train/test JSONL路径，沿用相同训练集前8题few-shot与test顺序，记录SHA，避免依赖宿主Python3.14上大且缓慢的datasets/pyarrow下载。客户端语料从已push的内网git镜像只读archive到本任务目录，模型环境没有global pip改动。原始请求留远端。

Goal保持active。19ms、客户端质量/时延、最优正式服务验收仍未完成。本轮源码、紧凑证据、报告将自检并推送GitHub和内网镜像。

## 证据与资料

* `evidence/formal_a321/results/formal_strict_profile_v2/`：启动、退出、112份解析manifest、两臂shape/通信摘要、时间线v2。
* `evidence/formal_a321/strict_profile_validation.json`：设备、步数、区间并集、phase分区与末端校验。
* 远端`/work/results/formal_strict_profile_v2/`：原始prof、`timeline_analysis_v2.json`及完整微架构报告。
* [当前12组正式性能报告v11](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-strict-stack-performance-20261010-v11.html)。
* [CANN HCCL_DETERMINISTIC](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/900/API/hcclug/hcclenvref_07_0010.html)。
* [CANN HCCL_NPU_SOCKET_PORT_RANGE](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/maintenref/envvar/envref_07_0144.html)。
* 本地CANNBot：`a2/agents/PROF_WIRE/evidence/cannbot/model-infer-profiling/SKILL.md`、`model-infer-perf-breakdown/SKILL.md`；`a2/agents/CANNBOT_DOC/upstream-cannbot/ops/ascendc-performance-best-practices/references/elementwise/double_buffer_design.md`及Scalar指南。本轮沿用已授权的正式模型、decode/rank范围与自定义精确去重分析，没有启动技能的交互式层sample确认流程。
