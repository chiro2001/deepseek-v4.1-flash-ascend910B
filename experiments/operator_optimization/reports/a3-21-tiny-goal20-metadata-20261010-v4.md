# Tiny metadata准备优化：CPU热点、动态坐标融合与20 ms目标验证

日期：2026-10-10 · a3-21 chip4 · `feat/tiny-operator-opt-20261009`

**最终组合达到19.209 ms/step、A=1、52.058 token/s。** 紧邻十组全部更快，且十组中位数均低于20 ms（19.029–19.343 ms）。
20 ms目标的正式性能与精度验证已通过，独立tiny服务已恢复并验收。

## 1. 为什么改变优化重点

上一轮已采纳GMM1/激活融合，紧邻十组结果为21.991 ms/step、A=1、45.474 token/s。
本轮首先测试相同BN16/BK512的multibuffer：default18.501 us、off18.591 us、on18.476 us，
全元素逐位相同，但on仅改善约0.025 us，没有实用收益，保留默认。

随后试验跳过TP1同步scheduler、单tokenFULL图replay前的stream同步。
三组实际路由、48个输出token和logprobs通过，却没有稳定性能收益（一组更慢），不采纳。
47个实际decode图的event中位12.144 ms、replay调用墙钟0.112 ms。
图event与约22 ms step的差值不能全部叫CPU时间：lm_head、sampler及图外设备准备仍在其间。

CPU诊断进一步对应到具体位置。39个诊断步中：

|CPU函数|调用总数|累计时间|解释|
|---|---:|---:|---|
|`_build_attention_metadata`|39|311.492 ms|约8 ms/步，包含下方子调用|
|`_build_attn_group_metadata`|975|217.132 ms|25次/步|
|`DeepseekV41MetadataBuilder.build`|975|191.645 ms|各资源builder重复准备|
|`_config_value`|4056|37.132 ms|静态配置重复读取|
|`Tensor.copy_`|2340|39.343 ms|大量小tensor准备|
|`build_c2_metadata`|39|22.660 ms|ring、mask、source positions及RoPE gather|

cProfile/event会扰动执行，表中累计值彼此包含，不能相加或当作正式可回收墙钟。
`_bookkeeping_sync`约487 ms的Event.synchronize主要等待设备，也不能再加到图时间上。
这组证据说明低效位置除了核内搬运/计算，还包括图外的metadata生成、Python分派和许多小任务。

## 2. 本轮实现及原理

三组候选均以已验证的`gmmact`为参照，模型计算、实际路由和同步定义保留。

- **mdstatic**：在builder初始化时缓存sliding_window、head数量/维度和index配置。每步seq_lens、positions、slot及长度仍来自当前scheduler，不缓存旧step结果。
- **mdslots**：在mdstatic基础上，将每个物理group的压缩slot转换、完成条件、padding mask、page/offset div/remainder和两次copy合成一个Triton Vector kernel，写回同一个persistent `[T,2]` buffer。
- **mdall**：再把C2 ring计数合成一个核，将complete mask、源位置和两张RoPE表的gather合成另一个核。任务仍提交到原来的COMPRESSOR阶段，inputs-ready、stage-ready、buffer-reuse等事件和group id不变。

原slot链读取的原始physical slot、C2位置奇偶和query有效范围全部保留。
负slot或未完成组输出`[-1,-1]`；跳过ring更新时禁止写入；不同物理group的输出buffer保持独立。
C2源位置只有有效完成token才使用`position-1`，否则为0；无效token仍按原定义读取RoPE第0行。
ring owner、已使用query长度、已完成上下文长度及buffer尾部均保留原语义。

最后加入**mdfull**：启用仓库已有的受保护多group原始slot mapping路径，将11次原始坐标计算合成一次二维grid启动。
它与上面的2D page/offset准备是不同步骤：前者从每group的block table和positions生成原始physical slot，
后者再为V4.1 cache plane形成`[page, offset]`及压缩写入mask。每个group仍使用自己的输入/输出指针，circular组PAD填充保留。
TP1、dtype、连续性和物理块布局前置条件仍由原代码检查；未覆盖的形态显式回退。

代码从安装vendor的build函数用严格唯一source anchor生成候选函数，vendor文件不被修改；
源码变化不匹配时直接失败。当前vendor build SHA256：
`7b3ed08e09d36afced350b36ead0e759183e6644910d5c2643a5f4144dfe5e94`。
同一步的batch共享、group-local共享和native metadata任务仍沿用原代码。

## 3. 精度与覆盖

slot的224个用例、C2 ring/RoPE的144个用例全部逐位一致。
包含负PAD、block边界、高physical page、C2奇偶、实际请求数为0、skip更新、prefill/padded形状及尾部不变。

完整模型三组非恒定gate、HC及专家权重审计通过。
三个候选的全部实际专家路由、48个输出token与参照一致，最大logprob差为0。
每套图仍有40层GMM1/激活融合、80次HC/HcPost、40次路由准备/汇合等真实覆盖，
metadata build计数按每请求累增，动态数据没有跨step冻结。

mdfull另做三组审计：原始slot的fused/native路径每次逐位verify、实际路由、token和logprobs全部通过，最大logprob差0。
审计末累计240次真实融合、覆盖11个group、fallback=0。正式计时关闭verify；匹配采集中累计624次真实融合、fallback=0。

## 4. 正式性能与匹配profiling

关闭profiler/审计的同进程十组正反交错，2048输入/48输出，A=1，
prefix关闭、KV=4 GiB、同CPU绑定，FULL_DECODE_ONLY `[1]`、static kernel真实启用。
每臂两次预热；审计计时、诊断计时和首次编译时间均不作为正式性能。

第一套会话用于消融（十组每个候选均快于参照）：

|模式|ms/step|A|token/s|
|---|---:|---:|---:|
|已采纳gmmact参照|22.265|1|44.913|
|静态配置缓存mdstatic|21.711|1|46.060|
|再加slot融合mdslots|20.386|1|49.054|
|再加C2融合mdall|**19.970**|**1**|**50.075**|

mdall的十组范围19.938–20.026 ms，有三组略高于20 ms。随后用紧邻第二套会话测试多group启动融合：

|模式|ms/step|A|token/s|
|---|---:|---:|---:|
|mdall参照|19.857|1|50.360|
|增加多group启动融合mdfull|**19.209**|**1**|**52.058**|

配对吞吐改善中位约3.43%，十组全快、十组均低于20 ms。两套会话不能相减或累计百分比。

两套匹配20步的Device4、trace解析、CSV哈希和替换计数均通过：

|同进程匹配采集|参照task/步|候选task/步|参照kernel累计|候选kernel累计|
|---|---:|---:|---:|---:|
|gmmact→mdall|1718|1536|14.619 ms|14.601 ms|
|mdall→mdfull|1536|1526|14.496 ms|14.412 ms|

第一套kernel累计几乎不变，而正式step明显改善，指向metadata生成和分派的优化。
第二套原始slot kernel从11次/步、累计33.865 us变为1次/步、6.681 us；减少的设备时间仍小于完整step改善。
每步保持12次group-local slot融合、1次ring counts、1次ring source及原有40层模型核覆盖。

PipeUtilization的task duration加权口径下，2D slot融合核Vector约53.3%、Scalar约41.9%、MTE2约0.46%、MTE3约0.92%。
这里主要是小规模坐标运算和控制，不能凭带宽低就判为HBM或UB容量瓶颈；增加dual-buffer并不是这项优化的依据。
上述比例有重叠，不代表整芯片利用率。没有把persistent GM buffer大小误称为UB Occupancy。
本轮没有新增容量Occupancy或指令级TimelineDetail；GMM与注意力已有Default/MemoryDetail证据见v3及前序报告。

mdall的39步CPU诊断中，`_config_value`已不再出现在采样调用中；metadata生成仍是热点，Triton JIT启动变得更显著。
此诊断与原诊断属于不同会话，仅用于解释调用结构，不据此计算CPU百分比收益。
采集态kernel累计不是端到端墙钟，微架构计数也不能相加。

## 5. 服务、边界和其他机会

独立服务默认`GOAL20_ARM=mdfull`，static启用、verify关闭；模型id归属正确，health=200，32输入→16输出token验收通过。
服务地址：`http://172.17.0.4:18971`。服务冒烟不作为跨进程吞吐证明，正式性能来自上述同进程基准。

主要证据目录：`goal20_metadata_audit_v1`、`goal20_metadata_perf_v1`、`goal20_blockmap_audit_v1`、`goal20_blockmap_perf_v1`，
以及slot/ring边界JSON、`goal20_mdfull_service_smoke.json`。coverage摘要包含实际路由与输出token哈希、动态builder计数和融合/回退次数。
小型结果和源码位于`experiments/operator_optimization/`；raw大trace及请求仍在远程`/work/operator_opt/results/`。

本轮始终排除chip14–15，不操作其他租户；chip6的AICPU注册失败无有效计时，继续保留故障证据。
SSH曾短暂中断，恢复后先核验容器进程和结果文件，没有把中断当成成功或失败的性能证据。

完整机会总账在`experiments/operator_optimization/OPTIMIZATION_OPPORTUNITIES.md`。
尚可研究跨group的2D slot单次启动、缓存已编译Triton launcher、将C2两个融合核再合一、
norm/RoPE/cache邻接融合、Q_a/KV合并分派及隔离重测head分组。
已有wo_a/GMM2/shared布局和HC/RMS精度失败不会无依据重做。

TP1 tiny dummy BF16的结果不外推到生产TP8/W4A8、真实checkpoint、DSpark或CED-PD。
当前vendor的RmsNormCast FP32输出=最终BF16.float()已由源码及上板证实，但固定路由精度门槛仍保留。
