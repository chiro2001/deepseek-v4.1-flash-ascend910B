# 前缀方案未带来端到端收益：112份微架构数据及下一轮metadata试验

2026-10-10 · a3-21 chip8–15 · 正式TP8/40层/W4A8/384专家top6/Engram int8及完整视觉 · strict

GMM1前缀两方案精度通过，但12组同实例交错配对未显示性能收益，保留此前metastack。当前约19ms目标未达到。新的metadata静态几何候选已通过源码生成预检，正式整网审计正在运行；尚未计收益。

## 公平性能配对结果

profiler、路由/cache/消费者clone及CPU诊断关闭；2K输入/48输出、batch1、无推测解码，A=1。prefill和前9步排除。

|方案|ms/step|A|tok/s|
|---|---:|---:|---:|
|tp8base|30.137425|1|33.181335|
|tp8metastack|25.728940|1|38.866739|
|tp8prefixup|26.969480|1|37.078950|
|tp8prefixuproute|25.852680|1|38.680709|

`tp8prefixup` 在12/12组慢于metastack，配对额外耗时中位1.335455ms，配对加速中位0.950618。`tp8prefixuproute` 仅3/12更快，额外耗时中位0.183255ms，加速中位0.992881。没有将精度通过当作优化成功，也没有累计跨会话收益。

正式精度与异常修正见[报告v21](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-prefix-up-formal-pass-20261010-v21.html)：GMM2 prefix发生MTE非法GM访问，保留其原生counts；两个仅GMM1 prefix的候选真实8rank×40层独立比较、九个整网配对delta0、3840 GMM消费者及576 cache group精确。

## 微架构诊断：优化的工作量不在主导位置

计时完成后才采集7指标×2方案×8rank＝112目录，5步warmup/10步active，每份kernel_details非空并通过设备/步数验证，112份全部离线解析。原始CSV/trace留远端，各方案56份形状汇总与通信时间线、文件SHA和解析清单入库。

两采集方案为base与prefixuproute，后者还包含已有metadata/Indexer优化；因此全图task数量之差不能都归因于prefix。GMM1相同形状族在chip8的PipeUtilization如下，均为采样task/core归一化均值：

|GMM1计数|base|prefixuproute|
|---|---:|---:|
|AIC Scalar ratio|0.621790|0.583640|
|AIC MTE2 ratio|0.092173|0.090910|
|AIV Scalar ratio|0.240785|0.198705|
|AIV MTE2 ratio|0.236138|0.264308|

Scalar比例仅小幅下降，没有消除大部分控制成本。GMM2保持原生counts，其AIC Scalar约0.527→0.531、AIV Scalar约0.518→0.521。Cube、Scalar、MTE/wait可重叠；不能把一个核的ratio理解为整个核阵或可回收step时间。跨层相同容量shape混合了不同有效局部token数，min/median也不能直接当同输入配对kernel收益。

Memory、MemoryL0、MemoryUB、L2Cache及ResourceConflictRatio都有对应计数。例base GMM1：L1读53.416GB/s、L0B读6.181/写12.362GB/s；主存读0.1226GB/s、AIV UB读0.04954/写0.06143GB/s；MemoryUB另给Vector/Scalar访存细分。它们是当前profiling的task/core归一化数据，包含cache驻留/稀疏任务影响，不是全芯片HBM带宽或容量occupancy。L2记录hit/miss-allocate等计数，未测到UB容量占用。

输入容量为6行、局部有效行在精度采样中0–3/中位1，当前小M任务必须优先考虑控制与启动成本。源码上，一次cumsum方案每层新增转换；direct routing方案还需在下投影恢复counts。chip8采集显示ConcatD从1到41次/step，Sub从9到45；后者同时受既有metadata优化减少Sub的影响。恢复counts的工作抵消了GMM1边界读取节省，实际E2E结果已经拒绝采用。

精确去重通信包络后，两臂仍每步82个AllReduce。未将通信累计、calibrated start spread或采集态gap等同于纯网络等待。

## UB双缓冲的判断边界

A8W4 pre/post主要队列仍单缓冲；post行预算 `8.5*row*n+4*alignUp(row,8)+6*n+64<=ubSize` 已保存。当前小M下应先核对每核行/tile迭代是否有steady阶段，不能仅把depth1改2。没有双缓冲实际收益，也不把MemoryUB带宽当容量occupancy。L1/L0/UB的更多重叠、标量循环削减和静态展开仍需匹配具体tile并独立验证。

## 下一项实际试验：metadata静态几何

已新增 `tp8hostmeta`＝metastack＋缓存builder不变的cache-kind、compress-ratio、storage-block几何以及slot/SMLA/QLI共享键。原来每step约25个builder重复做类型判断、属性查询和字符串构造；新方案在初始化计算一次。保留所有动态位置、query/seq长度、slot值、skip标记、任务提交/等待和group独立输出，不缓存跨step动态值。cache-spec对象改变时回退既有ring builder。Compressor的storage block仍完整发布。

`/work/src_hostmeta_v2`的源码锚点及生成方法预检通过；`formal_hostmeta_audit_v1`已在复查资源/授权Alarm后启动，要求六配对完整路由/Top5/logprob、缓存消费者精确与实际spec-build覆盖。守卫通过后才自动启动12组关闭审计配对。本候选尚无精度/性能结论。

剩余优先方向为metadata准备与注册开销、未接整网的固定序通信、正式W4A8小M控制/缓冲，及经过符号/offset/assist-bias和显存预算验证的等值W4整数解包表示。后者当前只有源码证据，没有实施或收益。已失败/不匹配的HC与TP1专家特化保持原生。

最近独立客户端metastack为 `(26.292508ms/step,A=1,38.033648tok/s)`，GSM8K100/100、Vision23/23；API为实验释放，恢复记录保留。待新性能结果选择最优方案后继续部署与客户端验证。代码/紧凑证据/从提交源码生成MANIFEST/自检后双远端push，报告COS与links-server发布。完整约19ms目标继续推进。
