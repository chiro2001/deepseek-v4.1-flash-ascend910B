# 正式异步TP8验收与真实权重tiny对齐，目标提高到17ms

2026-10-11 · a3-21 physical chip8–15 · 正式W4A8/Engram int8 · strict

用户将最终目标提高到 **正式客户端≤17ms/step**。当前正式异步metastack客户端已完成GSM8K **100/100**、视觉 **23/23**，八条串行请求测得 **(19.376255ms/step,A=1,51.609559tok/s)**，仍未达标。下一阶段优先解决快速tiny与正式TP8的契约差异，再继续优化。

## 已核对的正式结果

恢复的前一个本目录会话是 `01a11fdd-c5ae-7a31-b350-9f4bb665c166`；最后因上下文容量耗尽而中断。恢复后以落盘文件和实际进程为准，核对了当时未交接完的异步结果。

`formal_async_audit_v1`：三次原生控制、三bank六个配对通过；21个请求与同步正式参考的完整路由、token、Top5/logprob一致，最大差0。消费者检查沿用独立整网审计，性能测量关闭路由/clone/profiler。

`formal_async_perf_v1`：同实例、同八个worker、12组交错配对；异步按真实token到达墙钟除以生成token数计时，不用CPU polling次数报价。

| 方案 | ms/step | A | tok/s |
|---|---:|---:|---:|
| 原生bank |20.406689|1|49.003539|
| metastack |19.396068|1|51.556842|
| hostmeta |19.418334|1|51.497723|

metastack为这轮整体中位最快者。此前同步hostmeta客户端26.133047ms属于另一会话与调度模式，不能与本轮相减宣称配对收益。

`formal_best_async_service_v5`：正式40层/hidden5120/384专家top6、W4A8_DYNAMIC、两张Engram int8表及完整视觉；源码 `/work/src_async_serve_v3`，原loopback18765，独立模型名 `dsv41-a321-formal-best-v5-async-metastack-20261011`。客户端在发请求前核对 `/v1/models` 和唯一API argv，退出0、GSM8K100/100且无空答/错误、视觉23/23。串行八请求、2K输入/256输出、无推测解码，客户端三元组为 `(19.376255,1,51.609559)`。

17ms对应A=1时至少58.823529tok/s。当前还需约2.376255ms/step的时延下降。客户端入口新增 `--target-ms`，默认17；历史验收记录不改写。质量验收要求100/100与23/23。

## 为什么重做tiny

旧tiny为TP1、dummy BF16、8专家/top2、Engram关闭；正式路径为TP8/EP8、384专家/top6、本地48专家、W4A8融合GMM和真实Engram。旧tiny的HC、selected-GMM、router等候选已在正式迁移中失败或不匹配，不能用旧tiny的约19ms代替正式验收。

新tiny仍使用同一生产镜像和TP8/EP8，只缩减到八个代表层。保留hidden5120、FFN2304、384专家/top6、量化描述和实际张量、64注意力头、两张真实Engram表、RoPE/cache几何、FULL_DECODE_ONLY `[1]`、strict、图片额度4、A=1。所有张量在a3-21本地复制，逐张量计算源payload SHA256，再独立重读输出校验；Engram大表保持同一inode，不复制约206GiB表数据、不经SSH传输权重。

| 正式原层 | tiny层 | 保留的角色 |
|---:|---:|---|
|0|0|local/SWA|
|1|1|local/SWA＋Engram第一张表|
|2|2|C2 KV/Indexer源|
|3|3|C2共享消费者|
|14|4|另一C2源＋Engram第二张表|
|20|5|global KV/Indexer源、candidate源配置|
|21|6|global共享消费者|
|24|7|global共享KV、独立Indexer源|

生产开关 `V41_QLI_NO_CANDIDATE=1` 同样保留；没有宣称candidate筛选路径获得额外覆盖。缩层后的后半段激活与正式模型不同，八层最终输出没有正式质量意义。完整Engram映射仍有固定启动成本，缩层是否明显缩短重启时间须按实测判断。

一个容易漏掉的差异是Engram哈希种子：`compute_hash_multipliers` 使用 `10007*layer_id`。把原层14改成tiny层4会改变查表行，即使表inode相同也不能对齐。因此tiny worker在初始化前将哈希种子层号映射回 `[1,14]`，CPU history和device hash都沿用原种子。

## 当前tiny证据与下一道门

`tiny_tp8_build_v1`已退出0，输出 `/work/tiny_tp8_8layer_v1`，复制权重 **68.821985GiB**，复制和独立复核用时 **115.760秒**。完整逐张量manifest留远端；仓库归档9KB compact summary。正式源config SHA为 `40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424`，tiny config SHA为 `ff8424b6bcb67af0a5b45dd284ae1a498c51d125439fd245af9a20a61d1b42e1`。

首轮 `tiny_tp8_alignment_audit_v1` 在原生cache规划器的40层硬编码检查处失败，未进入数值验证；清理时只结束环境明确属于该失败作业的本任务进程，保留失败日志。

新增诊断专用cache规划适配：只把源层集合、状态源集合、SWA层数改为精确的 `[2,4,5]`、`[2,4]`、8层；原payload计算、KV/index偏移、state容量、唯一完整覆盖、dtype与allocator继续执行。原规划函数SHA守卫为 `328f2d755ec7a6cdcd7f0976d7f606e44d2aca0fd81941e9c91cc1bc943384ab`。适配仅在私有快照的 `STACK_TINY_PROFILE=1` 进程启用，未改安装态文件。

CPU真实spec验证通过：16个资源、三slot容量为131072/131072/147712字节，KV/index同页不重叠、别名在不同live block写入保持原值、null block不变；缺KV、state、SWA、index四种不完整拓扑均被拒绝。单独生产模式进程验证原规划函数SHA未变、诊断hook未启用。tiny预计五个cache group而正式模型为12组，因此不能用tiny估计完整metadata准备链收益。

已重跑 `tiny_tp8_alignment_audit_v2`，源码 `/work/src_tiny_align_v6`。三套与正式审计相同的2048-token输入；先逐位比较原样保留的前四层prefill路由，再三组tiny原生A/A比较完整八层路由、token、Top5/logprob，覆盖47/48输出两种C2末步状态。当前运行结果待完成，不能声明tiny已对齐。`tiny_profile_controller_v1` 将逐次检查审计runner存活；只有退出0、三组原生A/A与七次前缀检查均通过，才启动六组无审计计时和七组×八rank微架构采集。

验证边界保持清楚：前四层prefill路由对齐是一个真实数值见证，但不替代每个张量的全算子比较；后四层结构/权重一致且上下文变短，也不等于整网输出一致。新结果强制 `diagnostic_only=true`、`formal_weights=false`、`formal_quality_claim=false`。正式worker仍要求40层，正式比较默认40层，正式服务选择器会拒绝诊断结果。

完成对齐审计后，才采集八rank七组微架构数据、做无审计配对并在tiny筛选候选。通过tiny的候选仍须回到完整正式TP8，通过完整精度/消费者门、同实例交错配对及正式客户端17ms和质量验收。

## 设备、代码与交付

正式API客户端终态核对后，只SIGTERM argv完全一致的本任务API PID492907，为tiny实验释放8–15；保存停止/恢复记录，不reset、不停止其他租户。启动器再次确认8–15无占用、Alarm仍为已授权80C98001。当前没有运行中的正式best API，正在运行tiny诊断。

本轮恢复异步服务选择/worker守卫的未提交改动，补充tiny构建器、native worker、正式前缀/原生A/A验证入口，以及启动器诊断隔离。同步/异步正式选择守卫均通过；40层与8层严格比较及非法expert/NaN拒绝检查通过，默认正式比较拒绝8层。

最终要求见 `experiments/operator_stack/ACTIVE_OBJECTIVE.md`。goal工具拒绝覆盖未完成goal，原goal保持active；所有后续实验和最终完成审计按用户最新17ms要求进行。报告将发布COS/links-server，源码/紧凑证据提交后从HEAD生成MANIFEST、自检并双远端push。完整17ms目标尚未完成。
