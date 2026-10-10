# W4A8仅上投影前缀方案通过正式精度，性能配对进行中

2026-10-10 · a3-21 physical chip8–15 · 正式TP8/40层/5120/384专家top6/W4A8_DYNAMIC · strict

`formal_prefix_up_audit_v1`已退出0。两种仅GMM1采用prefix的候选，在正式权重和真实decode输入的8rank×40层独立比较通过；同实例四方案三组/九个完整路由、token、Top5/logprob配对全部一致，delta0。两候选合计3840份有效行GMM消费者比较逐位一致，576份cache-group整数坐标精确。已由完整守卫启动12组关闭审计的性能配对，随后单独采集七组微架构指标。**尚无新候选性能提升或19ms达标声明。**

## 本轮如何定位并修正前缀方案

先前试验将prefix同时交给GMM1和GMM2。在 `formal_prefix_audit_v2` 中，原生A/A及metastack捕获后的原生控制均通过；`tp8prefix`建图时，chip8上的AIV核21报507015、MTE非法GM访问或跨设备访存超时，作业退出1。性能控制器正确拒绝启动。

按cannbot plog诊断方法核对进程355270的debug/run日志和异常kernel dump，故障为下投影：`GroupedMatmul_a0be48d01b162435398ed52e1b6182d4_69325233410`，symbol `mix_aiv+0x4d74`。HCCL日志明确说失败task不是HCCL。kernel配置为INT8输入、INT4 FRACTAL_NZ权重、FP32 bias/pertoken scale、UINT64 weight scale、INT64 group list及BF16输出。已保存参数区和dump JSON；没有把其默认attrs当成完整的运行时参数解码，也没有把一次故障泛化为所有GroupedMatmul不支持prefix。

这不是完整路由/logprob的精度不一致，候选尚未完成该阶段。当前正式A8W4下投影prefix路径没有成功运行，因此修正为保留GMM2原生counts格式，继续检验GMM1的prefix机会。未reset、未停止其他租户；新进程复查8–15无占用，Alarm仍只有已授权的80C98001。

## 两种新候选的原理

|方案|Routing输出|GMM1输入|GMM2输入|新增表示转换|
|---|---|---|---|---|
|tp8metastack参考|counts|counts|counts|无|
|tp8prefixup|counts|prefix|保留原counts|GMM1前一次cumsum|
|tp8prefixuproute|prefix|直接prefix|原生helper恢复counts|GMM2前cat/diff|

GMM1源码在counts模式逐个累加专家边界，prefix模式直接读取边界。新方案试图减少其重复Scalar控制工作，同时保留已发生异常的下投影原生契约。权重、量化值、专家顺序、激活、通信和GMM算术不变。counts保留方案通过新的compute-input对象传递prefix，不修改两个GMM共享的原生输入。所有值每step更新，未缓存跨step的路由计数。

## 正式验证的覆盖

在编译候选bank之前，从原生图保存本次真实decode消费点输入，使用同一正式权重进行8rank×40层的GMM1 prefix比较。INT8激活输出与动态scale的有效行全部逐位一致。之后建立base、metastack和两个候选bank，原生四次稳定控制通过，再执行三组交错精度配对。

每个候选：24份rank消费者审计、1920份GMM1/2比较、288份cache group比较。总计3840份GMM消费者、576份cache group；logprob delta0。routing预留容量的未写入尾部不被masked unpermute消费，比较全部 `[0,prefix[-1])` 的有效行，INT8/FP32 scale/BF16均不放宽容差。原始请求和向量留远端，紧凑证据入库。

这些数据的实际局部有效token数为0–3、中位1，说明当前batch1下专家任务非常小。该数值是本次配对消费点观察，不是任意输入的全局上界，也不是UB容量occupancy。

## 性能与微架构数据如何采集

`formal_prefix_up_perf_v1`使用相同 `/work/src_prefix_v3`，关闭路由/cache审计、消费者clone以及CPU诊断，重新加载同一正式模型，同实例四bank做12组交错配对。prefill和前9步不进入内部engine-step中位数；A=1，必须同时给ms/step与tok/s。不能把前面的审计耗时或跨会话数字相减。

计时结束后，另采PipeUtilization、ArithmeticUtilization、Memory、MemoryL0、MemoryUB、L2Cache、ResourceConflictRatio，每组8rank、5步warmup/10步active。重点检查GMM1 Scalar、MTE搬运与cache/buffer带宽，GMM2仍保持原生counts；profiling结果不进入性能中位。容量occupancy若仍不可得，将继续明确缺失。

UB源码预算见[报告v20](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-w4a8-prefix-probe-20261010-v20.html)：pre/post主要队列单缓冲，A8W4 post行预算为 `8.5*row*n + 4*alignUp(row,8) + 6*n + 64 <= ubSize`。本次小M观察提示应先核对每核tile/行循环数；只有存在足够steady阶段才能通过dual-buffer获得重叠。没有仅修改buffer深度，也未宣称双缓冲收益。

## 当前最优服务和后续机会

最近一次独立客户端已验收的tp8metastack为 `(26.292508ms/step,A=1,38.033648tok/s)`，GSM8K100/100、Vision23/23，见[报告v19](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-formal-client-pass-20261010-v19.html)。该API为继续实验已释放，恢复记录保留；新候选只有整网精度通过，客户端质量和最优部署仍待性能选择后验证。目标约19ms仍未达到。

下一阶段按实际配对结果选择：有效收益才继续叠加；无收益保留负结果。剩余待尝试方向包括metadata静态几何与host准备开销、固定序通信整网接入、正式W4A8小M控制/缓冲，以及读取到的原生W8A8路径。若试验将现有4bit整数解包成8bit表示，必须检查有符号/offset/assist-bias语义、额外显存和原生消费者逐位一致；当前仅保存源码机会，未实现、未计收益。已失败的HC/static及不匹配的TP1专家特化继续保持原生。

本轮源码和紧凑证据、提交后生成的MANIFEST、自检及双远端push持续更新；报告发布COS并登记links-server。完整优化目标继续推进。
