# metadata取得小幅收益，strict通信复测确认数值边界

2026-10-10 · a3-21 chip8–15 · 正式TP8/40层/hidden5120/384专家top6/W4A8_DYNAMIC/Engram int8及完整视觉 · strict

缓存metadata静态几何的hostmeta候选通过正式精度，并在12组同实例交错配对中9/12更快，配对节省中位 **0.100410ms**、加速中位 **1.003924**。内部engine-step整体中位 **(25.636325ms/step,A=1,39.007151tok/s)**。收益很小，约19ms目标仍未达到。

## 代码与精度

`tp8hostmeta`在metastack基础上，初始化时缓存cache-kind、compress-ratio、storage-block几何及slot/SMLA/QLI共享键，省去约25个builder每step重复的类型/属性和字符串处理。所有动态位置、query/seq长度、slot、skip以及DeviceMetadataTask提交/等待保持每步更新。spec对象更换时回退原ring builder。Compressor仍发布正确storage block，未缓存动态值。

`formal_hostmeta_audit_v1`退出0：三次原生稳定控制、六个整网配对的完整路由、输出token、Top5/logprob全一致，delta0。hostmeta三组24份rank审计，288份cache group坐标与当前CPU整数参考精确，实际spec-build每rank至少3600次，拒绝覆盖0。

## 公平性能结果

同实例、2K输入/48输出、batch1、A=1，关闭审计/profiler/CPU诊断，排除prefill及前9步。

|方案|ms/step|A|tok/s|
|---|---:|---:|---:|
|base|30.665440|1|32.610000|
|metastack|25.719245|1|38.881390|
|metastack＋静态metadata几何|25.636325|1|39.007151|

不把两个整体中位差当稳定收益，也不与前一会话25.728940直接相减。按本轮最低中位选择hostmeta。前一轮GMM1 prefix两方案精度通过但性能未获益，保留负结果，见[报告v22](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-prefix-negative-microarch-20261010-v22.html)。

## 通信复测：稳定性与FP64数学结果不同

`formal_strict_collectives_v1`使用已保存的80份正式rank-local BF16向量、8rank。在新通信域初始化前设置strict，并把32次collective捕获进同一图，摊薄旧测试中较大的Python graph.replay开销。所有数据是独立事件计时，不是E2E；方法固定顺序，不宣称同进程交错配对收益。

|方法|8rank事件中位的中位μs/collective|对齐strict原生BF16|对齐FP64 sum转BF16|
|---|---:|---|---|
|strict原生BF16 AllReduce|21.503500|是|否|
|strict原生FP32 AllReduce再转BF16|24.852703|否|是|
|AllGather＋固定序FP32，block256|15.932478|否|是|
|AllGather＋补偿FP32，block256|17.452803|否|是|

六个固定序/补偿配置（block256/512/1024）均重复及图重放稳定、对齐FP64参考；但每个配置在8rank×80向量的640例均与strict原生BF16有差异。strict原生BF16自身重复/图重放稳定，并逐位对齐自己的参考。这里是不同规约累加/舍入方式产生的确定性数值差异，**不是当前服务再次发生规约不稳定**。

所以不能把更快、更接近FP64的结果直接当成保持原生行为的替换。当前最优服务继续使用strict原生规约；本轮没有计入通信端到端收益，也没有放宽完整路由/Top5门槛。报告同时保留native-reference和FP64-reference两种独立判据。

## 当前最优TP8与客户端

已复查同组8chip空闲及授权80C98001 Alarm，启动 `formal_best_strict_service_v4`，使用已审计/配对最低中位的 `tp8hostmeta`，源快照 `/work/src_hostmeta_v2`，loopback18764，独立模型名 `dsv41-a321-formal-best-v4-tp8hostmeta-20261010`。正式权重和strict、auto通信端口、4GiB KV、max_seqs1/max_len8192、完整视觉不变。

客户端runner已启动：有界等待选择记录/服务ready，核对模型名与唯一API进程后运行8条serial、2K/256输出、Vision23及GSM8K100。此时尚未拿到终态，不能移用旧服务100/100与23/23作为新服务结果。19ms仍须客户端实测。

## 下一处机会

CPU诊断仍有metadata准备链和设备等待。准备独立测试异步调度是否能重叠host准备与设备执行；该试验不更改权重或推测解码，仍A=1，并要求对齐本轮同步参考的完整路由/Top5、cache消费者。不同调度模式属于不同会话；不相减当算子收益。异步时必须按实际token到达的墙钟/数量计时，不能按可能只是在取队列的CPU engine.step次数报价。当前只新增实验入口，未启动异步模型、未计收益。

UB单/双缓冲、各级带宽及Scalar工作证据见v20–v22。没有实测UB容量occupancy或双缓冲收益。完整目标仍为正式端到端约19ms，最优部署、客户端质量以及全部源码/证据/MANIFEST自检/双远端push、COS/links报告持续交付。
