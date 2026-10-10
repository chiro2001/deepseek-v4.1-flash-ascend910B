# 正式 wo_a Cube：独立探针通过，真实消费者精度拒绝

2026-10-11 · a3-21 physical chip8–15 · 正式40层/W4A8_DYNAMIC/Engram int8 · strict · 当前客户端目标≤17ms/step

当前已通过质量验收的客户端仍为 **(19.376255ms/step,A=1,51.609559tok/s)**，GSM8K100/100、Vision23/23。新候选没有正式客户端结果，尚未达到17ms。

## 新候选与真实布局

正式图每步有40次单group wo_a，输入 `[1,4096]`、BF16权重 `[4096,1024]`；上一轮rank0采集态累计约0.928ms，仅用于选热点。此前TP1八group O投影的负结果不覆盖这个形状。

`formal_woa_gemv_probe_v1`退出0，执行芯片为physical8。读取正式层0、20的原始 `[8192,4096]` 权重，按ColumnParallel真实rank切片取1024输出行，再转置contiguous；保留原BF16权重和BF16中间输出。两层×八rank×三输入尺度×两seed，共96例/候选，每个输出元素保留原≤1BF16 ULP门槛。

|候选|独立精度|图内事件配对|判定|
|---|---|---|---|
|Cube BN64/BK256|96/96；80例逐位相同，16例最大1ULP|native20.336718μs→17.960176μs；配对中位1.132454，64/64更快|独立通过；真实消费者拒绝|
|Cube BN128/BK512|96/96；80例逐位相同，16例最大1ULP|17.942266μs；配对中位1.134238，63/64更快|保留独立证据|
|Vector BN8/BK512|第二例失败，1元素3ULP|不采有效性能|拒绝|
|Vector BN16/BK256|第二例失败，1元素2ULP|不采有效性能|拒绝|

独立事件时间、采集态累计和正式客户端报价互相独立；不能将1.132454倍加速外推为整网加速，也不能把40次独立差值直接从旧客户端时延中扣除。

## 接入与启动前检查

新 `tp8woa` 继承已验证metastack配置，仅替换登记的40个wo_a权重消费者，匹配M1/K4096/N1024、BF16、contiguous、权重版本。其他形状保留正常matmul。启动证据绑定独立probe结果和完整kernel文件SHA；编译在capture外进行。

修复接入时误写的tuple配置key。加载前检查执行真实install/switch函数的配置逻辑，确认新ARM与metastack配置相同、没有额外hostmeta spec缓存、正式HC保持原生；全部stack Python AST和diff空白检查通过。

每rank消费审计要求40次调用且40个不同层，逐元素≤1ULP。完整整网仍另外比较全部路由、token、Top5和logprob<1e-3，保留原生A/A控制、Indexer与cache消费者检查。单算子通过不能绕过这些门槛。

## 正式整网终态

不可变快照 `/work/src_woa_model_v1`，`formal_woa_model_audit_v1`退出1。已建立同一实例base/metastack/wo_a三bank，保持strict原生规约和异步调度；启动时chip8–15全部空闲，逐芯片Alarm均为已授权80C98001。

首个wo_a请求的真实消费者检查失败：`language_model.model.layers.0.self_attn.wo_a`最大184 BF16 ULP，超过原1ULP门槛。独立96case结果不能覆盖真实激活；没有放宽精度、没有判定整网路由或客户端通过，也没有有效E2E候选时延。

`formal_woa_perf_controller_v1`明确报出 `Audit failed; do not run timing`，未创建 `formal_woa_model_perf_v1`。候选不得进入API部署或最终最优选择。失败原因和已完成的原生控制/各方案覆盖保存为紧凑 `rejection.json`；完整请求仍留远端。

当前正式API为实验释放。完整17ms目标保持active，不能以独立算子或审计态时延宣布完成。

## 证据

- `results/formal_woa_gemv_probe_v1/{result.json,run.exit,launched.json}`：完整独立数值与64组配对，结果SHA256 `a8b3d38c398dcca7f317cd17fd820f295a6475493bd565f10b5b892a0eba82bf`。
- 新kernel文件SHA256 `bbd1b31b0532be220445f0a24555faa1967270671b40344a81d1f92976ce874b`。
- 完整权重、请求和trace留远端，仅紧凑证据入库；源码、报告提交后从HEAD生成MANIFEST、自检并双远端push。
