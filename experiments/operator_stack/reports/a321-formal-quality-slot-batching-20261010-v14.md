# 正式TP8质量验收通过，继续合并图外slot发射

2026-10-10 · a3-21 physical chip8–15 · 正式40层/W4A8/384专家top6权重 · strict

正式core第二版API客户端验收已正常退出0：**GSM8K100/100，Vision23/23**，无空答和请求错误。8条不同prompt的串行客户端性能为 **`(26.744123ms/step, A=1, 37.391393tok/s)`**，仍未达到≤约19ms目标。本轮继续优化完整模型，没有以单kernel通过代替端到端达标。

## 当前通过了什么

`formal_core_strict_service_v2`使用正式checkpoint、core、`HCCL_DETERMINISTIC=strict`及正式`deepseek_v41` tokenizer/renderer和parser前端。独立模型名`dsv41-a321-formal-core-strict-front-v2-20261010`、loopback18762及唯一API进程argv已核对。

|验证|范围|结果|
|---|---|---|
|原生及叠加图精度|原生3组A/A；base/core/stack6个配对|完整路由、Top5集合、logprob一致，delta0|
|消费者|72份rank的Indexer/cache审计|逐位一致|
|客户端性能|8条不同prompt，并发1；2K输入、256输出；无profiler/审计/spec|26.744123ms/step、A=1、37.391393tok/s|
|GSM8K|官方JSONL、相同训练集前8题few-shot、test前100题、chat模式|100/100，空答0，错误0|
|视觉|两张官方图片、23题及文本/换图/空图负控|23/23，PASS|

内部12组关闭审计的core结果仍为`(26.622860ms/step,A=1,37.561704tok/s)`；客户端采用不同输出长度和计时位置，因此不将两者直接相减或累计收益。第一版API缺chat template的HTTP400记录保留；修正前端后的23/23证明这项接口问题已解决。

GSM8K完整输出与视觉原始请求留远端。本地保存来源SHA、质量摘要、退出码、性能三元组及服务归属。验收完成后，仅向本任务自己的API进程发送SIGTERM，为后续算子实验释放同一组授权设备；未停止其他租户、reset设备或清page cache。

## 新候选的依据与原理

strict七组/112份微架构采集显示，chip8 core的较长kernel间隙集中在主图前。每步原有**12次slot_mapping_kernel**，其中一个约51µs，另外多个约23µs；这些累计与间隙仅作定位，包含采集/发射影响，不是可直接回收的端到端时间。

新`tp8meta`候选在已验证core上增加slot转换合并。原始block table准备保持原生core的多group融合；这项改动处理其后将每group raw slot转换成`[physical block, offset]`的12次发射，改为一次发射。

* 每个cache group保留独立输出buffer，没有将物理block编号共享成同一份结果。
* descriptor只保存地址和布局常量；raw slot、位置、query边界和实际token/请求数仍由当步输入读取，不缓存跨step动态数值。
* C2的有效结束边界、奇数位置完成规则、skip-ring语义与PAD=-1均保留。
* 只有batch1/token1、完整group登记、同输入身份和支持的dtype/layout才走候选；其他形态走原来的准备方法。
* 用原生runner每步重建的batch字典记录本步是否已合并；避免一个group的准备被错误沿用到下一步。

实现只在新实验的`STACK_METADATA_MANY_SLOTS_ENABLED=1`以及对应arm下使用，当前已验收core不会默认启用未验证候选。新增独立源码目录参数，让新实验运行固定源码快照，避免改动已经运行的服务代码。

## 独立验证与首轮修正

`formal_slots_batch_probe_v1`退出1，Ascend编译器在第二个标量地址store处报`expected result type with offset = 0 instead of 1`。没有将这次编译失败算作精度或性能通过。修正为一个带掩码的连续向量store，尾部必须保持不变。

同时按真实profiling列核对契约：raw slot和输出为**INT32**，位置为INT64；先前只接受INT64的prototype不足以覆盖正式路径。候选已支持真实INT32与INT64路径，独立验证分别覆盖两种dtype、PAD、block边界、C2奇偶、空/有效实际请求数、实际token数和skip-ring，并检查输出尾部及变化输入的图重放。

`formal_slots_batch_probe_v2/v3`随后在接近INT32上界的测试值发现余数不一致，诊断明确为旧core Triton helper与CPU整数数学参考不同。例如raw=2147483525、ratio2、block64时，正确坐标为`[16777215,2]`，旧helper为`[16777215,0]`，合并候选为`[16777215,2]`。未将旧helper当作全整数范围的数学真值，也未放宽误差门槛。

v4增加了对全部用例的CPU精确floor/remainder参考，同时继续要求本次正式池范围测试用例与旧路径逐位一致；生产集成只允许配置的raw pool范围小于2^24。v4已正常退出0：INT32、INT64各72组，共144组数学参考全部一致，输出尾部不变，变化输入图重放各3次共6次通过。独立12group操作的事件时间中位如下，包含发射影响；它不是完整模型的ms/step，也不计作整网收益。

|slot dtype|旧路径12次发射 µs/call|合并一次发射 µs/call|模型端到端收益|
|---|---:|---:|---|
|INT32（正式契约）|801.356|392.164|尚未验证|
|INT64|945.504|458.397|尚未验证|

两种dtype全部通过后，守卫已启动`formal_slots_batch_audit_v1`，接正式8rank完整路由、Top5/logprob及12group整数消费者对照，保持base/core/meta同实例与原始精度门槛。审计仍在加载/执行中；精度及实际覆盖通过后，守卫才启动12组关闭审计/profiler的交错配对和单独CPU诊断。尚无正式整网增益可计。

已增加15个稳态decode step的8rank cProfile诊断入口，安排在配对性能之后单独运行，用来找图外Python/C调用与同步位置。嵌套cumulative时间不可相加，同步包含设备/到达等待；不会将诊断态时间作为正式性能。

## 完整目标与后续

用户要求重新创建goal后，新goal已创建并为active，保留正式权重TP8 ≤约19ms/step、全部精度/消费者/客户端质量、最优服务部署、报告/COS/links-server、源码/manifest/自检及双远端push条件。

当前core质量已通过，19ms仍未达到，服务在算子实验期间已结束。新slot候选尚无正式整网收益可计；正式HC/router/GMM的失败或不匹配特化继续保持原生，W4A8激活候选覆盖0不计收益。后续仍需基于实测继续研究metadata准备、正式小M/专家控制、通信及适用的缓冲/预取方案，不能把这一个候选通过当作完成。

证据根：`experiments/operator_stack/evidence/formal_a321/results/formal_core_strict_service_v2/`、`formal_slots_batch_probe_v1/`至`v4/`及`formal_slots_batch_audit_v1/`；原始数据在a3-21的`/work/results/`。新实验源码快照`/work/src_manyslots_v6`保持与正在执行的作业一致。此前微架构口径与UB双缓冲边界见[报告v12](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-strict-microarch-20261010-v12.html)。
