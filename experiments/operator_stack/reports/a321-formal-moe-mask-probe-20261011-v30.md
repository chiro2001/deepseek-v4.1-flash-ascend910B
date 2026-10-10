# MoE 掩码与 BF16 转换逐位通过，正式 TP8 审计进行中

2026-10-11 · a3-21 physical chip8–15 · 正式40层/W4A8_DYNAMIC/Engram int8 · strict · 客户端目标≤17ms/step

当前通过客户端质量验收的结果仍为 **(19.376255ms/step,A=1,51.609559tok/s)**，尚未达到17ms。新掩码候选已有独立逐位精度和可信设备计时，完整模型审计正在运行，没有新整网收益或客户端通过声明。wo_a Cube 在真实消费者上超过1ULP而拒绝，详见v29，不叠加其独立收益。

## 候选与独立精度

正式 EP8/384 专家、每rank连续48专家、top6。原生路径先执行 Less、GreaterEqual、LogicalOr、MaskedFill，再把概率转为BF16供原生 token-unpermute 消费。新内核只合并这些逐元素操作，保留原生路由、分布式MoE、GMM、权重、激活和strict规约。

`formal_moe_mask_probe_v2`退出0，仅在physical8执行；八个rank范围分别验证，不宣称已运行完整TP8。八范围×四行数(1/2/8/16)×三概率风格，共96例，每例比较FP32掩码和BF16消费者两种输出，192次全部逐位一致。用例含范围边界、负概率、带符号零及BF16舍入边界。没有使用浮点容差；真实模型消费者仍须单独逐层审计。

## 计时修正与结果

第一版每图32调用的四条路径都约8.94μs，未建立提交开销余量，不能据此判断候选无收益或计入性能提升。第二版每图512个不同输入地址、每次重放8次，增加真实设备工作量，记录CPU提交时间，要求每个窗口的设备时间至少为提交时间2倍。64个配对的四条路径均通过，最低比例约2.155；没有把CPU重放开销当作kernel时间。

以下是同一独立试验、M=1、图内事件时间，profiler关闭：

|候选|原生μs/调用|融合μs/调用|配对加速中位|更快配对|
|---|---:|---:|---:|---:|
|FP32掩码|4.392957|0.841917|5.175139|64/64|
|掩码＋消费者BF16转换|5.506194|0.854939|6.464644|64/64|

单算子6.464644倍不能外推为整网加速，也不能从旧客户端19.376255ms跨会话扣除。正式每步40次消费者，最终收益由关闭观察器的同实例配对和新API客户端判断。

## 正式接入与当前作业

`tp8mask`继承metastack配置。仅在实际 `[1,6]` INT32 ID、FP32概率和连续48专家范围上使用候选；其他形状维持原生行为。每个捕获消费者通过实际GMM输入的layer对象绑定到模型层，不能用40次同一消费者冒充40层覆盖。

启动前验证独立probe完成、192次精确比较、提交余量和速度门槛；冻结kernel、原生dispatch及combine源码SHA。CPU适配器已执行真实source重写与关键字调用契约检查，局部契约另确认新ARM配置等同metastack、保留原生HC、没有额外hostmeta spec缓存；这些CPU结果不是模型精度验收。

不可变快照 `/work/src_mask_model_v2`，`formal_moe_mask_audit_v1`已在chip8–15全部空闲、逐芯片Alarm为已授权80C98001后启动。正式40层、异步调度、strict、同八worker比较base/metastack/mask三方案。要求三组完整路由/token/Top5/logprob<1e-3、原生A/A控制、每rank40个不同层的BF16消费者逐位一致，以及Indexer/cache消费者通过。

`formal_moe_mask_perf_controller_v1`只在全部审计通过后启动 `formal_moe_mask_perf_v1` 的12组无审计/profiler交错配对，比较新候选相对同实例metastack的增量。后续仍须最优方案的API归属、串行8请求性能、GSM8K100/100和Vision23/23验收。17ms目标保持active；当前没有17ms在线服务。

## 紧凑证据

- `results/formal_moe_mask_probe_v1/{result.json,run.exit,launched.json}`：第一版精度通过、计时不作为收益证据。
- `results/formal_moe_mask_probe_v2/{result.json,run.exit,launched.json,adapter_preflight.json}`：96例/192次逐位比较、64配对、提交余量和源码绑定。
- `results/formal_moe_mask_audit_v1/launched.json`及`results/formal_moe_mask_perf_controller_v1/launched.json`：资源检查、strict、启动参数与控制器归属。
- 完整请求、权重和trace留远端。源码与紧凑证据提交后，从HEAD生成MANIFEST、自检并双远端push。
