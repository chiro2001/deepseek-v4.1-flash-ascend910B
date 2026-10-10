# 正式 MoE 掩码19.229ms复测与40层逐位精度通过

2026-10-11 · a3-21 chip8–15 · 完整40层/384专家top6/W4A8_DYNAMIC/Engram int8 · strict · 客户端目标≤17ms/step

正式无审计/profiler的同实例12组交错配对已完成。新mask **(19.228652ms/step,A=1,52.005725tok/s)**，相对同实例metastack配对节省中位0.149769ms、12/12更快。这是正式内部token到达计时，尚未达到17ms，不能替代新API客户端报价。

|同实例方案|ms/step|A|tok/s|
|---|---:|---:|---:|
|base|20.393742|1|49.034650|
|metastack|19.372919|1|51.618446|
|mask|19.228652|1|52.005725|

`formal_moe_mask_audit_v1`退出0：三组完整配对中，metastack和新mask两方案相对原生的路由、token、Top5/logprob全部一致，六个比较的最大logprob差均为0。三次原生A/A控制通过。mask每rank40个不同层的概率消费者逐位一致，三组×八rank共960次真实消费者检查通过，Indexer及cache消费者检查通过。

当前已通过客户端质量验收的报价仍为 **(19.376255ms/step,A=1,51.609559tok/s)**，GSM8K100/100、Vision23/23；新方案需要自己的客户端验收，不能与旧客户端跨会话相减。

## 正式审计边界

- 同一八worker、同一正式checkpoint，base/metastack/mask三图bank；保持strict原生规约、异步调度、完整视觉模块与实际Engram表。
- Mask精度使用逐位相等；没有放宽BF16消费者、路由或logprob门槛。独立96例/192次比较只用于筛选，正式消费者另检查实际40层输入。
- 审计包括route导出、clone和cache验证，审计态时间不作为性能收益。wo_a Cube已被真实消费者184ULP拒绝，没有叠加其独立收益。

## 性能与客户端后续

完整守卫启动的 `formal_moe_mask_perf_v1` 已退出0。计时关闭route/clone审计与profiler，实际decode token到达为步数；没有以CPU轮询次数计算时延。

`formal_moe_mask_delivery_controller_v1` 已验证mask实测最快、12/12更快且配对节省0.149769ms，超过0.05ms部署门槛。`formal_best_async_service_v6`已以同一不可变快照 `/work/src_mask_model_v2` 在本任务私有loopback18766启动，独立模型名 `dsv41-a321-formal-best-v6-async-mask-20261011`；当前加载中。启动复查8–15全部空闲/已授权Alarm，不停止其他租户。控制器随后执行归属验证、串行8请求性能、Vision23和GSM8K100。

部署不等于17ms完成；客户端质量、目标门槛与最终发布分别核对。若增量不稳定则保留证据并拒绝部署该候选。目标继续active。

## 保存的证据

`results/formal_moe_mask_audit_v1/consumer_summary.json`记录24份rank审计/960次消费者、三原生控制、六完整比较及远端完整requests的SHA。`result.json`、`run.exit`、`banks.json`、`native_controls.json`、`comparisons.json`和`validated_for_mask_perf.json`保留原始紧凑证据；完整routes/请求留远端。性能和delivery两个启动记录分别绑定实际作业与源文件。

源码、v29/v30报告和独立证据已在`ecdff09`完成MANIFEST自检并双远端push；本轮正式审计/性能终态和自动客户端入口继续提交、发布并复核。
