# hostmeta正式客户端100/100与23/23通过，继续异步调度精度审计

2026-10-10 · a3-21 chip8–15 · 正式40层/5120/384专家top6/W4A8/Engram int8与完整视觉 · strict

最新hostmeta服务客户端已退出0：**GSM8K100/100、Vision23/23**，8条serial、2K输入/256输出、A=1为 **(26.133047ms/step,1,38.265726tok/s)**。profiler/路由/cache审计关闭，服务归属核对通过。19ms仍未达到。

该方案是core＋12组slot合并＋Indexer融合＋metadata静态几何缓存。其同实例12组内部engine-step配对为meta `(25.719245ms,1,38.881390tok/s)`、hostmeta `(25.636325ms,1,39.007151tok/s)`，9/12更快，配对节省中位0.100410ms。客户端与内部步计时不同，不相减；也不把旧服务26.292508当成同进程A/B。精度及strict通信数值边界详见[报告v23](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-hostmeta-small-gain-strict-comm-20261010-v23.html)。

## 客户端和设备记录

服务作业 `formal_best_strict_service_v4`，源 `/work/src_hostmeta_v2`，loopback18764，模型名 `dsv41-a321-formal-best-v4-tp8hostmeta-20261010`。先核对 `/v1/models` 与唯一API argv，再发请求。GSM8K100题无空答/请求错误，用时410.8s，官方train/test JSONL SHA保持相同；原始答案/请求留远端，只归档summary与acceptance。

质量终态100/100及23/23落盘后，验收控制器检查仍是本任务API再SIGTERM。保留服务选择、PID、源码/结果SHA及恢复记录，随后复查同组chip占用和授权80C98001 Alarm，启动下一精度实验。未reset、未停止其他租户。**当前本任务API已为异步试验释放**，不能把已验收服务描述为仍在运行。

## 下一轮异步调度的原理与边界

此前独立CPU/NPU数据中的主图前准备链和host阻塞较明显，Event.synchronize还包含设备等待，不能直接删除当收益。异步调度试图让下一步host准备与已有设备执行重叠，仍使用正式权重、strict、A=1与相同cache/算子方案，不启用推测解码。

`formal_async_audit_v1`已启动，源快照 `/work/src_async_v2`。除原生A/A、三bank完整路由/Top5/logprob及cache消费者门外，每个请求还与 `formal_hostmeta_audit_v1` 中同arm/同seed/同输出长度的同步正式参考比较，失败保留证据。未声明异步精度或性能通过。

计时入口新增真实token到达次数/时间戳。异步engine.step可能只取出已生成的队列数据，不能把CPU调用数当decode step：异步使用warmup后实际token到达墙钟除以输出token数，再汇总请求。`decode_mean_ms`与原同步`decode_median_ms`分别标注，统一测量字段和scope明确模式；跨模式会话数字不直接相减。

完整精度门通过后，守卫才启动12组关闭审计的异步同进程算子配对，最后仍需客户端8请求质量/性能验收。若未通过或没有收益，恢复已验收hostmeta原生规约路径。

本轮报告COS/links-server发布，代码/紧凑证据/提交源码生成MANIFEST及自检后双远端push。完整约19ms目标继续，尚未完成。
