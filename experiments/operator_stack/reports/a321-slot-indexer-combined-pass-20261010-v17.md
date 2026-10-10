# slot＋Indexer叠加正式精度通过

2026-10-10 · a3-21 chip8–15 · 正式权重TP8 · strict

`formal_slots_indexer_audit_v1`已退出0。base/core/meta/metastack四个bank同实例，三组配对共九个比较的token、完整路由、Top5集合和logprob全部一致，最大差0。四次原生控制通过；两个slot候选共48份rank消费者、各12个group的整数坐标逐位一致，并有实际合并覆盖。Indexer融合的消费者检查也正常通过，没有将只走回退的方案算成优化。

本版在参考方案学完布局后停止重复登记，继续读取当步输入调用原准备方法。独立probe_v6的144组整数/尾部检查与6次变化输入图重放也已退出0。

随后守卫启动`formal_slots_indexer_perf_v1`：四方案、12组交错配对、关闭路由/cache审计和NPU profiler，配对之后单独采CPU诊断。此时仍在执行，不能用审计态约29–32ms的数字选择最优方案或计算收益。

上一版三方案关闭审计的有效结果为meta `(26.234730ms/step,A=1,38.117411tok/s)`，详见[报告v16](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-slot-formal-pass-cpu-20261010-v16.html)。它比同实例base更快，但相对普通core的增量受旧版参考登记开销影响；本版用于重新核验这一增量及Indexer叠加效果，不跨会话累加。

当前正式core客户端已通过GSM8K100/100、Vision23/23，客户端`(26.744123ms/step,A=1,37.391393tok/s)`。新meta/metastack还需选择最快且通过门槛的方案，完成客户端质量/时延及服务归属验证。19ms和最终最佳服务仍未完成，goal保持完整范围并为active。

源码、精度与性能紧凑证据、报告、MANIFEST及自检随本轮双远端push。原始路由/请求留a3-21任务目录；新增W4A8前缀模式、专家控制与缓冲候选仍需继续实施验证，不能把本次叠加精度通过当成19ms达标。
