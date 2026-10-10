# 正式四方案公平配对：当前最快25.694ms/step

2026-10-10 · a3-21 chip8–15 · 正式40层/W4A8/384专家top6 · strict

`formal_slots_indexer_perf_v1`已正常退出0。在参考方案学完布局后停止重复登记，关闭路由/cache审计和NPU profiler，同实例完成四方案12组交错配对：

|方案|ms/step|A|tok/s|
|---|---:|---:|---:|
|base|30.183590|1|33.130585|
|core|26.840930|1|37.256533|
|core＋slot合并|25.724235|1|38.873848|
|core＋slot合并＋Indexer融合|25.694245|1|38.919221|

两个slot候选各11/12组快于core；slot相对core的配对加速中位1.031306。叠Indexer整体中位最低，但只在7/12组快于单独slot，配对差中位0.025080ms，增量很小。按已完成结果选metastack做客户端验证，不宣称Indexer有明显稳定提升。

本次为内部engine-step、2K输入/48输出、batch1、无推测解码A=1；prefill和前9步不进入中位数。先前26.234730ms的meta来自另一次会话及带持续参考登记的实现，不直接相减或累计收益。**当前仍比19ms目标慢约6.7ms。**

## 精度与覆盖

对应`formal_slots_indexer_audit_v1`同实例三组/九个配对的完整路由、Top5集合和logprob全一致/delta0，四次原生控制通过。两个slot候选共48份rank、576份group整数坐标逐位符合CPU参考；Indexer消费者正常通过。独立probe_v6的144组整数/尾部及6次变化输入图重放通过，生产池绑定保持实际V1配置与范围限制。

正式HC/router及不匹配的selected-GMM继续保留原生；BF16激活候选覆盖0不计收益。没有以降低精度或缩小模型换取数字。

## 最快服务与客户端进度

核验审计/性能结果后，已在同一组chip重新检查无占用和授权的80C98001 Alarm，启动`formal_best_strict_service_v3`，选择`tp8metastack`，正式权重、strict、auto通信端口、loopback18763，独立模型名`dsv41-a321-formal-best-v3-tp8metastack-20261010`。源码快照`/work/src_manyslots_v8`与已验收四方案一致。

串行客户端runner已启动，先核对模型名和唯一API进程，随后测8条2K/256输出性能、Vision23例及GSM8K100。此时新服务仍在加载/验证，不能把旧core的GSM8K100/100、Vision23/23或客户端26.744ms移用到新组合。服务启动选择器不是19ms达标守卫，最终客户端时延仍须实测。

## 剩余机会和完整交付

CPU诊断表明metadata准备链仍较重，同步包含实际设备等待；W4A8源码确认counts模式的专家前缀扫描、group-list 0/1支持与type2拒绝，以及A8W4若干单缓冲队列。接下来继续尝试正式W4A8前缀表示、控制循环、metadata准备与可证明的buffer/预取方案。资料和边界见[报告v16](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-slot-formal-pass-cpu-20261010-v16.html)，叠加精度见[报告v17](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-slot-indexer-combined-pass-20261010-v17.html)。

性能/精度/实际覆盖的紧凑证据、独立CPU诊断及服务选择/启动记录均保存，原始请求和向量留远端。新报告继续上传COS/links-server，源码与MANIFEST自检并双远端push。Goal保持active，完整19ms、最优服务客户端验收及后续优化工作均未缩小或视作完成。
