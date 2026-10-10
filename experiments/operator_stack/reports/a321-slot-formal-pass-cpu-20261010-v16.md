# slot合并正式精度通过，12组测得26.235ms/step

2026-10-10 · a3-21 chip8–15 · 正式W4A8/384专家top6 · strict

`formal_slots_batch_audit_v2`已退出0：base/core/meta六个配对的完整路由、Top5集合及logprob全部一致，最大差0；候选实际覆盖已确认，24份rank消费者各12个cache group的整数坐标逐位符合CPU参考。KV池绑定修正解决了上一轮候选只走回退的问题。

随后`formal_slots_batch_perf_v2`关闭审计和NPU profiler，完成同实例12组交错配对，正常退出0：

|方案|ms/step|A|tok/s|
|---|---:|---:|---:|
|base|31.147350|1|32.105460|
|core（本版仍做布局登记）|27.185700|1|36.784044|
|meta（core＋slot转换合并）|26.234730|1|38.117411|

meta在12/12组中快于base，配对加速中位1.194429；也在12/12组快于本版core。**相对普通core的增量还须复测**：本版core为候选持续登记布局，存在额外开销，不能将它与先前26.622860ms的core跨会话相减或累计收益。新源码已在学完12个group布局后让参考路径直接调用原准备方法，下一轮重新验证精度和性能。

这是内部engine-step的2K输入/48输出测量，prefill和前9步不进入中位数。新meta尚未完成客户端质量/时延验收，19ms未达到。此前通过GSM8K100/100、Vision23/23的是core API，客户端为`(26.744123ms/step,A=1,37.391393tok/s)`，不把该质量结论自动移给新候选。

## CPU诊断定位的下一段

配对性能之后，另用cProfile采集每arm15个稳态decode step、八个rank。它扰动Python开销，嵌套cumulative时间不能相加；同步时间包含设备执行和到达等待，不能当作纯CPU可回收时间。下面仅展示rank0的cumulative时间÷15：

|函数/区段|core ms/diagnostic step|meta ms/diagnostic step|意义|
|---|---:|---:|---|
|`_build_attention_metadata`|8.429|5.303|准备链仍较重|
|25次group metadata build/step|5.699|2.979|slot合并减少发射与调用负担|
|slot prepare，12次/step|3.016|0.936|与独立发射合并方向一致|
|布局record检查，12次/step|0.362|0.270|说明参考登记会影响比较|
|`Event.synchronize`|17.817|17.786|主要是等待；不能删除这个数作为优化收益|

下一步重点仍是metadata准备和实际GPU关键路径。已准备参考登记成本修正，并补测`tp8metastack`：在slot合并上再叠Indexer融合，不能凭此前Indexer小收益/无收益就假定新路径下的临界位置相同。

## 组合验证正在执行

新源码快照`/work/src_manyslots_v8`保留当前GPU整数算法，停止参考方案学完布局后的重复登记，并增加meta＋Indexer组合。`formal_slots_batch_probe_v6`的144组整数参考、尾部和6次变化输入图重放已退出0。

`formal_slots_indexer_audit_v1`已启动正式base/core/meta/metastack四bank、三组配对；此时仍在加载/审计。只有九个配对、四次原生控制、两个slot候选的实际覆盖和48份rank消费者全部通过，才起`formal_slots_indexer_perf_v1`的12组关闭审计配对及独立CPU诊断。原路由、Top5、logprob<1e-3门槛保持。

## W4A8源码对应的后续机会

本轮只读核对了镜像内真正执行的W4A8类与融合GMM源码，保存路径、SHA和代码片段。

* `AscendW4A8DynamicFusedMoEMethod`当前融合GMM支持group-list 0/1；Host代码明确拒绝type2。因此上游未量化8expert稀疏type2候选不能直接移植。旧上游稀疏GMM配对也未显示收益。
* A8W4 pipeline/mid/post在计数模式1下会扫描前面的专家计数；前缀模式0有不同的直接读取路径。Python方法还会把输入0转换回1。可研究让routing直接产生前缀、保留到两个GMM，先验证384/top6/本地48专家的routing与整数计数，再验证量化输出、路由和整网性能。这仍是未实施候选。
* A8W4 pre/post可见若干`InitBuffer(...,1,...)`单缓冲队列；源码也有双缓冲预算常量。不能仅将1改成2：须核对真正tiling、有效tile数、事件依赖和UB预算。结合当前GMM Scalar高读数，先查控制循环，不能宣称双缓冲已有收益。
* 小M wo_a/wo_b的MTE2热点与GMM的Scalar热点继续分开处理，流水/带宽计数和累计时间不替代端到端指标。

## 证据与完整目标

`evidence/formal_a321/results/formal_slots_batch_audit_v2/`保存完整配对、原生控制和消费者验证；`formal_slots_batch_perf_v2/`保存12组结果、三元组校验、cProfile原始紧凑数据及摘要；`formal_w4a8_source_opportunities_v1/source_evidence.json`保存代码片段与SHA。原始路由/请求留远端，不经SSH搬运大文件。

Goal仍active：正式TP8≤约19ms、最终最优服务部署/客户端验收、持续报告/COS/links-server、源代码/manifest/自检与双push均保持完整范围。本轮数据是新的正式优化证据，未完成目标，也没有把独立kernel或诊断态结果当作达标。
