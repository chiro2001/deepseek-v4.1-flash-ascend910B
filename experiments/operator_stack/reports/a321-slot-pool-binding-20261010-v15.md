# slot合并候选：修正实际KV池配置的接入

2026-10-10 · a3-21 chip8–15 · 正式TP8＋strict

当前已验收core的结果保持为GSM8K100/100、Vision23/23、客户端`(26.744123ms/step,A=1,37.391393tok/s)`。19ms尚未达到。slot合并候选的独立数学和图重放已通过，但尚无正式整网收益可计。

## 正式审计的实际覆盖检查拦住了什么

`formal_slots_batch_audit_v1`退出1：`registered_groups=0`、`fallback:contract=1752`。元数据消费者检查拒绝把只走旧路径的候选当作有效优化；后续性能守卫随之退出，未启动性能作业。原生控制记录与bank覆盖、退出码均保留。

这意味着首次接入没有真正执行合并路径，不能把独立kernel成功或输出相同当作整网候选通过。[报告v14](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-formal-quality-slot-batching-20261010-v14.html)发布时该审计仍在加载；本报告更新其终态。

## 接入修正及回归验证

核对当前镜像源码：`model_runner_v1.initialize_kv_cache()`把实际配置存入`self.kv_cache_config`，后续使用其`num_blocks`；旧接入检查只读`vllm_config.cache_config.num_gpu_blocks`，该旧字段不应作为唯一的池大小依据。

已在worker的KV初始化后、编译预热前，绑定`model_runner.kv_cache_config.num_blocks`。候选继续检查pool×logical-block范围，不移除范围约束。新增逐项接入契约诊断，保存dtype、形状、设备、连续性与绑定池大小；不在capture内进行D2H或打印。

`formal_slots_batch_probe_v5`已正常退出0。除144组CPU精确整数参考、尾部保护和6次变化输入图重放外，还验证了“旧字段为空时拒绝登记、绑定实际V1池后允许登记”的回归场景。两种dtype均通过，实验池7939 blocks。独立时间不用于声明模型收益；旧helper在近INT32上界的数学差异及正式池范围约束继续保留。

守卫已启动`formal_slots_batch_audit_v2`，使用独立源码快照`/work/src_manyslots_v7`、正式权重、base/core/meta三个bank、三组配对和strict。此时仍在加载/审计，接入是否真正覆盖12group还须由该作业证明。

审计通过完整路由、Top5/logprob、原生控制、24份rank的12group整数消费者及实际合并覆盖后，才允许启动`formal_slots_batch_perf_v2`的12组关闭审计/profiler配对与单独cProfile。没有放宽数值门槛，没有把覆盖0算成收益。

## 交付与剩余工作

v14源码、证据、manifest已自检并推送GitHub及内网，提交`f48db9c`。本次绑定修正与新证据继续提交/自检/双push，报告上传COS和links-server。仍需完成新候选整网验证、测出真实增益、继续逼近19ms，以及最终最优服务部署与客户端验收；goal保持active。

证据在`experiments/operator_stack/evidence/formal_a321/results/formal_slots_batch_audit_v1/`、`formal_slots_batch_probe_v5/`、`formal_slots_batch_audit_v2/`。原始路由/请求与完整日志留远端任务目录，不通过SSH传输≥1MB文件。
