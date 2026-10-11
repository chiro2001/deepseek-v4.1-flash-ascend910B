# DSpark 真实 AOT 参数接线与 tiny 2K/256 几何验证

2026-10-11。新 goal 已成功重建并保持 active：完整 TP8、真实权重、包含 DSpark 全部执行开销的推理达到 ≤20 ms/step。正式验收尚未达成。

## 并行代理与设备

主线程重新读取 Codex 线程元数据及最新 turn_context，确认 tiny_qkv、tiny_engram、tiny_ids 均实际使用 gpt-6.1-sol/max，三者明确 ACK 当前任务与设备权限。核验记录为 `evidence/formal_a321/subagent_model_recheck_20261011_v42.json`。

TP8 SDK 使用 physical8–15；tiny 使用独立 physical4。每次启动重新核查健康和占用，TP8 仅放行此前明确授权的 80C98001，不 reset、不停止其他租户、不降低正常 0.70 显存预算。tiny 的 initial、SDK、model 三次真实显存门均通过。

## TP8 v5：显存竞争使参数修复尚未被执行

v5 在正常初始化时，rank2 的 free36.64/61.27GiB 小于 0.70 所需 42.89GiB。其他租户短作业在提交后进入部分芯片；本任务没有降低预算或停止租户。

v5 未进入 SDK case，不能声称其 Tensor ABI 或 alias 通过。原始 null exitcodes 保留，另附主线程身份核验后仅收口 5 个自有 child 的清理证据；所有本任务进程最终退出。

## TP8 v6：真实内核已接受参数，捕获 stream 仍需接线

v6 严格冻结 56 文件，freeze SHA256 `f14f0049b51e1495e7856adc45c93e6a535e6f8c54c576a4ca15c9fd9723a5a1`。旧 76 门加真实 Linux 自有 child 清理 9 门，共 85 项 CPU 检查通过。忽略 SIGTERM 的自有测试 child 经 PID/PPID/boot/start_ticks、parent cmdline 及 kernel pidfd 目标核验后，以 pidfd SIGKILL 收口，实际 reaped exitcode=-9。错配身份拒绝发信号，旧 v1–v5 字节保持。

新 snapshot 的 70 个源码条目核验后，`formal_dspark_sdk_init_static_default_v6` 在 fresh8–15 运行。8 rank 标准 WorkerWrapper 正常 init_device 全部通过；没有加载模型权重。

rank2/3/5/6/7 取得完整 SDK 结果。真实 real-warmup 中，Dynamo 入口及 outer-AOT boxed→SDK unboxed 边界均为原 19 个 Tensor，身份与顺序一致、kwargs 保留、没有替换 Tensor 或绕过 generated kernel assert。每 rank 记录 15 次 copy 调用，此前的 `expected Tensor()` 阻碍已在这些 rank 的真实执行中解决。

随后 cold-capture 在原 `torch.npu.graph` 入口被拒绝：`NPU graphs must be captured on a non-default stream`。0 完整 epoch、alias_error=null，不能授予 32 代语义、alias 或正式消费者通过。rank0/1/4 没有完整 SDK 结果，其具体失败阶段不从其他 rank 推断。

v6 自有 8 个 child 全部真实 reaped，exitcode均为 -9，最终 exitcodes 完整；主线程再核查全部 PID 已消失、8–15 无进程。没有在失败 worker 内重试或修复 stream。真实 Ascend model_runner_v1.graph_capture 原生上下文会创建并持有非默认 NPU stream、等待原 current stream，退出恢复原 current。独立 v7 正按该源码准备，不能把原生新建 stream 记成 init-return 默认 stream 借用未变；不会调用包含 empty_cache 的完整 capture_model。

运行紧凑证据共 60 文件，SSH archive 77896B，SHA256 `f47ed5f7b0a41dbc6923fc45f083c5c98847d9ad39eafd6e3b7e4b4a1ee8afbe`，逐文件 SHA 已校验。证据在 `evidence/formal_a321/sdk_init_static_default_v6/`，逐 rank 验证在同级 `sdk_init_static_default_v6_validated.json`。

## tiny 2K/256

独立几何源冻结 28 文件，freeze SHA256 `c719101eeb04abae8e28b60bfc63bdbafeea3c5999ac71a7c9fad34435a85c2d`。真实冻结 native sparse-index builder 的 CPU 交叉检查及正常 HBM 预算正负控共 24 门通过。SWA128/K5、max_model_len4096、max_batched2048、capture代表3072；正常0.70预算与固定2GiB KV池保持。

physical4 的新任务 `dspark_lane_20261011_c4_graph_geometry2k256_v1` 已启动。三次 fresh 门 HBM used 为3289/3290/3298MiB，free均大于45875.2MiB预算要求；43 O投影派生owner初始化、capture后seal、1032参数view及43 router seal通过，capture不做D2H。真实模型类型为完整40层 tiny dummy/BF16、原生MTP3，不能代替正式权重。

本任务是几何/缓存/数值窗口：计划2个warmup和9个独立数值比较，实际rounds重新推导，保留initial/terminal原生bookkeeping和末轮fence。

v1终态exit1：首个OFF warmup实际生成256 token、2段prefill+128 decode、130次native propose/helper成功，target replay128/draft replay0、terminal fence1。原生prefill真实分为2044（sampled=[]）和4（sampled=[87930]）；测试脚本的单prefill断言在请求验证时拒绝该合法分段。43源/cache前后及capture后seal仍通过，target40各2个消费者、draft3各130个消费者。

0完成request、0 geometry step快照，9个数值比较尚未开始；3组真实reader绑定、动态block table、每组665可见读槽的逐step有限/bitwise证据仍缺。warmup关闭router/first-value审计，没有nonfinite记录不构成逐activation精度通过。输出为周期2复读，不能用于质量或性能准入。physical4终态HealthOK/无进程，SSH compact74673B。v1保持冻结；独立v2仅修正测试脚本对原生chunked prefill的计数与顺序判据，不关闭分段prefill、不改变模型或放宽缓存数值门。

旧128/32 tiny OFF/ON同实例计时虽观察到时延下降，输出全部复读，clean-rate=0，不能替代新2K/256或正式≤20ms验收。

## API 接线与后续正式门

旧 `serve_formal_tp8.py` 与 `run_formal_client_acceptance.py` 强制关闭 DSpark/A1，不能直接用于新 goal。新的独立 CPU 桥正在核验真实 EngineClient collective_rpc、external→core request ID 映射、完成时映射删除的生命周期以及客户端结束时钟。计数取回 RPC 必须在客户端 decode end 之后；原末轮 device fence/Gloo join 必须在 HTTP 输出前。CPU/mock 不能冒充实际 HTTP 或设备通过。

完整43层/17点 main/draft/verifier/cache消费者、INT8原位prepack的实际顺序、路由/token/Top5/logprob/接受前缀/rollback仍须正式实际审计。CPU context候选仅设计，其53个冻结文本文件哈希通过，没有设备graph实现、收益或正式准入。

最终正式客户端串行至少8个计数请求，warmup另列，固定2K输入/256输出预算，audit/profiler关闭；按实际 verifier rounds计算 `ms/step=1000*decode_s/rounds`、`A=decoded_tokens/rounds`、`tok/s=decoded_tokens/decode_s`，包含已提交initial/terminal草稿完成，同时报告clean-rate/复读/空答。

完整正式权重、HCCL strict、GSM8K100/100、Vision23/23、服务归属、包含DSpark≤20ms、最优部署及交付全部实际达成后才 complete。
