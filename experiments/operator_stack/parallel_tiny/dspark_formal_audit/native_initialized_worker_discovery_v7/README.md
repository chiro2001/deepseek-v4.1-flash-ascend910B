# SDK init-only 小门的原生 capture-stream 生命周期 v7

本目录完成原生 capture-stream 的最小修订，供主线程在新 snapshot、新 worker 中运行 `static_default` / 32 代小门。95 项 CPU 检查已通过；本 revision 的 NPU 权限和 NPU 作业数均为 0，实际 SDK 32 代、copy/alias、device、full43、prepack 与正式收据均为 false。full43/17 点 consumer 实施继续暂停。

v6 的 rank2 完整结果显示 real-warmup 已发生 15 次 copy，但 cold-capture 在正常 init-return 的 default stream 上被 SDK 拒绝，完成 epoch 为 0，`alias_error=null`。该原始结果仅说明已走到的阶段，不授予 v7 的运行结论。结果 SHA 与路径见 `evidence/cpu_native_capture/v6_cold_capture_reference.json`。

v7 调用已由正常 worker 导入的 `vllm_ascend.worker.model_runner_v1.graph_capture(worker.device)`。实际 owner、全文 SHA 和函数 AST SHA 都受 controller manifest 约束；返回值必须属于原生 `GraphCaptureContext`。原 factory 创建非默认 stream，让它等待原 current stream，并通过原生 `torch.npu.stream` 生命周期选择和恢复。

原 init-return 基线保存在 `initialized_worker_scope` 和原 stream 资源中。`native_capture_scope` 如实记录 `native_context_creates_stream=true`、`init_return_stream_borrowed_unchanged=false`，另存实际 native stream 和 scope。既有 `own_stream_created=false` 表示 gate 没有直接调用 stream constructor，不能据此宣称原 stream 未改变。原生 context、manager 和 stream 持有到 worker 退出；probe allocation、warmup、cold-capture、32 代都位于同一个原生 context 内。设备、runtime context、rank、PID 与 boot 身份保持原门约束。

正常退出后，gate 严格核验恢复到原 init-return stream 和 scope，且 idle / noncapturing。失败退出只交给原 factory 的退出生命周期，随后终止该新 audit job，不额外查询设备、修复 stream 或重试。v6 的强映射 ABI、19 Tensor/kwargs 的 boxing adapter、SDK/kernel assert、数学与 tuple、32 代及受身份保护的 pidfd 清理保持。六个 ABI/cleanup/math 源文件逐字节一致；旧 v1–v6 冻包的准确集合与全部 SHA 也已复核。

CPU 共 95 项：原 85 项重跑，加原生 capture lifecycle 10 项。新测试执行实际 factory 和 `StreamContext` 的源码 AST，使用 CPU fake state，验证非默认 stream、等待原 stream、正常恢复、来源/owner/设备/context/rank 错配拒绝、失败仅原生退出以及连续 32 次 scope 稳定。这些 CPU 事实不能替代实际 NPU 收据。语法、direct help/dry-run、无真实 torch/SDK 的 package import 也已通过，日志和 receipts 见 `evidence/` 与 `cpu_validation.json`。

主线程重新完成 fresh8–15 资源与 host 门后，使用独立新目录执行：

```bash
bash /work/NEW_PRIVATE_SNAPSHOT/stack/run_stack.sh \
  parallel_tiny/dspark_formal_audit/native_initialized_worker_discovery_v7/init_only_driver.py \
  --model /home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq \
  --physical-chips 8,9,10,11,12,13,14,15 \
  --output /work/NEW_JOB/gate --variant static_default --generations 32
```

`NEW_JOB/gate` 须尚不存在；每个 variant 使用 fresh worker。candidate 源集为 14，launcher 与既有 AA 依赖由主线程管理。本包只使用原 native factory，不调用完整 `capture_model`，不直接 patch stream，不清 cache、不 reset、不降低正式 0.7 显存预算。`freeze.json` 与 `SHA256SUMS` 给出准确 payload 集合；每个文件小于 1 MB。
