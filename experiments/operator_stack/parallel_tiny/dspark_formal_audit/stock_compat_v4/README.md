# 末轮全 TP 完成边界

正式 `formal_dspark_native_eager_aa_v2` 使用 stock v3，exit1：完整真实40层主模型及三层 MTP 加载、两个 warmup 和首条计数请求通过，第二条在 `Client completion precedes final worker sample completion` 门拒绝。输出 rank 可先发布结果，其他 rank 的已提交 terminal draft 和末轮计数尚未完成；单 rank 设备 fence 不足以证明整个 TP8 已完成。

v4 保留原拒绝门。在每个 rank 的末轮原生草稿/目标设备 fence 和 `sample_end_ns` 之后，使用真实 stock `GroupCoordinator.barrier()` 的 CPU Gloo TP8 group 完成一次 join，再返回末轮输出。所有 rank 的实际设备完成和时间戳均先于 barrier entry，输出 rank 返回才可能触发客户端结束。这次 join 的墙钟计入 decode，稳态不加额外设备同步或 D2H。不同 rank 的 barrier return 可以有微小时间差，设备完成仍全部在客户端结束前。

独立 guard 要求 actual world_size=8、rank 对应、CPU backend=gloo、真实设备 fence、时间戳先于 join；禁止 device group、以 join 代替 device fence、重复 join。不会将客户端时间戳改成 max(worker时间戳)，也不将取回计数器 RPC 的时间塞入 decode。

原始8rank counter与 external/core ID、最终 tokens、客户端结束时钟在完整性门前落盘；失败请求也保留证据。raw 大于1MB时留远端，以 compact/SHA归档。A/A 使用2K输入/47输出的审计请求；关闭审计的正式性能必须另跑2K/256、至少8serial和完整消费者/质量门。

32项 CPU 检查通过：继承 v3 的26项门，并增加错误 backend/world/rank、无设备 fence、重复 join 和8 CPU线程延后最后一个 rank 的实际 publication 回归。这些是 CPU 协议检查，不能冒充 NPU 或消费者通过。

```bash
python3 experiments/operator_stack/parallel_tiny/dspark_formal_audit/prepare_stock_v4.py
python3 experiments/operator_stack/parallel_tiny/dspark_formal_audit/stock_compat_v4/test_stock_v4.py
```

只覆盖新冻结快照 `/work/src_dspark_native_eager_v3` 的 bench/counter和prep freeze，不修改旧终态现场、SDK、数学体或权重。目标仍是包含 DSpark 的正式 TP8 ≤20ms/step，当前未达成。
