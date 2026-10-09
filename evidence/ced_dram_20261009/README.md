# CED DRAM 初版设备预检（2026-10-09）

【实测】这两份日志来自 a3-21 的独立实验容器，经 COS 取回到本目录。
实验全程持有 `/run/lock/expert-prefetch-npu-{Phy-ID}.lock`；结束后容器退出并释放锁。

| 日志 | 覆盖范围 | 结果 |
|---|---|---|
| `probe8.log` | Phy-ID 6–13，逐卡向量运算及 128×128 FP16 matmul，结果与 CPU 期望逐元素比较 | 8/8 |
| `hccl8.log` | Phy-ID 6–13，8 rank HCCL all-reduce，各 rank 输入 rank+1，三轮，期望和 36 | 8/8 × 3 |

镜像 `local/dsv41-a3-ced-pd:v3`，与既有起服脚本一致的 privileged + device +
只读 driver/firmware 挂载。测试没有加载模型、重置设备或操作其他容器。
设备健康查询中 8–13 仍显示 Alarm；这些预检证明基础计算和 TP8 通信可以运行，
尚不证明完整模型正确性、持续稳定性或性能。

另一次 Phy-ID 6 的自进程 Mooncake HBM→host 1024 B 读回逐元素校验通过，
SHA256 `8808405eec6fbe306fe3369f88daed79dd5613ddbb5e801f632b01d6218c5f08`。
该输出保存在会话工具结果中，本目录没有它的原始文件；不把自进程读回当作
八 rank 跨进程 mock 消费已完成的证据。

a3-21 时钟比本机慢约 7–8 分钟，以上日志沿用远端时间。
