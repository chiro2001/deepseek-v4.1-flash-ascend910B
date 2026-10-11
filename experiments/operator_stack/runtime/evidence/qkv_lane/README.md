# Tiny QKV lane 终态

完整 tiny 没有完成原生 A/A、真实 QKV 消费者比较、模型质量或性能门。40 层 BF16 dummy 参数曾成功加载并注册，随后捕获受 AiCPU 初始化/执行异常拦住。不得把这些启动尝试称为 tiny 数值通过或性能收益。

projection-only BF16 probe 正常退出，24 输入的合并输出最大 **8 ULP**，超过原 **1 ULP** 门，候选 **reject**，未测候选性能。最坏点为 KV、seed20、输入 scale=1e-3，最大绝对差 `1.4551915228366852e-11`。输入为 `[1,5120]`，两个输出均 `[1,512]`，权重是合成 BF16；scope 不含 RoPE、缓存、压缩器、MoE或整网。

请求 NZ(29) 时日志出现 `allow_internal_format=False` 的 base format 回退警告，旧 probe 没有记录实际 format。因此两种请求布局不能作为真实 ND/NZ 对比。当前源已经补实际格式检查与源哈希绑定，没有重跑 NPU。原生 A/A event 数组仅保留作诊断原始数据。

| 尝试 | physical | exit | 终态 |
|---|---:|---:|---|
| audit_v1 | 2 | 1 | 隔离命名空间的 visible 编号无效，aclInit 107001 |
| audit_v2 | 2 | 1 | 编号修正后 HDC/OPP 发送失败 |
| audit_v3 | 2 | 1 | privileged 容器 env2 仍失败；plog 明确目标侧无文件空间 |
| audit_v4 | 6 | 1 | 参数加载成功；CANN cache 目录遗漏，脚本已修复 |
| audit_v5 | 6 | 137 | initial multistream static dump 同步等待；设备 OOM 日志；仅结束核对归属的本任务进程 |
| audit_v6 | 6 | 143 | 为改用正式阶段顺序结束，未完成 A/A |
| audit_v7 | 6 | 137 | initial single-stream static仍 AiCPU 507018 / Device OOM，捕获不能前进 |
| probe_v8 | 6 | 0 | projection-only 完成；BF16 数值门失败，候选性能未运行 |

v7 的 `Device oom` 日志发生于 `03:59:23`，早于 `load_model` 的 `03:59:24`；HBM 当时约29/64GB。日志证实设备侧异常，尚未确定具体内存种类。没有 reset、删除设备文件、终止他人租户或清 pagecache。当前本 lane 无 NPU 活句柄。

[archive_status.json](archive_status.json) 给出完整口径、每次日志、冻结源快照和远端路径。[binding_review.json](binding_review.json) 保留只读正式接线发现：V4.1 实际使用独立 `multistream_preprocess`，V1 prolog 上注册权重并不能证明真实覆盖。正式模型的数值与性能由主线程独立验收。

`source_snapshots/` 保留实际运行版本。chip2 早期 v1/v2 的 worker 曾在同私有源目录修正，逐次 SHA 未记录；`chip2_latest` 明确为最终保存版本。chip6 v7 三个冻结 Python 文件已与远端源 SHA 逐项核对。`SHA256SUMS.txt` 覆盖本目录的全部其他归档文件；`../../parallel_tiny/qkv_lane/SHA256SUMS.txt` 覆盖当前可审查源。

两次 SSH 取回分别136428B、201004B，只含紧凑日志/JSON/源。权重、大 trace 和构建产物留远端。
