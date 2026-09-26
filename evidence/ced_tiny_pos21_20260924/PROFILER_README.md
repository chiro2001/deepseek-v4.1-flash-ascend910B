# 1+1 CED profiler 证据索引

**范围：**A3-22 `model-tiny`、TP1、D 角色，分别采集 eager 与 `FULL_DECODE_ONLY`
graph；Engram 关闭。D 用 `/dev/davinci7`，P 保持在 `/dev/davinci6`。A3-21
的 8+8 与本次测试不共用设备。此测试用于看事件与执行顺序，不代表真实权重、
长上下文或生产性能结论。

## 按请求窗口的关键结果

| 模式 / 请求 | HTTP wall time | SparseFlashMla 实际核数及 CANN stream | Host API `Record/Wait` | Device task `Record/Wait` |
|---|---:|---|---:|---:|
| eager M1 `cmpl-c008829a…` | 412 ms | 69 在 stream 47 | 716 / 569 | 716 / 495 |
| eager M8 `cmpl-c91f4790…` | 1,542 ms | 350 在 stream 47 | 3,621 / 2,875 | 3,621 / 2,550 |
| graph M1 `cmpl-bab7da9f…` | 332 ms | 38 在 stream 47 | 388 / 307 | 388 / 271 |
| graph M8 `cmpl-f6662a93…` | 419 ms | 240 在 stream 2，40 在 stream 47 | 468 / 348 | 2,860 / 2,227 |

Graph M8 中 stream 2 的 240 个 attention 核按时间分成 6 组，每组 40 个；同窗
出现 2,400 个 `EVENT_RESET`。这证明 graph 路径下出现了重复的设备事件任务，
不证明某个具体 wait 消费了哪一个 record。trace 没有 vLLM request ID 或 event
handle；KV connector 日志才直接记录 P→D KV 传输完成时间。

真实权重 A3-21 的单 token prompt-tail eager 对照比 stream 变化更直接：只强制
尾步 eager、保留后续 FULL graph 时短针恢复。因此本轮不建议为了定位故障而先
在正在跑的 8+8 上重启做 `DSA_OVERLAP=0` 消融；只有尾步 eager 复测仍失败或
需要单独量化 overlap 收益时再安排。

## 复核与归档

- 详细窗口、stream、event、KV 顺序和证据限制：[`PROFILE_WINDOW_ANALYSIS.md`](PROFILE_WINDOW_ANALYSIS.md)
- 逐请求机器摘要：[`profile_window_summary.json`](profile_window_summary.json)
- 分析程序：[`analyze_prof_windows.py`](analyze_prof_windows.py)
- COS keys、文件 SHA 和下载命令：[`PROFILE_ARCHIVES.md`](PROFILE_ARCHIVES.md)
- Graph 原始 trace（私有）：`share/xfer/ced_tiny_stream_prof_d_graph_20260924.tar.gz`，SHA-256
  `e59f6081406f34a36fdf4f24c514a1bbd085403165e12d9b37dda437f8439abf`
- Graph controls（私有）：`share/xfer/ced_tiny_stream_prof_d_graph_20260924_controls.tar.gz`，SHA-256
  `d8a2b47f2f26c32de5009d51f5e52729c191ab2eda8aa1774d8447b2f1be47a6`
- Eager 原始 trace（私有）：`share/xfer/ced_tiny_stream_prof_d_eager_20260924.tar.gz`，SHA-256
  `f70045e5264228eb40f32e14b9c80520c0b85fc36f8f5bbb45a2be71d85ea7d9`

`/start_profile` 与 `/stop_profile` 均 HTTP 200，CANN 成功导出 CSV/trace；两臂
都记录了 profiler 在 `RECORD` 状态停止的 warning，graph 臂另有 external callback
thread warning，解读时需保留这些限制。

采集完成后已停止并移除 Graph D 与 proxy。P 仍在 `/dev/davinci6` running；
`npu-smi` 未列出 Phy-ID 7 上的进程，但仍报告约 2.9 GB HBM。清理快照 key 与
SHA 见 `PROFILE_ARCHIVES.md`。A3-22 chip0/1 未使用，A3-21 的 8+8 未中断。
