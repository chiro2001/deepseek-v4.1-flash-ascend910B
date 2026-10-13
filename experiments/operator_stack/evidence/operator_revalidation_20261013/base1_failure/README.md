# base1 一次性启动失败证据

状态：按 root 授权仅启动一次；EngineCore 未完成初始化，端口未开放。没有重试、调参、重启 D 容器、人工停止进程或发送生成/客户端请求。完整原始日志为本目录 `run.log`，SHA-256 与 D 容器源文件一致。

## 第一失败点

`run.log` 中最早的硬错误位于 2026-10-13 02:27:40（日志时区相当于 +08 的 10:27:40）：Worker pid `1625867` 启动失败，日志明确为：

> Free memory on device (10.97/61.27 GiB) on startup is less than desired GPU memory utilization (0.7, 42.89 GiB).

EngineCore pid `1625665` 于 02:28:14 报 failed to start，外层 `run.exit=1`。日志未指出发生该 memory check 的物理 die 编号，也未给出 rank4→physical-chip 映射。不能仅凭该错误把检查设备认定为 physical 12。

## rank 与设备证据边界

服务选择文件的计划请求 physical chips `8–15`，配置 TP=8、DCP=1、`VLLM_BATCH_INVARIANT=1` 和正确的 A3 BI OPP 路径。运行日志中可确认的初始化进度如下；日志给出 local rank，不给出物理 die 编号：

| run.log 内 PID | rank / local_rank | 初始化证据 |
|---:|---:|---|
| 1625711 | 0 / 0 | BI enable，world size 8 |
| 1625730 | 1 / 1 | BI enable，world size 8 |
| 1625774 | 2 / 2 | BI enable，world size 8 |
| 1625821 | 3 / 3 | BI enable，world size 8 |
| 1625867 | rank 未显式打印（按 spawn 顺序位于 rank 3 后、rank 5 前，可能是 rank 4） | WorkerProc failed；free-memory 检查失败，未记录 rank/device 行 |
| 1625918 | 5 / 5 | BI enable，world size 8 |
| 1625966 | 6 / 6 | BI enable，world size 8 |
| — | 7 未见完成行 | 八 rank 未全部就绪 |

API/Engine 日志 PID 分别为 service `1624804`、EngineCore `1625665`。同一启动轮次的 D-container `docker top` 曾列出 host PID `3427861`（launcher）、`3427871`（service）、`3430410`（VLLM::EngineCore）、`3431024`、`3431065`、`3431129`（VLLM::Worker）及 worker spawn 子进程 `3431287`、`3431415`。因为进程退出后 `/proc/<hostPID>/status`、`NSpid` 与 cgroup 均不可读，内部 PID↔host PID 的完整交叉表无法事后恢复；不按进程创建顺序臆测各 worker rank 的 host PID。

## 启动窗口 NPU 与归属核对

同一轮启动窗口的 `npu-smi` 进程表记录（采样命令无单独 wall-clock 时间戳，故仅能定位在 10:27–10:28+08 的初始化/失败窗口）：

| physical ID | NPU 进程表 PID / 名称 | 报告显存 | 同轮 D `docker top` / 事后归属 |
|---:|---|---:|---|
| 8 | `3431024` / `VLLMWorker` | 117 MB | 同 PID 出现在目标 D 容器 `docker top`，属于本容器 worker；rank 未能交叉 |
| 9 | `3431065` / `VLLMWorker` | 117 MB | 同 PID 出现在目标 D 容器 `docker top`，属于本容器 worker；rank 未能交叉 |
| 10 | `3431129` / `VLLMWorker` | 117 MB | 同 PID 出现在目标 D 容器 `docker top`，属于本容器 worker；rank 未能交叉 |
| 12 | `3428147` / `python3` | 45,387 MB；die HBM 49,230/65,536 MB | **不在同轮目标 D 的 `docker top` 名单中**；NPU 工具的 container PID 字段为 `NA`。事后 `/proc/3428147/status` 与 `/proc/3428147/cgroup` 均 absent，容器/租户归属 unknown。不能据此归为本次服务或 `fyf_1009`，也不能据自然释放推断来源。|

这份 NPU 快照里 physical 11、13–15 未列进程行；snapshot 的 PID→rank 映射和 rank4 的目标 die 均未记录。失败后 10:28:47+08 的最终快照中 NPU process rows 全空、HBM 回到基线附近；这只能描述最终现场，不能追溯归属 3428147。

## 服务与留存状态

- `service_selection.json`：service instance ID `a1e248a17c744195a120354545ec9d87`；`engine_initialized=false`、`ASGI_startup_completed=false`、quality/performance 均未通过/观测。
- 其中 `runtime_fingerprint=3311e992…` 是 CPU-plan fingerprint，且 metadata 标为 `simulated_hardware=true`；它不是实际 runtime fingerprint。
- 最终检查：D 容器 CID `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0` 仍 running、restart=0；18960 无 listener；NPU process rows 空。未访问 `/v1/models`，因为服务未监听。
- root 已要求停止现场轮询；本回执后不做进一步轮询或清理，也不停止外部 `fyf_1009`。

## 归档文件校验

- `run.log`：从 D 容器单文件 SSH/base64 传回后解码；46,825 bytes，SHA-256 `192558e37f8be4e3d6e40528a5cafc5bd5d8143c2081e0ddcac8b7b34c1c491c`，与远端原件一致。
- `run.log.text_projection`：经文本接口生成的可读投影，SHA 不同且字节数 46,777；不作为原始日志或哈希证据，原始证据以 `run.log` 为准。
- `run.exit`（内容 `1`）：SHA-256 `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`。
- `plan.json`：SHA-256 `242295357597407e36eb865bb3ecf551f4add662ed7e8a6dbcfc2257944c495f`。
- `service_selection.json`：SHA-256 `e25982c728d09a78abe023e983fc0d77bf7dc9558ae11eb824ab6fe12f17278d`。
