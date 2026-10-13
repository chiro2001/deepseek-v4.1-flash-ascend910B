# 旧 CED P/proxy 计划内 SIGTERM 收尾收据（2026-10-13）

本收据独立记录旧 P/proxy 的计划内停服和资源检查，不修改前面的 canary、D 根因或旧
run-log 快照。完整事故分析仍见
[`../failure_diagnostics.md`](../failure_diagnostics.md)，该文件 SHA-256 为
`61e0686596d9bca7977c616053db228316d3c04105d5871c823c327983c32816`。

## 操作边界与结果

- 发送 SIGTERM 前再次确认 P `/delivery/metrics` 的八个 rank 都各有唯一旧 tag
  `dm-canary-3e48e9fe86e8454ab214308d2910f6a1`；原始 4,501 B JSON 保存在
  [`../prefill/metrics_after_canary_20261013.json`](../prefill/metrics_after_canary_20261013.json)，
  SHA-256 `667a6f7b7bd47273e1fba3376577187512aeb3ac1afb90a8e87cd43495bae7a1`。
- 停服前精确核对 P container 内 proxy PID `339897` argv 后，对该 proxy 进程发送
  SIGTERM。日志记录 `Shutting down`、`Application shutdown complete` 和
  `Finished server process [339897]`；随后 `127.0.0.1:18992` listener 消失。
- 停服前精确核对 runner PID 文件 `115` 与 API PID `121` 的完整 argv。仅向 P API PID
  `121` 发送 SIGTERM，未向 runner、EngineCore、worker 发信号；runner 留在原位等待 API
  退出并写入退出码。日志记录 API shutdown、EngineCore 收到 SIGTERM并开始 resource
  teardown、HTTP/FastAPI shutdown 完成。
- P API 在 native cleanup 末尾输出 `corrupted size vs. prev_size`，runner 写出
  `run.exit=134`。因此这次是**带清理期 heap-abort 的异常进程退出**，不是 clean exit；
  但容器保持 running 且 P API/EngineCore/workers、producer listeners 和 HBM 进程均释放。
- 没有 `docker stop`/restart、kill P worker、删除 cache、NPU reset 或接触 D container。
  P 与 D 容器最终都仍为 Running、RestartCount=0。D 的 API 在此前 canary 中已退出，本次
  没有触碰 D。

## 停服前后资源门

完整采样原文：

- [`pre_stop_snapshot.txt`](pre_stop_snapshot.txt)，SHA-256
  `0222844c730a11d188b2dcbd70d35ed43a560c12c2669c14caea365888949473`。
- [`post_stop_snapshot.txt`](post_stop_snapshot.txt)，采集于
  `2026-10-13T06:19:49+08:00`，SHA-256
  `3964ca24f15aca94c3320fc09cc690474fcd8a79438b52e63bcd44dd1dd52c80`。

检查结果：

- P 的 `18990`、proxy `18992` 和 P producer KV `19090–19097` 均无 listener；D API
  `18991` 也无 listener。P runner/API/EngineCore 与八个活动 worker 均退出，P container
  本身继续运行；八个 worker PID 仍作为 `Z`/`<defunct>` 僵尸留在原容器 namespace，未清理。
- NPU Health 全为 OK。P 原服务对应的 NPU0–3 上已无 P worker；HBM 从停服前约
  59–60 GB/每 die 降到停服后约 2.9–3.2 GB/每 die。D 对应 NPU4–7 继续无活动进程。
- 停服后 host `free -h` 为 2.0 TiB 总量、1.9 TiB available。停止后没有 CED listener 或
  P worker 仍占用 host RAM/HBM。
- **不能把物理 0–7 整组视为可供 P4 使用：**停服后只读发现其他项目容器
  `m00933363_catlass_runtime_bz_a3_1` 在 host NPU2（对应 chips 4/5）运行外部
  `a3_terminal_gate.py --benchmark matmul`，PID 2703751 及其子进程，cgroup
  `/docker/6c515b1ce0865f7e48ca3dcd28055b31f6078d4cde3a6262e6cfe82f85d4a164`。该项目与
  CED 无关；未终止/接触其容器或进程。P4 使用 chips 0–7，需等其自然释放或由用户协调，
  不能在它活动时启动 P4。

## 停服后的原日志副本

以下文件均从仍保留的 P container job path 或 P 内 proxy job path复制，原件未覆盖：

| 本地副本 | 远端原件 | 字节数 | SHA-256 |
|---|---|---:|---|
| [`prefill_run_final.log`](prefill_run_final.log) | P `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/prefill_18990/run.log` | 146,519 | `ee17fd6554086b16e5ce4d25006217b23ca535156da2c4b3c783ebbfd9eb007d` |
| [`prefill_run.exit`](prefill_run.exit) | P `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/prefill_18990/run.exit` | 4 | `e726b440cc164001999e718b51780d80314709efe06e7857282377be9111127b` |
| [`prefill_runner.pid`](prefill_runner.pid) | P `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/prefill_18990/runner.pid` | 4 | `e623772cc677b6536e0545bc9870ad53a37b5cf0187fcc9851641b52344c7ee8` |
| [`prefill_service_selection_final.json`](prefill_service_selection_final.json) | P `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/prefill_18990/service/service_selection.json` | 12,540 | `0bd93dca17f568b73b93a8ef13925a6eeac82124391025d70ca442820bd23ab7` |
| [`proxy_final.log`](proxy_final.log) | P `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/proxy_18992/proxy.log` | 8,882 | `88fbe1e9272ae3ef251a15776f71bcb805e7d85600995f471bdfe7947436e819` |

v4 新 pair 尚未启动。D4 可在新包逐文件/SHA、role/core/OPP 与 D-only compressor 来源门验证后，
于物理 8–15 独立初始化且不发送请求；P4 必须等物理 0–7 无外部进程后才能开始。
