# DCP8 v2 启动监督归档

## 启动前检查与计划

- 远端新包：`/work/delivery_matrix_20261013_v4/delivery_matrix`；24 个指定文件（7 个主文件、bootstrap、12 个 runtime 文件、2 个 draft 文件及两个 manifest）逐项 SHA-256 与本机一致，均小于 1,000,000 B。递归复制带入的两个本机 `.pyc` 已从新远端包移除，冻结源文件未改。
- CPU-only plan：`TP=8 / DCP=8`、`max_model_len=196608`、async、DSpark5、draft graph、BI=1、Engram index=0；`DELIVERY_DCP_OVERLAY` 与 `DELIVERY_DCP_DRAFT` 均指向 v4 新目录。`service_launch_requested=false`，`runner.sh` 的 `bash -n` 通过。
- 两次宿主 gate 均通过：2026-10-13 02:58:32.908812 +08、02:59:10.648307 +08。两次均确认 NPU 4–7 无进程、18960 无监听、目标容器 ID/image 未变、物理设备映射为 8–15、无旧 delivery runner/service。
- 02:59:33 +08 按独立 runner 请求启动，`launched.json` 记录 `service_launch_requested=true`。启动后没有发送 canary 或其他 chat 请求。

## 终态与首因

服务在 engine/worker 初始化阶段失败，尚未 ready；`run.exit=1`。`stock_graph_observer.py:file_receipt` 抛出 `ValueError: Stock graph observation source mismatch`：实际 DCP draft 文件
`/work/delivery_matrix_20261013_v4/delivery_matrix/dcp_draft_20261013_v2/llm_base_proposer.py`
SHA-256 为 `cbe8f6c913d2c9b4a406c9f9cde03234686dc8f19cace278d306ef24327613f3`；observer 对原 best snapshot 的 `36d7c887…` 仅在路径含 `draft_graph_files` 时允许，而新 overlay 路径/哈希不满足该 guard。没有通过放宽任意哈希、关闭 draft 或重启绕过。

2026-10-13 03:07:29.830130 +08 复核：项目容器仍 running，18960 无 listener，容器内无 service/worker/EngineCore 残留，宿主 NPU 4–7（physical chips 8–15）无进程。

## 文件哈希

| 文件 | SHA-256 |
|---|---|
| `runner.sh` | `b7e6cefcc709fc76c0ad18fd14bd0e0573186b2b214cfe499bed8cdbf996888c` |
| `prepare_launch_cpu_only.json` | `73945b2dad96cfcb89475649686cf19411507120d9f1a152c626667f4483dd61` |
| `service_plan_cpu_only/plan.json` | `55265b4985750be7c55057e7b0d5ae0339266b7112b46905f95a09d36b0a47aa` |
| `launched.json` | `a690bed7dcaa2ec7dcdc6627e4df5034c6430794a882b9bd32f1ca6d924c338d` |
| `failure_receipt.json` | `092ab89ae11d0ec781b3d5e8604c4387b785964b688593011afb8a7dfe4318b9` |
| `failure_trace_excerpt.log` | `ca88d8cc67895016470382fcd82e6fc8c04996f02631e9c4235923bce35a9a18` |
| `run.exit` | `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865` |
| `service/plan.json` | `d9823ac20d1dce896a993501a0ea14429b3aad223a26debbf19891bb0e148399` |
| `service/service_selection.json` | `c33bbf7ef7f834349fa34b30f02bddae10026d7dd33a03f4d086af01c42c1806` |

完整远端原日志保留于 `/work/results/delivery_dcp8_v2/run.log`，大小 189,939 B，
SHA-256 `a48b817be16db206ff9491e5afac16160e6656009daf585a78afa91f05386733`；仅复制相关 trace 摘录。
本次没有 canary 或质量/性能通过收据。等待主代理修复 stock observer 与受控 draft overlay 的兼容后再启动新的独立 job。
