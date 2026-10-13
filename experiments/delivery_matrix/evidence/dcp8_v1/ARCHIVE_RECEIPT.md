# DCP8 v1 启动与 canary 归档

## 结论

- v3 包已复制到远端新目录 `/work/delivery_matrix_20261013_v3/delivery_matrix`；仅含当前入口文件、bootstrap 与 `dcp_runtime_20261013_v3`，未复制 v1/v2 runtime 或旧 raw overlay。20 个包文件逐项 SHA-256 与本机一致。
- CPU-only runner/plan 使用 TP8+DCP8、max model len 196608、async、DSpark5、draft graph、BI=1、Engram device index=0；runner `bash -n` 通过，plan 未请求启动服务。
- 服务在 2026-10-13 02:32:36 +08 发起，02:41:56 +08 首次 ready。身份检查确认 8 rank 全为 TP8/DCP8、async、draft graph、DSpark5、BI=1、Engram index=0。
- 唯一 canary 随后失败：HTTP 200 但 body 为空，metrics 返回 500。首因是 frozen `llm_base_proposer.py:1367` 在 DCP 首次 draft 时断言 `long_seq_args is not None`；counter 的 `execute_model without previous sample completion` 是后续连带错误。EngineCore fatal，服务自行退出。未发正式 code8/Dihuo；没有停服、重启容器、改源码或关闭 draft。
- 02:46:20 +08 只读复核：项目容器仍 running，18960 无 listener，容器内无 DCP8 runner/worker/EngineCore 残留，宿主 NPU 4–7（physical chips 8–15）均无进程。没有需要清理的存活服务进程。

## 包与计划校验

| 文件 | SHA-256 |
|---|---|
| `service.py` | `204de912fed5884e11c1a770a4973d8ba09b31a6652ebf31566592826a6dd428` |
| `worker.py` | `4f18899ec4aa3ca3e6b66f91b19f599aa025384303c85908ea430e5565b1383d` |
| `prepare_launch.py` | `9848210a2a53eec1a346561982364f7828d8b5c6270cd105a3d5e3e1b11781fc` |
| `run_service.sh` | `cc5bcd644c78e5259b82f2cfc8aff3627ab6482b52fb36e8bd7f197ec5a1cdd9` |
| `dcp_overlay.py` | `40e6f1fcd4caa5333c061e497104ca06115d8edc4eb13a9b67effc74f69337b0` |
| `bootstrap/sitecustomize.py` | `f65de4040f924fbb34694964f0a7d88ca5a43edd0db3ba1a3dcd9c2cec0efe34` |
| `dcp_runtime_20261013_v3/manifest.json` | `8d09cb35615b1a5bfd66c7e4b8f7602c263d2c580423f3599dcc9816ed90b556` |
| 本地 CPU-only runner | `runner.sh` — `7de45a3a2b72c1416e472c86fdee888803471f268eff92caca2bd9518629f2e9` |
| 本地 prepare 输出 | `prepare_launch_cpu_only.json` — `9d608b1988adb3a47bd94882cd641856fc1c3d5fa65d6ddea921b5838c6916e5` |
| CPU-only plan | `service_plan_cpu_only/plan.json` — `a1ccb4c97b5c053c427896fc81f10835fdfd8daf63c345a49ccc535f842e1f0f` |

`plan.json` 根目录副本与 `service/plan.json` 内容相同，SHA-256 均为
`a497f05431808265681e092bef7dd42ae548ce190895927c933766b20f2a011a`。

## 失败证据与原件映射

| 证据 | 来源/说明 | SHA-256 |
|---|---|---|
| `quality/client/dcp8_code_canary_20261013_v1.json` | 由 `/home/chiro/tmp/dsv41-workload-bench/evidence/dcp8_code_canary_20261013_v1.json` 复制；原件保留 | `dc4d36f73262fa3e21265470105d7a11775c3c8bf1ab49992d09356ce192cec4` |
| `failure_receipt.json` | canary、首因、退出状态与释放检查的绑定收据 | `9f5e521661a067f198cc7ee5c7fbc6fe79c1e31e3f4d7d1c8b4f15985f740bb1` |
| `failure_trace_excerpt.log` | 原始远端日志的相关行摘录 | `d3fc63ca58128eaf5d8706e77b289d9e15286ac6ed5294dec9bf272e10665adb` |
| `run.exit` | runner 退出码记录（`0`；不代表 canary 成功） | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `launched_ready_snapshot.json` | canary 前、服务 ready 时的独立启动收据快照 | `eb94b080f1a9a6e4be951d23dca557674f5d3c552029e2c15d6b055359988734` |
| `launched.json` | 加入 canary fatal 与当前资源状态后的收据 | `708b8ffabcdecb5054bd0e56e11ed4746295873c82e5b0a072a0b6e37f95fde9` |
| `service_selection.json` | 服务 ready 时快照：engine/ASGI 均 initialized | `773fedd97aa8db2bceb13d773fce9cee9c1308bcb8319c1457d2a304d3d9585c` |
| `service/service_selection.json` | 服务 terminal shutdown 后快照：engine 曾 initialized，ASGI 已关闭 | `b5d07f67b9515653aac95cad6058a1b84414035e37a6130b8233ec0969111e4a` |

完整原始日志留在远端 `/work/results/delivery_dcp8_v1/run.log`，大小 405,084 B，
SHA-256 `a3dbbffc275bdfcd799a477788b4b313c7622e1d5b3c71463fc739befb16aee3`；未复制整份日志。
失败后物理芯片 8–15 已确认空闲。该 DCP8 runtime 不得记为质量或性能通过；应先由主代理修复 frozen draft 的 DCP 初始 metadata，再重新生成独立候选/计划与启动收据。
