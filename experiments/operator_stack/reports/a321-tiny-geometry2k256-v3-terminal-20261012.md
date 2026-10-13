# a321 tiny geometry 2K/256 v3 终态报告（20261012）

## 结论

独立 2K/256 geometry/缓存/数值门 job（dummy 模型 + 真实 DSpark 推理链）已自然到达终态：

- **exit=0**，**20/20 请求完成**（18 个带审计的 geometry 请求全部远端原子复核）。
- **9/9 数值门全部通过**：3×`native_off_AA`、3×`off_on`（candidate-0/1/2）、3×`native_on_AA`；每项 `token_equal=true`、`top5_keys_equal=true`、**`max_logprob_delta=0.0`**。
- 3828 个 geometry step 原始文件（141,359,997 B）在远端完成逐文件重构校验且按设计**未取回**；ON 臂捕获绑定点地址跨请求稳定；三组均为 665 slots / 615 native padding、无 active null。
- `failure_lines=[]`、`nonfinite_files=[]`；physical4 已无 NPU 进程，容器内 bench 进程已退出。
- 冻结门 `complete_9_diagnostic_gate_passed=true`（本地 freeze 全量断言通过）。

口径纪律：dummy 输出周期复读、clean-rate=0；本 job 只验证机制与数值，**不得外推为正式 TP8 20ms 或质量结论**。所有 admission/performance/formal 标志显式为 false（见证据 JSON `declarations`）。

## 证据路径与 SHA256

基准目录 `BASE = operator-stack-tp8/experiments/operator_stack/parallel_tiny/dspark_lane/fixture_timing_diagnostic_v2/geometry_2k_256_v3_cpu_prepare/`：

| 证据 | 路径 | SHA256 |
| --- | --- | --- |
| 终态证据 JSON（本任务交付 a） | `operator-stack-tp8/experiments/operator_stack/evidence/formal_a321/tiny_geometry2k256_v3_terminal_20261012.json`（10,097 B） | `c459fa9154d2068db3f1dca4de0904dc00552820eb5568956db359459c3d431e` |
| 冻结终端分析 | `BASE/frozen_runtime_terminal/dspark_lane_20261011_c4_graph_geometry2k256_v3/terminal_analysis.json` | `d9faf88acb282bb93cdf09f02d6ce92bb99741cbe19fb39014d24478b600416b` |
| 终端交付清单（306 文件，13,008,381 B） | `BASE/delivery_geometry2k256_v3_terminal_files.json` | `ce18c503acdcf772374b9122e673a8120ab67b3b9dcbee60fa744f41306b79ac` |
| compact 取回回执（252 文件） | `BASE/runtime_evidence/dspark_lane_20261011_c4_graph_geometry2k256_v3_fetch.json` | `1d383cf6415fccfb887ff318f23e917bc38f6eaab927ddb6e15c54af02cda21c` |
| 终态状态快照 | `BASE/runtime_evidence/dspark_lane_20261011_c4_graph_geometry2k256_v3_status.json` | `cdcc7f63d69247fd589409753dd0eb94c51874ebd9ef2767360a840993247d5e` |
| 远端 CPU 几何聚合 | `BASE/runtime_evidence/dspark_lane_20261011_c4_graph_geometry2k256_v3_geometry_terminal_aggregate.json` | `e72daa34e709430f7700c6d959d70bc473f0341ca71f2772028e7b1ec6ba1e1a` |

冻结原始件主树：`BASE/frozen_runtime_terminal/dspark_lane_20261011_c4_graph_geometry2k256_v3/`（取回原件 + preflight/run.log/checks/results 等）。

## 传输口径

- compact 取回 **252 个文件 / 11,967,292 B**，逐个原件 SHA 校验通过（组装后 `original_compact_bytes_verified_after_assembly=true`）；`results/requests.json`（1,064,345 B，SHA `a01abe2be7fb327a3f82e6d6ed66bfba78b3141b0183ff128c4a73f16d3eb1ed`）超 1MB 阈值按设计留远端，其字节/SHA 由远端聚合证明。
- SSH 单响应硬上限 900,000 B（脚本内断言）；实测**单次最大 58,769 B**；取回总打包 1,345,703 B，聚合响应 21,069 B，终态状态响应 725 B。

## 未完成 / 失败项

- 无失败项：无 failure_lines、无 nonfinite、9/9 门通过、exit=0。
- 遗留说明（非缺陷）：3828 个 step 原件与 `requests.json` 原件按设计留远端（`remote_step_files_transferred=false`）；dummy 输出周期复读、clean-rate=0 为预期现象。

## 过程摘要

- 只读轮询 6 次（每 ~10 分钟，17→18→19→20 请求），全程仅 status 查询；终态（run.exit=0，时间戳 1791745016）后依次执行 bounded compact 取回 → 远端 CPU 只读聚合 → 本地 freeze。
- 未启动任何新 NPU 作业、未重启/停止容器、未触碰 physical8-15、远端全程只读（fetch/aggregate/status 响应均 <1MB）。
- 监控快照归档于 `~/tmp/dsv41-tiny-geometry/`（含 5 个只读状态快照及 SHA；首次 17/20 轮询快照未单独归档，其余 5 次均有留档）。
