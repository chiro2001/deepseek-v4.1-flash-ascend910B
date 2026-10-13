# 固定 owner fixture 的 2048/256 机制计时诊断包（v3_2k_256，CPU 准备终态）

本目录是 v2（`fixture_timing_diagnostic_v2`，128/32）之后的下一套几何/缓存计时窗口准备件。当前只完成纯 CPU 准备与冻结：未启动任何 NPU 作业，`npu_permission=0`，`--launch` 在 v3 参考 token 取回前硬拒绝。128/32 的计时数字与几何不为本包提供准入；也不构成正式 TP8 ≤20 ms/step 或任何质量声明。

## 几何与配置（自 v3 geometry 契约与准备件提取）

| 项 | 值 |
| --- | --- |
| 输入/输出 | 2048 输入、256 输出（greedy、ignore_eos、detokenize=False） |
| 投机 | dspark，K=5，verify 调度 6 token，capture sizes `[6]`（FULL_DECODE_ONLY） |
| 窗口/长度 | SWA128；max_model_len 4096；max_num_batched_tokens 2048 |
| capture 代表 | `DSPARK_CAPTURE_SEQ_LEN=3072`（draft 图冷 capture 一次） |
| 显存/缓存 | `gpu_memory_utilization=0.70`、`kv_cache_memory_bytes=2 GiB`、block 128、prefix caching off |
| 实际 prefill | native chunked prefill 两段 `[2044, 4]`，唯一可采样 seed 在第二段 |
| decode 轮次 | 按 fresh receipt 推导，不假定 16/128；实测 v2 2K/256 暖机：OFF 128 轮/130 helper，ON 255 轮/2 个 prefill helper |
| 设备 | physical 4、logical 0、容器 `dsv41-tiny-goal20-20261010-c6`（launcher 门：fresh 健康/空 owner/容器映射 + 空闲 HBM ≥70%） |

## 保留的计时纪律（与 v2 相同口径）

- 2 个 warmup + 8 个 OFF/ON 交错配对（每臂 8 个计数请求；同 pair 同 prompt/seed，OFF/ON 顺序按 pair 奇偶交错）。
- decode 窗口：从 `rounds[0]`（首个 prefill chunk）的 target bookkeeping 返回开始，覆盖 partial-prefill draft、seed draft、全部 target decode 与 terminal 已提交 draft，到请求末 `read_receipts` 的设备完成 fence 结束；`ms/step` 分母 = 该请求实际 target decode 轮次。
- 每请求核对：两段 prefill 几何、seed token、decode 每轮 `scheduled=6/verified=5`、进度链、token 与冻结参考一致、target 每 decode 轮 1 次 native replay、ON 每 decode 轮 1 次 draft replay 且仅在 propose 窗口内、OFF 0 次、partial/seed 0 次、terminal（ON）1 次、`terminal_sync_count=1`、`steady_extra_sync_count=0`、helper 计数（OFF=全部行，ON=两个 prefill 行）与行级 `actual_round_at_entry` 绑定、冷 capture helper 全局 1/1。
- 每请求与汇总的 `(ms/step, A, tok/s)` 三元组按实际步数/实际 decode token 数自洽；按臂分别聚合。
- 窗口外 43 owner/source/cache 封印（每请求 before/after 各 43 项），窗口内禁止诊断 D2H/哈希/参数 metadata；profiler、route/logprob audit、first-value trace 全关（`metadata` RPC 明确回报 false）。
- 周期 2 复读分类与 dummy clean-rate=0 如实报告；`tiny/formal admission=false`、`formal_tp8_claim=false`；dummy 复读 A 不得当作收益。
- 计时数字只描述该 tiny 机制诊断；不构成 20 ms/step、质量或正式准入证明。

## 文件与 SHA（当前冻结字节）

| 文件 | SHA256（前 16） | 说明 |
| --- | --- | --- |
| bench_fixture_timing.py | 87edfb01b0e21794 | 2K/256 同实例 OFF/ON 交错计时 bench（参考校验在 torch import 之前） |
| fixture_timing_worker.py | e74f3f5b8630b50e | 冻结 native 算术 + 43 封印 + host-only 计数器 worker |
| timing_controls.py | bf04be1c051284cd | WindowGuard、2K/256 validate_geometry（轮次推导）、load_reference、三元组 |
| launch_fixture_timing.py | c6f80a568216dc8c | 两段式冻结 + physical4 门 + NPU 启动分离（本包未执行 --launch） |
| check_fixture_timing_cpu.py | 8b0d9faae8be45f9 | CPU 门（72 项） |
| build_reference_2k256.py | 4722e7a2ce2cb389 | 从 v3 终态 requests.json 提取参考（不伪造字节） |
| analyze_fixture_timing.py | 874941a97f6e2093 | 终态分析（执行后产出 analysis JSON） |
| fetch_fixture_timing.py | 6ea405e215fe90a3 | 有界只读取回 + physical4 释放证据 |
| native_sources.json | c21003fa6d833973 | 与 v2 逐字节相同（shared/native/vendor/model 冻结 SHA） |
| frozen_shared/*.py | 见冻结 manifest | 6 个共享源逐字节复制（含 `o_proj_cache_fixture.py` = 7dfa1133…） |
| cpu_cases/v1_actual_warmup_off.json | 3d5a0d57a4c4fcab | v1 geometry job 的真实 2K/256 OFF receipt（CPU 门正例底稿） |
| cpu_cases/v2_actual_warmup_off.json | ff447811c9dc2642 | v2 geometry job 的真实 2K/256 OFF receipt |
| cpu_cases/v2_actual_warmup_on.json | 577acdbead2f1929 | v2 geometry job 的真实 2K/256 ON receipt（255 decode、2 helper） |

冻结与就绪产物：

- `evidence/cpu_gate.json`（SHA `217427726b4c6db5…`）：CPU 门结果（72 项控制、18 项参考接口控制、无 torch/NPU import）。
- `evidence/dspark_lane_20261011_c4_fixture_timing_diagnostic_v3_2k_256_freeze_staged.json`（SHA `5442666d4c1fab6e…`）：staged freeze，逐文件 SHA + 纪律标志 + `launch_blocked_by_pending_reference=true`。
- `readiness.json`：`cpu_ready=true`、`npu_authorization.authorized=false / npu_permission=0`、未决项清单与未来命令原文。

## 与 v2 的差异清单（逐文件）

1. `timing_controls.py`：新增 2K/256 契约常量（PROMPT/OUTPUT/K/PREFILL_CHUNKS/REFERENCE_JOB）；新增 `load_reference()`（pending 语义、job、schema、vocab 校验）；`validate_geometry()` 重写为 2048/256：两段 prefill、seed 唯一、decode 轮次推导、进度链、逐行 replay 归属、partial/seed/terminal 相位 replay、helper 行级绑定；`WindowGuard`、`period2_after_prefill`、`triplet`、`pooled_triplet` 逐字节未改。
2. `bench_fixture_timing.py`：docstring；新增 `--prompt-tokens/--tokens`；参考改为 `reference_token_ids_2k256.json`（`load_reference`，在 `import torch` 之前拒绝 pending）；`max_model_len=4096`、`max_num_batched_tokens=2048`；config.json 记录参考来源；prompt 2048、max_tokens 256；窗口断言新增 seed 行（`rounds[1]`）；record 增加 `partial_prefill_seed_and_terminal_draft_included_in_window`；summary 记录外部前置与按臂口径字段。
3. `fixture_timing_worker.py`：`required` flags 增加 `DSPARK_CAPTURE_SEQ_LEN=3072`；`load_model` 增加几何断言（max_model_len 4096、max_num_batched_tokens 2048、speculative K=5、draft `sliding_window=128`）；窗口守卫安装、43 封印、helper 计数、`begin_request/finish_request/metadata` 与 v2 一致。
4. `launch_fixture_timing.py`：2K/256 `GEOMETRY`；注入空闲 HBM ≥70% 门；flags `DSPARK_CAPTURE_SEQ_LEN=3072`；bench 命令行带 `--prompt-tokens 2048 --tokens 256`；参考 token 两段式冻结（pending→`_freeze_staged.json`，materialized→`_freeze.json`）；`assert_launchable` + 远端首行 assert 双重阻断 pending；新增纯函数 `build_manifest/build_readiness`；写 `readiness.json`；`--launch` 需 final freeze 且参考来源必须是 v3 作业；远端程序仍走 v2 的 CPU_DISCOVERY（fresh 健康/空 owner/挂载/模型 sha/审计输出路径）。
5. `check_fixture_timing_cpu.py`：以 v2 风格扩充为 72 项控制：窗口拒绝语义、consumer/helper AST 级保持、HBM 预算门（3289/7465 通过、预算不足拒绝）、真实 v2 2K/256 OFF/ON 正例、26 项负例（丢 partial/seed/terminal draft、几何/进度/replay/helper/窗口守卫/Fence 篡改等）、三元组与 pooled 正负例、参考接口 18 项（pending/missing/错 job/缺 seed/长度/vocab/提取器一致性与 fail-closed、包内参考必须保持缺失）、freeze/readiness 语义、远端 payload base64 往返 SHA 校验、gate→launcher 前置一致性、cpu_cases 证据完整性。
6. `analyze_fixture_timing.py`：默认 job、参考 2k256、窗口 seed 行断言、几何按臂输出（实际轮次）、steady 计数从 geometry 推导、scope 改 2048/256。
7. `fetch_fixture_timing.py`：默认 job 与 docstring（远端 REMOTE 逻辑不变，仍顺带取回 physical4 健康/容器进程证据）。
8. `build_reference_2k256.py`：新增提取器（纯函数 + CLI），同 seed 全部记录必须 bitwise 一致，必须含 canonical 六 tag（`native-off-S-A/B`、`candidate-S-off/on`、`native-on-S-A/B`），256 长度、词表范围、三 seed 不碰撞；仅允许从路径含 v3 作业名的本地 `results/requests.json` 生成，禁止伪造。
9. `native_sources.json`、`frozen_shared/`：与 v2 逐字节相同（共享源 SHA 全部一致，含 43 owner fixture 的 `7dfa1133…`）。
10. `cpu_cases/`：新增真实 2K/256 证据底稿（v1 off、v2 off/on）。

## 2K/256 参考 token 接口（pending v3 terminal evidence）

- 期望文件：本目录 `reference_token_ids_2k256.json`（当前不存在；包内任何脚本/门都不创建它）。
- 来源：v3 作业 `dspark_lane_20261011_c4_graph_geometry2k256_v3` 终态本地取回的 `results/requests.json`（另一子代理落地；路径必须包含该 job 名）。
- 字段：`pending=false`、`source_job`、`source_requests_path`、`source_requests_sha256`、`source_records`、`extractor`、`extracted_unix`、`tag_subset_allowed`、`fabricated_reference_bytes=false`、`seeds.{0,1,2}.{tags,token_ids(256 ints)}`。
- 提取命令：`python3 build_reference_2k256.py --job dspark_lane_20261011_c4_graph_geometry2k256_v3 --source <本地 v3 runtime>/results/requests.json --write`
- 提取规则 fail-closed：同 seed 的所有记录（含 warmup）token 必须完全一致；每 seed 必须含 canonical 六 tag；长度 256、id ∈ [0,129280)、三 seed 互不相同。任何不满足即拒绝写出。
- bench/launcher 在文件缺失或 `pending=true` 时于任何 torch/NPU 动作之前失败（CPU 门第 71 项控制验证该行为）。

## 未来启动命令（原文）

```text
python3 build_reference_2k256.py --job dspark_lane_20261011_c4_graph_geometry2k256_v3 --source <本地 v3 runtime>/results/requests.json --write
python3 launch_fixture_timing.py --freeze-only
python3 launch_fixture_timing.py --job dspark_lane_20261011_c4_fixture_timing_diagnostic_v3_2k_256 --pairs 8 --launch
```

说明：第 2 步在参考 materialized 后写 `evidence/dspark_lane_20261011_c4_fixture_timing_diagnostic_v3_2k_256_freeze.json`（final freeze）；第 3 步只有在主线程另行授权 fresh physical4 窗口后才可执行，且必须 final freeze 一致。本包未执行第 3 步。

## 未决依赖

1. v3 终态参考 token（阻塞 `--launch`）；取回后需重跑 `--freeze-only` 生成 final freeze，`readiness.json` 自动刷新为 present。
2. 主线程 NPU 窗口授权（fresh physical4 健康 + 空 owner + 空闲 HBM ≥70%）。
3. 终态 physical4 健康与释放证据取回（fetch 脚本已备）。
4. v3 终态如有 ON 臂实测 receipt，可按需替换 `cpu_cases/` 底稿并重跑 CPU 门（非阻塞：正例已含真实 ON 形态）。

## 遗留风险

- 计时按臂口径：v2/v3 真实暖机证据显示 2K/256 ON（draft graph 路径）decode 轮数可与 OFF 不同（accept 行为不同），包按 receipt 推导，不假设两臂轮数一致；报告必须按臂如实呈现 A 与 tok/s，不得跨臂用 A 声称收益。
- draft 图冷 capture 断言为「恰 1 个 `[6]` 条目」（与 128/32 冻结行为一致）；若 2K/256 实际条目集合不同，worker 会在启动早期失败并留证，需修正后重新门控/冻结。
- `analyze_fixture_timing.py` 断言 startup `parameter_metadata_views==1032`（同模型、同 43 封印来源）；若终态不同需人工复核后再分析。
- launcher 的 HBM 门要求 fresh 空闲 ≥ 70% total；不满足即拒绝启动（禁止降低预算）。
- 远端目的目录必须不存在（CPU_DISCOVERY 已断言），防止覆盖历史证据。
- 本包所有数字均为 dummy 周期复读机制级；dummy clean-rate=0；任何外推（收益/质量/20ms）均不成立。

## 纪律

不改动 `fixture_timing_diagnostic_v2` 与 v7 冻结字节；不 commit/push；远端只读访问有界（取回证据除外）；不 reset/dropcache/停止未知进程；临时文件仅落 `~/tmp/dsv41-tiny-timing2k`。
