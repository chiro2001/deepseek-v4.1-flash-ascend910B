# a321 tiny 2K/256 同实例 OFF/ON 机制计时终态报告（20261012）

job：`dspark_lane_20261011_c4_fixture_timing_diagnostic_v3_2k_256`（tiny DSpark 线，physical4 / TP1 / 容器 `dsv41-tiny-goal20-20261010-c6`）。
本报告只描述该 tiny 机制诊断；dummy 周期复读，clean-rate=0，tiny/formal admission 全 false，不得外推正式 20 ms/step 或质量结论。

## 结论

- **终态 exit=0**：`run.exit`=`0`，`done.json` exit=0、finished_unix 1791746024.77（CST 2026-10-12 03:13:44）；**18/18 请求完成** = 2 warmup + 16 measured（8 个 OFF/ON 交错 pair，每臂 8 个计数请求）。
- 失败行 0、非有限 0；`run.log`（151 行）以 `FIXTURE_TIMING_COMPLETE` 结束；supervisor PID 2251067 已退出；physical4 终态健康 OK、无模型进程，容器仅剩 `sleep infinity`。
- **18/18 请求 token 与`reference_token_ids_2k256.json`（v3 终态提取）逐位一致**；8/8 pair 内 off/on token 相等；period2 复读 18/18、dummy clean-rate=0.0。
- 机制计数与封印全通过：initial/terminal draft 均入窗、terminal fence 每请求 1、helper/replay 计数按臂符合冻结口径、冷 capture 恰 1 个 `[6]` 条目、43 项 fixture 封印 before/after 每请求通过。
- 按臂观测（同实例机制计时）：**OFF 46.236890121 ms/step、26.594704279 tok/s；ON 38.780688221 ms/step、31.707957644 tok/s**。OFF→ON 观测差 7.456201899 ms（-16.13%），仅适用于此同实例 tiny 机制诊断；不得跨臂用 A 或该差值声称收益或正式吞吐。

## 按臂三元组（按窗口聚合）

口径：`ms/step = 1000 × Σdecode_s / Σrounds`；`A = Σdecode_tokens / Σrounds`；`tok/s = Σdecode_tokens / Σdecode_s`。逐请求三元组先单独自洽（`ms = 1000·A/tok/s`，rel_tol 1e-12），再按臂求和聚合。

| Arm | 请求 | decode_s（窗口合计） | steps(=Σrounds，receipt 推导) | decode tokens | ms/step | A | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OFF | 8 | 76.7070007105358 | 1659 | 2040 | 46.23689012087751 | 1.2296564195298372 | 26.594704278664405 |
| ON | 8 | 64.33716175938025 | 1659 | 2040 | 38.78068822144681 | 1.2296564195298372 | 31.707957643975046 |

轮数按每条 fresh receipt 推导（不假定）：measured 两臂均为 3×128 轮 + 5×255 轮；seed0→128 轮、seed1/2→255 轮（同 pair 两臂轮数相同，由同一 seed 决定）。warmup-off=128 轮、warmup-on=255 轮（未计入聚合），与 v2 2K/256 暖机形态一致。聚合 A 两臂同为 1.2297 是「两臂合计步数/Token 恰好相同」的算术结果，不构成任何跨臂收益声明。

## 逐请求明细（执行顺序，18 个请求）

measured=√ 的 16 个请求进入聚合；窗口含 initial/terminal 已提交草稿；helpers=该请求 native helper 调用（=成功返回）；replay 为窗口内 native replay 返回数。

| # | tag | arm | seed | measured | decode_s | rounds | dec_tok | ms/step | A | tok/s | helpers | tgt_replay | draft_replay |
| ---: | --- | --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | warmup-off | off | 0 | — | 6.059987290 | 128 | 255 | 47.343650702 | 1.992187500 | 42.079296177 | 130 | 128 | 0 |
| 2 | warmup-on | on | 1 | — | 9.914481660 | 255 | 255 | 38.880320235 | 1.000000000 | 25.719952767 | 2 | 255 | 255 |
| 3 | pair-0-off | off | 0 | √ | 5.973901130 | 128 | 255 | 46.671102580 | 1.992187500 | 42.685674644 | 130 | 128 | 0 |
| 4 | pair-0-on | on | 0 | √ | 5.047856690 | 128 | 255 | 39.436380393 | 1.992187500 | 50.516489600 | 2 | 128 | 128 |
| 5 | pair-1-on | on | 1 | √ | 9.858940120 | 255 | 255 | 38.662510274 | 1.000000000 | 25.864849254 | 2 | 255 | 255 |
| 6 | pair-1-off | off | 1 | √ | 11.716086820 | 255 | 255 | 45.945438509 | 1.000000000 | 21.764946259 | 257 | 255 | 0 |
| 7 | pair-2-off | off | 2 | √ | 11.647154330 | 255 | 255 | 45.675115021 | 1.000000000 | 21.893759863 | 257 | 255 | 0 |
| 8 | pair-2-on | on | 2 | √ | 9.830103200 | 255 | 255 | 38.549424313 | 1.000000000 | 25.940724610 | 2 | 255 | 255 |
| 9 | pair-3-on | on | 0 | √ | 4.998325610 | 128 | 255 | 39.049418829 | 1.992187500 | 51.017084499 | 2 | 128 | 128 |
| 10 | pair-3-off | off | 0 | √ | 5.853721880 | 128 | 255 | 45.732202190 | 1.992187500 | 43.562028605 | 130 | 128 | 0 |
| 11 | pair-4-off | off | 1 | √ | 11.842283740 | 255 | 255 | 46.440328393 | 1.000000000 | 21.533008801 | 257 | 255 | 0 |
| 12 | pair-4-on | on | 1 | √ | 9.908804870 | 255 | 255 | 38.858058313 | 1.000000000 | 25.734687821 | 2 | 255 | 255 |
| 13 | pair-5-on | on | 2 | √ | 9.791411940 | 255 | 255 | 38.397693882 | 1.000000000 | 26.043230697 | 2 | 255 | 255 |
| 14 | pair-5-off | off | 2 | √ | 11.817361100 | 255 | 255 | 46.342592549 | 1.000000000 | 21.578421599 | 257 | 255 | 0 |
| 15 | pair-6-off | off | 0 | √ | 5.967432330 | 128 | 255 | 46.620565077 | 1.992187500 | 42.731946657 | 130 | 128 | 0 |
| 16 | pair-6-on | on | 0 | √ | 5.017738210 | 128 | 255 | 39.201079762 | 1.992187500 | 50.819709867 | 2 | 128 | 128 |
| 17 | pair-7-on | on | 1 | √ | 9.883981120 | 255 | 255 | 38.760710274 | 1.000000000 | 25.799320831 | 2 | 255 | 255 |
| 18 | pair-7-off | off | 1 | √ | 11.889059380 | 255 | 255 | 46.623762273 | 1.000000000 | 21.448290555 | 257 | 255 | 0 |

pair 执行顺序按奇偶交错：pair0/2/4/6 = off→on，pair1/3/5/7 = on→off（pairs.json，8/8 `token_ids_equal`）。
ON 每请求 terminal draft replay=1；OFF/ON 的 partial/seed draft replay 均 0；`other_native_explicit_fences`=0。

## 计数核对

- **initial/terminal draft**：每请求 prefill[0]（initial，partial）与 decode[-1]（terminal）`draft_calls`=1；`initial_and_terminal_submitted_draft_included_in_every_window` 与 `partial_prefill_seed_and_terminal_draft_included_in_window` 全 true。
- **fence**：每请求 `explicit_terminal_device_fences`=1、`terminal_sync_count`=1、`steady_extra_sync_count`=0、terminal 之外 native 显式 fence=0；窗口内 diagnostic D2H/hash 调用=0；route/logprob audit、first-value trace、profiler 全 false。
- **helper**：OFF=全部行（128 轮请求 130 次、255 轮请求 257 次），ON=两个 prefill 行（恒 2 次）；调用=成功返回；helper 历史逐行绑定 `actual_round_at_entry`（req_id+execute_begin_s 与对应行相等）。
- **replay**：target replay=每 decode 轮 1（两臂均 = rounds：128/255）；draft replay ON=rounds、OFF=0；全部 replay 事件 `existing_entry`/`positional_address_match`/`mode=FULL`；ON draft replay 仅在 propose 窗口内。
- **冷 capture**：`actual_entries=1`、descriptor `num_tokens=6`（capture sizes `[6]`，FULL_DECODE_ONLY）、`constructor_use_cuda_graph=true`、`bool_flip=false`；冷 helper 全局 1 调用/1 成功返回（capturing=true）；`parameter_metadata_views=1032` 与冻结预期一致；19 个实际源码 SHA guard 全通过。
- **固定 fixture 封印**：43 owner buffers；before-capture 43 entries + after-capture 43 checks 通过；每请求 before/after 各 43 项，18 请求共 1548 次检查（measured 1376 次）；`fixture_source_sha256`=7dfa1133…（与冻结一致）；`original_parameters_unchanged=true`、`capture_or_warmup_weight_mutation=false`、`source_or_cache_values_mutated=false`；全部封印读取在计时窗口外。
- **几何**：2048/256、两段 prefill [2044,4]（第二段唯一 seed）、decode 每轮 `scheduled=6/verified=5`、进度链连续、`computed_before` 链闭合；18/18 通过。

## 参考 token 逐请求比对

- 参考：`reference_token_ids_2k256.json`，pending=false、source_job=`dspark_lane_20261011_c4_graph_geometry2k256_v3`、source_requests_sha256=`a01abe2be7fb327a3f82e6d6ed66bfba78b3141b0183ff128c4a73f16d3eb1ed`、extracted_unix 1791745879.65（CST 03:11:19）。
- 结果：**18/18 请求（3 seed × 全部记录，含 warmup）token 序列与参考逐位一致**；8/8 pair off/on 相等；period2 复读分类 18/18；dummy clean-rate=0.0。

## 失败项 / 未完成项

- **无失败项**：无 failure/error 行、无非有限、无 traceback；run.exit=0。
- **无阻塞未完成项**。以下为需主线程知悉的包缺陷与本轮操作说明（未改动任何冻结字节）：
- **包缺陷（analyzer 不可原样运行）**：final freeze 的 `runtime_source_files` 含 `reference_token_ids_2k256.json`（9 项），但 `launch_fixture_timing.RUNTIME_FILES` 缺少该键（v2 版本含），`analyze_fixture_timing.py` 第 45 行原样运行即 `KeyError`（日志 `~/tmp/dsv41-tiny-timing2k-monitor/analyzer_direct_run.log`，rc=1）。
- **本轮处置**：以一次性本地驱动 `~/tmp/dsv41-tiny-timing2k-monitor/run_analyzer_with_reference.py` 把 reference 条目补回*运行时*字典后调用同一 analyzer（其自身断言照旧执行，并额外核验 reference 文件 SHA 与 final freeze 一致）；包内文件（含 freeze/launch/analyzer）零改动。建议后续包修复：把 `REFERENCE_NAME` 加回 `RUNTIME_FILES`。

## 证据与 SHA256

BASE = `operator-stack-tp8/experiments/operator_stack/parallel_tiny/dspark_lane/fixture_timing_diagnostic_v3_2k_256/`（除注明外）

| 证据 | 路径 | SHA256 |
| --- | --- | --- |
| 终态分析（本 job 聚合产物） | `evidence/dspark_lane_20261011_c4_fixture_timing_diagnostic_v3_2k_256_analysis.json`（31,846 B） | `1fef32befcddc19e453a6a7ea83170e10e92f07ef87fd2b3b5179aa2fd3bee6d` |
| bounded 取回回执（67 文件） | `evidence/…_fetch.json`（12,163 B） | `2930daf166b0ce341ae251aa9d01ad39369cf0c3fa98ba2ada1a90df7f8062b5` |
| 终态状态快照（第二次只读轮询） | `evidence/…_status.json`（736 B） | `7c1611fb5c82fda47cb53add7ff28e3d10ca735c676cb2a44d58aa1db2952470` |
| 启动回执（launch） | `evidence/…_launch.json`（6,979 B） | `7410c64c816bb5c25f92b287df655a8f39a2e0688e4075845420b0f2bc3a6c4e` |
| final freeze manifest | `evidence/…_freeze.json`（5,914 B） | `83004ebe2a6c696b35b9876d5715436e62392e1e5b4c2fe4af570e74415454d4` |
| staged freeze | `evidence/…_freeze_staged.json` | `5442666d4c1fab6e0b0af2de87690b55197b2c702efb078e37fd785996c765c7` |
| 取回原件主树 | `runtime/dspark_lane_20261011_c4_fixture_timing_diagnostic_v3_2k_256/` | 逐文件 SHA 见 fetch 回执（67/67 通过） |
| 逐请求记录 | `runtime/…/results/requests.json`（89,515 B） | `e5130194701574f7c8965bc0f0ab09a8576c9e8e9e3fbe894962fc0ea32f41bd` |
| 按臂汇总（远端） | `runtime/…/results/summary.json`（1,457 B） | `6692f29525e4e49a1d8f17ceb86d3d15b5550e52a30ff56fbce6ea076485172e` |
| pair 记录 | `runtime/…/results/pairs.json`（3,370 B） | `045b3bf2662983fd46c342abb2bc1a93db6eec4b67273efbdd7ea56001854354` |
| startup（冷 capture/guard） | `runtime/…/results/startup.json`（3,009 B） | `4100b529fe0a89ee4d0caa2a8639335ba3f130c15882a8bbff70bae4c1a071e8` |
| run.exit | `runtime/…/run.exit`（`0\n`，2 B） | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| done.json | `runtime/…/done.json`（48 B） | `a6af3810bf5aa15ce50b810e5fb1cbb2407eaf7d7eed3e6d1a38eecdd6227ad1` |
| run.log | `runtime/…/run.log`（62,208 B，151 行） | `fec892bbdff757b1bd47e972c91a9b1c2b942bd6597adceafe6f58dd4fc33830` |
| 监控快照 1（03:06:02，进行中） | `~/tmp/dsv41-tiny-timing2k-monitor/status-20261012-031358.json` | `29d359d35f1ec7a4…` |
| 监控快照 2（03:16:05，终态） | `~/tmp/dsv41-tiny-timing2k-monitor/status-20261012-032401.json` | `7c1611fb…`（同 status 证据） |

- 取回：67 文件 / 8,227,145 B，逐文件原件 SHA 校验通过；SSH 单响应打包 738,725 B（<1 MB 上限）；单文件最大 547,212 B（`results/raw_014.json`）。本 job 无 raw `.pt`/trace 大件需留远端。
- 远端 job 根：`/home/l00886679/projects/dsv41-tiny-prof-20261009/dspark_lane_20261011_c4_fixture_timing_diagnostic_v3_2k_256`。supervisor PID 2251067（本轮未触碰）。

## 显式声明

- performance/formal admission 全 false：`tiny_candidate_admitted=false`、`formal_candidate_admitted=false`、`formal_tp8_claim=false`、`formal_full_main_mtp_verify_scheduler_cache_20ms_goal_achieved=false`；无 performance receipt。
- dummy 周期复读、`dummy_clean_rate=0.0`；不得以 dummy 接受长度或 A 跨臂声称收益；上述 (ms/step, A, tok/s) 仅描述该 tiny 机制诊断，**不得外推正式 20 ms/step、质量或任何正式准入**。
- 合规：本轮未启动任何新 NPU 作业、未执行 `--launch`、未重启/停止容器或远端进程、未 reset/清 pagecache/删远端文件；未触碰 physical8-15；远端访问全程只读（status 轮询×2 + bounded 取回×1）。

## 监控过程摘要

- 只读 status 轮询 2 次：CST 03:06:02（进行中，0/18）、03:16:05（终态，18/18，exit=0）；job 于 03:13:44 在两次轮询之间自然终态。终态确认后 03:18:10 执行 bounded 取回（738,725 B），随后聚合分析（03:19–03:27）。
- 监控快照留档于 `~/tmp/dsv41-tiny-timing2k-monitor/`（含 SHA 与 analyzer 直跑失败日志/修复驱动）。
