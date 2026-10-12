# 正式 API 窗口 32.97→28.09 ms/step：桥状态写尾部伪影修复（−14.8%）

2026-10-13 · a3-21 physical8–15 · 正式40层/W4A8_DYNAMIC/Engram int8/strict · DSpark 真实 ON（graph+async）
· 目标 ≤20ms/step **未达成**（当前 28.0921）

## 1. 发现：每请求 133–725ms 的窗口内尾部伪影

对既有正式窗口（v4 async、v5 async、api_v1 sync）逐请求做窗口分解
（`decode_start → 首轮 execute_start` = pre；`Σ 逐轮 span` = rounds；`末轮 sample_end → client_end` = post）：

| 窗口 | pre 合计 | rounds 逐轮中位 | post 合计（8 请求） |
|---|---:|---:|---:|
| v4 async（32.97） | 238ms | 27.1–28.1ms | **3533ms** |
| v5 async（32.89） | 254ms | 26.9–28.0ms | **3514ms** |
| api_v1 sync（46.49） | 134ms | 37.1–39.1ms | **3602ms** |

post 逐请求 133、182、336、391、458、645、664、725ms（两臂几乎相同、随请求序号递增）——
定位到桥的状态机：`CounterBridge._write()` 每个事件把**全部历史窗口（含 8 rank 全量 receipts）**
以 `indent=2` 同步重写 `bridge_windows.json`，运行结束时该文件 **33.2MB**；随着历史增长，
每次同步序列化阻塞 ASGI 事件循环数百毫秒，表现为"响应就绪 → http_output_send_begin"之间的空洞
（v5 窗口 5：+21.7ms 响应就绪 → +643.9ms 才开始发送）以及随后客户端 finish RPC 同样被推迟。

**结论：post 是诊断 harness 自身的证据写开销，不是模型/引擎/解码工作**；它在两臂等量存在，
不改变 async 相对收益，但把 ms/step 抬高约 5.1ms。

## 2. 修复（新快照 `src_dspark_native_graph_api_async_v2`）

仅改 `_write()` 的序列化视图：状态文件里每个窗口剥离 `rank_receipts`
（全量 receipts 仍原样经 HTTP finish 返回并由客户端落盘）；stage 机、校验、事件、收据字节全部不变。
CPU 自检：8 窗口历史 + 全量 receipts（内存）时 `_write` = **0.17ms / 1716 字节**（原 33MB 级）。
基线 `api_async_v1` 快照及其全部历史证据保持字节不变。

## 3. v6 正式窗口结果（同负载 8 串行 × 2K/256、审计/profiler 关）

job `formal_dspark_native_graph_api_v6_async_states`（快照 api_async_v2、async 三 env、`--draft-graph`），
客户端 exit0、服务正常关闭（run.exit=0）：

```
(28.092118429394812 ms/step, A=2.9394812680115274, 104.63722326244132 tok/s)
694 actual rounds / 2040 decoded tokens / decode_elapsed_s_sum=19.49593019
clean 7/8、复读 1（与基线同口径）；窗口=pre 386ms + rounds 18849ms + post 61ms
```

对照：

| 版本 | ms/step | A | tok/s | 说明 |
|---|---:|---:|---:|---|
| v4 async（含伪影） | 32.9705 | 2.9395 | 89.155 | 旧最优 |
| **v6 async（修复后）** | **28.0921** | 2.9395 | **104.637** | −14.8% |
| api_v1 sync（历史基线） | 46.4874 | 2.9651 | 63.783 | 参考 |

**等价性**：v6 的 8/8 请求 token 流与冻结 sync 基线窗口逐位一致
（`client/token_identity_vs_sync_baseline.json`，逐请求 sha256 相同）。

## 4. 剩余差距的精确位置（设备侧）

逐轮 span 中位 27.2–27.4ms，且 post 已压到 0–30ms/请求 ⇒ **轮时 ≈ 设备忙时**
（既有 profile：整步全流 union 27.29ms、目标图 21.59ms）。⇒ 到 ≤20ms/step 需把每轮设备工作
再砍约 **7.2ms（−26%）**：目标图 21.6→约14.5、draft/verify/engram/eager 约5.7→约5，
并保持宿主已隐藏的现状。已排除方向：MC2 非融合（+3.1ms/目标图）、slot-map 融合（客户端噪声级）。

## 5. 下一步（进行中）

- 候选 A（已启动）：`V41_MLS_SHARED_EXPERT=1`（vendor `multistream_overlap_shared_expert`，把共享专家
  gate_up 投影搬到辅助流与路由专家重叠；模型确有 1 个共享专家）。job
  `formal_dspark_native_graph_api_v7_mls_shared`（快照 api_mls_v1，env 门控，默认关闭时
  additional_config 逐字节不变）。
- 候选 B（准备中）：`V41_MLS_DSA_OVERLAP=1`（vendor `multistream_dsv4_dsa_overlap`，冻结合同此前显式
  置 False；其 MLA prolog/compressor 在辅助流上与主计算重叠）。
- 每个候选以 API 窗口做筛选（token 必须与冻结基线逐位一致），胜出者再跑 bench async 数值门
  （token/logprob vs sync256 参考）与同实例配对，最后提交正式窗口与质量门。

## 6. 候选 A 结果：shared-expert 多流重叠 **有效（−1.14ms）**

job `formal_dspark_native_graph_api_v7_mls_shared`（快照 `api_mls_v1` = api_async_v2 + env 门控
`V41_MLS_SHARED_EXPERT=1`；其余 env 与 v6 相同）exit0、服务正常关闭：

| 版本 | ms/step | A | tok/s | rounds | 逐轮 span 中位 | clean |
|---|---:|---:|---:|---:|---:|---:|
| v6 async | 28.0921 | 2.9395 | 104.637 | 694 | 27.2–27.4ms | 7/8 |
| **v7 +shared重叠** | **26.9477** | 2.9395 | **109.081** | 694 | **26.3–26.5ms** | 7/8 |

- **Δ −1.1445 ms/step（−4.07%）**，A 完全不变（694 轮、2040 token 相同）；
  token 8/8 与冻结 sync 基线逐位一致（`client/token_identity_vs_sync_baseline.json`）。
- 与预期机制一致：共享专家（n_shared_experts=1）的 gate_up 投影移到辅助流，与路由专家
  dispatch/combine 重叠，节省的正是其在串行链上的暴露时间。
- 证据：`evidence/formal_a321/dspark_native_graph_api_v7_mls_shared/`。

候选 B（`V41_MLS_DSA_OVERLAP=1`）已启动：`formal_dspark_native_graph_api_v8_mls_dsa`（port 18623）。

## 7. 候选 B 结果：DSA 多流重叠 **也有效（−0.77ms）**

job `formal_dspark_native_graph_api_v8_mls_dsa`（同快照 + `V41_MLS_DSA_OVERLAP=1`）exit0、服务正常关闭：

| 版本 | ms/step | A | tok/s | rounds | clean |
|---|---:|---:|---:|---:|---:|
| v6 async（基准） | 28.0921 | 2.9395 | 104.637 | 694 | 7/8 |
| v7 +shared重叠 | 26.9477 | 2.9395 | 109.081 | 694 | 7/8 |
| **v8 +DSA重叠** | **27.3195** | 2.9395 | 107.596 | 694 | 7/8 |

- v8 Δ = **−0.7726 ms/step（−2.75%）**；token 8/8 与冻结 sync 基线逐位一致；A/轮数/token 数均不变。
- 两个开关机制不同、路径不同（共享专家 gate_up 辅助流 vs MLA prolog/compressor 辅助流），
  组合臂 v9（`V41_MLS_SHARED_EXPERT=1` + `V41_MLS_DSA_OVERLAP=1`）已启动
  （`formal_dspark_native_graph_api_v9_mls_both`，port 18633），预期再加性收益。
- 证据：`evidence/formal_a321/dspark_native_graph_api_v8_mls_dsa/`。

## 8. 当前正式最优序列

| 版本 | ms/step | A | tok/s | 相对上一步 |
|---|---:|---:|---:|---:|
| api_v1 sync（历史） | 46.4874 | 2.9651 | 63.783 | — |
| v4 async（含桥伪影） | 32.9705 | 2.9395 | 89.155 | −29.1% |
| v6 async（桥修复） | 28.0921 | 2.9395 | 104.637 | −14.8% |
| v7 +shared重叠 | 26.9477 | 2.9395 | 109.081 | −4.1% |
| v8 +DSA重叠 | 27.3195 | 2.9395 | 107.596 | −2.8% |
| **v9 两者全开（当前最优）** | **26.6134** | 2.9395 | **110.451** | — |

目标 ≤20ms/step 仍需**再砍约 7.0ms**，全部在设备侧（逐轮 span≈设备 union）。

## 9. 组合臂 v9（当前最优）与数值门

job `formal_dspark_native_graph_api_v9_mls_both`（`V41_MLS_SHARED_EXPERT=1` + `V41_MLS_DSA_OVERLAP=1`）
exit0、服务正常关闭：

```
(26.613435590778096 ms/step, A=2.9394812680115274, 110.45102606106579 tok/s)
694 rounds / 2040 tokens / decode_elapsed_s_sum=18.4697243；clean 7/8、复读1；token 8/8 与冻结 sync 基线逐位一致
```

- 相对 v6 −1.4787ms（−5.26%）；两个开关的贡献近似可加（−1.144 与 −0.773 → 实测 −1.479）。
- A、轮数、token 数完全不变 ⇒ 只是把重叠隐藏的设备时间从串行链上拿掉。
- 证据：`evidence/formal_a321/dspark_native_graph_api_v9_mls_both/`。
- 数值门（bench async vs sync256 参考，token 逐位 + Top5 logprob delta<1e-3）：
  job `formal_dspark_native_mls_gate_v1` 已启动（快照 `src_dspark_native_mls_v1`，freeze 18/18 自检 0 mismatch）。

## 10. 本轮累计

32.97 → **26.61 ms/step**（−19.3%）、tok/s 89.2 → 110.5（+23.9%），全部建立在
token 逐位一致（8/8）之上；A/复读/clean 口径不变。距 ≤20ms 仍差 6.6ms（−24.8%），
且差距全部在设备侧（目标图 21.6ms + draft/verify/pre_target 设备 ≈5.7ms）。

## 11. 数值门（v9 配置）通过

job `formal_dspark_native_mls_gate_v2`（bench async，`V41_MLS_SHARED_EXPERT=1` + `V41_MLS_DSA_OVERLAP=1`，
`--sync-reference formal_dspark_native_graph_sync256_v1`）：exit0、
`async_reference_passed=True`、`passed=True`；**6/6 请求 token 逐位一致、max logprob delta 0.0**
（Top-5 logprob 门 <1e-3）；同 job 三元组 `(42.13495041666667, A=5.795454545454546, 137.545066225167)`。
证据：`evidence/formal_a321/dspark_native_mls_gate_v2/`。

备注（透明记录）：gate v1 因设备显存未完全回落（前一窗口 workers 尚在释放）在权重加载阶段 OOM，
job `formal_dspark_native_mls_gate_v1` exit=1；清理自有残留 worker、确认 physical8–15 无进程且
HBM 归零后重发 v2 通过。失败原文保留在远端该 job 目录。

## 12. 下一步（进行中）

- `formal_dspark_native_mls_aa_v1`：mls 配置的 **A/A 审计**（audit 模式，route/token/Top5/logprob
  完整门 + 同臂 A/A），既是 goal 要求的审计证据，也用于解锁下一步设备时间线剖析
  （profile 启动器要求同源 AA 收据；新快照 `src_dspark_native_mls_profile_v1` 已建，freeze 23/23 自检 0 mismatch）。
- 之后：对 v9 配置做设备时间线（比较 pre_target/target/draft/verify/engram 与 15899/21591/… 基线），
  定位剩余 ~6.6ms 的下一个可执行切点。

## 13. A/A 审计（v9 配置）通过

job `formal_dspark_native_mls_aa_v1`（audit 模式、`V41_MLS_SHARED_EXPERT=1`+`V41_MLS_DSA_OVERLAP=1`，
draft-graph、45-token 逻辑、3 对 A/A）exit0、`native_aa_passed=True`：

| 对 | tokens_equal | routes_equal（2094×40×6） | route_set_differing | prefill/decode differing | max logprob delta |
|---|---|---|---|---|---|
| pair-0/1/2 | true | true | 0 | 0 / 0 | **0.0** |

⇒ 多流重叠配置通过 route/token/Top5/logprob 全套审计门；配合数值门（gate_v2 6/6 与冻结参考逐位一致）
与正式窗口 token 8/8 一致，v9 配置的数值等价链完整。证据
`evidence/formal_a321/dspark_native_mls_aa_v1/`。

设备时间线剖析 `formal_dspark_native_mls_profile_v2`（同配置、profiler 开启、15 步）已启动，
用于对比基线 profile（pre_target 15899us / target union 21591us / draft 2.6ms / engram 0.37ms）
并定位剩余 ~6.6ms 的下一批切点。
