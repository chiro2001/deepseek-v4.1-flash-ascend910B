# CANNBot 性能优化指南回顾 + 本部署 Device Free 归因

> 2026-09-15 23:2x CST｜A3-node1 chips 8-15｜容器 `dsv41-a21-perf`（端口 8020）
> 配置：`static_kernel=1` + `npugraph_ex=1` + `mtpq` + Engram(on, int8, gate CHUNK=0) + local-owner + hash fast + jemalloc

---

## 1. CANNBot 里的性能优化指南（两处，别混）

| 层 | 仓库/路径 | 内容 |
|---|---|---|
| **编排层** | `cannbot` 仓 `plugins/model-infer-optimize/`（`$P/src/cannbot/plugins/model-infer-optimize/`） | 阶段 0–6：模型分析/基线 →（1 并行化）→（2 KV+FA）→（3 融合算子）→（4 量化）→（5 图模式）→（6 总结）；另有 `sota-approach-workflow.md` 探索流与 `references/decision-rules.md`（通过/淘汰判据）、Dashboard/报告模板 |
| **知识层** | `cannbot-skills` 仓。**submodule 之前是空 gitlink，本轮已补**：`$P/src/cannbot/vendor/cannbot-skills` 已 checkout 到 pinned `38728be`（2026-08-19）；另在 `$P/src/cannbot-skills`（新 clone，`6bf582b`）留了一份 master 参考 | `model/model-infer-*`（15 个）+ `graph/torch-npugraph-ex-*`（8 个） |

> 注意：`$P/src` 是指向 `~/backup/dsv41-legacy-20260915/src` 的软链，submodule 实际落在 backup 树里；
> `model-recommend-analysis`（措施表）只在 master clone（`6bf582b`）里有，pinned `38728be` 不含。

### 1.1 与本问题（设备 Free / host bound）直接相关的 6 个技能

| 技能 | 一句话 | 关键规则 |
|---|---|---|
| `model/model-infer-perf-breakdown` | 把 `kernel_details.csv` 按模型结构切 component/cluster，出 **wall_ms / bubble_ms** 中位数 + 异常 layer 的单页 HTML | 五类 insight：`module_bubble` / `operator_jitter` / `theoretical_deviation` / `vector_sequence_candidates` / `data_movement_ops`；cluster 覆盖率 <80% 要红字告警 |
| `model/model-infer-multi-stream` | 多流/重叠分析 | **`overlap_pct` 判真假并行**：真 ≥0.5、假 ≤0.05；每个并行点至少派生 2 种编排；强调查"假并行"根因是 GE 把副流输出的轻量 precompute 拉回主流形成 barrier |
| `graph/torch-npugraph-ex-performance-diagnosis` | **正是我们这条路径**（npugraph_ex/aclgraph 已跑通但慢/Device 利用率低） | FX 图静态审计：reinplace 未命中会留下三类冗余 tensor move —— 输入侧未折叠 `copy_`、`auto_functionalized` 物化出的 `clone`+写回、out-of-place 原样保留；用 `TORCH_COMPILE_DEBUG=1` 出 FX 序列 + `debug.log` 的 `missed opportunities` |
| `model/model-recommend-analysis` | 通用优化措施表（22 条） | 与本问题相关的：措施1 **算子自动融合**（"vector 占比大、算子数多、单算子耗时短 → 调度 bound"）；措施5 **多线程调度**（`TASK_QUEUE_ENABLE=1`，需配合绑核）；措施6 **调度线程绑核**（`CPU_AFFINITY_CONF`，预期 ~10%+）；措施8 AICPU→AICore；措施9 静态图多流 |
| `model/model-infer-profiling` | 采集规范 | `ExperimentalConfig(Level1 + PipeUtilization)`；prefill/decode 分开、decode ≥10 步、首编译放窗口外；产物验收（列数、`run failed`=0） |
| `model/model-infer-profiling` / `graph/torch-npugraph-ex-knowledge` | 指标口径 | `step_trace_time.csv` 里 **不要用 Stage 当迭代耗时**（Stage 含 Free）；用 `Computing` 表示 NPU 活跃，`Free` 表示空闲占比 |

其余（`fusion` / `prefetch` / `kvcache` / `graph-mode` / `superkernel` / `quantization` / `parallel-analysis`）本轮不直接命中：
`superkernel` 仅 GE 图 + A3 Decode，**与 aclgraph 互斥**；`prefetch` 只对 memory-bound 热点。

---

## 2. 本部署实测：decode 段 Device Free = **25.8%**

采集：`PROFILE=1` 起服 → `/start_profile` → 32K 单流（warmup + 192 token）→ `/stop_profile` → `msprof --export=on`。
产物：`logs/prof_a21/dp0_pp0_tp0_*/PROF_*/mindstudio_profiler_output/`；
提取：`logs/prof_a21/extract/op_summary_r0.csv`（178,552 tasks）。

### 2.1 总账（窗口 16.4–19.85 s，94 个 decode step）

| 指标 | 值 |
|---|---|
| busy（所有 task 区间并集） | **27.22 ms/step（74.2%）** |
| **FREE** | **9.49 ms/step（25.8%）** |
| tasks | 1,900 个/step，**平均每个 kernel 只有 14.3 µs** |

### 2.2 Free 的构成（按空档大小分桶）

| 桶 | n | 总时长 | 占 Free | 折算/step |
|---|---|---|---|---|
| <5 µs | 78,182 | 100.8 ms | 11.3% | 1.07 ms |
| 5–20 µs | 7,653 | 80.8 ms | 9.1% | 0.86 ms |
| 20–100 µs | 5,072 | 190.9 ms | 21.4% | 2.03 ms |
| **0.1–0.5 ms** | 2,017 | **367.3 ms** | **41.2%** | **3.91 ms** |
| 0.5–1 ms | 80 | 46.1 ms | 5.2% | 0.49 ms |
| **1–3 ms** | 67 | 98.5 ms | 11.0% | **1.05 ms** |

空档数 93,071 个 / 94 步 ≈ **990 个/step**，平均每个 9.5 µs；按步内相位直方图**平铺**（20 个相位桶各 ~44 ms），
⇒ **不是"每步一个固定尾巴气泡"，而是全步均匀分布的海量微气泡**（调度/依赖 bound）。

### 2.3 两类气泡的归属

**（a）1–3 ms 桶：Engram route。** 67 个空档里 66 个的**前一个算子**是 `hcom_broadcast_`/`Fill`，
后一个算子是 `hcom_alltoallv_`（stream 41）——即设备在等 host 把 Engram 的 all_to_all 下发下来，
合计 1.05 ms/step。把 ≥0.5 ms 桶一起算，Engram 相关空档 = **144.5 ms / 94 步 ≈ 1.54 ms/step**。

**（b）0.1–0.5 ms 桶：stream 47 上的 eager 链。** 2,017 个空档中，**680 个前一个算子是 `ViewCopy`**，
后一个算子 **786 次是 `Cast`（stream 47）**、202 次 `IndexCheck`（stream 47）、89 次 `hcom_allReduce_`（stream 38）。
stream 47 是 Engram 回程解包（a2a 回值 → ViewCopy → Cast → index/scatter）所在的辅助流，
这条链与 host 侧的查表/H2D/事件串起来，形成 0.2 ms 量级的串行气泡。

### 2.4 与目标的关系（重要）

`tok/s = A × 1000 / ms_per_step`：

- 32K 实测 A≈3.0；若 Free 能压到 0，步时 = 27.2 ms → **≈110 tok/s**。
- 128K 实测 A≈2.66–2.9；要 110 tok/s 需步时降到 **24–26 ms**，而 128K 的 busy 还会因 `SparseFlashMla` 增长。

⇒ **"清 Free"就是 G1 的唯一现实路径**；Free 不清零，就只能靠拆算子提效（profile 里 top 算子
`SparseFlashMla` 13.4% / `HcPre` 9.2% / `QuantLightningIndexerV2` 8.6% / `GroupedMatmulSwigluQuantV2` 7.7%）。

---

## 3. 下一步候选（按 cannbot 判据排序）

| # | 动作 | 依据 | 预期 |
|---|---|---|---|
| 1 | 把 Engram route 的 host→device 串行链拆开（提前一 step 算 / 预取 / 让 a2a 不挡在图外） | 实测 1.54 ms/step 空档归属明确 | ≤1.5 ms |
| 2 | 查 npugraph_ex 冗余 tensor move（`copy_`/`clone`/out-of-place），尤其 stream 47 的 `ViewCopy`+`Cast` | `torch-npugraph-ex-performance-diagnosis` | 未测 |
| 3 | `enable_cpu_binding`：当前 **CPU_BIND=0**，cannbot 措施6 建议 `CPU_AFFINITY_CONF` 细粒度绑核 | `model-recommend-analysis` 措施6（~10%+） | 需实测 |
| 4 | 算子融合/减数量：1,900 task/step、均值 14 µs → 调度 bound；`HcPre/HcPost`、gate 内逐元素算子 | 措施1/2 + perf-breakdown 的 `vector_sequence_candidates` | 未测 |
| 5 | 用 `model-infer-perf-breakdown` 走一遍单 step component/cluster/bubble 归因，量化"每层 bubble" | 它的主产物就是 bubble_ms | 归因工具 |

> 已在 CANNBOT_TUNING_NOTES.md（更早一版、基于旧 checkout 行号）里做过的索引仍然有效；
> 本文件用新 clone（`6bf582b`）复核了路径与结论。
