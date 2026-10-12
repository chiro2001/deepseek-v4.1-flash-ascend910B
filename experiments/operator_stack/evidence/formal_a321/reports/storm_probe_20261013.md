# storm_probe：pre_target「431 小算子风暴」归因与压缩候选（CPU/离线）

> 2026-10-13｜rank0（physical chip 8）｜数据源：a3-21 容器 `dsv41-real-stack-tp8-20261010-a321`
> 冻结 profile `formal_dspark_native_graph_profile_v1`（graph 臂）与 `formal_dspark_native_profile_v1`（eager 臂）
> 证据 JSON：`../storm_probe_20261013.json`（含 128 个 cluster 全表、step5 全序列、SHA 清单）
> 边界：只读离线分析；未启动任何 NPU 作业、未重启/停止容器、未碰 physical4/缓存/远端文件。**离线估算不是性能报价。**

## 0. 一句话结论

pre_target 窗口（15.4ms）里**设备只忙 2.1ms**，其余约 13ms 是「约 650 次 aclnn 主机派发 + 少量阻塞同步（H2D、`.item()`、allReduce 等）后的 Python 胶水」；
405/431 个算子按来源分四块：**eager slot-mapping（12 次）= 可一键融合（代码已就绪）**、**Engram 查表已入图（87 kernel/次 replay，静态）**、**DSA 稀疏注意力元数据（37 个/步、kernel 合计 ~0.5ms）**、**约 250 个 eager 输入准备/缓冲小算子（含 26 次 `.item()`、13 次阻塞 H2D）**。
与 eager 臂逐键对照**完全一致**——draft 入图改造**没有**覆盖 pre_target；57.58→46.49ms 的收益全部来自 draft 窗口。

## 1. 口径修正：431 vs 405

| 口径 | 定义 | 每步算子数 |
|---|---|---|
| W2（父任务叙述） | `start < 首个 target HcPre(6,4,5120) 开始` | **431** |
| W1（冻结解析器 `analyze_timeline_graph.py`） | `start < target 图模型(id=48)最早 kernel 开始` | **405** |

差出的 26 个是 target 图入口 kernel（model id 48 的 Cast/Sub 等）；本地 `timeline_compact.json` 的 rank0 pre_target span 15899.18µs 与 W1 的 10 步 span 中位数 15899.5µs 吻合 ⇒ 冻结口径为 W1。
窗口结构（每步）：`[0–2ms 稀疏小算子] [3.5–11.5ms stream47 eager 爆发] [11.5–13.5ms stream37 DSA 元数据] [14.0ms allReduce] [14.5–15.45ms stream141 Engram 图 replay] [15.37ms 3×ZerosLike]`。

## 2. 分布（(Type, OP State, Input Shape, Stream) 聚类，128 类）

每步 405 个（W1）；按家族合并（计数/步，耗时合计/10 步）：

| 家族（流） | 计数/步 | 耗时/步 | 说明 |
|---|---|---|---|
| Cast 69（47/141/37） | | 108µs | 多为 `aclnnDivMods_/GeScalar_/FloorDivides_` 的融合内部 Cast |
| Fill 38（47/141） | | 44µs | `fill_`（缓冲填充；含 Engram pad/掩码） |
| SelectV2 33 = `torch.where` | | 65µs | 输入准备/掩码 |
| BroadcastTo 31 | | 48µs | 输入准备 |
| ViewCopy 24（47，4096;1;1;1;6;1;1;1）| | 192µs | 视图拷贝（大缓冲复制） |
| FloorMod 22 / FloorDiv 17 | | 83µs | `%`/`//`（位置/槽位/哈希） |
| ClipByValueV2 20 = clamp | | 42µs | |
| GreaterEqual 18 / Less 8 / Equal 4 | | 66µs | |
| GatherV3 17（47 的 1048576,1,1,64 ×12；141 的 384M 表 ×4）| | 104µs | 1M×64 表与 Engram L1/L14（384006168/384016682 行）查表 |
| `_compute_slot_mapping_kernel` 12（47，static）| | 37µs | **每步 12 次 = KV cache group 数** |
| MaskedFill 5 / ScatterElements 1 / SearchSorted 1 / Cummin 1（141，N/A 态）| | 353µs | Engram running-min/采样类尾部 |
| SparseFlashMlaMetadata 3 + QuantLightningIndexerV2Metadata 2 + SparseAttnSharedkvMetadata 1（37）| 6 | ~421µs | DSA 元数据 |
| hcom_allReduce 1/步（stream 40，另有 1 条 N/A 重复行）| | 中位 22µs、步5/7 210µs、**步11/12 4.3ms** | vocab-parallel 嵌入的 TP allReduce（见 §4.5） |
| 其余 ~50 类（Index/IndexCheck/Range/Slice/Sub/Add/Mul/Stack/Embedding、BitwiseXor、LogicalNot 等）| ~77 | ~150µs | 输入准备胶水 |

「每类中位入队间隔」（gap=本算子 start − 前一枚算子 end，全序列）：slot-mapping 91µs、ViewCopy 16µs、where 18µs、Cast(6) 45µs、GatherV3(1M) 42µs；**10 步逐键一致**，唯一跨步变化是 allReduce 时长（见 §5/§6）。

## 3. 算子→源码映射（证据行号）

| 链 | 源码证据（容器内 `/vllm-workspace/vllm-ascend/vllm_ascend/`）| 数据佐证 |
|---|---|---|
| **slot mapping ×12（static）** | `worker/block_table.py:19-20`（kernel 导入）、`:26-42`（团队注释：12 次=KV cache group 数、每步 ~0.8ms host 串行、`V41_SLOT_MAP_FUSED`）、`:260-303/616/656/691-712`；调用点 `worker/model_runner_v1.py:1488` | 形状 `2;6;1,64`；间隔中位 91µs×12≈1.1ms |
| **Engram 查表图 replay（model id 49，87 kernel）** | `models/deepseek_v41/model.py:1069`（建 `EngramGraphCache`）、`:1180-1201`（零拷贝 run/replay）、`:1356-1410`（`prepare_engram_inputs` pad/copy）；`engram_graph.py:126-140`（searchsorted+hash+查表 node）；`engram_hash.py:436-507/650-711`（xor/%primes/cat、clamp_min、masked_fill、cummin）；`engram_device_index.py:1082-1119`（token_map index_select、masked_fill、`//`、clamp_、where、scatter_ 页镜像）；`engram_hbm.py:558-576`（HostMappedEngramTable.lookup_local） | run.log:309（L1/L14 行数、256B/行）、run.log:876-883（`graphs=2 replays=38 eager_fallbacks=1 ptr_mismatch=0 keys=[(4,1,64),(6,1,64)]`）；config：n_heads=8、max_ngram=4 ⇒ cols=24，6 token×24=144（GatherV3 索引长度） |
| **Engram pad/拷贝（ZerosLike 2048×6144×2 等）** | `model.py:1368-1398`（容量缓冲 zero_/copy_）、`:973-975`（容量=2048） | bneck `pad=0.12ms/step`；窗口内 3×ZerosLike（2048,6144）×2 + mask |
| **DSA 稀疏注意力元数据（37）** | `attention/dsa_v41.py:873`（`npu_sparse_flash_mla_metadata`）、`:911`（`npu_quant_lightning_indexer_v2_metadata`）、`attention/dsa_attn_kv_plan.py:148`（`npu_sparse_attn_sharedkv_metadata`） | 主机 trace：t+12.5–13.5ms 三个 API 各 48–74µs |
| **vocab-parallel 嵌入 + TP allReduce** | `ops/vocab_parallel_embedding.py:229-257`（mask→embedding→`masked_fill_`→`torch.ops.vllm.all_reduce`）；触发点候选 `spec_decode/llm_base_proposer.py:789-794/1026-1031/1512` | 主机 trace：t+14.0ms `vllm::all_reduce` 123µs + `c10d::allreduce_` 62µs；设备 hcom 22µs–4.3ms；随后 `SynchronizeStreamWithTimeout`（常规 0.5ms、11/12 步 4.3ms 与其等长） |
| **其余 ~250 个 eager 小算子（12–24 轮 divmods/where/copy_/fill_ 模式）** | 候选：`worker/model_runner_v1.py` 输入准备路径、`spec_decode/dspark_proposer.py:223-344` 与 `llm_base_proposer.py:2477-2493`（draft 缓冲 fill_）、`engram_device_index.py:1082-1119` | trace 无 python 栈；见 §8 未决问题 |

## 4. 静态 vs 动态判定

* **几何静态**：405 个算子的 (Type, State, Shape, Stream) 在 10 步中**逐键零差异**（计数、时长 ±50µs），无按 seq_len 增长的形状 ⇒ 全部静态几何。
* **值动态**：依赖上一步接受结果的算子：Engram 哈希/查表（input_ids/positions）、slot mapping（positions/block table）、DSA 元数据（seq_len 每步增长）、draft seed 缓冲（`next_token_ids.copy_`）、嵌入（当步 6 个 token id）。
* **地址静态**：Engram 零拷贝图要求 `input_ids/positions/query_start_loc/block_table` 的 `data_ptr` 不变（`engram_graph.py:29-33/160-175`），本 run `ptr_mismatch=0`。
* **主机控制流读设备值**：窗口内 **26 次 `aten::item`/`_local_scalar_dense`**（单次 0.6–5.5µs），是「每步变化」的强判据，也是可去掉的同步点候选。

## 5. eager 臂对照（draft eager）

同一 harness、rank0：**128 个 cluster 键集合与计数、405 个/步全部一致，增删项为零**。唯一差异：

| 项 | graph 臂 | eager 臂 |
|---|---|---|
| hcom_allReduce（pre 窗口，1/步） | 中位 22µs；步5/7 210µs；步11/12 4.3ms（10 步均 ~0.9ms） | 10 步均 ~2.2ms/步 |
| 其余 404 个算子（类型/形状/流/计数/耗时） | 完全一致 | 完全一致 |

⇒ pre_target 风暴与 draft 入图（已带来 57.58→46.49ms 的 11ms 收益）**无关**；要压 pre_target 必须在目标侧输入准备/Engram/元数据侧动手。

## 6. 主机预算与同步时间线（rank0，15 步归一）

* 主机 API：`aclnn*` 合计 **2.84ms/步**（75×copy_、76×fill_、34×where、19×remainder、15×divmods…）；`launch` 1.75ms/步；`aclrtSynchronizeStream` 14.6ms/步（**1 次/步、单次 ~21.9ms**——位于 target 图开始后 ~16.9ms，疑似 harness 计量同步，`formal_dspark_counters.py:359`，未证实前不要当作生产成本）。
* pre 窗口 step5（15.45ms）：设备 kernel 并集 **2.11ms**；窗口内 aclnn API 事件 **653 次/2.64ms**；cpu_op 叶子自时间 3.28ms；⇒ **约 10.6ms 为未追踪主机胶水**。
* 同步事件：`SynchronizeStreamWithTimeout` t+14.9ms（0.45–0.53ms；步11/12 = 4.32/4.33ms，与 allReduce 等长）；`aclrtMemcpy`（同步 D2H）1 次/步 35–59µs；窗口内阻塞 `acl_memcpy_host_to_device` **13 次**。

## 7. 压缩候选（按性价比排序；**均需设备验证，离线估算≠报价**）

| # | 候选 | 预计省 | 风险 | 验证方法 | 配置/源码 | 先设备验证？ |
|---|---|---|---|---|---|---|
| 1 | `V41_SLOT_MAP_FUSED=1`：12 次 slot-mapping 启动 → 1 次（代码已就绪，默认 off） | **0.5–1.1ms/步**（团队实测 host 1.774→1.048ms；本 profile 间隔 91µs×12） | 低（带 verify 模式与显式回落告警） | 先 `verify` 逐元素比对；再同 harness 复测该 cluster 12→1、api launch 计数下降 | 配置为主 | 是 |
| 2 | 去掉/挪走 pre 窗口 26×`.item()` 与 13×阻塞 H2D（持久缓冲化、控制流去同步） | 0.3–1ms/步 | 中（涉控制流重构） | 计数对比（item=0、H2D=0）+ 数值对账 + 分段 profile | 源码 | 是 |
| 3 | 把 target 输入准备链（~250 eager 小算子，其中 12–24 轮 mini-block）折叠成「元数据图 replay」——复用 `EngramGraphCache`（零拷贝+指针校验）与 `device_metadata_executor` 机制 | 3–5ms/步（保守；窗口 ~10.6ms 主机胶水） | 高（数据依赖形态） | 先在 shadow 模式捕获+逐元素对账；再分段 profile 对比 host 时间 | 源码 | **必须先做** |
| 4 | Engram pad 再优化（`PAD-SKIP` 已省 0.13ms；再减 zero 容量/只做一次） | 0.1–0.2ms/步 | 低 | bneck `pad` 计数 + kernel 计数 | 源码/配置 | 是 |
| 5 | 嵌入 allReduce / `inputs_embeds` 预构建并入 draft 图或 Engram 图（消除尾部同步等待） | 0.1–0.5ms/步；尾延迟更多（步11/12 的 4.3ms 等待消失） | 中 | 同步事件与 allReduce 时长对比 | 源码 | 是 |
| 6 | 步11/12 的 4.3ms allReduce 抖动：查 rank 到达偏斜（straggler），用 event 等待替代 `SynchronizeStreamWithTimeout` | 均值 ~0.8ms/步（2/10 步），p90 改善显著 | 低-中 | 8 rank allReduce 起止对齐分析 + 复测 | 源码 | 是 |
| 7 | DSA 元数据（37 个中除 6 个 metadata 外的 31 个小算子）合并/提前并行 | 0.2–0.4ms/步 | 中 | kernel union/计数对比 | 源码 | 是 |

不做的事（本报告范围外）：不启动 NPU 作业、不改设备侧算子、不针对 target 图（另一线）。

## 8. 未决问题（需下一轮更小代价的取证）

1. ~250 个 eager mini-block 与 26 次 `.item()` 的**精确 Python 调用方**：trace_view.json 无 python 栈；建议下一次 profile 开 `with_stack` 或稳态 py-spy 采样（**不改设备**）。
2. `aclrtSynchronizeStream` 21.9ms/次为 harness 计量还是引擎成本——需对照 `formal_dspark_counters.py` 的同步点。
3. `ViewCopy(4096;…)` 的缓冲身份（4096 = Engram pages boot 容量，未证实）。

## 9. 复核入口

* JSON：`../storm_probe_20261013.json`（cluster 全表、step5 序列、SHA、主机预算、模型 id 映射）
* 本地分析脚本与中间件：`~/tmp/dsv41-storm-probe/`（`pretarget_cluster.py`、`pretarget_perstep.py`、`eager_cluster.py` 等，均可对冻结 CSV 复跑）
* 原始 SHA：kernel_details `a25ceb5b…`、task_time `7e84ff68…`（graph 臂 rank0）；eager 臂见 JSON `meta.sources`
