# ★ 步的 28% 在"小算子海"尾巴：~1000 个算子 × ~4 µs —— 剩余最大的结构性杠杆（2026-10-05）

> 数据：`k6full_1004_100156` PROF_000003（rank0，TP8 + `SP_TOKENS=5` + `DRAFT_GRAPH=1`，
> **`ENGRAM_DEVICE_INDEX=0`**，batch 6，626 步 / 17.69 s）。
> 切步用**锚点法**（主模型 stream 上 `GroupedMatmulSwigluQuantV2` 每步 40 个），
> 并对 16 步取平均 —— 比手填时间窗可靠（手填窗口会跨步）。
> 工具：`tools/excl_steps.py`、`tools/idle_who.py`、`tools/stream_timeline.py`、
> `tools/op_chain_cluster.py`、`tools/prof_report.sh`。

## 0. 一句话

一个 decode 步（**27.41 ms**）= **[主模型图 ~19.5 ms] + [eager 尾巴 ~7.9 ms]**，
尾巴里塞了 **~1000 个 1–10 µs 的小算子**，它们**背靠背、几乎无空隙**，
按实测的 **~4 µs/算子**地板算，光"发射"就占 **~4 ms/步（15%）**。
⇒ 继续优化**只能靠"少发算子"（融合 / 消除）**，没有任何"填气泡"的空间。

## 1. 三段式：主图 / 尾巴 / 真空闲

| 量 | 值 | 占比 |
|---|---:|---:|
| 步长（16 步平均） | **27.41 ms** | 100% |
| 全部设备任务并集 | 24.545 ms | 89.6% |
| 真空闲（≥20 µs 的全系统无任务段） | **1.470 ms** | 5.36% |
| 任务数 | 2734 /步 | — |

尾巴的时间位置（把 27.1 ms 的步切成 4 段看各 stream 的工作量）：

| 窗口 | 谁在工作（busy 之和） |
|---|---|
| 0 – 19.5 ms | **主模型图 s158 为主** |
| **19.5 – 21.0** | **s47 0.87 ms**（采样/输入准备），s158 只剩 0.23 ms |
| **21.0 – 23.0** | **s154 1.59 ms**（DSpark draft） |
| **23.0 – 25.0** | **s35 0.89**（AICPU metadata）+ s47 0.75 + s154 0.46 |
| **25.0 – 27.2** | s35 0.60 + s158 0.29 + s47 0.11 |

⇒ **19.5 ms 之后主图几乎没有活**（合计 0.52 ms）：这 7.6 ms 是**纯 eager 尾巴**。

## 2. 各 stream 的独占贡献（多步平均）

| stream | 身份 | 任务数 | busy | **独占** | 占比 |
|---:|---|---:|---:|---:|---:|
| 158 | 主模型 40 层（图） | 1487 | 16.49 ms | **14.76 ms** | **53.8%** |
| 154 | DSpark draft（3 层 + LM head） | 250 | 2.06 | **1.89** | 6.9% |
| 47 | 采样 / 输入准备 / engram gather | 424 | 1.73 | **1.66** | 6.1% |
| 35 | AICPU metadata ×7 | 37 | 1.22 | **1.09** | 4.0% |
| 156 | KV·wkv 路径 | 160 | 0.94 | 0.34 | 1.2% |
| 155 | 共享专家 | 160 | 1.28 | 0.18 | 0.6% |
| 151/152 | 小副流 | 12/12 | 0.15/0.09 | 0.03/0.04 | 0.1% |
| 153/157/38/41 | 通信/伴随流 | — | 2.9 | **0.00** | 0.0% |

**读法（重要）**："独占"= `union(全部) − union(去掉该 stream)`，
即"把它整条删掉，并集最多能缩多少"。它是**并集上的上界**，
不等于墙钟收益（墙钟收益还要求那段在关键路径上）。
本表的用法是**排序**：s158 ≫ s154 ≈ s47 > s35。

## 3. 尾巴里到底是什么（按每步次数聚类）

### 3.1 stream 47（424 个/步）

| 每步 | ms/步 | 算子 | 形状 | 归属 |
|---:|---:|---|---|---|
| 1.00 | 0.200 | `SparseAttnSharedkvMetadata` | `"2;;;;1"` | AICPU，SWA 切分 |
| 24.00 | 0.193 | `ViewCopy` | `"16384;1;1;1;6;1;1;1"` | **一串 24 次小拷贝，挤在 0.43 ms 内**（每步 1.6%） |
| 1.00 | 0.143 | `MatMulV2` | `"6,5120;16160,5120"` | LM head |
| 2.00 | 0.063 | `ZerosLike` | `"8192,6144"` | padding 底噪 |
| 46.00 | 0.063 | `Fill` | `"1;"` | `aclnnInplaceFillScalar`，**每步 46 次** |
| 53.00 | 0.062 | `Cast` | `"6"` | 其中 28 次 `aclnnDivMods_Cast`、16 次 `aclnnGeScalar_Cast`（dtype 提升） |
| 18.00 | 0.059 | `GatherV3` | `"1048576,1,1,64;6;1"` | RoPE 表行查询（`rope_dsv4.py`） |
| 25.00 | 0.048 | `SelectV2` | `"6;6;"` | `aclnnSWhere` |
| 12.00 | 0.039 | `_compute_slot_mapping_kernel` | `"2;6;32,8192"` | 已有的融合先例 |
| 16.00 | 0.036 | `FloorMod` | `"6;6"` | 位置/槽位链 |
| 16.00 | 0.035 | `GreaterEqual` | `"6;"` | 同上 |
| 13.00 | 0.031 | `ClipByValueV2` | `"6;;"` | 同上 |
| 15.00 | 0.026 | `FloorDiv` | `"6;"` | 同上 |
| 19.00 | — | `IndexCheck` | `"1;6"` 等 | 高级索引的边界检查（与 `Index` 成对） |

### 3.2 stream 35（37 个/步，全是 AICPU metadata）

| 每步 | ms/步 | 算子 |
|---:|---:|---|
| 3 | 0.213 + 0.204 + 0.193 | `SparseFlashMlaMetadata`（长 KV，按 ratio 各一次） |
| 2 | 0.188 + 0.157 | `QuantLightningIndexerV2Metadata` |
| 1 | 0.172 | `SparseAttnSharedkvMetadata` |

⇒ **6–7 次 AICPU 调用 = 1.13–1.33 ms/步（4–5%）**，单次 **157–213 µs**。

## 4. 为什么"发射"这么贵：~4 µs/算子

把 23.4–24.7 ms 这段（s47/35/41/38）摊开看：**333 个任务 / 1.3 ms**，
相邻任务的间隙几乎全是 `+0.000 ~ +0.011 ms`。典型条目：

```
23.420 IndexCheck 3.0 µs → 23.423 Index 9.3 µs → 23.432 LinearIndex 2.3 µs
→ 23.436 ScatterElementsV2 2.7 µs → 23.447 Cast 1.6 µs → 23.449 IndexCheck …
```

* 每个算子的**设备时间只有 1.4–9 µs**，但**启动间隔 ≈ 其自身时长**；
* 也就是说：**连续小算子之间几乎不能重叠**（同一 stream 的依赖链 + 4 µs 级调度地板）；
* 与之对照：`metadata` 的**宿主下发**（`GetWorkspaceSize` 209–258 µs/次）已被证伪为瓶颈
  （`META-HOST-VERDICT`：注入到 1000 µs/次、7 ms/步仍 0.0% 影响，隐藏预算 1–5 ms/次）。

⇒ **不是"host 太慢"，是"设备必须一个接一个地把 1000 个小算子跑完"。**

## 5. 可做的融合清单（按"每步次数 × 结构"排序）

### 5.0 ★ 另一条独立证据：Python 侧每步 42 次标量取值 + 36 次 H2D

从 `FRAMEWORK/torch.op_range`（Python 侧 aten 算子时间线，按时间顺序抽取）统计同一份 profile：

| Python 侧算子 | 全 profile | **每步** | 含义 |
|---|---:|---:|---|
| `aten::as_strided` / `aten::slice` | 284k / 232k | 454 / 371 | 视图（几乎免费） |
| `empty_tensor` / `aten::empty` | 206k / 65k | 329 / 103 | **每次申请临时张量** |
| `aten::copy_` | 92.8k | 148 | 拷贝 |
| `aclnnInplaceCopy` | 50.1k | 80 | → 上一节 F3 的 24× ViewCopy 即其中一部分 |
| `aten::fill_` / `aclnnInplaceFillScalar` | 54.0k / 47.7k | 86 / 76 | 常量填充 |
| **`aten::item` / `aten::_local_scalar_dense`** | **26.4k / 26.4k** | **42.2** | 标量取值；**只有落在 device 张量上的那些才会抽干流水线**（见下方口径修正） |
| `aten::to` / `aten::_to_copy` | 61.7k / 24.8k | 99 / 40 | dtype 提升 |
| `aten::where` / `aclnnSWhere` | 39.6k / 22.6k | 63 / 36 | mask 选择 |
| `aten::scalar_tensor` | 19.5k | 31 | 标量转张量 |
| **`acl_memcpy_host_to_device`** | **22.6k** | **36.1** | **每步 36 次 H2D** |
| `aten::remainder` / `aten::div` | 11.9k / 9.4k | 19 / 15 | `%` / `//`（位置与槽位） |
| `aten::index_select` | 15.1k | 24 | RoPE 快路径 + 其它 |

把 `aten::item` 的**上下文窗口**（op_range 里按顺序取前 14 / 后 8 个算子）摊开看，三种典型形态：

```
… aten::sub | detach | to | as_strided | _local_scalar_dense | item | aclnnSubs | sub …
… fill_ | aten::max | _local_scalar_dense | item | as_strided | slice | to …
… copy_ | _to_copy | to | fill_ | sum | sum | _local_scalar_dense | item | empty | as_strided_ | nonzero | select …
```

⇒ 形态都是**"求标量再拿去做 Python 控制流"**（`x.max().item()` / `mask.sum().item()` / `(a-b).item()`）。

**口径修正（避免过度断言）**：`.item()` 落在 **CPU 张量**上是廉价的（不碰设备），
只有落在 **device 张量**上才会同步抽干流水线。同一份 profile 的 `api_statistic` 里
"同步类"调用只有 **`aclrtSynchronizeStream` 6/步 + `aclrtSynchronizeEvent` 5/步 ≈ 11/步**，
而 `.item()` 是 42/步 ⇒ **其中大多数是 CPU 标量读取**（例如 `query_start_loc_cpu`、
`packed_tensor`（显式 `device="cpu"`）这类）。
⇒ **"42 次设备同步"是错的**；正确说法是"**42 次标量取值，其中 ≤11 次会抽干设备**"。
（H2D 那 36 次/步是另一回事，见下。）

> ⚠️ 口径提醒：本 profile 是 `ENGRAM_DEVICE_INDEX=0`，host engram 路径本身会做 D2H/CPU 查表，
> **会抬高 `.item()`/`copy_` 的计数**；交付口径（`=1`）应重采一张同表再对比，
> 上表的用途是**给出量级与形态**，不是交付口径的绝对值。

| # | 目标链 | 每步算子 | 现耗时 | 融合后 | 预估省 |
|---|---|---:|---:|---|---:|
| **F1** | **spec-decode 后处理链**（`IndexCheck+Index` → `NotEqual/Less/LogicalAnd/ReduceSum/Sub/Clip/GatherElements/SelectV2` → `IndexFill`） | ~120 | ~0.6 ms | 1 个 kernel（输入：logits + 已接受 token + mask） | **0.3–0.5 ms** |
| **F2** | **位置/槽位链**（`FloorDiv+FloorMod+ClipByValueV2+GreaterEqual+SelectV2+BroadcastTo+Range`，形状 `6`/`6;6`） | ~110 | ~0.25 ms | 1 个 kernel（`build_swa_indices` 全链下沉，仓库里已有 `_compute_slot_mapping_kernel` 先例） | **0.15–0.25 ms** |
| **F3** | **24× `ViewCopy(16384)`**（聚在 0.43 ms 内的同形状拷贝） | 24 | 0.193 ms | 1 次批量拷贝（若是连续切片） | **0.15 ms** |
| **F4** | **`Fill "1;"` ×46 + `Cast "6"` ×53**（dtype 提升 + 常量 fill） | ~99 | 0.13 ms | 常量张量预建 + dtype 对齐（纯 Python 侧） | **0.08–0.12 ms** |
| **F5** | **RoPE 表查询**（`GatherV3` ×18 + `IndexCheck` ×23 + `BroadcastTo` ×22） | ~63 | ~0.15 ms | 每步只查一次、按 (config,group) 复用 | **0.05–0.1 ms** |
| **F6** | **AICPU metadata 6–7 次** | 7 | **1.13–1.33 ms** | 见下 | **0.3–1.0 ms** |

### 5.1 F6（metadata）的三条路，按风险从低到高

1. **跨步复用**：序列长度远大于窗口（长上下文稳态）时，SWA 的核间切分**结构不变**，
   只有起点平移 ⇒ `SparseAttnSharedkvMetadata`（0.17–0.20 ms/步）可能可缓存。
   *前提*：先用夹具证明"输入只差一个平移量时，输出切分逐位相同"。
2. **提前提交**（时间线显示它在 draft **之后**才跑）：它只依赖采样后的 `seq_lens/block_table`，
   理论上可与 2 ms 的 draft 重叠 ⇒ 最多隐藏 ~1.2 ms。
   *难点*：`DeviceMetadataExecutor.submit` 出现在**下一步**的 forward context 建立时。
3. **host 参考实现 + H2D**（`GetWorkspaceSize` 209–258 µs，迁移后靠 META-HOST-VERDICT 的
   1–5 ms/次隐藏预算；device 侧 1.2 ms 的 AICPU 时间直接消失）。
   *代价*：要在 Python 里复刻 `csrc/attention/sparse_flash_mla_metadata/op_kernel_aicpu/*.cpp`
   的切分逻辑（1638 行），产出需逐位一致。

## 6. 已排除，不要重走（连同本轮复核）

| 靶点 | 结论 | 出处 |
|---|---|---|
| 动态 K 表（N≥2 关推测） | **静态 K=5 全面更优**（N≥2 时 A 从 2.7 掉到 1.0） | `REMAINING-TARGETS-20261004` |
| K=7 / K=3 | 第 6/7 位几乎不被接受；每步成本 ≈ 固定 + 每流 ⇒ 加 K 更慢 | 同上 |
| metadata **host** 路径 | 隐藏预算 1–5 ms/次，实际 0.21–0.26 ms/次 ⇒ **不在关键路径** | `META-HOST-VERDICT` |
| HcPre A1（自适应 K_L0） | 服务内**从未执行**（static kernel 缓存未失效）；澄清后效应也低于噪声底 | `KERNEL-CACHE-STALE` |
| HcPre+RMSNorm 融合 | M=6 实测净亏 672 µs/步 | `LEVERS-R5` |
| 内置算子 tiling 覆盖（MoE 路由候选 A） | 本 CANN **不允许** vendor 覆盖内置算子 tiling（5 条替代解释已排除） | `MOE-ROUTING-PATH-Y` |
| wo_a 量化 / FUSED_MC2 / MC2 | 精度不通过 / 全面略差 | `LEVERS-R5` |
| "host 太慢所以 step 慢" | acl 33.9 ms/步里 **65% 是同步阻塞**；且 `[bneck] hp` 是**步周期**不是 host 开销 | `DECODE-EXCLUSIVE-HOST-20261005` §4.1/§4.2 |

## 7. 复现

```bash
BASE=$HOME/cedpd-repo/results/k6full_1004_100156/prof/dp0_pp0_tp0_dcp0_ep0_rank0_1435_20261004042337710_ascend_pt/PROF_000003_20261004042337723_00001435KBJORREQ/mindstudio_profiler_output
bash tools/prof_report.sh $BASE 158 16          # 独占贡献 + 空闲归因 + 链聚类
python3 tools/stream_timeline.py $BASE 47,35,41,38 40 0 40 23.4 24.7   # 尾巴现场
```
