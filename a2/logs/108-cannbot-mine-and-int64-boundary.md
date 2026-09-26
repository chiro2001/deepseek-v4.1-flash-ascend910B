# 108 — cannbot 挖掘结果 + **一条正确性优先的边界风险**（我们距 2³² 只差 1.03 MiB）

> 2026-09-23 02:1x–02:2x CST。执行：子代理 **`CANNBOT_MINE`**（纯只读、未占卡；含 4 个它自己派的子代理）
> + **主代理**（外推与裁决）。产物：`agents/CANNBOT_MINE/REPORT.md`（995 行）+ `out/`（38 MB）。
> 标记：**【实测】/【推断】/【未确认】/【仓】= cannbot 仓库原文**。

---

## 0. 一句话

两条**改变结论**的发现：
① ★★★ **cannbot 上游新 commit `135cb59` 直指我们的边界** —— `blockTablesStride` 类 **stride 必须提升 int64**，
   而**我们距 `2³²` 只差 1.03 MiB**（7938 × 540,928 vs 4,294,967,296）⇒ **正确性风险，优先级高于性能**（已派专项核查）。
② ★★ **`index_select` 的"隐式连续化"由源码证实**（不再是推断）：`aclnn_index_select.cpp:186-187` **无条件** `l0op::Contiguous(self)`
   ⇒ "只有真连续才 5–8 µs"有了机制解释；同时**"直接换个官方 gather 就收工"这条路被否掉**（我们是页粒度 + 页步长 ≠ 载荷）。

---

## 1. ★★★ 正确性优先：我们距 `2³²` 只差 **1.03 MiB**

`prof_int8` 已定案：生产页数 **7938**、`pool_bytes_per_block = 540,928` ⇒

```
7938 x 540,928 = 4,293,886,464 B
2^32           = 4,294,967,296 B
gap            =     1,080,832 B = 1.03 MiB (0.025%)
```

而 cannbot 更新 `6bf582b → 25f645a` 的新 commit **`135cb59`** 正好点名
「`blockTablesStride` 类 stride 变量**必须在第一次乘法前显式提升到 int64/uint64**」，
并附 **910B2 在位置 `4294969344` 起全 0** 的生产案例 —— **我们跑的是 910B（A2），同族。**
★ 命中代价是**静默错数**（不是崩溃）⇒ **上线级缺陷，优先级高于性能**。
⇒ 已派 `PROF_int8` 做**纯读的代码级核查**（列出所有 `pages × 每页字节` 类乘法、算 7938 与 28,577 页下的值、指出哪一行是 int32）。

---

## 2. ★★ `index_select` 的"隐式连续化"：**源码证实**（不再是推断）

| 证据 | 原文 |
|---|---|
| **算子实现**【仓】 | `ops-nn/index/gather_v2/op_api/aclnn_index_select.cpp:**186-187**` —— **无条件** `l0op::Contiguous(self)`；`:40-55` 流程图为 `Contiguous → GatherV2 → ViewCopy` |
| **cannbot 命名**【仓】 | `graph/torch-npugraph-ex-performance-diagnosis/references/case-001:26`：`*_SliceAiCore_Slice` = 「**完成非连续输入的隐式连续化**……不是模型显式表达的切片语义」 |
| **我们的实测**【实测】 | 只有**真连续**掉到 **5–8 µs**；`gap 512` 与 `gap 65,536` 同价（422/421 µs）；66,560 的"紧凑页"仍 443.7 µs |

⇒ 机制闭合：**连续输入时 `Contiguous` 退化为 no-op**，否则**物化整个平面**（我们已实测到 **0.04%** 吻合，见 `logs/106`）。

### 2.1 ★ 但它同时**否掉**了一条捷径（重要）
cannbot 的 `model-infer-kvcache/SKILL.md:139` 说 PA 模式下 KV 由 FA 凭 `block_table` 内部读，
范式里没有 host 侧 gather ⇒ 看起来"换个官方 gather 就收工"。**不成立**，因为：
`dsa_v41.py:338-342` 的 `kv8_page_view` 是 `as_strided((pages, per_page), (page, 1))`，
**行步长 476,416 B > 行宽 65,536 B** ⇒ 我们是 **页粒度 + 页步长 ≠ 载荷**。
⇒ **判据必须改成"目标算子有没有显式 block stride 参数"**，而不是"是不是官方 gather"。

---

## 3. cannbot 的其它可用件（按我们的 P 编号）

| 我们的问题 | cannbot 的答案 | 等级 |
|---|---|---|
| **P1** int8 KV 该怎么摆 | ★ **packed record**：`deepseek_v4_1/models/modules/common_modules.py:65-80` 原文 `# FP8 KV caches use a packed record (nope + rope + scales + padding)`、`align_up(..., 32)`；与 `quantization-structure-cards.md:181`、`npu_kv_quant_sparse_flash_attention:93`、`gather_selection_kv_cache_tiling.cpp:44-45`（`MAX_KV_CACHE_DIM=656`）**四处一致** ⇒ **我们 `(payload, scale)` 两块等宽平面是异类** | 【仓】 |
| **P1 替代路径** | 短期 `npu_gather_pa_kv_cache_functional`（tiling 有 **12 个 stride 字段**，gather 家族里唯一按 stride 零拷贝散写）；长期 `npu_kv_quant_sparse_flash_attention`（`AddConfig("ascend910_93", ...)` 真注册） | 【仓】+【未确认：装机版本能否调起】 |
| **P2** 8 卡 trace 怎么做 | `model-infer-profiling/SKILL.md:13,198,201,224`：`Level1+PipeUtilization` ⇒ **47 列**；**`communication*.json` 仅多卡生成**（= 我们缺的关键产物）；收尾 `prof.step()` 次数规则。★ `:148-152` 坑：**"某些挂载让 msprof 产不出 CSV ⇒ 先换输出盘，不是代码问题"** | 【仓】 |
| **P2 清洗口径** | `perf-breakdown/SKILL.md:126`：**只保留 collective summary 行、丢弃可匹配的 `AivKernel` fragment** ⇒ **我们旧分析把通信 fragment 当算子热点的根因** | 【仓】 |
| **P3** offload 描述符翻倍 | 根因同 P1（scale 摆成第二块平面）+ `datacopy_optimization_design.md:47-53` 阈值表：20–91 KiB/段属"好~极佳"⇒ **瓶颈是描述符/调用次数，不是带宽** | 【仓】 |
| **P4** gate padding | `graph-mode/SKILL.md:90,153,155-160`（`.item()` 禁用、shape 变化→重编译、**"通过参数控制"**）；★ **cannbot 只有约束、没有"动态 pad 被接受"的正面结论** | 【仓】 |
| **P5** prefix 对齐粒度 | **cannbot 完全没覆盖**（1,031 行逐条看过） | — |
| **P6** 自定义 kernel | `scatter/patterns.md:318,334-336`：明令**单 kernel、禁止两次 launch，固定开销 ~15–20 µs/call**；`datacopy_optimization_design.md`（**910B 实测**，比我们平台更近）；`core_shrink_design.md:5,47-52`（小 shape 裁核）；★ `triton-latency-optimizer/references/{device-side-gather,discrete_memory_access}.md` **直接对应我们已有的 `kv8_fuse_triton.py`** | 【仓】 |

### 3.1 可复用工具
`analyze_kernels.py` / `detect_structure.py` / `render.py` / `compare_runs.py`（P2 标准链）；
`ops/ops-profiling/scripts/perf_summary.py`（纯 stdlib、离线跑 CSV）；
★ **`operator_index.tsv`（2,593 行、带 `has_910_93` 判据）** —— 查"我们平台支不支持某算子"应从它开始；
`asc-devkit/docs/api/`（2,720 个 .md，**本地 CANN 文档树**；`ascendc-docs-search` 因 `ASC_DEVKIT_DIR` 为空不可用，但可绕过）。

### 3.2 cannbot **没**覆盖的（避免抱错期望）
`index_select/IndexSelect/aclnnIndexSelect` **零覆盖**；`as_strided` 只有一条目录索引；
无"小数据 UB→GM 窄写"专题；无 launch 开销独立文档；`ascendc-perf-optimize` 核间流水章节**为空**；
prefix cache 对齐粒度**完全没有**；稀疏 FA 契约**明写"占位/未经算子验证"**；
★ **`cannbot-insight` 不是性能工具**（是 coding agent 会话可观测性）；`blaze`/`mc2` 对我们**原文写死不适用**。

---

## 4. ★ 追加：int64 溢出的**核查结果**（`PROF_int8` 纯读审计，649 行报告 §12）

**裁决：在我能看到的全部代码里「会溢出」不成立；但 A2 把"单张量字节/槽页偏移"推进了 `2³¹–2³²` 带，而闭源算子内部看不清。**

### 4.1 各量数值（7,938 页 vs A2 的 28,577 页）
| 量 | A3 | A2 | /2³¹ | /2³² |
|---|---:|---:|---:|---:|
| 单张量字节（页 131,072） | 1.04 GB | **3.75 GB** | 0.484 → **1.744** | 0.872 |
| **槽页 stride 乘积（147,712）** | 1.17 GB | **4.22 GB** | 0.546 → **1.966** | **0.983**（★ 距 2³² 剩 **73.8 MB / 1.7%**） |
| 对照：池总量 | **4,293,886,464** | 15.46 GB | **1.999** | **0.9997** |

### 4.2 ★★ 反向强判据（比正向审计更强）
**A3 自己的池总量 = `1.999 × 2³¹`、`0.9997 × 2³²`** ⇒ **若总量走 signed int32，A3 现在就已经坏**；
而 A3 的 int8 路径已过**逐位等价与精度门** ⇒ **总量必为 64 位**。
⇒ 用"已经发生的事实"证明类型正确，**不依赖对源码的解读**。

### 4.3 逐处类型审计（两处**恰好就是 cannbot 处方要求的写法**）
| 位置 | 类型 | 判定 |
|---|---|---|
| `dsa_v41.py:349-351 kv8_page_view` | Python int → ATen `IntArrayRef`(int64) | ✅ |
| `dsa_v41.py:364` | 索引**显式 `.to(torch.int64)`** | ✅ |
| **`dsa_v41.py:373 kv8_gather_rows`** | `ids = phys**.to(torch.int64)** * per_page + offs**.to(torch.int64)** * per_row` | ✅ **乘法前已提升** |
| `dsa_v41.py:592/894/941/981` | `torch.gather(…**.to(torch.int64)**, …)` | ✅ |
| `dsa_v41.py:608`（`.to(torch.int32)`） | 装的是 **page id（≤28,576）**，不是字节偏移 | ✅ 余 4 个数量级 |
| `kv_offload/cpu/gpu_worker.py:107` | `base_ptr + block_ids**.astype(np.uint64)**[:n] * row_stride` | ✅ **正是处方** |
| `gpu_worker.py:113-116` | `np.arange(bpc, dtype=**np.uint64**) * block_page_size`；`+ block_ids**.astype(np.uint64)**[:,None]*row_stride + sub_offsets` | ✅ 全 uint64 |

### 4.4 ★ 新的真实暴露面
A2 让**单张量**量跨过 signed-int32 上限（**1.744–1.966 × 2³¹**），**距 2³² 只剩 1.7%**。
cannbot 实测失败位置 `4,294,969,344` 比 2³² 高 2,048 B ⇒ **uint32 级路径确实存在** ⇒ **A2 正落在同一条带的边缘**。

### 4.5 三处「看不清」（**不猜**）
`swap_blocks_batch`、`npu_scatter_nd_update_sk`（**均闭源**；调用侧是 int64 指针/大小数组，**算子内部索引算法看不到**）、
ATen `index_select`/`as_strided` 内部 offset 计算（容器内无源码）。
⇒ **关闭路径**：对这三个做**双端哨兵断言**（构造 `pages=28,577` 的 int8 平面 ≈3.75 GB，首/末块各写哨兵、读回比对；
末块若全 0 即命中 cannbot 那个症状），**约 5 分钟**，待 c0 锁。
