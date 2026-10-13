# ★ decode 里"并行"到底是什么在并行、掩盖了哪些延迟

> 起因：用户追问 —— 「并行计算到底是什么在并行，具体掩盖了什么延迟？
> 能否用 AIC/AIV 交替并行，把下一个 batch 或 draft 的带宽/计算掩盖掉？」
> 数据：`results/armF_r6_base/prof/...rank0.../kernel_details.csv`
> （交付口径、纯 decode、`conc=1`、72 步稳态、profile 步长 40.02 ms / 服务实测 24.59 ms）。
> 全部为【实测】。工具：`tools/prof_{overlap_who,overlap_where,stream_ident,draft_locate}.py`。

## 0. 一句话

**看着 96% 忙，其实最稀缺的资源（AIC）只用了 55%。**
`Free` 4.9% 是"整机有没有活"的意思，误导性极强；真正的问题是
**AIC 闲着的 18 ms 里，11.25 ms 在等 AIV、4.2 ms 在等通信、2.1 ms 在等 AICPU —— 三者都没和 AIC 重叠。**
而"AIC/AIV 交替并行"在同一 token 批内**做不了**：主流上 272 对 AIC↔AIV 是**直接依赖**交替的。

---

## 1. 资源账：把"忙 96%"拆开

| 资源 | busytime/步 | 占步长 |
|---|---:|---:|
| **AIC**（24 cube/die） | **22.05 ms** | **55%** |
| **AIV**（48 vector/die） | **14.75 ms** | **37%** |
| 通信（union） | 4.22 ms | 11% |
| AICPU | 2.07 ms | 5% |
| 全资源并集 | 39.05 ms | **97.6%** |

**关键恒等式（实测闭合）**：

```
AIC 闲 = 40.02 − 22.05 = 17.97 ms
        = AIV 未重叠 11.25
        + 通信(∩AIC = 0.000) 4.22
        + AICPU 2.07
        = 17.54 ms     ← 差 0.43 ms，在噪声内 ✓
```

⇒ **步长的 45% 是"AIC 在等别人"**，不是"机器在干活"。

> 这解释了那个悖论：**图下发 + `Free` 低 ≠ 有用**。
> `Free` 低是因为 AIV/通信/AICPU 一直在跑，而它们**都不是**这一档的稀缺资源。

---

## 2. "并行"发生在哪里：主流空着，侧流在跑

### 2.1 主流只忙 51%

主计算流（stream 109）：**1167 算子/步，busy 20.36 ms/步（51%）**，其余 49% 在等。

### 2.2 主流空闲窗口被谁填（>50 µs 的窗口，22 个/步，合计 7.89 ms/步 = 20%）

| 填充者 | 占用 | 内容 |
|---|---:|---|
| stream 105 | 2.55 ms/步 | `Cast`×1595 + `IndexSelect`×990 + `Matmul`×660（混 AIC） |
| **stream 108** | 1.89 ms/步 | **`AivKernel`×4454 = engram 的 WKV all_gather**（纯通信+vector） |
| stream 47 | 1.73 ms/步 | `FillScalar`×3456 + `SelectV2`×2376 + `DivMods`×2160（**纯 AIV**） |
| stream 35 | 1.02 ms/步 | `SparseFlashMlaMetadata`×216（258 µs 级 AICPU） |
| stream 110 | 0.84 ms/步 | `Remainder`×324 + `IndexSelect`×270 |
| 其它 | 0.54 ms/步 | 103/104/102 |

⇒ **目前真正被掩盖的只有 ~7.9 ms，其中"有价值"的只有 engram 那 1.89 ms**
（其它是把侧流工作挪到主流空闲处，属于"填坑"而不是"重叠"）。

---

## 3. ★ 步内位置分布：一图看清"哪些是被串行排到最后的"

以 engram `allgatherAicpuKernel` 为步标志（每步一次，实测 77 次 / 间隔中位 40.63 ms / 变异 0.08），
把每类算子的时间按"落在步的第几成"分类：

| 算子 | 个数/步 | ms/步 | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% | **80%** | **90%** |
|---|---:|---:|---|---|---|---|---|---|---|---|---|---|
| `SparseFlashMla` | 43 | 2.988 | 9 | 12 | 11 | 13 | 12 | 12 | 13 | 11 | 2 | 4 |
| `GroupedMatmulSwigluQuant` | 43 | 5.185 | 9 | 12 | 11 | 11 | 12 | 12 | 13 | 11 | 5 | 5 |
| `GroupedMatmul`（w2） | 43 | 2.740 | 9 | 12 | 11 | 10 | 12 | 12 | 12 | 11 | 5 | 5 |
| `HcPre` | 86 | 3.279 | 9 | 12 | 11 | 12 | 12 | 12 | 13 | 11 | 5 | 4 |
| `HcPost` | 86 | 1.401 | 9 | 12 | 11 | 11 | 12 | 12 | 12 | 11 | 5 | 5 |
| `RmsNorm` | 186 | 2.496 | 10 | 12 | 10 | 12 | 12 | 11 | 11 | 11 | 7 | 4 |
| **`SparseFlashMlaMetadata`** | 3 | 0.787 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **17** | **83** |
| **`SparseAttnSharedkvMetadata`** | 2 | 0.453 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | **72** | 19 |
| **`ArgMax`** | 19 | 0.163 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | **26** | **69** |

**两类算子的分布完全相反**：

* **计算类**（SparseFlashMla / GroupedMatmul / HcPre / HcPost / RmsNorm）：**0–80% 每格均匀 9–13%**，
  80% 之后断崖跌到 2–5% ⇒ **前 80% 在算 43 层，最后 20% 无事可做**；
* **元数据/采样类**（SparseFlashMlaMetadata 83% 落最后 10%、SparseAttnSharedkvMetadata 72% 落 80–90%、
  **ArgMax 69% 落最后 10%**）⇒ **采样与"下一步的元数据"全被排在主流停止之后**。

### 3.1 步尾那 5.2 ms 是谁

最后 20% 的活动归属（主流此时只剩 0.52 ms）：

| 侧流 | ms/步 | 占比 | 内容 |
|---|---:|---:|---|
| stream 105 | 2.558 | 31% | `Cast` + `IndexSelect` + `Matmul` |
| stream 47 | 1.969 | 24% | `FillScalar` + `SelectV2` + `DivMods`（**采样/掩码类**） |
| stream 35 | 0.704 | 9% | **`SparseFlashMlaMetadata` 258 µs × 3** |
| stream 109（主流） | 0.522 | 6% | 少量收尾 |
| stream 96 / 110 / 104 | 0.90 | 11% | `IndexSelect` / `Remainder` / `AivKernel` |

⇒ **步尾约 5.2 ms = 采样链 + 下步注意力元数据，全在侧流，主流空着。**
这正是另一份文档说的"尾部 5.5 ms"，本轮把它**定位到了算子级**。

---

## 4. 主流内部：272 对 AIC↔AIV 是**硬串行**

主流上把算子按 AIC/AIV 分段：

| 类别 | 段数/步 | 块内算子数（中位） | 单块时长（中位） | 每步合计 |
|---|---:|---:|---:|---:|
| **AIC** | **271.9** | **1**（均值 1.2） | **37.3 µs** | 14.07 ms |
| **AIV** | **272.0** | 2（均值 3.1） | **21.2 µs** | 6.31 ms |

⇒ **主流 = 272 组「AIC 块(37 µs) → AIV 块(21 µs)」严格交替**，平均 **每 37.5 µs 切一次**。
（272 × 51.7 µs ≈ 14.07 ms ✓；272 × 23.2 µs ≈ 6.31 ms ✓ —— 和上面的资源账对得上。）

**这 272 对就是那个串行链**（每层约 6.3 对）：
```
HcPre(AIC) → RmsNorm(AIV) → DynamicQuant(AIV) → QuantMatmul(AIC) → RoPE(AIV)
→ SparseFlashMla(AIC) → MatMulV2(AIC) → HcPost(AIV) → MoE gating/routing(AIV)
→ GroupedMatmul(AIC) → ...
```

**AIC 块中位只有 1 个算子** ⇒ 每次 AIC 只跑 37 µs 就要让位给 AIV，而 AIV 又只跑 21 µs
⇒ 两边都在**半空转**。

---

## 5. draft（DSpark）到底在哪儿：**串在主流里，没有并行**

用层计数反推（每层应有 2 次 `HcPre`/`HcPost`，1 次 MoE gating）：

| 算子 | 个数/步 | 推断 |
|---|---:|---|
| `HcPre` / `HcPost` | **86** | 86 / 2 = **43 层** |
| `MoeGatingTopKHash` | **43** | **43 层** ✓ |
| `GroupedMatmulSwigluQuant` | **43** | 43 层 ✓ |
| **`SparseFlashMla`** | **40** | **只有主层有** |

⇒ **43 层 = 40 主层 + 3 个 DSpark proposal 层**（`num_nextn_predict_layers=3`）。
**draft 的 3 层与 40 个主层在同一条主流上串行执行**，没有任何并行。

（另：搜 draft 专有算子名，只找到 `rejection_greedy_sample_triton` **1 个/步、0.003 ms** ——
DSpark 的 proposal 层复用的是主层同款算子，所以无法用名字区分。）

---

## 6. 两个想法的评估

### 6.1 「AIC/AIV 交替并行」

**上限很明确**：主流内 AIV 6.31 ms 若全藏进 AIC ⇒ 40.02 → 33.7 ms（**1.19×**）。

**但在同一 token 批内做不了**，因为那 272 对是**直接依赖**（AIV 吃 AIC 的输出、AIC 吃 AIV 的输出）。
主流上 AIC 块中位只有 **1 个算子**，这个"细碎"本身就是依赖紧耦合的证据。

> ## ⛔ 更正（2026-10-07）：下面这段"跨请求"的结论**是错的**，当天晚些时候被实测推翻
>
> 原文：~~"`conc≥2` 时不同请求的 AIC/AIV 天然错峰 —— 这已经在 continuous batching 里做了
> （这也解释了为什么 `conc=8` 时 AICore 能到 100%）"~~
>
> **更正**：vLLM 的 continuous batching 把**所有并发请求塞进同一个 batch、同一张图、同一次前向、
> 同一条流** ⇒ 请求之间**不产生相位多样性**。实测（`UBATCHING-VERDICT-TINY-PROFILE`，conc=1 → conc=8）：
>
> | 指标 | conc=1 | conc=8 |
> |---|---:|---:|
> | **串行度**（Σ各资源 busy / 步长） | 79% | **93%（更高）** |
> | **AIC 利用率** | 39% | **35%（更低）** |
> | 每步算子数 | 3,349 | 1,603（**更少、更宽**） |
>
> ⇒ **并发提高只会让算子"变宽变少"（摊薄固定开销），不会产生 AIC/AIV 重叠。**
> `npu-smi` 报的 `AICore%` 从 79% 升到 100% 是**"busy"口径，包含等数据/等指令**，
> 不等于在做 MAC，不能当作重叠的证据。
>
> **并且"跨请求并行"在量化上也是亏的**：若改成"每个请求一次独立前向 + 多流并行"，
> 每步的 AIC 需求量 = 8 × 13.55 = **108 ms**（每个前向都要重读全部权重），
> 而当前批处理实测 conc=8 步长只有 **41.5 ms** ⇒ **批处理好 2.6×**。
> ⇒ **"跨请求"不是一条可用的杠杆**，`conc=1` 那 45% 的 AIC 空转是**结构性的**。

### 6.2 「把下一个 batch / draft 的带宽和计算掩盖」

**draft 这条路走不通**（§5：draft 依赖主模型 verify 的输出，且它已经串在主流里）。

**但"下一个 batch 的准备工作"这条路是对的，而且目标已经定位到算子级**（§3）：

| 步尾成分 | 能否提前 | 依据 |
|---|---|---|
| **`ArgMax`**（采样，19 个/步，69% 在最后 10%） | ❌ | 依赖本步 LM head 的 logits |
| **`SparseFlashMlaMetadata`**（3 个/步，258 µs 级，83% 在最后 10%） | ✅ **大概率可以** | 只依赖 block table / KV 长度（步开始时就知道） |
| `SparseAttnSharedkvMetadata`（2 个/步，72% 在 80–90%） | ✅ 同上 | 同上 |
| `FillScalar`/`SelectV2`/`DivMods`（stream 47，1.97 ms） | 需逐项核 | 可能是采样索引构造（依赖 logits）或路由掩码（不依赖） |

⇒ **可提前的那部分（metadata 类，约 1.2–2 ms/步）就是"把下一步的元数据与这一歩的计算重叠"**。

### 6.3 还有一块被忽略的：**通信 4.22 ms 与 AIC 零重叠**

```
通信 ∩ AIC   = 0.000 ms
通信 ∩ AIV纯 = 0.028 ms
```

通信是**唯一一块"既零重叠、又不受 token 批依赖限制"**的时间 ——
allreduce 只需要等本层 GEMM 的结果，而**下一层的 GEMM 完全可以同时跑**
（这就是标准的"计算-通信重叠"）。本仓已排除的路径（`FUSED_MC2` −3.8%、`enable_sp` 更慢、
custom allreduce 无实现）都是在**算子融合**层面，**没有试过"多流 + 控核"**这条路。
官方那边对应的正是 `CCU 展开 AllReduce` + 多流。

---

## 7. 结论：可掩盖延迟的清单（按证据强度排序）

| # | 目标 | 可回收 | 依赖是否允许 | 证据 |
|---|---|---:|---|---|
| **1** | **通信（4.22 ms）与 AIC 重叠** | ≤4.2 ms | ✅ 允许（下一层 GEMM 可同时跑） | §6.3 零重叠 |
| **2** | **步尾 metadata 提前**（SparseFlashMlaMetadata / SharedkvMetadata） | ~1.2–2 ms | ✅ 大概率允许 | §3 位置分布 83%/72% |
| 3 | 主流内 AIV 跨层流水 | ≤6.31 ms | ⚠️ 同批内是依赖；需跨请求 | §4 272 对中位 1 个算子 |
| 4 | draft 与 verify 并行 | ~0 | ❌ 依赖 + 已串在主流里 | §5 层计数 43=40+3 |
| 5 | 采样（ArgMax）提前 | 0 | ❌ 依赖 logits | §3 |

**一句话回答用户的问题**：
现在被掩盖的只有 engram 那 ~1.9 ms 的 all_gather（以及若干"填坑"式侧流）；
**AIC 闲着的 18 ms 里没有任何一项被真正藏住**。
而"下一个 batch"在 `conc=1` 时不存在，
所以真正能动的是：**① 通信与计算重叠（4.2 ms）、② 把下一步的元数据提前（1~2 ms）**。

## 8. 复现

```bash
PROF=~/cedpd-repo/results/armF_r6_base/prof/dp0_pp0_tp0_dcp0_ep0_rank0_1434_20261004195431495_ascend_pt
python3 tools/prof_overlap_who.py   $PROF/ASCEND_PROFILER_OUTPUT   # 各流构成 + 主流空闲窗口归因
python3 tools/prof_overlap_where.py $PROF/ASCEND_PROFILER_OUTPUT   # 步内位置分布 + AIC/AIV 块长
python3 tools/prof_stream_ident.py  $PROF/ASCEND_PROFILER_OUTPUT   # 给每条流命名
python3 tools/prof_draft_locate.py  $PROF/ASCEND_PROFILER_OUTPUT   # draft 层计数反推
```

> 需 pandas/numpy ⇒ 用交付镜像跑（宿主没有 pandas）。
