# 优化空间估计（2026-09-16，静态内核修复后的基线）

> ## ⚠️ 2026-09-16 04:15 重要修正：步数分母错了 10%
>
> 实测：客户端 70 步窗口内 `DispatchFFNCombineW4A8` 有 **3076 个事件 = 43.94/步**，
> 不是 40/步（40 层 target + 4 层 draft 路径）。此前用 40 当分母 ⇒ **每步设备值偏小 ~10%**。
>
> **修正后的 32K profiled 设备账**（窗口 2611 ms / 63.7 步）：
>
> | 项 | 修正前（错误） | **修正后** |
> |---|---|---|
> | 设备跨度 | 37.29 | **40.98** |
> | busy | 30.94 | **33.99** |
> | compute | 27.64 | **30.36** |
> | comm | 3.30 | **3.63** |
> | FREE | 6.36 | **6.98** |
>
> 与同会话客户端 43.08 ms/step 的差 = **2.1 ms host 开销**（比之前估计的 5.5 小得多，更自洽）。
>
> **折算生产态（unprofiled 35.85，缩放系数 35.85/43.08 = 0.832）**：
>
> | 项 | ms/step | 说明 |
> |---|---|---|
> | **compute** | **25.3** | 这是真正的"地板" |
> | comm（全暴露） | 3.0 | 可重叠则归零 |
> | FREE | 5.8 | |
> | host | 1.8 | |
> | **合计** | **35.85** | ✓ |
>
> ⇒ **"完美重叠 + 零 FREE + 零 host"的地板 = 25.3 ms/step（32K）**，
> 比我之前估的 25.2 略高；128K 的地板还要再加 ~3.3（长上下文注意力）≈ **28.5 ms**。

> 问题：FREE 仍在，通信与计算仍未互相掩盖，**还剩多少空间？**
> 方法：全部基于**实测**（unprofiled 客户端墙钟 + 同会话配对 A/B + 设备 profile 的结构比例），
> 不做纸面上限的乐观外推。**每条给出置信度。**

---

## 1. 当前基线与"地板"的算法

### 1.1 已知量

| 量 | 值 | 来源 |
|---|---|---|
| 32K 客户端墙钟（unprofiled） | **35.85 ms/step** | `measure_fixed.log` |
| 128K 客户端墙钟（unprofiled） | **39.10 ms/step**，A=2.763 | `measure_lws128.log` |
| 设备 busy（**profiled**） | 30.94 ms/step（83.0%） | `dev_account.py`，步数由同会话客户端给 |
| 设备 FREE（profiled） | 6.36 ms/step（17.0%） | 同上 |
| comm（profiled） | 3.30 ms/step，**与 compute 零重叠** | 同上 + `op_wall.py` |
| compute（profiled） | 27.64 ms/step | 同上 |

### 1.2 把 profiled 数值折算到生产态（unprofiled）

profiler 会同时抬高客户端与设备侧时间。用**两次独立的同会话对照**定标：

| 对照 | profiled 客户端 | unprofiled 客户端 | 差 |
|---|---|---|---|
| `fmc2_prof_32k` → `fmc2b_32768` | 40.97 | 34.35 | **6.62** |
| `fixed_prof` → `fixed_32768` | 43.08 | 35.85 | **7.23** |

取 **≈6.9 ms/step** 作为 profiler 的附加量，则生产态近似的设备跨度
≈（profiled 设备跨度 37.30）− 6.9 ≈ **30.4 ms/step**，按比例拆：

| 生产态近似 | ms/step |
|---|---|
| 设备 **compute** | **≈22.5** |
| 设备 **comm（全暴露）** | **≈2.7** |
| 设备 **FREE** | **≈5.2** |
| 客户端 − 设备跨度（纯 host 侧外露） | ≈5.5 |
| **合计** | **35.85** ✓ 与实测吻合 |

> ⚠️ 这是"按比例折算"的近似（置信度**中**）。它只用于**分配优化空间**，不用于对外报绝对数。

---

## 2. FREE 的来源是**高度集中**的（实测，置信度**高**）

对 decode 窗口按"空档大小 × 前→后算子"分解（`free_sources.py`）：

| 前 → 后 | ms/step | 占 FREE |
|---|---|---|
| **`Fill → hcom_alltoallv_`** | **1.392** | 22% |
| `hcom_broadcast_ → hcom_alltoallv_` | 0.551 | 9% |
| `hcom_alltoallv_ → IndexCheck` | 0.435 | 7% |
| `Sub → Sub` | 0.198 | 3% |
| `hcom_broadcast_ → ZerosLike` | 0.147 | 2% |
| 其余（>30 种组合） | ~3.6 | 57% |

**前三项 2.38 ms/step 全部是 Engram route 的 all_to_all/broadcast 前后** ——
即 host 在把 route 的集合通信下发下去时，设备在空等。**这与之前"Engram 相关空档 1.54 ms/step"的结论一脉相承**，
现在的量更大（因为现在 host 侧更快了，暴露反而更清楚）。

> 注意：`hcom_alltoallv_` 是 **Engram** 的值回传，不是 MoE 的（MoE 走 `DispatchFFNCombineW4A8`）。

---

## 3. 逐项可回收空间（按置信度排序）

### 3.1 高置信（机制已定位，有配对证据）

| # | 项 | 可回收 | 依据 | 落地难度 |
|---|---|---|---|---|
| **A** | **Engram route 的 host→device 下发暴露** | **≤2.4 ms/step** | §2 的 2.38 ms/step 直接就是"等下发" | 中（需把 route 提前一步/预取，历史多轮收窄过） |
| **B** | **host post-D2H 路径**（hash 0.42 + route 1.0–1.5 + pad 0.13） | **≤1.6–2.0 ms** | `delaypost5 → +5.80`（116% 暴露） | 中（同上，部分与 A 重叠**不可重复计**） |
| **C** | **TP allReduce 与计算重叠** | **≤2.7 ms** | comm 3.30 profiled 全暴露、交集 = 0 | **高**（需接 `npu_mm_all_reduce_base`，本 build 是死代码） |

> **A 与 B 高度重叠**：route 的 host 工作既在免费里也在暴露里，合并计**上限 ~2.5 ms**，不是 2.4+2.0。

### 3.2 中置信（结构清楚，需改算子）

| # | 项 | 可回收 | 依据 |
|---|---|---|---|
| D | HcPre + HcPost 融合 | ≤2.4 ms（profiled 2.99 折算） | 两个逐元素算子每层各 2 次，共 160 次/步 |
| E | 剩余 FREE（30+ 种小组合） | 部分不可回收 | 依赖链天然的空隙；乐观 1–2 ms |

### 3.3 低置信 / 已否决

| 项 | 结论 |
|---|---|
| 加大 `SP_TOKENS` | ❌ 逐位置接受率几何衰减，pos5/6 恒 0 |
| BF16 draft 换接受率 | ❌ 净 tok/s 不变/变差 + KV < 3M（`draft-precision-acceptance-ab.md`） |
| `dyamic_spec_config` | ❌ async 下死开关 |
| `CPU_BIND=1` | ❌ 实测更慢 |
| MC2/hierarchy | ❌ EP=8 不满足 `epWorldSize` 16 对齐 |
| MegaMoe（`enable_fused_mc2=2`） | ❌ `moe_intermediate_size=2304` 不被 `%512` 整除，会静默退回 |
| `multistream_overlap_shared_expert` | ❌ 与 `enable_fused_mc2` 互斥；而 ALLTOALL 路径有 host 同步风暴（实测 41.8→35.85 的差距） |

---

## 4. 三档目标（诚实区间）

以 32K 的 35.85 与 128K 的 39.10 为起点：

### 档 1：保守（只做 A+B）—— 置信度 **高**

| | 32K | 128K | tok/s @128K（A=2.763） |
|---|---|---|---|
| 现在 | 35.85 | 39.10 | 70.1 |
| 档 1 | **≈33.5** | **≈36.8** | **≈75** |

手段：把 Engram route 的 host 工作移出关键路径（预取/提前一步）。
**代价低、无结构风险**；但历史上多轮尝试已被收窄（route 流水化实测无收益），
所以这一档更可能落在 **34–35**（+3~5%）。

### 档 2：中等（档 1 + C 的通信重叠）—— 置信度 **中**

| | 32K | 128K | tok/s @128K |
|---|---|---|---|
| 档 2 | **≈30.8** | **≈34.1** | **≈81** |

手段：接上 `npu_mm_all_reduce_base`（或等价的 mm+allreduce 融合），让 2.7 ms/step 的 TP allReduce
进入算子内部流水。**这是当前唯一能实质吃掉"零重叠"的路子**，
但需要改图内代码 + 重捕获，属结构性改动。

### 档 3：激进（档 2 + D 的 HcPre/HcPost 融合 + 剩余 FREE）—— 置信度 **低**

| | 32K | 128K | tok/s @128K |
|---|---|---|---|
| 档 3 | **≈28.5** | **≈31.5** | **≈88** |

---

## 5. 结论：**110 tok/s 不可达**（在当前硬件/算子/接受率下）

`tok/s = A × 1000 / ms`。128K 要 >110：

| A | 需要 ms/step | 我们的地板（完美重叠 + 零 FREE + 不动 compute） |
|---|---|---|
| 2.763（实测） | **≤25.1** | ≈25.2（= 22.5 compute + 2.7 不可避免的通信尾） |
| 3.0 | ≤27.3 | 同上 |
| 4.0（不可达） | ≤36.4 | — |

**修正后的地板**：32K 的 compute = **25.3 ms/step**；128K 再加 ~3.3（长上下文注意力）
⇒ **128K 的地板 ≈ 28.5 ms/step**（= 39.10 的 73%）。

| A | 110 tok/s 需要 ms | 与 128K 地板（28.5）的关系 |
|---|---|---|
| 2.763（实测） | ≤25.1 | **不可能**（地板就超了 13%） |
| 3.0 | ≤27.3 | 不可能（差 4%） |
| **3.15** | **≤27.3** | 仍需把地板再压 4% |
| 3.5 | ≤31.4 | **有可能**（地板 28.5 + 一些不可消除开销） |

⇒ 结论比之前更精确：**110 tok/s 不是"差 40%"，而是"差 13%"**，
但**单靠消除 comm/FREE/host 不够**（那只能到 28.5），必须**同时**
把 compute 压低 ≥12% **或** 把 A 从 2.76 提到 ≥3.15。

**现实可达区间：128K 约 31–35 ms/step ⇒ 79–89 tok/s。**
要突破到 110，必须动 **compute 本身**：
`DispatchFFNCombineW4A8`（9.41 profiled / ~7.8 生产态，MoE 专家权重流，受带宽限制）、
`SparseFlashMla` + `QuantLightningIndexerV2`（长上下文注意力）、
`HcPre/HcPost`（2.99）、`QuantBatchMatmulV3`（2.17）——
或**改 draft 模型提高 A**。前者需要算子级优化（cannbot 措施2/3/8），后者超出当前范围。

### 因此建议把验收目标调整为可达的分档

| 指标 | 当前 | 档 1 | 档 2 | 档 3 |
|---|---|---|---|---|
| 32K ms/step | 35.85 | 34.5 | 30.8 | 28.5 |
| 128K ms/step | 39.10 | 37.0 | 34.1 | 31.5 |
| **128K tok/s** | **70.1** | **~75** | **~81** | **~88** |
| 32K device FREE | 17%（profiled） | — | <10% | <5% |

---

## 6. 建议的下一步（按性价比）

1. **先试档 1 的低成本路径**：Engram route 的 host 工作与下一步重叠
   （`Fill → alltoallv` 那 1.39 ms/step 是最集中的单点）。
2. **同时评估档 2 的接线成本**：`npu_mm_all_reduce_base` 需要改 MLA 的 o_proj 输出与 MoE 输出两处，
   都要在图内 + 重捕获。**建议先做一次离线可行性验证**（能否拿到 hcom 名、能否在图内调用），
   再决定是否投入。
3. **档 3 的 HcPre/HcPost 融合**可用现有热更新通道低成本迭代（但改图内代码仍需重捕获）。

---

## 7. 证据路径

| 内容 | 路径 |
|---|---|
| FREE 来源分解 | `scripts/free_sources.py` → `/tmp/op_summary_fixed.csv`（本报告 §2） |
| 设备账 | `scripts/dev_account.py`、`scripts/op_wall.py` |
| profiler 定标 | `profiler-overhead-analysis.md` §3（两次同会话对照） |
| 基线测量 | `measure_fixed.log`、`measure_lws.log`、`measure_lws128.log` |
| 接受率上界 | `acceptance-and-hotreload.md` §1 |
| 零重叠与 cannbot 解法 | `comm-compute-overlap-cannbot.md` |
| draft 精度 A/B | `draft-precision-acceptance-ab.md` |
