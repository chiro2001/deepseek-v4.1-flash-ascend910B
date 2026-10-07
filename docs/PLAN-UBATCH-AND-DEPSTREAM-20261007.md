# ubatching 与 DepStream 两条线：后续怎么做（2026-10-07）

> 本文件是 `feat/depstream` 分支的**开篇计划**，覆盖**两条互相独立的线**：
> * **U1/U2 = ubatching（DBO）** —— **切批**，已验证净亏，需先测判据；
> * **DepStream** —— **不切批**，按真依赖做侧流化（旧文档里叫 U3 / Path A，**该叫法已废弃**）。
>
> 基线 `feat/v41-dcp8@336ef05`；分支/worktree 见 §7。结论标【实测】/【推断】/【未确认】。

---

## 0.0 ★ 术语（先读这一节，防止后面被误导）

### 正式名：**DepStream**（依赖驱动侧流化 / Dependency-driven Side-Streaming）

| | |
|---|---|
| **拆解** | **Dep**endency-driven + **Stream**-ing |
| **一句话定义** | 在**不改批、不改 shape、不改数值**的前提下，按**真实数据依赖**判定主流上哪些算子不构成硬前驱，把它们放到**侧流**执行，用 event 只保留真依赖，从而填满稀缺资源（AIC）的空转窗口 |
| **机制三要素** | ① **依赖判据**（不是算子类型）② **侧流承载**（不是新 batch）③ **event 保序**（只保真依赖） |
| **判据** | §3.6：依赖审计产出可挪候选 ≥0.5 ms，且过答案稳定性门 |

### ⛔ 五个必须避免的叫法（会把人带偏）

| ❌ 不要叫 | 为什么危险 |
|---|---|
| **"U3" / "Path A"** | 编号无语义；Path A 是旧文档里**被高估**的说法（当时写低难度 4.2 ms，实测修正为三个子机制、难度差异极大）。**此后一律用 DepStream。** |
| **"AIC∥AIV 并行"** | 会被读成把 AIC 算子放一条流、AIV 算子在另一条 —— **那样做收益为零**（相邻 AIC/AIV 是硬依赖，链上仍严格交替）。微基准已实锤：**仅切流无效，难点在找独立工作**。 |
| **"ubatch" / "ubatching"** | 那是**切批**（DBO）：工作量膨胀 1.70×、数值会变。DepStream **不切批**，二者是**相反方向**。本分支旧名 `feat/ubatch-aic-aiv` 已改，就是为了断掉这个联想。 |
| **"multi-stream"（泛称）** | 与 CED 已有的 `MULTISTREAM=1` 混淆。后者是**硬编码的少数侧流**（engram / DSA / 采样），DepStream 是**依赖驱动、系统性**的扩展。 |
| **"stream parallel"** | 太泛，会与 **DP**（不同请求走不同流）混为一谈。 |

### ✅ 允许的简称

* 正式：**DepStream**（文档 / commit / 分支 / 汇报）
* 中文口语：**侧流化** 或 **依赖侧流**
* 子机制：**DepStream-A**（侧流重排）、**DepStream-B**（主流 AIV 拆分）、**DepStream-C**（尾部提前）

> 分支 `feat/depstream`　worktree `~/projects/dsv41/depstream`

---

## 0. 判据先行：两个数决定这条线的生死

| 量 | 定义 | 值 | 来源 |
|---|---|---|---|
| **U** | **完美重叠上界** = 步长 ÷ AIC_busy | tp8 conc=1：40.02/22.05 = **1.82×**；tiny conc=8：23.88/8.43 = **2.83×** | 【实测】`DECODE-AIC-AIV-PIPELINE` / `UBATCHING-VERDICT-TINY-PROFILE` |
| **k** | **拆批代价**（把一次前向拆成 2 个 micro-batch 的净损耗） | **1.70×**（tiny conc=4/8：98.9→58.0、130.6→75.1） | 【实测】`DBO-GRAPH-VERDICT-NEGATIVE` |

**判据：`U / k ≥ 1.15` 才值得继续。**

| 场景 | U | k | **U/k** | 判定 |
|---|---:|---:|---:|---|
| tp8 conc=1 | 1.82 | 1.70【跨体制外推】 | **1.07** | ❌ 不值得 |
| tiny conc=8 | 2.83 | 1.70 | **1.66** | ✅ 值得 |

### ⇒ 关键未知量是 **k 随批大小 M 怎么变**，而它从未被测过

* 上界 U 随并发**升高**（AIC 利用率从 39% 掉到 35%）⇒ 高并发反而更有空间；
* 但拆批代价 k 也随 M 变化：M=6 → 2×3 时算子数≈翻倍（**结构决定**）；
  **M=48 → 2×24 时算子数是否仍翻倍，没测**。
* 若 k 在 M≥24 时降到 ~1.0~1.2 ⇒ G1 有真实空间；若 k 恒 ≈1.7 ⇒ **G1 在单流上死了**。

**这就是第一件要做的事（U1）。**

---

## 1. U1：判据实验 —— 测出 k(M) 曲面（1~2 天，只用 tiny）

**复用现成资产，几乎不需要新代码**：DBO overlay 已建好（`tools/dbo_build_overlay.sh` + 27 个
`dbo_fix_*.py`），图模式能跑，`V41_DBO_GRAPH_SERIAL` 判别开关已在。

### 扫点

* `dbo_decode_token_threshold` ∈ {4, 8, 16, 32}（默认 32 ⇒ conc=1 不触发，必须下调才能测小 M）
* conc ∈ {1, 2, 4, 8, 16}
* 每格两臂：
  * `V41_DBO_GRAPH_SERIAL=1` ⇒ **纯拆批代价** k(M) = 基线 ÷ 串行臂
  * `V41_DBO_GRAPH_SERIAL=0` ⇒ **拆批 + 重叠**，净收益 = 基线 ÷ 并发臂

### 产出（三张曲线）

1. **k(M)**：拆批代价 vs 批大小
2. **重叠净收益(M)**：并发臂 vs 串行臂
3. **U(M)**：该 M 下的完美重叠上界

### 判据（硬门）

> 在**目标并发 8~16**（用户真实负载）上 `U/k ≥ 1.15` ⇒ 进 U2；否则 **ubatching 线关闭归档**。

**建议先测 3 个点（conc=2/8/16）即可看出趋势**，不必扫满网格。

---

## 2. U2：若 U1 通过 —— 真正形态的 G1（每 ubatch 独立 compute_stream）

必须改的三件事（其余都已解）：

| # | 改动 | 现状 |
|---|---|---|
| **1** | **每个 ubatch 一条独立 `compute_stream`** | 🔴 **现在只有一条、所有 ubatch 共用** ⇒ 计算**物理上不可能并行**。这是我们唯一没做的关键改动 |
| **2** | **图内 fork/join** | ✅ **规则已解**：fork event 必须 record 在**捕获根流**上；`tools/tiny_graph_ms.py` 验证 1.31× |
| **3** | **地址生命周期** | 🔴 捕获期 metadata 张量必须改成"**预分配静态 buffer + 每步 `copy_`**"。保活（`_DBO_KEEPALIVE`）只是 workaround —— replay 读到的仍是**陈旧内容**，不是正确性修复 |

补充：yield 钩子已在唯一咽喉点（`GroupCoordinator._all_reduce_out_place`）验证触发
（`allreduce-yields=103600`），但**只测过 eager**，图模式下需重做。

**Kill criterion**：图模式下测重叠系数 **ρ < 0.3** ⇒ 停。

**成本估计**：骨架改造 + 地址审计 + 精度门 ≈ **1~2 周**。

---

## 3. DepStream：★ 不依赖 U1 的主攻方向 —— 「只切流、不切数据」

> **2026-10-07 补充实测**（`depstream_probe*.py`，`armF_r6_base` 交付口径 profile，3 步窗口）：

### 3.2 现状覆盖【实测】

**步长分解（40.644 ms/步，profile 口径）**：

```
主流(109) busy 25.423 (62.5%)  ├─ AIC 17.193
                               └─ AIV  8.231
主流 gap      13.157 (32.4%)   ├─ 大 gap ×1   7.711  ← 步的 70~80% 位置
                               ├─ 中 gap ×26  2.186  (50~200µs)
                               └─ 小 gap ×467 3.260  (<50µs)
全卡 AIC busy 21.177 (52.1%)   ⇒ AIC 空闲 19.47，其中：
                                  被 AIV 挡 11.60 / 被 COMM 挡 3.96 / AICPU 1.02
                                  真正空转 1.65
```

**三个决定性事实**：

| # | 实测 | 含义 |
|---|---|---|
| **1** | **主流 AIV 窗口 8.231 ms 期间，AIC 只跑了 1.802 ms（7.3%）** ⇒ **6.43 ms AIC 空转** | AIV 挡住的 AIC 时间 |
| **2** | **主流 AIV 块（658 个/步）结束后，到下一个算子的间隔中位 1.0 µs**（p25 0.2 / p75 1.8） | 主流 AIV 是**硬串在链上**的——后面立刻接 `HcPre`(163) / `MatMulV2`(124) / `GroupedMatmul`(120) / `QuantMatmul`(110) |
| **3** | **7.71 ms 大 gap 里，侧流跑了 882 个算子 / 8.598 ms，覆盖 gap 的 93%**；含 `MatMulV2`、`HcPre`、`GroupedMatmul`、`SparseAttnSharedkv`、`AivKernel`、`SparseFlashMlaMetadata`、`QuantLightningIndexerV2Metadata` | 尾部空洞 = **采样 + 下一步注意力/索引元数据**（跨步依赖），**不是**主流在等自己 |

### 3.1 它和 DBO 的区别（为什么"没有拆"）

| | DBO / G2 / G3 | **DepStream（不切数据，只切下发流）** |
|---|---|---|
| **切什么** | **切数据**：M 个 token → 两个 M/2 批，各跑一次完整前向 | **切下发顺序**：同一批、同一组 kernel、**同样的 shape** |
| **工作量** | **膨胀 1.70×**（MoE 专家利用率↓、allreduce 次数×2 且更小） | **一点不变** |
| **数值** | 归约形状改变 ⇒ **会变** ⇒ 要过四道门 | 算子不变 ⇒ **逐位一致** |
| **代价** | k=1.70×（结构性） | 只在"**是否有可挪的工作**"上 |
| 机制 | 通信点主动让出 | 把无依赖的算子放到另一条流 + 补 event |
| 先例 | 无（净亏 42%） | ✅ **有**：`AivKernel`（engram all_gather）4.11 ms **已 100% 与 HCCL 并行** |

> **⚠️ 修正前一版的表述**：此前把这条线写成「难度低、上界 4.2 ms」，那是照抄旧文档里 Path A 的说法（**这也是 Path A 这个叫法要废弃的原因**）。按新数据，**DepStream 的真实结构是三个子机制，难度与收益完全不同**（见 §3.5）。

主流（stream 109）busy 只有 51%，其空闲被 5 条侧流填掉 **7.89 ms/步（20%）**：

| 侧流 | ms/步 | 内容 | 算不算真重叠 |
|---|---:|---|---|
| stream 105 | 2.55 | `Cast` + `IndexSelect` + `Matmul` | 部分用 AIC |
| **stream 108** | **1.89** | **engram `AivKernel`（all_gather）** | ✅ **是**（真正藏进通信） |
| stream 47 | 1.73 | `FillScalar` + `SelectV2` + `DivMods` | ❌ 只是填坑 |
| stream 35 | 1.02 | `SparseFlashMlaMetadata`（AICPU） | ❌ 填坑 |
| stream 110 | 0.84 | `Remainder` + `IndexSelect` | ❌ 填坑 |

**⇒ 真正被掩盖的只有 engram 那 1.89 ms；`通信 ∩ AIC = 0.000` ⇒ 集合通信这块完全空白。**

### 3.3 DepStream 的三个子机制（按"能不能兑现"排序）

#### DepStream-A：把**已在侧流上的 AIC 工作**对齐到主流 AIV 窗口（低风险，~2 ms）

* 侧流已承担 **3.98 ms/步** 的 AIC 工作，构成：
  `s106 QuantBatchMatmul 1.292`(80 次/步)、`s107 QuantBatchMatmul 0.593`(40)、
  `s105 MatMulV2 0.537`(12) + `SparseAttnSharedkv 0.389`(3) + `HcPre 0.266`(6) + `GroupedMatmul 0.180`(3)、
  `s47 MatMulV2 0.222`(2)
* **但落在主流 AIV 窗口里的只有 1.80 ms** ⇒ 纯**调度重排**，不碰任何依赖，最多再加 ~2 ms。
* 判据：只改 `torch.npu.stream(...)` 布置 + 答案稳定性门。

#### DepStream-B：★ 把**主流 AIV 块里"非真前驱"的那部分**挪出去（最大单项，难度高）

* **依据**：主流 AIV 块 **median 2 个算子、mean 3.1 个**（658 块/步、8.231 ms）。
  如果块里只有 1 个是后继 AIC 的真前驱，另外 1~2 个可以延后
  ⇒ 理论可挪 **≈ 2/3 × 8.231 ≈ 5.5 ms**。
* **反证依据**：AIV→下一个算子的间隔**中位 1.0 µs** ⇒ 至少"块尾那个"是真前驱，不能整体挪。
* **必须先做代码级依赖审计**（§3.4）才能知道到底有几个可挪。
* 难度：**高**（逐块判定 + 改下发流 + 图捕获）。

#### DepStream-C：★ 把**尾部 7.7 ms 空洞里"与采样无关"的部分提前**（新发现，中等难度）

* 尾部空洞 7.711 ms 位置固定在步的 **70~80%**，其间侧流 882 个算子 / 8.598 ms：
  采样链 + `SparseFlashMlaMetadata`(AICPU) + `QuantLightningIndexerV2Metadata`(AICPU)
  + `AivKernel`(engram all_gather) + `IndexSelect` / `Index` / `ViewCopy` + 少量 `MatMulV2`/`HcPre`/`GroupedMatmul`。
* **其中"下一步的元数据 / engram 预取"并不依赖采样结果**（只有采样本身依赖）⇒ 可以提前到
  前 70% 的计算窗口里，与主流 AIC 重叠。
* 保守可回收 **1~3 ms**；且这是**跨步**重叠，不碰本步的数据依赖，**精度风险最低**。
* 难度：**中**（在 model runner 里把 metadata 准备提前；图捕获下要保证地址与 event 正确）。

### 3.4 依赖审计怎么做（1~2 天，纯分析，不动服务）

**问题**：主流上 658 个 AIV 块 / 步，哪些的输出**不是**紧跟其后那个 AIC 算子的真前驱？

* 已有工具：`tools/prof_chain_core.py`、`prof_chain_blocks.py`、`prof_stream_ident.py`、
  `prof_consumers*.py`、`prof_op_neighbors.py`、`prof_overlap_who.py`
* 新增探针（本轮）：`depstream_probe.py`（AIC/AIV 窗口重叠）、`depstream_probe2.py`（主流 gap 归因 +
  AIV 块后继分布）、`depstream_probe3.py`（gap 大小/位置分布）、`depstream_probe4.py`（大 gap 内幕）
* 判定方法：**代码级**读 `vllm_ascend/models/deepseek_v41/model.py` 的层循环，
  对每个 AIV 块列出"输出张量 → 后续消费者"，消费者是紧跟的 AIC ⇒ 硬前驱；
  消费者在后面 ⇒ **可挪候选**。
* 另一条通用手段（复用已有方法论）：**延迟注入探针** —— 给候选算子加 Δ 延迟，
  若步长不涨 ⇒ 它已不在关键路径上；若涨满 Δ ⇒ 硬前驱。

### 3.5 修正后的预期（诚实版）

| 子机制 | 可回收 | 难度 | 精度风险 | 依赖 U1？ |
|---|---:|---|---|---|
| **DepStream-A** 侧流 AIC 重排 | **~2 ms（5%）** | 中低 | 无（逐位） | 否 |
| **DepStream-B** 主流 AIV 块拆分 | 上界 **~5.5 ms（13%）**，实际取决于审计 | **高** | 无（逐位） | 否 |
| **DepStream-C** 尾部元数据提前 | **1~3 ms（3~7%）** | 中 | 无（逐位） | 否 |

> **三者可叠加**，乐观合计 **4~8 ms（10~20%）**——但这个数字**必须先由 §3.4 的审计证实**，
> 现在只能算"上界"。我此前说的"低难度、4.2 ms"应以此表为准。

### 3.6 判据

> 审计产出"可挪候选"合计 **≥ 0.5 ms** 且答案稳定性门通过 ⇒ 进实现；
> 否则 DepStream 关闭（与 U1 同处置）。
> **DepStream-C 单独可做**：即使 A/B 全否，尾部元数据提前仍独立成立。

## 4. 明确不做（已被实测或数学关闭，避免重复投入）

| 线 | 关闭理由 |
|---|---|
| **G2 通信∥计算** | 上界 **≤1.10×**（通信暴露 2.5 ms / 24.6 ms）**< 拆批代价 1.70×** ⇒ **数学上不可能赢** |
| **跨层软件流水** | **无合法调度**：层 L+1 的第①个算子吃层 L 的**最后一个**输出（不是难度问题，是目标不可达） |
| **「不同请求走不同流各跑完整前向」** | 那就是 **DP**；应与 **DP2TP4 / DP4TP2 的实测**直接比较，不走 DBO 骨架 |
| 用合成微基准外推 ubatching 收益 | 1.37× 是**固定总工作量切多流**（kernel 效率不随 shape 变）；真实拆批**会改变 kernel 效率** ⇒ 不可外推 |

---

## 5. 与交付配置的关系

| 项 | 状态 |
|---|---|
| 交付配置 `armRESTORE7_1007_090030` | **DCP=1**（实测日志 `decode_context_parallel_size=1`）⇒ 与 PR 的"DCP/PCP 不支持"**不冲突** |
| G1 与 **DCP8 线** | ⚠️ **现有实现下互斥**（上游限制）⇒ 若两条都要，需先解决 DCP 兼容，或 G1 只在 DCP=1 档提供 |
| tp8k5 交付实例 | **全程不动**；所有实验在 **tiny（a3-21 chips 2–3，端口 19310）** |

---

## 6. 资产索引（都在本仓）

| 类别 | 路径 |
|---|---|
| 上游存档 | `tools/ref_pr11273.diff`、`tools/ref_pr11273_npu_ubatch_wrapper.py` |
| 多流图捕获 | `tools/tiny_graph_ms.py`（成功版）、`tools/tiny_graph_multistream.py`（失败版反例） |
| DBO 补丁链 | `tools/dbo_build_overlay.sh`、`tools/dbo_fix_*.py`（27 个） |
| 负结果存档 | `tools/dbo_negative_disable_devmeta.py` |
| 分析工具 | `tools/prof_{chain_core,chain_blocks,stream_ident,overlap_who,overlap_where,ab_overlap,consumers*}.py` |
| 关键文档 | `DBO-A3-PROGRESS-AND-BLOCKER-20261006.md`（错误链 1–16）、`DBO-RUNTIME-VERDICT-20261006.md`、`COMM-STRUCTURE-AND-DBO-REGIME-20261006.md`、`DBO-GRAPH-VERDICT-NEGATIVE-20261006.md`、`MULTISTREAM-GRAPH-CAPTURE-BREAKTHROUGH-20261006.md`、`UBATCHING-VERDICT-TINY-PROFILE-20261006.md`、`DECODE-AIC-AIV-PIPELINE-20261006.md`、`DECODE-PARALLELISM-WHAT-IS-HIDDEN-20261006.md`、`METRIC-DEFINITION-AND-CONC-LOSS-20261007.md` §4 |

---

## 7. 分支与工作目录

```bash
# 分支 feat/depstream（基线 feat/v41-dcp8@336ef05）
cd ~/projects/dsv41/depstream

# 隔离原因：main-merge 是活跃开发目录；容器挂载来自 a3-21 的 cedpd-repo，与本 worktree 无关
```

**为什么单独开分支**：

1. 这条线的结论**可能整体是负的**（k 若恒定 ≈1.7 就直接死）⇒ 不应污染 `feat/v41-dcp8` 的交付叙事；
2. 它的改动面**深**（`npu_ubatch_wrapper.py` / `dsa_v41.py` / `model_runner_v1.py` / `platform.py`
   四处），与单流固定成本线、DCP 线的文件重叠；
3. 便于用 `git diff feat/v41-dcp8` 一眼看清"这条线到底改了什么"。

---

## 8. 执行顺序（建议）

```
① U1 判据实验（tiny，1~2 天）        ← 便宜、决定性、先做
   ├─ 通过 ⇒ U2（1~2 周，高风险）
   └─ 不通过 ⇒ 关闭归档
② DepStream 依赖审计（tiny/profile，1~2 天） ← 与 U1 并行，无条件值得做
   └─ 有候选 ⇒ 挂侧流 ⇒ 精度门 ⇒ 上 tp8
```

> **DepStream 与 U1 无依赖，可以并行**；且 DepStream 即使 ubatching 全盘失败也**独立成立**。
