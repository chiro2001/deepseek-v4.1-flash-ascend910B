# ubatching / AIC∥AIV 线：后续怎么做（2026-10-07）

> 本文件是 `feat/ubatch-aic-aiv` 分支的**开篇计划**。
> 基线 `feat/v41-dcp8@336ef05`；分支/worktree 见 §7。
> 结论标【实测】/【推断】/【未确认】。

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

## 3. U3：★ 不依赖 U1 的主攻方向 —— Path A「无拆批的重叠」

**这是我建议真正投入的方向。**

### 3.1 它和 DBO 的区别

| | DBO / G2 / G3 | **Path A（G1 的无拆批形态）** |
|---|---|---|
| 拆批？ | 是 ⇒ 付 k=1.70× | **否** ⇒ **不付拆批代价** |
| 精度风险 | 拆批改变数值 ⇒ 要过四道门 | 不动数值 ⇒ **风险低** |
| 机制 | 通信点主动让出 | 把**无依赖的 AIV 算子**挂到侧流 |
| 上界 | ≤1.10×（G2） | **4.2 ms** |
| 先例 | 无（净亏 42%） | ✅ **有**：`AivKernel`（engram wkv all_gather）**4.11 ms 已 100% 与 HCCL 并行** |

### 3.2 现状覆盖【实测】

主流（stream 109）busy 只有 51%，其空闲被 5 条侧流填掉 **7.89 ms/步（20%）**：

| 侧流 | ms/步 | 内容 | 算不算真重叠 |
|---|---:|---|---|
| stream 105 | 2.55 | `Cast` + `IndexSelect` + `Matmul` | 部分用 AIC |
| **stream 108** | **1.89** | **engram `AivKernel`（all_gather）** | ✅ **是**（真正藏进通信） |
| stream 47 | 1.73 | `FillScalar` + `SelectV2` + `DivMods` | ❌ 只是填坑 |
| stream 35 | 1.02 | `SparseFlashMlaMetadata`（AICPU） | ❌ 填坑 |
| stream 110 | 0.84 | `Remainder` + `IndexSelect` | ❌ 填坑 |

**⇒ 真正被掩盖的只有 engram 那 1.89 ms；`通信 ∩ AIC = 0.000` ⇒ 集合通信这块完全空白。**

### 3.3 第一步（1~2 天，纯分析，不动服务）

**依赖审计**：对主流上每个 AIV 算子，判定"它的输出是否有 AIC 消费者、是否在关键路径上"，
产出**可挂侧流的候选清单**（含各自可回收 ms 与依赖证据）。

* 已有工具：`tools/prof_chain_core.py`、`prof_chain_blocks.py`、`prof_stream_ident.py`、
  `prof_consumers*.py`、`prof_overlap_who.py`
* 候选族的先验（来自 `DECODE-AIC-AIV-PIPELINE` §3）：
  `HcPost`(AIV, 1.401 ms)、`InplacePartialRotaryMul`(1.285)、`ScatterNdUpdateSk`(1.180)、
  `RmsNorm`(2.012)、`DynamicQuantV2`(0.747)
* ⚠️ 但**同层内 `RmsNorm→Quant→Matmul` 是硬依赖**，能不能挂必须先证依赖（这正是审计要回答的）

### 3.4 判据

> 候选可回收 **≥ 0.5 ms** 且**答案稳定性门通过**（`walk_blocks` 逐位门已证不适用，用答案稳定性门）。

---

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
# 分支 feat/ubatch-aic-aiv（基线 feat/v41-dcp8@336ef05）
cd ~/projects/dsv41/ubatch-aic-aiv

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
② U3 依赖审计（tiny/profile，1~2 天） ← 与 U1 并行，无条件值得做
   └─ 有候选 ⇒ 挂侧流 ⇒ 精度门 ⇒ 上 tp8
```

> U3 与 U1 **无依赖**，可以并行；且 U3 即使 ubatching 全盘失败也**独立成立**。
