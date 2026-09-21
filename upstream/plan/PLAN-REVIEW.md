# 规划评审：怎么真正影响他们的开发计划

> 2026-09-21｜触发：用户指出「**很多 PR 不是社区 PR，而是他们内部团队的 PR**」
>
> 这条修正**改变了整个打法**。下面是重审结果。

---

## 1. ★ 决定性发现：RFC #16375 就是他们的开发计划，而且**公开在招人**

`[RFC]: DeepSeek V4.1 Roadmap`（#16375，作者 `weijinqian0`，label `RFC`，**open**）

### 1.1 三条硬事实（实测）

| 事实 | 数值 |
|---|---|
| 条目总数 | **50** |
| 已完成 | **0**（`- [x]` 计数 = 0） |
| **评论数** | **0** —— **一个条目都没人认领** |
| 引用它的 issue | 只有 2 个，**都是 bug 报告**，不是条目认领 |

### 1.2 ★★ 三处明确邀请（原文）

> 行号口径：`[NN]` = RFC 正文行号；快照 `pr/refs/RFC-16375-body.md`（sha256 `459c6328…`）。
> 以下三句**逐字**引自该快照。

> **第 3 行**：*"Release targets and **owners can be attached to individual implementation issues** as they are agreed."*
>
> **第 108 行**（Feedback Period）：*"This issue can remain open as the umbrella tracker;
> **contributors are welcome to propose owners, implementation issues, and target releases**."*
>
> **第 104 行**（完成标准）：*"Performance items **additionally require reproducible comparisons**;
> overlap items require **trace evidence**."*

⇒ **他们要的正是我们有的东西**：
* "owners can be attached" ⇒ 我们可以**认领条目**
* "reproducible comparisons" ⇒ 我们手上有大量同会话 A/B
* "trace evidence" ⇒ 我们有 profile（`[bneck]`、op-level 计数、stream 分析）

### 1.3 这条发现为什么推翻了我原来的框架

| 我原来的框架 | 修正后的框架 |
|---|---|
| "我们的实现 vs 他们的实现"，比谁快 | ❌ **错** —— 那是在公开质疑一个内部团队的工作 |
| 写一份"我们远超他们"的总纲 | ❌ **错** —— 会被读成 PR 宣传，且树敌 |
| **"他们的路线图有 50 个空白条目，我们能填其中 N 个"** | ✅ **对** —— 帮他们**打勾** |

> **一句话**：不要当挑战者，要当**能帮他们交付的人**。
> 内部团队最需要的东西不是"你比我强"，而是"**这 5 个条目我有数据，可以帮你标完成**"。

---

## 2. ★ 资产 ↔ RFC 条目映射（这是今晚最有价值的产出）

### 2.1 Engram 节（RFC line 46–50）—— 我们命中 **5/5**

| RFC 条目（原文摘要） | 我们已有的东西 | 证据强度 |
|---|---|---|
| **[46]** Support Engram CPU offload with CPU-resident tables, **bounded pinned-memory staging**, batched lookups, async H2D prefetch | **比他们更进一步**：表常驻 host DRAM（INT8，206 GiB），**device-index 直接索引，根本不需要 H2D**；pinned 暂存那条路我们**量过并放弃了**（见下） | ★★★ 实测 |
| **[47]** Define CPU/HBM residency & hot-row caching; **measure table footprint, lookup latency, transfer volume, NUMA/bandwidth sensitivity under realistic concurrency** | **这五项我们全有**：<br>· footprint：206 GiB（4 文件：11.4/91.6/11.4/91.6）<br>· lookup latency：hash `0.427→0.076`、plan `0.261→0.068`（numba JIT）<br>· transfer volume：`d2h` 0.19–3.4 ms、`route` 1.34–2.89 ms 逐 rank<br>· **A2 vs A3 的机型差异**（`host_mem_pool` 判据）<br>· 真实并发下的 profile | ★★★ 实测 |
| **[48]** Engram TP with explicit ownership; **distinguish node-level table sharding from model TP** | `LOCAL_OWNER=fast`：把整表广播/分片改成**本地全表 + 设备索引**；我们**量过分片注册的代价**（a2a+bcast ≈0.5 ms，净收益上限仅 ~6%） | ★★★ 实测 |
| **[49]** **Eliminate unnecessary duplicate queries across TP ranks** | **这正是 `local-owner` 做的事** —— 省掉 metadata `all_gather` 与 ids `all_to_all`；`route` 从 **2.462 → 0.058 ms** | ★★★ 实测 |
| **[50]** Validate offload/TP with SP, DCP, PD, and **graph replay** | **graph replay 我们有**（device-index 入图）；SP/DCP/PD 未覆盖 ⇒ **诚实标注** | ★★ 部分 |

### 2.2 图执行节（RFC line 73–77）—— 我们命中 **2.5/5**

| RFC 条目 | 我们 | 强度 |
|---|---|---|
| **[73]** Establish V4.1 **ACLGraph decode support**, explicit eager fallback rules | **DSpark draft 入图四件套 + 明确的回退规则**（adapter 不支持 / 动态 spec / 非 FULL_DECODE_ONLY ⇒ 回退 eager） | ★★★ |
| **[75]** Integrate `npugraph_ex` + static-kernel optimizations for **Engram integration** and MoE | `NPUGRAPH_EX=1` + `STATIC_KERNEL=1` + Engram device-index 入图 | ★★★ |
| **[77]** ★ **Define the eager/graph boundary for CPU Engram lookup and dynamic communication metadata; keep host synchronization OFF the captured path** | **这条是我们的 0009 的核心命题**：同步 host 时间 **3.379 → 0.058 ms**；并给出"哪些必须留在图外"的边界定义 | ★★★ **命中靶心** |
| [74] [76] | 无（TopK/compressor 状态稳定性、multistream 图兼容） | — |

### 2.3 融合算子节（line 90–91）—— 我们命中 **2/2**

| RFC 条目 | 我们 |
|---|---|
| **[90]** Optimize **Engram gather/dequantization/gating fusion** | 0006（分块 gate，去掉 2048 行 padding）+ 0008（JIT） |
| **[91]** Validate numerical accuracy, non-contiguous strides, empty/padded batches; **benchmark both individual kernels and the full pipeline** | 我们有 op-level profile **和** e2e 两个口径 |

### 2.4 MoE 节（line 63–69）—— **需要重新措辞**

⚠️ RFC 的硬件矩阵写的是：**A2/A3 = W8A8，A5 = W4A8**。而我们在 A2/A3 上做的是 **W4A8**。

**修正**：我们的 0001（MoE AllGather）与 0002（mask 范围比较）**都是量化无关的**——
它们改的是通信方式与掩码计算，**W8A8 同样受益**。所以：

* **不要**把它们挂在"W4A8"名下（那会撞上他们的矩阵）
* **要**把它们挂在 **[63]"Enable and tune the routed-expert W8A8 paths on A2/A3"** 与
  **[65]"Compare against existing collective strategies … document selection rules"** 名下
* 我们的 W4A8 配方（msmodelslim）**单独说明**是"超出当前矩阵的实验"，不要求他们接受

### 2.5 验证节（line 97）—— 完全命中

> **[97]** *"Publish reproducible TTFT, inter-token latency, throughput, HBM usage,
> **Engram transfer cost**, and communication/compute-overlap measurements against the corresponding baseline."*

⇒ 我们**每一条都有**（TTFT、ms/step、tok/s、HBM、Engram transfer cost、overlap 分析）。

---

## 3. 当前规划的四个缺口

### 缺口 ① 框架错位（最严重，已在上文修正）

原计划的 D12"总纲"写的是 `UPSTREAM-GAP-ANALYSIS.md`（**差距分析**）——
这个名字本身就在暗示"我们比你们强"。**必须改名改框架**：

| 原名 | 改为 |
|---|---|
| `docs/UPSTREAM-GAP-ANALYSIS.md` | **`docs/RFC-16375-CONTRIBUTION.md`**（**对 RFC 的贡献报告**） |
| 八节结构：第 3 节"函数级正面对比" | 改为**"可复现对比（RFC 要求的 reproducible comparisons）"** |
| 第 6 节"鲁棒性：他们崩、我们稳" | 改为**"NPC-style 边界数据：并发 24–32 的失败与我们的规避方案"** —— 定位成**帮他们 debug**，不是打脸 |

### 缺口 ② 没有"认领"这个动作

RFC 说 "**owners can be attached**"。原计划只在评论区给数据，**没有明确认领条目**。

**应改为**：每条数据都**显式对应 RFC 的一条**，格式：

```
Re: RFC #16375 — Engram memory management

[47] "measure table footprint, lookup latency, transfer volume, and NUMA/bandwidth
      sensitivity under realistic concurrency"

We have measured four of the five requested quantities on A2 (8×910B3) and A3 (8×910C):
  · table footprint : 206.0 GiB (4 shards: 11.4 / 91.6 / 11.4 / 91.6)
  · lookup latency  : hash 0.427 → 0.076 ms, plan 0.261 → 0.068 ms (after JIT)
  · transfer volume : d2h 0.19–3.41 ms, route 1.34–2.89 ms, per-rank
  · machine split   : A2 host_mem_pool=0 (PCI 19e5:d802) vs A3 =1 (19e5:d803)
  · NUMA/bandwidth  : <我们有的/没有的，诚实写>

Raw numbers: <链接到我们的仓库>
We can attach these as the completion evidence for this item if useful.
```

**关键在最后一句** —— 把决定权留给他们。

### 缺口 ③ 没有"投稿窗口"的概念

RFC 的 **Feedback Period** 写着 "**At least one week after opening**" —— 虽然已经过了，
但**它仍然 open 且 0 评论**，说明窗口事实上还开着。

**应利用这点**：第一份沟通**直接发在 RFC 上**（而不是新开 issue）。
好处：① 留下公开记录；② 让所有 CC 的人看到；③ 不需要 maintainer 先 approve 一个 issue。

> ⚠️ 但这**同样需要你授权才能发**。我今晚只准备草稿。

### 缺口 ④ 没有考虑"他们的激励"

内部团队为什么要理我们？**因为 50 个条目全空，而他们要交差。**

我们提供的价值应该是：
1. **可复用的证据**（他们不用自己跑 A2/A3 对比）
2. **可复用的工具**（单卡探针、对比 harness）
3. **他们 bug 的根因判据**（#16828 的 `host_mem_pool`）

而**不是**："我们的实现比你们的好"。

---

## 4. 修正后的打法（三层）

### 第 1 层：立即可交付的"帮助他们打勾"的证据包

| 交付物 | 对应 RFC | 形态 |
|---|---|---|
| **L1-A** | **[47]** | CPU/HBM residency + 五项测量（footprint/latency/volume/机型/并发） |
| **L1-B** | **[77] ︎** | eager/graph 边界定义 + 同步 host 时间 3.379→0.058 ms + 哪四件事必须在图外 |
| **L1-C** | **[46][48][49]** | local-owner 机制 + 分片注册的代价测算（为什么整表索引更优） |
| **L1-D** | **[90][91]** | gate 分块 + JIT 的 op-level 收益与逐位一致性 |
| **L1-E** | **[63][65]** | MoE AllGather 与 mask 范围（**量化无关**的措辞） |

**统一形态**：每条 = **RFC 条目引用 + 数字 + 一条可复跑命令 + 我们仓库的链接**。

### 第 2 层：函数级可复现对比（原 H2，**改名改框架**）

仍然做（它是"reproducible comparisons"的最强形态），但定位改为：

> **"RFC #16375 要求 reproducible comparisons，这里是我们的"**
>
> 而不是"我们的实现比你们的快"。

具体：`bench/engram_gate_head2head.py` —— 上游代码**逐字照抄**（含行号来源），
两边背靠背跑，给 时间/峰值显存/`torch.equal`。

### 第 3 层：**代号化的"轨道路线"**（让他们能规划我们）

这是**新的想法**：不要给 11 个零散补丁，而给**三条轨**，每条对应 RFC 的一组条目：

| 轨道 | 内容 | 对应 RFC | 我们的状态 |
|---|---|---|---|
| **轨道 A：Engram host 路径** | host 常驻 + 设备索引 + 入图 | [46][47][48][49][50][77] | ✅ A2/A3 实测 |
| **轨道 B：MoE 通信与掩码** | AllGather + 范围比较 | [63][65] | ✅ 实测（量化无关） |
| **轨道 C：图执行边界** | DSpark 四件套 + npugraph_ex | [73][75][77] | ✅ A2/A3 实测 |

**为什么这有用**：他们说 "owners can be attached to **individual implementation issues**" ——
**三条轨 = 三个 implementation issue**，正好是 RFC 认领的单位。

---

## 5. 修正后的执行清单（替换原 OVERNIGHT-PLAN 的对应项）

| 原编号 | 改动 |
|---|---|
| **D12** | 改名：`docs/UPSTREAM-GAP-ANALYSIS.md` → **`docs/RFC-16375-CONTRIBUTION.md`**；八节按 §4 第 1 层重写 |
| **D13**（H2） | 定位改为"**RFC 要求的 reproducible comparison**"；脚本名保留 `head2head` 但注释改成中性 |
| **新增 D14** | **三条轨的 implementation issue 草稿**（对应 RFC 的认领格式） |
| **新增 D15** | **RFC #16375 评论草稿**（按 §3 缺口② 的格式，一条一条对） |
| D3（MoE mask） | 措辞改为**量化无关**，挂 [63][65]，不挂 W4A8 |
| D8（#16828 评论） | **保留**，且与 RFC 评论**同一套证据**，只是受众不同 |

---

## 6. 还剩什么没想清楚（诚实列出）

1. **我们能否拿到 A2 复测？** RFC [47] 要 "NUMA/bandwidth sensitivity"，
   我们 A2 的数据是**单机 8 卡**、A3 是 **8 卡**；但 A3-node1 是 16 逻辑 die。
   口径是否能对上他们的期望，**不确定**。
2. **W4A8 的处理**：我倾向"单独说明、不要求接受"。但如果他们反问
   "你们为什么不做 W8A8"，我们需要一个答案（现实是：我们的权重就是 W4A8）。
3. **是谁拍板**：RFC 作者 `weijinqian0` 大概率是内部规划者，但**谁是 Engram 条目的 owner 不确定**。
   可能需要先问一句"who owns the Engram items?"。
4. **中国区 vs 海外**：仓库是国际项目（英文），但 V4.1 这条线明显偏中国团队。
   沟通语言、时区、review 节奏都要按这个来（我们本来就用英文写 PR，没问题）。
5. **要不要 CC 特定人**：RFC 的 CC List 是泛化的（没有具体 @）。
   是直接在 RFC 下评论，还是 @ 上 #16925/#16828 的作者？**建议先 RFC，再在被引用的 issue 里补链接。**

---

## 附：本次评审用到的原始侦察

```
RFC #16375          : 50 条目 / 0 完成 / 0 评论 / label=RFC / open
                      快照 pr/refs/RFC-16375-body.md (sha256 459c6328…, 119 行)
  line 3            : "owners can be attached to individual implementation issues"
  line 97           : "Publish reproducible TTFT, inter-token latency, throughput, HBM usage,
                       Engram transfer cost, and communication/compute-overlap measurements"
  line 104          : "Performance items additionally require reproducible comparisons;
                       overlap items require trace evidence"
  line 108          : "contributors are welcome to propose owners, implementation issues,
                       and target releases"
  ⚠️ 早先这里写的是 line 8 / 117 / 111 —— 三处都对不上原文，已于 2026-09-21 03:1x 修正
     （见 logs/14-20260921-rfc-citation-audit.md）
  Engram 节         : line 46-50（5 条，全空）
  图执行节          : line 73-77（5 条，全空）
  融合算子节        : line 90-91（2 条，全空）
  硬件矩阵          : A2/A3=W8A8, A5=W4A8

参考 RFC           : #13452 [RFC]: Kimi K3 Roadmap（maoxx241，3 条评论）
引用 #16375 的     : 只有 #16419 / #16828（都是 bug 报告，非条目认领）
```
