# DSpark 开销拆解方案（2026-10-01）

> 用户要求：**DSpark 要开**（为了整体 token 输出速度），因此必须**把它带来的开销
> 逐个拆解**：为什么会增加、以及能不能用**单算子优化**去降。

---

## 0. 先明确两个已确立的事实

### 0.1 DSpark 对 ms/step 是**明确代价**（同条件 A/B，历史实测）

| 档 | ms/**step** | A | ms/token | decode tok/s |
|---|---:|---:|---:|---:|
| 静态 `SPEC=0` | **24.35** | 1.0 | 24.35 | 41.06 |
| 静态 `SPEC=1 DRAFT_GRAPH=1`（SP_TOKENS=7） | **32.68** | 3.10 | 10.54 | 94.79 |

⇒ **ms/step +34.2%（+8.33 ms/step）**，换来 **ms/token ÷2.31**。

> 出处：`docs/CED-PD-DYNAMIC-SPEC-20260926.md` §11.4、`docs/CED-PD-SPEC-MODE-20260928.md` §6。
> ⚠️ 但这两份是 **CED-PD 拓扑（无 DCP）**。**我们自己的 DCP8 拓扑下没有 DSpark 的 profile**
> ⇒ 必须重跑（见 §3）。

### 0.2 【实测·本次】SPEC=0 的 DCP8 基线（72 个 decode step，rank0）

| op | 次/**step** | ms/step | 备注 |
|---|---:|---:|---|
| HcPre | **80** | 2.28 | = **40 层 × 2**（每层 attention 前 + ffn 前各一次） |
| GroupedMatmulSwigluQuantV2（MoE gmm1） | **40** | 2.11 | |
| QuantBatchMatmulV3 | 168 | 1.71 | |
| MatMulV2 | 94 | 1.64 | |
| SparseFlashMla | 78 | 1.46 | DCP merge 的双调用 |
| GroupedMatmul（MoE gmm2） | **40** | 1.33 | |
| **Sort** | **8** | 0.65 | DCP remap |
| MatMulV3 | 46 | 0.64 | |
| RmsNorm | 129 | 0.53 | |
| HcPost | **80** | 0.52 | = 40 × 2 |
| Cast | 382 | 0.49 | |
| … | | | |
| **设备合计** | | **19.30** | 1389.84 ms / 72 step |

来源：run `dcpcap_1001_091729`，`op_statistic.csv`（已解析）。
**这是 DSpark 拆解的基线**：凡是 DSpark 打开后**次数变化**的算子，就是它的开销。

---

## 1. DSpark 在代码里到底多做了什么

### 1.1 一个 decode step 的工作单位（历史 profiler 文档 §0）

> 一个 decode step = 1 次 **40 层 target forward（M = 1 + SP_TOKENS = 8 行）**
> + 1 次 **3 层 draft forward** + 验证

### 1.2 draft 不是"3 个小层"，而是 **3 个与 target 同规格的层**【实测·本次查 checkpoint】

`mtpq_manifest.json`（`v41-flat-verify3`）里属于 `mtp.*` 的张量：

| 组件 | 规模 | 对 target 的比例 |
|---|---|---|
| `mtp.N.ffn.experts.{w1,w2,w3}` | **384 专家 × 3 张 × 3 层 = 3456 个张量**，3.03 GiB（量化后） | **1:1** |
| `mtp.N.attn.{wq_a,wkv,wq_b,wo_a,wo_b}` | 与 target 层同结构，0.12 GiB/层 | **1:1** |
| `mtp.N.hc_attn_fn` / `hc_ffn_fn` | 与 target 同 | **1:1** |
| `mtp.N.head.weight`（markov head） | **1.26 GiB** | 无对应 |
| `mtp.N.embed.weight` | 1.23 GiB | 无对应 |
| `mtp.N.main_proj.weight`（3 个 target 层残差 → hidden） | 150 MiB BF16 | 无对应 |

⇒ **draft 的权重体量 ≈ 3 个 target 层**（约 target 的 7.5%）。
按"decode 是权重带宽 bound"的直觉，draft 应该约 +7.5%。**但实测 +34.2%。**
⇒ **差额（约 27%）必须来自别处** —— 这正是要拆的。

### 1.3 次数证据（可从 profiler 直接读）

| op | `SPEC=0`（本次实测） | `SPEC=1`（历史 CED 文档） | 差额 |
|---|---:|---:|---|
| `HcPre` | **80** = 40×2 | **86** | **+6** = **3 draft 层 × 2** |
| `GroupedMatmulSwigluQuantV2`（MoE gmm1） | **40** | **43** | **+3** = 3 draft 层各一个 MoE |
| `HcPost` | **80** | **86** | **+6** |
| `SparseFlashMla` | 78（DCP merge 双调用） | 40 | 拓扑不同，**不可比** |
| `GroupedMatmul`（gmm2） | **40** | **43** | +3 |

⇒ **draft 的"额外算子"在 profiler 里是可逐项辨认的**（层数从 40 变 43）。

---

## 2. 拆解方案（三个桶 + 一个必答项）

### 桶 A：**draft 前向**（3 个同规格层）
- 判据：`HcPre/HcPost` 从 80 → 86、`GroupedMatmul*` 从 40 → 43。
- 直接量化：把**层数**差对应的算子时间加起来。
- 单算子优化机会：draft 与 target **共用同一批 kernel**，所以邻居的 wo_a / QBMV3 优化
  **自动覆盖 draft 的 3 层**（这是"DSpark 让对称优化更值"的正面例子）。

### 桶 B：**target forward 的 M 从 1 变 8**
- 判据：**同类算子次数不变、单次耗时变大**（权重流量不变、激活/KV 流量 ×8）。
- 直接量化：对比同名算子的 `Avg Time(us)`（SPEC=0 vs SPEC=1）。
- 单算子优化机会：
  * ★ **M 补零到 16**（邻居实测 vendor matmul `M=8→14.99 µs`、`M=16→10.45 µs`，**1.43×**）
    —— 只在开了 DSpark（M=8）时才存在；
  * attention 的 KV 写入/读取 ×8 行的带宽项。

### 桶 C：**验证 / 采样 / 接受逻辑 + 3 个 head**
- 包含 `markov_head`（1.26 GiB 权重！）、`confidence_head`、`lm_head`、
  以及 vLLM 的 speculate/verify 编排（每步一组 host↔device）。
- 判据：这些算子在 `SPEC=0` 下**不存在或次数不同**。
- 单算子优化机会：
  * `markov_head` 是 vocab 级投影 ⇒ 可能可以**只对候选 token 计算**（类似 P 侧
    `corpus_mean` 的做法），而不是全 vocab；
  * `confidence_head` 很小（10 KB），但**可能带来一次额外同步**。

### ★ 必答项：**DCP 专属开销是否被 A 摊薄**
- 我之前的推断：DCP 开销是 per-step 的，会被 A 摊薄。
- **但必须实测**：draft 无 DCP（复制态），而 **verify 涉及 DCP merge**
  ⇒ `SparseFlashMla` 的次数会从 78 变成多少？merge allReduce 从 76 变成多少？
- 这决定"DCP8 开 DSpark 后每 token 的 DCP 惩罚是多少"。

---

## 3. 需要的实测（一次重启 + 两次 profile）

### 3.1 前置：现有 raw profile 情况【实测】

| run | 配置 | prof |
|---|---|---|
| `dcpcap_1001_091729` | `SPEC=0`（当前） | ✅ 已解析（**基线**） |
| `ced_d4b_0927_015401` | `SPEC=1 SP_TOKENS=7`（历史交付验收） | ❌ **`prof/` 目录是空的** |

⇒ **DSpark 的 raw profile 不存在，必须重跑。** 不能靠历史数据。

### 3.2 计划（一次重启，DSpark 单变量）

1. 重启 `dsv41-gen1`：`SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1`，其余**逐项不变**
   （`PREFIX=0 BAT_TOKENS=2048 EAGER=1 ENGRAM=0 DCP=8`）。
2. `V41_PROFILE=1` 已开 ⇒ 抓 decode 段 profile。
3. **同时测**：
   - `ms/step`（**用 `tools/ced_pd_bench.py`**，它按
     `steps = Δvllm:spec_decode_num_draft_tokens_total / SP_TOKENS` 算 step 数；
     **不能用 `dcp_perf.py`**，它按 token 间隔算，开了推测解码后失效）；
   - `A`（`Mean acceptance length`）；
   - `decode tok/s`。
4. **正确性回归（必须全过）**：T=904 针 = `Q7`、短问答 `17×23` = `391`、
   长针 2000/8000/16000 = 6/6、容量 = 6,082,458。

### 3.3 拆解的具体做法

| 桶 | 怎么从 profile 里取 |
|---|---|
| A（draft 前向） | 逐算子求 `(Count_spec1 − Count_spec0) × AvgTime` 之和 |
| B（M 1→8） | 逐算子求 `Count × (AvgTime_spec1 − AvgTime_spec0)`（Count 不变的那些） |
| C（验证/head） | `SPEC=0` 里不存在（Count=0）而 `SPEC=1` 里出现的算子 |
| ★ DCP 摊薄 | 看 `AivKernel`/`SparseFlashMla`/`Sort` 的 **次数**变化 |

**判据**：三个桶之和应≈ +8.33 ms/step（历史 CED 数字；我们的 DCP8 数字待测）。
若对不上，说明有桶漏了 —— 那就先补上再谈优化。

---

## 4. 单算子优化的候选清单（按桶）

| 桶 | 候选 | 依据 | 预估 |
|---|---|---|---|
| B | **M 补零到 16** | 邻居实测 1.43×（`12-WOA-FINAL-VERDICT.md`） | ~0.3 ms/step |
| B | `wo_a` Triton kernel | 邻居实测 1.59×，**自动覆盖 target+draft** | ~0.6 ms/step |
| B | QBMV3 三合一（wq_a/wkv/gate_up） | 邻居实测 2.196× | ~2.2 ms/step |
| A | 同上（draft 的 3 层同构 ⇒ 同批 kernel 受益） | 次数 40→43 ⇒ 收益 ×1.075 | — |
| C | `markov_head` 只算候选 token | 待验证（可能已在做） | 【未确认】 |
| C | 验证/采样编排的同步次数 | 需 profile 里看 host 段 | 【未确认】 |
| A/C | **draft 入图**（`DRAFT_GRAPH=1` 已开）是否真的消除 host 开销 | 需对比 DRAFT_GRAPH=0/1 | 【未确认】 |

---

## 5. 风险与未确认

1. **我们自己的 DCP8 拓扑下 DSpark 从未跑过** ⇒ 所有数字都要重测。
2. `dcp_perf.py` 开的推测解码后会失效（按 token 间隔）⇒ 必须换 `ced_pd_bench.py`。
3. 【未确认】draft 的 SWA 缓存组（G12，复制态 + aliasing）在 DCP 分槽下的**容量**账。
4. 【未确认】`aux_hidden_state` 取自 target 层 37/38/39 残差，DCP 下是 TP-all-reduce 后的
   复制态 ⇒ 理论没问题，但 8 卡未验。
5. 历史记录：**四并发时 DSpark 几乎没收益**（本线程早期）⇒ 高并发策略另测。
