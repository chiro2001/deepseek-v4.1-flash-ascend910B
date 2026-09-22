# 059 — ★★ 路线级发现：**②a（draft INT8）在两个档上都优于 ②c，且档 D 达 ×1.9126 ≥ 原始的 ×1.84 目标**

> 2026-09-22 13:4x CST。执行：**主代理**（纯算术 + 读源码，**不占卡**）。
> 起因：横向对账 `050`/`051`/`054` 的容量数字时，顺手把"②a vs ②c"放进同一张模型里算了一遍。
> 标记：【实测】/【算术】/【推断】/【未确认】。

---

## 0. 一句话

**`050` 把 ②c（draft BF16 + block 64）列为序 1、把 ②a（draft INT8）列为序 2，理由是 ②a 被
"图捕获期 `.item()`"挡住 —— 而那个阻塞正是 `049` 修掉的那一个。**
用已验证的零参数模型算：**②a 与 ②c 的 Σ（页容量）完全相同，但 ②a 保持 block=128 ⇒ BPR 更小**
⇒ **②a 严格优于 ②c（+5.2% token）**，且 **档 D = `817,898` = ×1.9126 ≥ 原始目标 ×1.84**。
⇒ ★ **②a 应当升为序 1**（`D_draftINT8` 已在 c1 上开始验）。

---

## 1. 算术（【算术】，全部由 `050 §1.6` 的零参数模型给出）

模型（`050`，9 个实测点逐字命中；`C2_draft64` 的 `c2_model.py` 是它的可复跑实现）：
```
Σslot_pages = 3 × max(kv+index(ratio2), state, swa, draft) + max(kv+index(ratio1), swa)
BPR         = cdiv(max_len, 128) + 1 + 10 × P(128) + P(draft_block)      ← draft 关时末项 = 0
P(b)        = cdiv(min(window−1 + max_in_flight, max_len), b) + 1
num_blocks  = avail_bytes // Σslot_pages − 1
tokens      = int(num_blocks / BPR × max_len)
```
8 卡口径（`avail = 4 GiB`，`max_len = 133120`）：

| 路线 | draft 页 | block | **Σslot_pages** | **BPR** | **tokens** | vs 档 B |
|---|---:|---:|---:|---:|---:|---:|
| 档 B（基线） | — | — | 540,928 | 2471 | 427,643 | ×1.0000 |
| 档 D（基线） | 131,072 | 128 | 476,416 | 2471 | 485,610 | ×1.1356 |
| **②c** draft BF16 | 65,536 | **64** | **282,880** | **2600** | 777,318 | ×1.8177 |
| ★ **②a** draft INT8 | **66,560** | **128** | **282,880** | **2471** | ★ **817,898** | ★ **×1.9126** |

★ **关键一行**：两者的 `Σ` **完全相同（282,880）** —— 因为档 D 的 slots 0–2 由
`max(kv2 41,600, state 65,536, swa 66,560, draft)` 决定，②a 的 draft 页 **66,560** 与 ②c 的 **65,536**
**都顶不过/顶平 `swa=66,560`** ⇒ 页容量落点一样。
**但 BPR 不同**：②a **保持 block=128** ⇒ `P_draft = 130`；②c 的 block=64 ⇒ `P_draft = 259`
（窗口 128 跨 2 块）⇒ BPR 2471 vs 2600。
⇒ ★★ **②a 用"更小的 BPR"赢，与页容量无关** —— 这正是 `051` 自己标出来的那个副作用，②a 天然没有。

档 C 同理（Σ 由 `max(kv2 73,856, …)` 决定，两者都 = 369,280）：
**②a = 626,488（×1.4650）vs ②c = 595,404（×1.3923）** ⇒ ②a 同样占优。

> ★ 与 `050` 的原始数字对账：②a 档 C `626,419` / 档 D `817,746` —— 与我这里差 **0.01%**（`int()` 取整口径），
> ⇒ **两份独立实现的算术一致**。

---

## 2. ★★ ②a 的"阻塞"已经被 `049` 修掉了（【推断·强】，正在实测）

`050` 给 ②a 的判词是"**不能单独上**"：
> draft 是 **non-causal multi-token decode**，q_len=6 ⇒ 走 `kv8_ori_plane` 的
> **prefill 分支（带 `.item()` 宿主同步）** ⇒ **图捕获期必炸** = `048` 的 `S_graphfix` 阻塞。

而 `049` 改的**就是这个判据**。`a2/publish/kv8-graphsafe/dsa_v41.py:548-553` 的注释逐字写着：
```
[S_graphfix] Capture-safe rebuild for decode-shaped batches, **including speculative
decoding**, where every request carries ``1 + num_spec_tokens`` query rows instead of one.
The old predicate (``query_rows == num_reqs``) only recognised the pure-decode shape, so a
spec-decode batch fell into the eager prefill branch below, whose ``.max().item()`` syncs
the stream and aborts graph capture (Not_Supported(EE1016), logs/048).
```
判据本身（`:500-505`）：
```python
bound = min(max(max_query_len, 0) or query_rows, query_rows); bound = max(1, bound)
if num_prefills > 0 and not _sg_is_capturing():
    return False, None          # 真·eager prefill 才走 legacy 分支
return True, bound              # ★ 其余（含 spec-decode 的 q_len=6）走 capture-safe 上界分支
```
⇒ **②a 面对的那个条件（q_len=6 的 spec-decode 批）现在落进的是上界分支，不是 `.item()` 分支。**

★ 旁证（【实测】，来自 `053` 的单 die 判定）：
`[SG-PPR] native_attention capturing=True num_reqs=32 query_rows=192 num_prefills=0 max_query_len=6 ... rows_bound=6`
—— **q_len=6 的捕获期批确实走上了新分支**。

⚠️ **未确认的部分**：上面是**目标模型**的 trace。**draft 模型自己那次 `kv8_ori_plane` 调用**
是否也满足 `num_prefills == 0`、以及 draft 的 `metadata.swa` 是否带 `max_query_len` ——
**必须用真机臂验**，不能只看目标模型的证据。⇒ 已派 `D_draftINT8`（c1）。

---

## 3. ① 改动面（`050` 给的 3 处，待 `D_draftINT8` 核源码）

| # | 位置 | 动作 |
|---|---|---|
| 1 | `DeepseekV41DraftSWASpec.__post_init__` | **放行 int8 + scale**（现在硬要求 BF16 ⇒ raise） |
| 2 | `dspark.py::DeepseekV41DSparkSWACache.get_kv_cache_spec` | 给 `dtype=int8, scale_dim=4, scale_dtype=float16` |
| 3 | `plan_cache_slots` 的 draft 检查 | `050` 说**已满足**（draft 66,560 ≤ slot 66,560）—— **仍需实测确认** |

★ 读路径**理论上自动复用** KV8 SWA（`reshape_cache` 用 `getattr(spec,'scale_dim',0)` 决定是否返回 tuple）
—— **要用探针证实，不能只读代码**。

---

## 4. ★★ ②a 的真正风险：**它动的是"投机解码"本身**（用户的硬约束）

用户 2026-09-22 明确：**必须保留投机解码**。②a 与 ②c 在这一点上**风险等级不同**：

| | ②a（draft INT8） | ②c（draft BF16 + block 64） |
|---|---|---|
| draft 的 KV dtype | **改了**（BF16 → INT8+scale） | **没改**（仍 BF16） |
| 对输出分布 | 理论上**不改**（只影响 draft 的**提案**，target 仍用 BF16 全精度验证） | 不改 |
| **对接受率** | ⚠️ **可能有影响** ⇒ **必须做接受率 A/B** | 理论上无影响；`054` 已实测"提案序列逐条相同" |
| 副作用 | 无（block 不变） | ★ DRAM 池 **+5.0%**（`sw_chunks` 1→2）；且 BPR 变大 |

⇒ ★ **判据**：`SpecDecoding` 四项（`MeanAccLen` / `AvgDraftAcc` / `per-pos` / `Accepted|Drafted`）**相对同几何 BF16 基线不许回退**，
且 **`max_tokens ≥ 64`**（`max_tokens=1` 只有 ~15 个 drafted token，**无统计功效**；`S_graphfix` 已在 `049` 里踩过并写死"不用绝对值"）。

---

## 5. 建议的路线顺序（**取代 `050`/`051` 的"②c 序 1"**）

```
序 1  ★ ②a（draft INT8，block 128）—— 期望 ×1.9126（C）/ ×1.9126（D），**超过原始 ×1.84 目标**
       前置：单 die 验 Q1（图捕获）/ Q2（容量）/ Q3（接受率）
序 2     ②c（draft BF16，block 64）—— ×1.3923（C）/ ×1.8177（D）；**②a 失败时的回落**
       现状：单 die 三问已过（054），8 卡端到端在 c0 排队
```
★ **②c 的 8 卡臂不必取消** —— 它是 ②a 失败时的保险，且两者**共用同一套判据与 harness**（可互相校准）。
但**交付选型要等 ②a 的 Q1/Q3 出来再定**。

---

## 6. 诚实边界

1. ★ **本日志全部是【算术】+【推断】**，**没有任何新的实测**；
2. ②a 的**容量预测（817,898 / 626,488）尚未在真机上验过**；
3. ②a 的**图捕获是否真的解除阻塞【未确认】**（`D_draftINT8` 在验；若失败则 ②a 仍被挡、②c 仍是唯一路线）；
4. ②a 的**接受率影响【未确认】** —— 这是它相对 ②c 唯一的实质风险；
5. `050` 说"`plan_cache_slots` 的 draft 检查已满足"**未经我复核**。

## 7. 交付

| 件 | 位置 |
|---|---|
| 本日志（路线级发现 + 算术） | `a2/logs/059-20260922-draft-int8-roadmap.md` |
| 零参数模型的可复跑实现（9 点逐字命中） | `a2/agents/C2_draft64/scripts/c2_model.py`（`C2_draft64` 交付） |
| 正在实测 ②a 的子代理 | `D_draftINT8`（c1）⇒ 将产出 `logs/056` |
