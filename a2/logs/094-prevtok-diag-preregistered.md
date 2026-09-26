# 094 — 预注册：`unavailable` 的三个候选机制与各自的判别性预测（DIAG 落地前就写死）

> 2026-09-23 00:5x CST。执行：**主代理**（读源码 + 已有臂读数，**在拿到 DIAG 之前**写下预测）。
> 目的：避免"事后编解释"——DIAG 一落地就能**逐条接受/否决**。
> 标记：**【实测】/【推断】/【待验】**。

---

## 0. 已知的硬事实（先说清楚，后面全靠它们）

| # | 事实 | 出处 |
|---|---|---|
| F1 | `exact` 臂 `calls=4`：计划 **24**、`unavailable` **18**、★ 拿到真值 **6**、`mismatch` **6** | `logs/088` |
| F2 | `exact` 臂 `calls=200`：计划 **1200**、`unavailable` **1194**、★ 拿到真值**仍是 6** | `logs/090` |
| F3 | ⇒ **前 1–2 次调用把 6 个全拿到了；其后每一次都是 0** | F1+F2 的算术 |
| F4 | `plan_repair_slots` 的**代价上界**：每请求 ≤ 6 槽位（`lookback=4`） | `engram_repair.py:44-59` |
| F5 | `bneck` 实测 `n=6 padded=6` ⇒ 每步是 **1 个请求 × 6 行** | 两臂的 `bneck` 行 |
| F6 | ★ **上游语义**：`num_tokens_no_spec[req_idx] = end_idx`，而 `end_idx` 是写进 `token_ids_cpu` 的下标 ⇒ **`num_tokens_no_spec` = 该请求"不含投机的已提交长度"** | `npu_input_batch.py:109-115` + `model_runner_v1.py:2902` |
| F7 | `positions` = 本步这 6 行的位置；`q = positions[row] - sh` | `engram_repair.py:222` |

---

## 1. 先做一个**纯算术推论**（很关键，它能排除掉一个直觉假设）

设本步首位置 = `p0`（= 该请求已提交长度 ⇒ 由 F6 **`ntok = p0`**，或 `p0+1` 若已含本步那枚非投机 token）。
`positions` = `p0, p0+1, …, p0+5`；`plan_repair_slots` 只保留 `r < sh` 的槽位 ⇒ 只有
`(r=0,sh=1,2,3) (r=1,sh=2,3) (r=2,sh=3)` 这 6 个，且它们的 `q = p0-1 / p0-2 / p0-3`（**全部 `< p0`**）。

⇒ ★★ **若 `ntok = p0`（或 `p0+1`），则这 6 个槽位全部满足 `q < ntok` ⇒ 全部应"拿得到"**，
**不可能出现 1194/1200 的不可用**。

⇒ 所以 **`q_ge_ntok` 这个直觉假设被算术**排除**（除非 `ntok` 远小于 `p0`）**。
这与"我原本最怀疑 `q >= ntok`"相反 —— **幸好没直接下结论**（`logs/090 §2` 就是怕这个）。

---

## 2. 三个候选机制 + 各自的判别性预测（**待 DIAG 验证**）

### 候选 A：`ntok` 明显小于 `p0`（口径/索引错）
**机制**：`num_tokens_no_spec[row]` 取到的不是"该请求的已提交长度"，而是一个**小得多的数**
（例如被重置、或索引到了别的槽位）。
**预测**：DIAG 的 `q_ge_ntok` 占**绝大多数**，且 `q - ntok` 会**很大**（≈ `p0` 量级，不是 0–3）。
★ 判别要点：**看 `q - ntok` 的绝对值** —— 若只有 1–5，说明是"差一点点"；若几百/上万，说明是口径错。

### 候选 B：`tok_neg`（占位符）
**机制**：`token_ids_cpu[row, q] < 0`（`PLACEHOLDER_TOKEN_ID` 之类）⇒ 被跳过。
**预测**：DIAG 的 `tok_neg` 占绝大多数，且 `tok` 恒为一个固定负值。
★ 为什么可能：投机解码的 verify 步里，**投机 token 位置**可能还没被写进 `token_ids_cpu`。

### 候选 C：★ **`token_ids_cpu` 的行宽/行号与实际不符**（返回 `None` 而非计数）
**机制**：`q` 超出 `token_ids_cpu.shape[1]`（或 `row` 越界）⇒ 抛 `IndexError` ⇒
被 `_engram_build_prev_tok` 的 `except Exception: return None` **整段吞掉** ⇒
**这一调用根本不该产生 `unavailable` 计数**。
**预测**：DIAG **一行都不打**（因为根本没进 `build_prev_tok`），而 `ENGRAM-TRUE-TOKENS` 仍可能为 0。
★ 这条与 F1/F2 **矛盾**（我们有计数）⇒ 先验上**应该被排除**；留着是因为它能解释"前 6 个之后全 0"。

---

## 3. 还有一个候选 D（**专门解释 F3 的"前 6 个之后全 0"**）

### 候选 D：**行序/请求序**在稳态下不再对得上，但**校验失守**
**机制**：`_engram_build_prev_tok` 的 fail-closed 校验是
`ncomp[r] == first`（`ncomp` = 模型自己发布的 `num_computed`）——
若这个校验**通过**但 `row→token_ids_cpu` 的行号映射其实不同，则 `q` 会落到**错误但合法**的位置，
拿到**错的 token**（那会表现为 `mismatch` 而不是 `unavailable`，**与 F2 不符**）⇒ 先验弱。
**预测**：DIAG 的 `reason` 分布**不集中**，且 `mismatch` 会随时间增长 —— **与 F2（mismatch 恒 6）矛盾**。

⇒ ★ 综合：**候选 A 与 B 是主嫌**，而判别它们只需 DIAG 里 `q_ge_ntok` vs `tok_neg` 的**占比**。

---

## 4. 我对结果的**明确预测**（写死，方便打脸）

> 【推断·强，待 DIAG 验证】**`tok_neg` 会占绝大多数**，而不是 `q_ge_ntok`。
>
> 理由：§1 的算术证明 —— 在 `ntok = p0` 或 `p0+1` 的口径下，那 6 个 `q` 全 `< ntok`，
> **`q_ge_ntok` 在数学上不可能发生**；既然实测有 1194 个不可用，那它就**必须**来自另两类。
> 而候选 C 会表现为"完全没有计数"（与 F1/F2 矛盾）⇒ 只剩 **`tok_neg`**。

★ 若 DIAG 显示 `q_ge_ntok` 占多数，则说明 **§1 的算术前提错了**（`ntok ≠ p0`），
那要回头查 `num_tokens_no_spec` 的**实际取值**（DIAG 会打印 `ntok`）。

---

## 5. 修法预案（按 DIAG 结果分支，**先写下来**）

| DIAG 结果 | 根因 | 修法 |
|---|---|---|
| `tok_neg` 占多数 | 投机 token 的位置在 verify 步还没写进 `token_ids_cpu` | ① 只对**非投机行**做修补（`r < 1`？需确认行序）② 或让小批 (`n == 1+sp`) 时**跳过**（承认 pad 兜底）③ 或把"已发布长度"放宽到**含投机** |
| `q_ge_ntok` 占多数 | `ntok` 口径/索引错 | 改 `num_tokens` 的来源（用**含本步**的计数），或改 `q >= ntok` 的判据为 `q >= token_ids_cpu.shape[1]` |
| 一行都不打 | 候选 C：`except` 吞了异常 | 把 `except` 改成**计数 + 首次打印**（现在是静默 `return None`）——★ 这本身就是个**静默降级**，该修 |

★ **无论哪种结果**，都要求把 `_engram_build_prev_tok` 的 `except Exception: return None`
改成**至少计数并打印一次** —— 现在它会把"发布失败/索引越界"整段吞掉，
那正是本仓反复出现的失败模式（`079 §3` 同族：**沉默 ≠ 没跑**）。
