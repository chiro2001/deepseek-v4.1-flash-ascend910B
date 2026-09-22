# 098 — ★★★★ 我给 `097 §3` 的覆盖率公式**有双计 bug**（子代理单测抓到）；并记一个**发布方向**的流程坑

> 2026-09-23 01:1x CST。执行：子代理 **`engram_repro_fix_arm`** 发现 + 修，**主代理独立核实**（读代码 + 读 md5 + 复跑测试）。
> 前置：`091`（验收总表）、`097`（撤回 0.5% 覆盖率 + 改动单）。标记：**【实测】**。

---

## 0. 一句话

我在 `097 §4` 写的覆盖率公式 —— `(absent + filled + overwrote) / planned` —— **是错的**：
`absent` 与 `filled` **记的是同一个槽位**的两个视角（"发现时缺页" / "补上了"），相加就是**双计**。
子代理的单测当场抓到：**"修复率 = 2.000"**（分子 6 / 分母 3）。

---

## 1. 【实测】双计的证据（主代理读代码核实）

`engram_repair.apply_repairs()` 的结构（`patches/engram-true-tokens/engram_repair.py`）：

```python
if first_touch[page]:                     # 本页是我们初始化的（原先是缺页）
    stats["absent"] += 1
    if mode >= 1:
        pages[page, slot] = comp          # ← 同一个槽位，被写进去
        stats["filled"] += 1
else:                                     # 页已 present（镜像里有值）
    if int(pages[page, slot]) != comp:
        stats["mismatch"] += 1
        if mode >= 2:
            pages[page, slot] = comp      # ← 同一个槽位，被覆写
            stats["overwrote"] += 1
```

⇒ ★ **两组都是"同一槽位的两个视角"**：
| 组 | "发现问题" | "真的写了" |
|---|---|---|
| 缺页类 | `absent` | `filled` |
| 陈旧类 | `mismatch` | `overwrote` |

⇒ 所以：
* ❌ 我给的 `(absent+filled+overwrote)/planned` **双计**了 `absent` 与 `filled`；
* ✅ 正确分母是 **`planned`**，而"真的写进去了"的分子是 **`filled + overwrote`**。

★ 这同时解释了 `088 §2` 里那个一直没被点破的疑点：
**`exact` 臂 `absent=0 / filled=0` 却 `mismatch=6`** ——
因为 `absent/filled` 与 `mismatch/overwrote` 统计的是**两个不相交的槽位集合**（缺页 vs 陈旧页）。

---

## 2. 【实测】修法（子代理已落，主代理复跑通过）

### 2.1 分成**两个**会分叉的问题（这一点很关键）

单测里的反例：`mismatch=3 / unavailable=0` ⇒ **"可用率=1.000 但修复率=0.000"** ——
**只给一个数会误判**（看起来像"全好了"，其实一个字节都没写）。
⇒ 现在**一次调用打两个数**：
```
[ENGRAM-PLAN-DIAG] call=N 汇总 planned=6 本次={...} ★可用率=0.667 ★修复率=0.167 不可用=2 (q_neg=0 q_ge_ntok=1 tok_neg=1)
```
| 指标 | 公式 | 回答的问题 |
|---|---|---|
| ★ **可用率** | `1 - unavailable/planned` | **真 token 拿到了吗**（拿到才可能比对/覆盖） |
| ★ **修复率** | `(filled + overwrote)/planned` | **真的写进去了吗** |

### 2.2 同时落地的四条（`097 §4` 的改动单）

| # | 内容 |
|---|---|
| ① | ★ `[ENGRAM-PLAN-DIAG]` 移到 **`apply_repairs`**，**逐计划槽位**打印（`reason ∈ {q_neg, q_ge_ntok, tok_neg, page_neg, page_oob}`），每 rank 前 `V41_ENGRAM_PLAN_DIAG`（默认 3）次调用 |
| ② | `build_prev_tok` 的 `stats["planned"]` → ★ **`scanned`**，其汇总行显式标 `★扫描口径(全表 n×lookback)`（防止再被当分母 —— 这正是 `097` 的坑） |
| ③ | `apply_repairs` 新增 **`planned` / `plan_ok` / `plan_avail` / `plan_scanned`**；`engram_hash` 的 `_TT_CUM` 一并累计 |
| ④ | `build_prev_tok` 把每个 `(row,shift)` 的 reason 编码进 `_PREVTOK_CODES`，供 plan 口径 DIAG 读取（零额外开销） |
| ★ | `[ENGRAM-PREVTOK-ERR]` 在 `mode2` 臂实测 **0 行** ⇒ 构造 `prev_tok` **没有**失败（不是"静默停用"） |

| 文件 | 新 md5 |
|---|---|
| `engram_repair.py` | ★ **`6766424c408bf391dd2853f17a4374ba`** |
| `engram_hash.patched.py` | ★ **`bf56bc134a67349ae4239056918c667d`** |
| `engram_hash.true_tokens.diff` | 已按同基线（`dc63b40b`）重生成，`patch -p1 --dry-run` 干净、应用后与 `patched.py` 逐字节相同 |

★ 主代理复跑：`py_compile` ✓ + `run_all.sh` **全绿**（含 4002 例覆盖性测试）。

---

## 3. ★★★ 顺带抓到的一个**流程坑**（比公式 bug 更值得记）

子代理把改动**直接写进了发布仓的目标文件**：

```
发布仓（dsv41-release/a2/patches/engram-true-tokens/engram_repair.py）  = 6766424c （新）
工作区（a2/agents/Engram_exactfix/patches/engram_repair.py）            = 1fe3ec88 （旧）
```

★ 而交付方向是 **工作区 → 发布仓**（`prepare_publish.sh` 的 MAP）⇒
**下一次 `PUBLISH=1` 会把子代理的改动覆盖回旧版** —— **静默丢失**，而且没有任何报错。

⇒ **处置**（主代理已做）：把发布仓那两个文件 **`cp` 回工作区**（权威源），
使两边逐字节一致 ⇒ 现在 publish 是幂等的。

⇒ ★ **纪律（写进协作约定）**：
> **改交付件要改"工作区的源头"**（`a2/agents/<agent>/patches/…` 或 `a2/patches/…`），
> **不要改发布仓里的目标文件** —— 那份是 `prepare_publish.sh` 的**产物**。
> 若为方便直接改了发布仓，**必须立刻回报**，由主代理同步回工作区（否则下次发布会静默回退）。

★ 这条与今天其它条同族（都是"**改动没落到真正生效的那一份上**"）：
`085`（配置选错代码路径）/ `081`（合并件过期）/ `093`（dsa 挂错份）⇒ 本条是**方向反了**（发布仓 → 工作区反了）。

---

## 4. 对目标的意义

| 项 | 之前 | ★ 现在 |
|---|---|---|
| "Engram 在取回前缀的历史完全正确" | 用 `090` 的 **0.5%** 量化（`097` 已撤回） | ★ 现在有**唯一、可解释**的两个指标（**可用率 / 修复率**）+ **plan 口径**的 reason 分布 ⇒ 下一次带 `TRUE_TOKENS≥1` 的臂就能给出**可信**的覆盖率 |
| `mode2` 臂的三条新事实 | — | ★ **`PAGELESS` 仍 8**（⇒ `mode=2` 修不了 `unavailable` 那类，`097 §2` 的预测【实测】成立）；★ `overwrote=3 / mismatch=3`（⇒ **能修的陈旧槽位是 3 个且都被改写**；且 `mismatch` 不是常数，要 per-arm 看）；★ 宿主 `7.130 GiB/rank`（与 `092` 逐字相同） |

---

## 5. ★ 附：final 臂与 fit 臂挂的 `engram_hash.py` **不同**（一个变量），主代理核实**行为等价**

按 `089` 的纪律（"比数前先 diff 对手的**全部**变量"），主代理发现：

| 臂 | 挂的 `engram_hash.py` | 备注 |
|---|---|---|
| `r8-4axis-fit`（3/3 那条） | `2c17545865c04d958209839cedc256cf` | 旧版（只含 `_TT_CUM`） |
| `r8-4axis-final`（正在跑） | ★ `bf56bc134a67349ae4239056918c667d` | 新版（含 `planned/plan_ok/plan_avail`） |

### 5.1 核实过程（可复算）

```
git -C dsv41-release show HEAD~6:a2/patches/engram-true-tokens/engram_hash.patched.py > /tmp/eh_old.py
diff /tmp/eh_old.py <新版>  ⇒ 31 行 diff，**新增 18 行，集中在行 42..81**
```
逐行看这 18 行的**归属**：
* `_TT_CUM = {...}`（**模块级 dict 定义**，含新增的三个键 `planned/plan_ok/plan_avail`）—— **定义本身无副作用**；
* `_engram_true_tokens_note()` 的**函数体**（注释 + 打印格式里多两个 `%.3f`）——
  ★ 该函数的**唯一调用点**在 `if repair_stats:` 分支内，而 `repair_stats` 只在
  **`_ENGRAM_TRUE_TOKENS` 为真**时才非 `None`。

### 5.2 结论

> ★ **`VLLM_V41_ENGRAM_TRUE_TOKENS=0` 时**：`_TT_CUM` 被定义但**从不被使用**（无副作用），
> `_engram_true_tokens_note()` **从不被调用** ⇒ **行为与旧版等价**。

⇒ **`r8-4axis-final` 与 `r8-4axis-fit` 在这个变量上可比**（两臂都是 `TRUE_TOKENS=0`）。
★ 但仍**如实标注**：这是**推断·强**（基于"调用点不可达"的代码事实），不是"两次跑的字节对比"。
⇒ 若 final 的结果与 fit **不一致**，第一个要查的就是这个变量（而不是先怀疑别的）。
