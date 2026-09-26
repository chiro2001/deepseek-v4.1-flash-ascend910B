# 075 — ★★★ `ENGRAM=1` × 卸载 的 P0 修复：**缺页不再 `KeyError`，降级为 barrier（pad 历史）**

> 2026-09-22 21:0x CST。执行：**主代理**（本机；代码级定位 + numba 真编译 + 两个 selftest case）。
> 上游病灶：`073`（判决）/`072` §3.3（anchor）。标记：**【实测】/【推断】/【未确认】**。
> ★ 红线：只改 `patches/files/{engram_hash.py,engram_jit_kernel.py}`，未碰 `upstream-v41/`，未占卡。

---

## 0. 一句话

`ENGRAM=1` + 卸载时 replay 轮必崩的那个 `KeyError: 2486`，根因是 **Engram 的 page 镜像
只由"流经 `update()` 的 token"写入**；被卸载池**取回**的前缀块从未流经 `update()` ⇒
命中边界上（block 对齐）第一个 token 需要的 lookback 前 1–3 个位置**必然**落在已取回的旧块里
⇒ `page_present == 0` ⇒ JIT kernel 把页号上报 ⇒ `raise KeyError(err)` ⇒ **引擎死**。

⇒ 修复：**把"镜像缺页"定义成与「槽位 = -1」完全同义的 barrier** —— 该行该 shift 起保持
`pad_id`，**哈希照常计算**，缺页只通过计数器 `miss_rows` / `pageless_history_rows` 上报。

---

## 1. 【实测】机制：为什么是"必然"而不是"偶发"

| 事实 | 出处 |
|---|---|
| `engram_max_ngram_size = 4` ⇒ **lookback = 4**（要 `p, p-1, p-2, p-3` 四个 token） | `~/models/out/v41-*/config.json`（A3 上读的） |
| 崩溃请求：`num_computed_tokens=56320`、`num_scheduled_tokens=8064` | `logs/072` §3.3（`dump_input` 原文） |
| `56320 / 32 = 1760` ⇒ **命中边界是 block 对齐的**（前缀缓存本来就是块粒度） | 同上 |
| ⇒ 第一个续算 token（位置 56320）的 `sh=1..3` 需要位置 56319/56318/56317 ⇒ 块 **1759**（上一个块，**已取回**） | 算术 |
| ⇒ 该块在镜像里 `page_present == 0` ⇒ 走标量回退 ⇒ `err_page = 2486` ⇒ `raise` | `engram_hash.py:463`（修复前） |

★ 与 `ENGRAM=0` 的对照（`p1b` replay 全过）一起看：**唯一变量就是 Engram 有没有这个镜像**。
★ 为什么 fill 轮没事：fill 是完整 prefill，所有 token 都流经 `update()` ⇒ 镜像齐。

---

## 2. 改了什么（两个文件，三处语义 + 一处计数）

### 2.1 `engram_jit_kernel.py`

1. **缺页不再中止本批**（decode 的标量回退 + prefill 的逐 shift 收窄两条路径都改）：
   缺页时该行**停用**（= 与 `tok < 0` 的 barrier 同路径），其余行照常取值；
2. **哈希照算**：删掉 `if err_page < 0:` 这个开关 —— 否则"一个缺页 ⇒ 整批没有哈希"，
   比 pad 历史更糟（会把整批 token 的 Engram 注入清零）；
3. **新增第 4 个返回值 `miss_rows`**（本次因缺页而降级的行数）；`err_page` 保留成
   **软信息**（首个缺页页号），不再等于"致命"。

### 2.2 `engram_hash.py`

1. `_engram_update_jit`：`miss_rows > 0` ⇒ 累加 `self.pageless_history_rows` +
   **一次性**打印 `[ENGRAM-PAGELESS]` 提示 + 若 `V41_ENGRAM_PAGELESS_STRICT=1` 才 `raise KeyError`；
2. **非 JIT（torch）路径同样修**：新增 `_mirror_row()`（缺页 ⇒ 补一行全 `-1` 的 barrier 行 + 计数），
   替换掉 `slab = torch.stack([self.pages[page] ...])`、`_vectorized_history`（原来缺页返回 None
   再把控制权交给会 KeyError 的标量走法）、以及标量走法里的 `self.pages[...]` 三处；
3. 模块级加 `import os` + `_pageless_strict()` / `_pageless_note()`。

★ **可回退**：`V41_ENGRAM_PAGELESS_STRICT=1` 完全恢复旧的致命行为（给需要 fail-closed 的臂）。
★ **不静默**：首次缺页一定打印一行，并给出成因与计数器名。

---

## 3. 【实测】本机验证（不需要 NPU）

```
$ pip install numba          # 本机没有 numba ⇒ 先装，才能真编译（njit）
$ python3 -c "... import engram_jit_kernel; selftest()"
numba 编译 + 两个 selftest case: True
```

`selftest()` 里新增 **case 2（缺页）**，逐条断言：

| 断言 | 期望 | 为什么 |
|---|---|---|
| `oob2 == -1` | 不越界 | — |
| `miss2 == 1`、`err2 == 0` | 缺页被**记录**、但**不致命** | 新语义的核心 |
| `hist2[0,0] == 7` | 本行自己那个 token（本步写入）仍然取到 | 降级只影响缺的 shift |
| `hist2[0,1] == pad` | 缺页 ⇒ barrier ⇒ pad | 与 `tok<0` 同义 |
| `oh2[0,0,0] == 6` | **哈希仍然算出来了**（`7^1^1^1 = 6`，`6 % 7 = 6`） | 否则整批 Engram 注入会归零 |

---

## 4. ⚠️ 诚实边界（这条不是"零代价"）

| # | 事项 | 说明 |
|---|---|---|
| 1 | ★ **这是"有界降级"，不是"零精度损失"** | 每个"取回边界"最多 `1 + (lookback-1) = 4` 个 token 位置的 Engram 历史退化成 `pad`（= 序列从此处开始的语义）。`lookback=4` ⇒ 影响面很小，**但确实不是 0**。 |
| 2 | ★ **精确修复（零损失）正在另派** | 思路是把命中边界前 1–3 个 token 的**真实 token id** 从 runner 侧（`prompt_token_ids`/`output_token_ids`）送进 `update()`。见子代理 `engram_exact_fix_design` 的 `docs/ENGRAM-OFFLOAD-EXACT-FIX.md`。 |
| 3 | **A3 端到端验证【未完成】** | 判据：同一条 p2e 形态的臂（`ENGRAM=1 + 卸载 + replay`）replay `failed=0`、`KeyError=0`，且 `pageless_history_rows > 0`（证明"触发过但被优雅降级"，不是"没触发"）。 |
| 4 | **A2 上的实际命中率/边界数量未知** | A2 是 `host_mem_pool=0` + `MAX_SEQS=4`，取回边界的频率会比 A3 臂低（或不同），但**结构上必然发生**。⇒ 上线后看 `pageless_history_rows` 计数。 |
| 5 | `_stock_small_history`（`V41_HASH_MODE=stock` 的 A/B 诊断路径）**未改** | 那条路只在显式做 hash A/B 时走；默认 `fast`。 |

---

## 5. 对三条路的影响

```
用户约束：★ ENGRAM 必须为 1（关掉 Engram = 模型能力下降不可接受）
  ⇒ 路 1（ENGRAM=1 + draft入图 + 档B/C + 不开卸载）  = 现状，天然可用
  ⇒ 路 2（ENGRAM=1 + 卸载）                        = ★ 本修复要打开的那条
  ⇒ 路 3（ENGRAM=1 + 卸载 + 档C）                  = 仍是最激进的一条
```
