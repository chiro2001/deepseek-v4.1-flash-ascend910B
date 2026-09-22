# 073 — ⛔⛔⛔ **`ENGRAM=1` + 卸载：replay 轮引擎死（`KeyError: 2486` @ `engram_hash.py:463`）**

> ★★★ **本文件与 [`072`](072-text-correctness-p2e.md) 是【同一臂的两份独立观测】**
> （主编 19:46 起臂、子代理 `a3_text_correctness` 事后独立读日志）。
> **两份的读数逐项一致**（`failed=13` / `wall=5.771 s` / 同一个 `KeyError` 栈 / 同一个 `65536` 请求），
> 但 **`072` 多给了四件我没做的**（见 §8）：**8/8 rank 逐字相同**、**泄漏链的完整因果**、
> **自建的对照矩阵**、**以及一条对 `056` 的定性更正**。
> ⇒ **两份都要读**；本文件的编号**原为 072，因撞号改为 073**。

> 2026-09-22 19:46–20:04（**远端 A3-node1 时间**）。执行：**主代理**（起臂 + 独立读栈）。
> 臂：`p2e-engram1-dev0-dg1-offload`（c0 / Phy-ID 8–15 / 真权重）。
> 标记：**【实测】/【推断】/【未确认】**。

---

## 0. 一句话

**`ENGRAM=1` + 卸载 + draft入图（`ENGRAM_DEVICE_INDEX=0`）—— 起服成功、fill 轮 16/16 成功，
但 ★★ replay 轮 13/16 失败、引擎死**，根因是 **Engram 的 host 路径**：
```
KeyError: 2486   @ engram_hash.py:463  (raise KeyError(err)，err 来自 JIT kernel 的返回)
栈：model.py:1256 prepare_engram_inputs → 1129 prepare_engram → ★1023 _prepare_engram_host
    → 1092 self.engram_history.update(...) → engram_hash.py:282 _engram_update_jit → 463 raise
```
★ **这是全仓第一次出现这个栈**（`grep -rl "KeyError" logs/` **零命中**；
`grep -rl "_engram_update_jit" shadow-pkg/results/*/serve.log` **只命中 p2e 这一条**）。

---

## 1. ★★ 这条臂是什么（**它就是 A2 要上的配置**）

| 项 | 值 |
|---|---|
| 模型 | `~/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`（**真权重**，不是 dummy） |
| ★ `ENGRAM` | **1** |
| ★ `ENGRAM_DEVICE_INDEX` | **0**（与 A2 生产一致；**主代理显式设的**） |
| 卸载 | `offload_bytes=60,666,413,056`、`bpc={"default":8,"swa":1}` |
| 投机 / 图 | `DRAFT_GRAPH=1`（draft 真入图）+ `GRAPH=1` |
| TIER | B（无 int8） |
| 几何 | `MAX_LEN=133120`、`MAX_SEQS=32`、fill 16×131072 → replay 16×**65536** |

**起服【实测】全绿**（这些判据都过了）：
```
Application startup complete = 1     GPU KV cache size = 427,643
★ DEVICE-INDEX 行数 = 0（证明 device-index 真关着）  ★ EH0012 = 0   ★ 207001 = 0
池注册 P1_pinned ... registered = 136 行    Engram 相关 = 24 行
```

---

## 2. ★★★ 判决读数（fill 全过 → replay 死）

```
[warmup]  ok=True  wall=18.23 s
[fill]    failed=0     16/16 成功   out_sha256=1fc2a9cef2b07a04…   Σttft mean=20251.2 ms
★[replay1] failed=13   ★ 只 3 条成功  out_sha256=7431691f87fc626f…   wall=★ 5.771 s
[sha256]  fill != replay   common=3   mismatched=[0, 1, 2]
```
★ **replay 只用了 5.77 s**（fill 用了 374 s）⇒ **不是超时，是引擎当场死**：
```
(EngineCore pid=1405) ERROR 09-22 12:03:56 [core.py:1351] EngineCore encountered a fatal error.
(APIServer pid=719)   ERROR … vllm.v1.engine.exceptions.EngineDeadError
⇒ 13 条 replay 全部 500
```

**死时那个请求【实测】**（`dump_input` 的 scheduler output 原文）：
```
scheduled_new_reqs=[NewRequestData(req_id=cmpl-b0829522dfd75ae9-…,
                                   ★ prompt_token_ids_len=65536, …)]
```
⇒ ★ **正是 replay 请求**（65536 = replay 的 prompt 长度）⇒ **崩溃发生在"取回前缀"这一步**。

---

## 3. ★★★ 决定性对照（**唯一变量 = `ENGRAM`**）

| 臂 | `ENGRAM` | `DEVICE_INDEX` | 池 | **fill** | **replay** |
|---|---:|---|---:|---|---|
| `p1b-tierB-dg1-offload` | **0** | auto | 56 GiB | failed=0 | ★ **failed=0**（16/16，TTFT 1,420 ms，`hits=901,120`、`CPU→GPU=21.5 GB`） |
| **`p2e-engram1-dev0-dg1-offload`** | **1** | **0** | 56 GiB | failed=0 | ★★ **failed=13**（引擎死） |

⇒ ★★ **两条臂的其余配置逐字相同**（同 runner / 同几何 / 同池 / 同 `DRAFT_GRAPH` / 同 `MAX_TOKENS=128`）
⇒ **唯一差别是 `ENGRAM`**。

---

## 4. 机制【推断·强】（**未做对照，别当结论**）

```
engram_hash.update() 的 docstring 原文：「page numbers come from **full SWA KV**」
⇒ Engram 的 hash 更新需要该请求的【完整页历史】（lookback 窗口）。
★ 而 replay 轮里，前缀是【从 CPU 池取回】的 ⇒ 只处理未命中的尾部 token
⇒ Engram 看到的是"不完整"的页历史 / 或者页号不在它预期范围内
⇒ JIT kernel 返回 err = 2486（行号）⇒ raise KeyError(2486) ⇒ 引擎死
```
★ **为什么 A2 现在没事（待证）**：A2 生产 `ENGRAM=1 + PREFIX=1`、命中率 96%，但
**它的命中来自 GPU 侧 prefix cache，不是卸载池**。⇒ 若"GPU 命中不触发、池命中才触发"，
则 **A2 一上卸载就会踩**。
⚠️ **但这只是【推断】** —— 见 §5 的三条判据缺口。

---

## 5. ⚠️ 诚实边界（**这条很重要，别把它读成"卸载不能上"**）

| # | 未确认 | 为什么重要 | 怎么判 |
|---|---|---|---|
| **1** | ★ **是"卸载命中"还是"65536 长度"？** | p1b 也是 65536 且没事 ⇒ 但它是 `ENGRAM=0` ⇒ **两个变量仍纠缠** | 一条臂：`ENGRAM=1 + 池 1 MiB`（**无命中**）+ 同 replay 几何 ⇒ 若也崩 = 与命中无关 |
| **2** | **是不是 `DEVICE_INDEX=0` 特有的？** | A3 上 `ENGRAM=1` 的另三条臂都开着 device-index（且**更早**就崩在 `EH0012`）⇒ 无法比较 | 需要 `ENGRAM=1 + DEVICE_INDEX=auto` 的成功起服臂（目前不存在） |
| **3** | ★ **A2 会不会踩？** | A2 = `ENGRAM=1 + 卸载 + DEVICE_INDEX=0` = **正是这条臂** | ★ **只有在 A2 上跑才能答**（或先在 A3 把 #1#2 判清） |
| 4 | **填充期为什么没事** | fill 是"完整 prefill"，Engram 能看到全部页 | 与机制一致 |
| **5** | `metrics_after.txt` 是 **0 字节**（引擎死时采集不到）⇒ **replay 轮的 `hits`/`CPU→GPU` 拿不到** | 无法直接证明"发生了池命中" | 需在崩溃前抓 metrics，或看 ZMQ 事件 |

---

## 6. 对 A2 窗口的直接影响

```
★ 这条发现**降低了"ENGRAM=1 + 卸载"的可信度** —— 它是 A2 要上的配置，
  而它在 A3 上【起服成功、fill 成功、replay 死】。
⇒ 窗口计划必须调整：
   ① ★★ 把"replay 轮"提到**起服之后立刻**做（而不是等 fill 跑完）——
      因为崩溃发生在 replay，而 fill 会白花 ~6 min
   ② ★★ 新增一道门：**测一次"带前缀命中的请求"**（同一 prompt 连发两次），
      第一次 cold、第二次 hit ⇒ 看第二次是否 500 / 引擎是否死
   ③ 若 A2 上也崩 ⇒ ★ **"卸载"要挡到 Engram 这条修好**（这是第 3 个独立阻塞：
      EH0012 / 207001 / 本条的 KeyError）
```

---

## 7. 原始证据

| 件 | 位置 |
|---|---|
| 引擎日志（含完整栈） | A3：`shadow-pkg/results/r8_p2e-engram1-dev0-dg1-offload_20260922_194605/serve.log` |
| 客户端结果（fill/replay 读数） | A3：`agents/R_8card_int8/out/p2e-engram1-dev0-dg1-offload/` |
| 对照臂 | A3：`…/p1b-tierB-dg1-offload/`（`ENGRAM=0`，replay 16/16 成功） |
| 启动命令 | 主代理 19:46 起的：`TAG=p2e-engram1-dev0-dg1-offload TIER=B GRAPH=1 DRAFT_GRAPH=1 ENGRAM=1 MAX_TOKENS=128` + **`export ENGRAM_DEVICE_INDEX=0`** |

---

## 8. ★★★ `072` 补的四件（**本文件 §4 的推断被它们收紧/更正**）

### 8.1 ★ **不是"我推的"，是 8/8 rank 逐字相同**（确定性）

`072` 独立核到：**8 个 rank 的 `KeyError: 2486` 栈逐字相同** ⇒ 这不是抖动、不是个别 rank。
⇒ **本文件 §4 那条"机制【推断·强】"可以去粗**：崩溃的**确定性**已被 8/8 证实。

### 8.2 ★★ 泄漏链的完整因果（**本文件没写的那一半**）

```
异常从 _model_forward 逃逸
  ⇒ 正好落在引擎 submit() / release() 之间
  ⇒ device-metadata 标志【永久置位】
  ⇒ 其后 32 次 RuntimeError: The previous device metadata submission has not been released
```
★ 我独立复核（读 A3 的 serve.log）：
```
臂      KeyError=32   "has not been released"=32   EngineDeadError=6
p1b          0                    0                        0
★ p2e       32                   32                        6
```
⇒ ★★ **32 = 32**：**每一条 `KeyError` 都对应一次泄漏** ⇒ **因果链成立**。

### 8.3 ★★★ **对 `056` 的一条定性更正（`072` 提出，我复核后采纳）**

> **`056` 把那条 leakage 定性为「②a 特有」—— 该定性【不成立】。**
>
> ```
> p2e 的配置：ENGRAM=1 + ENGRAM_DEVICE_INDEX=0 + 卸载 + DRAFT_GRAPH=1
>             ★ 无 int8、无 draft-int8
> ⇒ 它以【完全相同】的形态复现了 leakage
> ```
> ⇒ **正确表述**：它是「**异常从 forward 逃逸**」的**通用后果**；
> **②a 只是其中一个逃逸者**，**不是必要条件**。
> ★ 这条比 `KeyError` 本身更重要 —— 它说明**这类泄漏是结构性的，不是某个特性的副作用**。

### 8.4 ★ **`072` 的对照矩阵**（我独立复核过，与它一致）

```
臂                    ENGRAM  leakage  EngineDead  KeyError
p1a / p1b / p2c          0       0         0          0      ← 三个 ENGRAM=0 臂全干净
p2b / p2d2               1      8/4       0/5         0
★ p2e (dev0)             1      32         6        ★16      ← 新模式
```
⇒ ★ **Engram 是共同因素**；**只有 p2e 有 `KeyError`** ——
即 **`DEVICE_INDEX=0` 消除了起服期的 `EH0012` 之后，暴露出了排在后面的这个**。
⇒ ★★ **我的假设"`ENGRAM_DEVICE_INDEX=0` 是那个开关"对【起服期】成立，但不足以让目标配置可用。**

### 8.5 ★ 两条 `072` 提出、我认同的**方法论**要点

1. ★ **`rc=0` 是假的**：`replay 13/16 失败` 却被 runner 判为成功 ⇒ **runner 的 gate 不查 `requests_failed`**
   ⇒ ★ 这与 `071 §3` 的判据失效清单同族（**"通过"是假象**）。
2. ★★ **语义判据仍然空缺**（`071 §A1` 原样存在）：`072` 指出
   **p2e 的 prompt 是随机 token id**（`make_prompt()` = `1000 + ((base + i*7919) % 100000)`）⇒
   **输出必然是乱码** ⇒ **不能当质量证据**。
   ⇒ ★ 正确做法：**改用自然语言 needle + 长生成**，而不是随机 token + sha。

---

## 9. 因此 §6 的"窗口调整"要再加一条

```
新增 ④：★★ 语义判据必须用【自然语言 prompt】——
        已证实：本仓所有 8 卡臂的 prompt 都是随机 token id ⇒ 输出必是乱码 ⇒
        它们【从来没验过语义正确性】。
        ⇒ 用户的"A2 服务在正常用"（真实自然语言）是【当前唯一的语义证据】，
          但它是 A2 现网（【无卸载】）⇒ 不能覆盖"卸载开启后"这一格。
        ⇒ ★ 窗口里第 8 道门（段落 4 的 Q() 命令）用的就是自然语言 ⇒ 保留，且它是**唯一**的语义判据。
```
