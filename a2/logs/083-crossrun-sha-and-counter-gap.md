# 083 — ★★★★ 两条判据层的硬事实：① 跨运行的 sha 对比在这套几何下无效；② 一次性提示不能承载判据

> 2026-09-22 22:5x–23:1x CST（远端 A3-node1）。执行：子代理 **`engram_repro_fix_arm`** 发现，**主代理独立复核**。
> 前置：`077`（修复判决）/`080`（OOM 门）/`081`（合并件）/`082`（graphsafe 接线）。标记：**【实测】/【推断】/【未确认】**。

---

## 0. 一句话

`p3b2`（`TRUE_TOKENS=1` + `ROW_IDS=1`）**全绿**（fill 16/16 + replay 16/16、`failed=0`、`TypeError`/`KeyError:`/`EngineDead` 全 0），
而且 **`ENGRAM-PAGELESS` = 0**（对照 `p2g` 的 8 行）⇒ **pad 降级被真正消掉了**。
但同一轮里挖到两条**判据层**的事实，它们**改变我们该怎么判**：

1. ★★★ **跨运行的 sha 对比**（"用 A 臂的 sha 比 B 臂"）在这套几何下**不可用** —— 同配置两次独立运行 sha 就不同；
2. ★★★ **"只打一次"的提示不能承载判据** —— `absent/filled/mismatch` 的累计值**从未落盘**，
   而 `mismatch>0`（"陈旧页 = 静默算错"）正是决定要不要升 `mode=2` 的唯一信号。

---

## 1. 【实测】`p3b2` 的读数（修复真的生效了）

| 判据 | `p2g`（`TRUE_TOKENS=0`，pad 版） | ★ `p3b2`（`TRUE_TOKENS=1`） |
|---|---|---|
| fill / replay | 16/16 · 16/16 | **16/16 · 16/16** |
| `failed` | 0 | **0** |
| `TypeError` / `KeyError:` / `EngineDeadError` | 0 | **0 / 0 / 0** |
| ★ `[ENGRAM-PAGELESS]` 行数 | **8**（8 rank 各一次降级） | ★ **0** |
| `[ENGRAM-TRUE-TOKENS]` 行数 | —（未开） | **8**（8 rank 各一次） |
| `CPU→GPU` | 21.5 GB 级 | **21.52 GB**（取回确实发生） |

⇒ ★ **【推断·强】`ENGRAM-PAGELESS=0` 只能由一个原因解释**：`apply_repairs()` 先把页写满真 token
并置 `present=1` ⇒ kernel 的读路径**再也看不到缺页** ⇒ **pad 降级被消掉**，而不是"没触发"。
（`080 §3.1` 记过"沉默 ≠ 没跑"的口径，这里用 `p2g` 的 8 行作**阳性对照**把两者区分开了。）

### 1.1 ★ 一条我自己的错判，当场撤回

我在 `080 §3` 里怀疑 `V41_ENGRAM_ROW_IDS` 没送进容器。子代理实测：**该怀疑不成立**，
更准确的说法是 **`/proc/<pid>/environ` 在这台机器上不可信**（worker 的白名单里看不到传进去的 env，
但 `apply_repairs` **确实被调用**）。
⇒ **判 env 是否生效，要看"代码痕迹"（日志/计数），不要只看 `/proc`。** 已写进 `080 §3`。

---

## 2. ★★★★ 【实测】跨运行的 sha 对比**无效** —— 判据基础要改

子代理做了熵测试：`p2e` 与 `p2h` 是**同一配置**（文件逐字节相同、env 相同、salt 相同）的**两次独立运行**：

| 臂 | `fill_out_sha256_all` | fill | replay |
|---|---|---|---|
| `p2e-engram1-dev0-dg1-offload` | ★ `1fc2a9cef2b07a0464cd2848…` | 16/16 ok | 3/16（崩） |
| `p2h-base075-rerun` | ★ `a98f0645ce13637987ace98f…` | 16/16 ok | 3/16（崩） |

★ 主代理**独立复核**：两条 `client.json` 的 `fill_out_sha256_all` 前缀确实如上，
`salt=20260922 / prompts=16 / ptok=131072` 三者逐字相同，rounds 结构也一致（`(16,0,374.4)` / `(16,0,364.6)`）。

⇒ **同一配置、两次运行、fill sha 不同**。原因链（与 `072 §1` 自洽）：
`kv_offload_client.make_prompt()` 用的是**随机 token id** ⇒ 输出分布近均匀
⇒ 采样/规约噪声一翻就翻 ⇒ **sha 不是"内容指纹"，只是"这次抽签的签名"**。

### 2.1 因此**不能**用这些当判据（逐条）

| 被否掉的判据 | 为什么不能 |
|---|---|
| "`p3b2` 的 replay sha == 无取回臂的 replay sha ⇒ 零损失" | 跨运行 ⇒ 固定会被噪声打翻（实测只有 4/16 相同，**无信息**） |
| "`p2g` 的 replay sha != `p3b2` 的 replay sha ⇒ 修复生效" | 同上；**而且本次修复生效的证据不在这里**（在 `ENGRAM-PAGELESS` 计数） |

### 2.2 ✅ 可用的替代判据（**同一次运行内**）

```
① --rounds 3           ⇒ 比较 replay1 vs replay2（两次都走池取回）—— 同运行、同 salt、同进程
② a2/scripts/text_correctness_probe.py --mode prefix-pair
                       ⇒ 同一段长前缀 + 同一问题连发 N 次，比**第 2 发起**是否逐字相同
③ 计数与代码痕迹        ⇒ ENGRAM-PAGELESS 8→0、ENGRAM-TRUE-TOKENS 出现、CPU→GPU 量级
```

### 2.3 ⚠️ 连带影响（必须回头复核的一格）

`logs/069` 有一句"**卸载判据两臂逐字相同**"，随后列了一串读数（`CPU→GPU=21,519,269,888`、
`hits=901,120`、`BlockStored=29,436`、`BlockRemoved=0`、`GPU KV cache size=427,643`）。
那一串是**计数器**（跨运行可比，没问题）；但同一段里也列了 `fill sha`（`a119d9f6…`）——
那一项**属于本节否掉的类别**。
⇒ 标 **【未确认】**：需要复核当时那句"逐字相同"到底指**计数器**还是**含 sha**；
若是后者，应改写成"计数器逐字相同"。

---

## 3. ★★★ 【实测】一次性提示不能承载判据（计数器缺口，已修）

`p3b2` 的日志只有这一行（×8 rank）：
```
[ENGRAM-TRUE-TOKENS] mode=1 首次修补：n=6 计数={'unavailable': 6}
```
⇒ **`absent/filled/mismatch` 的累计值一次都没落盘**。根因两条：

1. `_engram_true_tokens_note()` 与 pageless 提示**共用** `_PAGELESS_WARNED[0]` 的"只打一次"旗标；
2. 第一次调用发生在 **warm-up decode（n=6）** ⇒ 那一次自然全是 `unavailable`（还没发布真实 token）
   ⇒ 之后所有更有信息量的调用都被"只打一次"吞掉。

★ 后果很具体：**`mismatch > 0`（"镜像里是别人的 token" = 陈旧页静默算错）至今没有真机读数**
⇒ **无法判定要不要升到 `mode=2`**（而 `ENGRAM-OFFLOAD-EXACT-FIX.md` §2 早就把"陈旧页"列为残余静默点②）。

### 3.1 修法（已落 `engram_hash.patched.py`）

| # | 改法 |
|---|---|
| ① | **独立旗标**（不再共用 `_PAGELESS_WARNED`） |
| ② | **按键累计** `_TT_CUM`（`absent/filled/mismatch/overwrote/unavailable/oob`），不再只加一个总数 |
| ③ | 打印规则：首次 + 累计四元组变化后每 `V41_ENGRAM_TRUE_TOKENS_LOG_EVERY`（默认 200）次再打 + ★ **`mismatch` 一出现立刻打** |
| ④ | `self.engram_true_token_stats_cum` 按键累计（原来 `engram_true_token_total` 只留总数，**丢掉分解**） |

★ 新防线：`tests/test_callsite_contract.py` 增加"**计数上报口径**"检查
—— 用 **AST** 断言 `_engram_true_tokens_note` 的函数体里**不引用** `_PAGELESS_WARNED`，
并断言存在 `_TT_CUM` 与 `LOG_EVERY`。

### 3.2 ★ 写这个检查时我自己又踩了同一个坑（值得记）

第一版检查写的是 `grep 字符串`：断言 `"_PAGELESS_WARNED"` 不出现在函数体源码里。
结果被我**自己的注释**（"旧实现把 `_PAGELESS_WARNED[0]` 共用…"）判成"仍在共用" ⇒ **假失败**。
⇒ 与 `079 §3`"判据被自己的文本污染"**是同一天第二次**。
**结论：判代码要用 AST/语法树；字符串搜索只适合判"存在性"，不适合判"不存在"。** 已改用 AST。

---

## 4. 对目标（四轴同开）的含义

| 轴 | 状态 |
|---|---|
| ① `ENGRAM=1` | ✅ 修复已验（`077`）；pad 降级已被消除（本节 §1） |
| ② 卸载 | ✅（`077`/`078` 逐字节保真） |
| ③ int8 档 C | ✅ 单轴验过；四轴需要 `logs/082` 那两个 env |
| ④ `DRAFT_GRAPH=1` | ✅ 单轴验过（`069`） |
| **四轴同开** | ⏳ 前置已补齐（`080`/`081`/`082` + 一键脚本 `run_4axis_arm.sh`）；**判据已按本节改成同运行内** |

★ 验收时**不要**再用跨运行 sha；改用 §2.2 的三类判据 + `text_correctness_probe.py`。
