# 079 — 三条「发布件自己的失效」：`global` 晚于使用 / tag 三处不一致 / 判据被提示文本污染

> 2026-09-22 21:2x–21:5x CST。执行：**主代理**（本机；A3 子代理报了第一条）。
> 性质：**发布纪律 + 防线**（不是实验结论）。标记：**【实测】**。

---

## 0. 一句话

这一轮**一个新功能都没写**，三个问题全部出在"**交付件本身**"：
一个**编译不过**（起服必挂）、一个**三处默认值不一致**（build 与 serve 对不上）、
一个**判据把提示文本数进去了**（真异常 0 却读出 8）。
★ 共同点：**它们都不会在"跑通一次"里被抓到** —— 只有**门**能抓到。

---

## 1. 【实测】`model_runner_v1.patched.py` `py_compile` 不过（子代理在 A3 首次发现）

```
SyntaxError: name '_ENGRAM_ROW_TOKENS_DISABLED' is used prior to global declaration   (line 3011)
```

| 项 | 内容 |
|---|---|
| 根因 | `_model_forward()` 里第 2999 行**先读**该全局，而 `global` 声明写在下面的 `except` 里 ⇒ Python 要求 `global` 出现在该作用域**任何使用之前** |
| 后果 | **import 期就崩、起服必挂**（不是运行期才出问题）|
| ★★ 陷阱 | **`ast.parse()` 能过**；只有 `py_compile` / `compile()` 的 **symtable 阶段**才报 ⇒ 只做 AST 检查的工会漏 |
| 修法 | `global` 提到函数体顶部，删 `except` 里那一行 —— 纯 hoist，无语义变化 |
| 收敛 | 我先按描述独立修，md5 与 A3 那份**不一致**（我多写了注释）⇒ **从 A3 把子代理那份原样取回作为权威**，发布仓改用同一份（`a94887de05bb63370a6604260b358101`）★ 教训：**多人在同一交付件上改 ⇒ 必须用 md5 收敛，不能"各自都对"** |
| 新防线 | `a2/patches/engram-true-tokens/tests/run_all.sh` 新增 **门 0：py_compile 全部交付件**（四个 .py），并在注释里写明"AST 会漏" |

## 2. 【实测】镜像 tag **三处不一致**（`selfcheck_pkg.sh` 抓到）

```
FAIL  镜像 tag 不一致！build_image 产出 'v8'，但 serve_a2/run_test 找 'v9'/'v8'
```

我改 `serve_a2.sh` 的默认 `IMAGE` 到 v9 时，**漏改了 `build_image.sh` 与 `run_test.sh`**。
⇒ 后果会是：**build 出来叫 v8、serve 去找 v9** ⇒ 要么找不到、要么**静默用旧镜像**（= 修复没上车）。
★ 这说明本仓那条门 **`tools/selfcheck_pkg.sh` 的"三处 tag 必须一致"是有效的**（它当场拦下）。
⇒ 顺手把 `MANIFEST.sha256` 从 307 项重算到 **447 项**（新增的 `engram-true-tokens` 包此前没进清单，
完整性校验会漏）。重算后 `selfcheck` **全绿**。

## 3. 【实测】判据被**自己的提示文本**污染

`075` 的提示写的是「…并继续（**不再 KeyError**）」，而交付脚本的判据是
`grep -c 'KeyError' <serve.log>` ⇒ **8 个 rank 打出 8 条提示** ⇒ 读数 `KeyError=8`
（看着像引擎炸了 8 次，**真异常 0**）。

⇒ 两条纪律（都写进了 077）：
1. **运行期打印的文本里不得出现裸的 `KeyError` / `Traceback` / `EngineDead` 子串**；
2. **真异常判据一律用带冒号的精确模式** `grep -ac 'KeyError:'`。

★ 编号说明：本条**让位**给 KV 逐字节保真探针（它先认领了 078）⇒ 本条编 079。
★ 同族先例：`066c` 的"判据与它要测的对象共用同一个字符串空间"；`039` §1 的"0 比较却报全绿"。
**共性：判据的字符串空间与它观察的对象的字符串空间重叠。**

---

## 4. 沉淀成三条可复用的门（都已落地）

| # | 门 | 落在哪 | 能抓住什么 |
|---|---|---|---|
| **G-py** | `py_compile` 全部交付件（不是 AST） | `a2/patches/engram-true-tokens/tests/run_all.sh` 门 0 | `global` 晚于使用这类**symtable 级**错误 |
| **G-tag** | `build_image` / `serve_a2` / `run_test` 的镜像 tag 必须三处一致 | `tools/selfcheck_pkg.sh`（**既有**，本轮首次生效） | "build 出来的名字 ≠ serve 找的名字"⇒ 静默用旧镜像 |
| **G-img** | 起服前比对 host 权威副本 vs **镜像内实际那份**的 md5 | `a2/scripts/serve_a2_offload.sh`（v9 起） | "补丁改了但镜像还是旧的"⇒ 白等 30 分钟 |

★ 另加一条**协作纪律**：同一交付件被多人改时，**以 md5 收敛**（本轮 `model_runner_v1.patched.py`
出现过两份"都对但不同"的版本，靠取回权威文件 + md5 比对解决）。

---

## 5. 状态

```
发布仓 HEAD: e52ad8e（selfcheck 全绿：tag 三处 v9 / MANIFEST 447 项 / 三方一致）
A3 现状:     p3a-true0-rowids1 起服中（TRUE_TOKENS=0 + ROW_IDS=1 的隔离臂）
下一步:      B 臂（TRUE_TOKENS=1）⇒ 量化 pad 降级的语义影响；同时 KV 逐字节保真探针（= 078，另一条战线）
```
