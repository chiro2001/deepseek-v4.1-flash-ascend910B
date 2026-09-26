# 134 — ★★★★★ **520k 单发 prefill × Agent 场景：A3(main) 上打满所有已知形态仍**不复现**；交付 A2 复现手册 + 探针（并修掉探针两处"判据自己骗自己"）**

> 2026-09-23 14:0x–15:0x CST。来源：**用户**给的可复现场景（"一次 agent 的 context ≈520k 直接进入，
> 完整一次 prefill（可能还有其它流同时请求），然后出现错误；**A2 main 也可复现**"）
> ＋ **主代理**在 a3-21 上的成组实测。
> 标记：**【实测】/【推断】/【未确认】**。关联：`130`（乱码方向）、`main:docs/CTX-AGENT-REPRO.md`、`main:tools/ctx_agent_probe.py`。

---

## 0. 一句话

**【实测】在 A3-21（main 检出 + 官方 a3 镜像 + `MAX_LEN=1M` + `CPU_BIND=0`）上，
"520k 大 prefill + 并发/追问/混合流"这套组合拳全部干净**（逐字命中、乱码指纹全 0、`bad_ctx=0`）
⇒ 与现场仍差**两条结构性变量**（**模型**、**SoC**）⇒ 本轮把 **A2(main) 的复现手册**做成一键，等现场数据。

★ 同时**修掉探针两处致命缺陷**（都是"判据自己骗自己"）：见 §4。

---

## 1. 用户现场（本卷的靶子）

* 一次 Agent 的 context **≈520k token 直接进入**；
* **完整做一次 prefill**；
* **可能还有其它流同时请求**；
* 然后**出现错误**（此前口径：长上下文 + Agent 场景出乱码，**且不开 DRAM 卸载也有**）；
* **在 A2 的 main 分支上也可复现**。

⇒ 关键推论：**触发条件需要 `MAX_LEN ≥ 520k`**。本仓 A3 的部署默认是 `MAX_LEN=133120`
（A3 已验证口径）⇒ **我此前所有扫描都被"128K 天花板"挡住了**，这也是我先前没复现的原因之一。

---

## 2. 为对齐现场做的两个新模式（`tools/ctx_agent_probe.py`）

| 模式 | 覆盖现场的哪一句 | 做法 |
|---|---|---|
| `biggrow` | "**完成一次 prefill 之后**出现错误" | 先用 520k 的 Agent 轨迹（system + user + tool_call + 520k 工具结果）做一次完整 prefill，再在**同一会话**里追问 3 轮（每轮问一个埋在不同深度 20/40/60/80% 的针）⇒ 覆盖"**复用 520k 前缀**"这条路 |
| `mixed` | "可能还有**其他的流同时请求**" | **1 路 520k 大 prefill 在飞**的同时，4 路**短流**（几百 token 的正常聊天）打进来；并带"**空闲时**的短流基线"做对照（同一问题、`temperature=0`） |

两者都带**上下文长度门**（`bad_ctx`）：不达标就不算通过（见 §4）。

---

## 3. A3-21 实测（**全部干净**，逐条可复算）

环境：main @ `16e9b40`、官方 a3 镜像、`MODEL=…engram-dr-vision-qrot-mtpq`、
`MAX_LEN=1048576 MAX_SEQS=4`、**`CPU_BIND=0`**（否则会被 NUMA 迁移卡死，见 `133`）。

| 臂 | 规模 | 结果 |
|---|---|---|
| `bigprefill` 单发 | **520,852 token**，单发 **111.4 s** | PASS |
| `bigprefill` + 2 路并发（各 520k） | 6 次请求 / **662 s** | 全 PASS |
| `biggrow` 520k → 同会话追问 3 轮 | 4 次请求（prefill 108.5 s、追问各 1.6 s） | 全 PASS |
| `mixed`：1 路 520k ＋ 4 路短流 | 9 次请求（含 4 条空闲基线） | 全 PASS |
| `needle`/`grow`/`reuse`/`toolargs`/`evict`/`conc` | 8K / 32K / **131K** | 全 PASS |

★ 另外几条读数（顺带确认这臂是"真 1M 口径"）：`GPU KV cache size = 1,394,931 tokens`；
`enable_cpu_binding=false`（`CPU_BIND=0` 真生效）；`PATCH_MODE=mount`。

**【实测】结论**：**单靠"520k 大 prefill + 并发/追问/混合流"，在 A3 + main 上不足以触发**。

---

## 4. ★★ 本轮真正的收获：探针自己的两处"判据骗自己"

### 4.1 语料短于目标 ⇒ 切出空串 ⇒ **拿 91 token 冒充 520k，还报 PASS**

第一次跑 `bigprefill` 时打印 `ctx≈91` 却判 **PASS**。逐层查：
`data/hongloumeng.txt` 是 2.47 MB 的 **UTF-8 中文** ⇒ **只有 826,639 个字符**；
而代码写成 `corpus[offset:offset+n]`，我要 520k 时 `offset=1,000,000` **已越过文件末尾**
⇒ **切出空串**；`embed_needles` 再把 4 条针插进去 ⇒ **实际上下文只有 91 token**。

⇒ 修法三条：① `build_context()` **平铺**语料到目标长度（并用 `rotate` 让不同 lane 拿到**不同内容**）；
② 新增**上下文长度门** `ctx_ratio_ok(real, target, --min-ctx-ratio=0.9)`：
不达标 ⇒ 标 `ctx_ok=False`、计入 `bad_ctx`、**不能算通过**；
③ `--selfcheck` 增四条判据把它钉死（含"91/520000 必须判 False"）。

### 4.2 "问 B 却插了 A 的针" —— **模型是对的，判据是错的**

32K 那一轮 `grow` 的 turn1–3 全 FAIL，而模型的回答是
「app_1.log 里没有"运维备忘 B"，**只有重复出现的"运维备忘 A"**」——
根因：`embed_needles` 用**位置下标**取针（`NEEDLES[i % 4]`），与 key 无关 ⇒ 问 B/C/D 时插的仍是 A。

⇒ 改为**按 key 取针**（`KEY_TEXT`），未知 key 直接 `raise`；并新增 `--selfcheck`：
逐个 key 验"只插自己的针、不混入别的针、四针按深度递增"。

★ 这两条与 `126`/`127`（"文本对 ≠ 行为不变"、"取最后一行是未经检验的前提"）**同族**：
**判据本身也是要被检验的对象**；`--selfcheck` 就是为此存在的。

---

## 5. 交付（`origin/main`）

| 文件 | 作用 |
|---|---|
| `tools/ctx_agent_probe.py` | 8 个模式：`needle`/`grow`/`reuse`/`toolargs`/`evict`/`conc`/`bigprefill`/`biggrow`/`mixed`；`--selfcheck`；上下文长度门 |
| `docs/CTX-AGENT-REPRO.md` | ★ **A2(main) 粘贴即用的复现手册**：前置（含 `--selfcheck`）/ 起服 / 四条复现命令 / 服务侧证据收集 / 已排除清单 / 失败后的**单变量切法** |
| `tools/selftest_ctx_agent_probe.sh`（+ `_fake_vllm.py`/`_check_probe_json.py`） | **20 条**沙箱自测（假服务）：干净必过 / 乱码必抓 / 工具参数逐字 / 上下文不达标必判无效 / 无 `/tokenize` 回退 / 连不上 rc=2 |
| `tools/selfcheck_pkg.sh` | 纳入 9c 项（含一处自身误报修正：`.md` 不能用 `py_compile`） |

---

## 6. 还没覆盖的两条结构性差异（**这就是下一步**）

| # | 差异 | 为什么可能决定成败 |
|---|---|---|
| 1 | **模型不同**：现场 `v41-w4a8-flat`；A3 上只有 `v41-w4a8-engram-dr-vision-qrot-mtpq`（`engram_layer_ids=[1,14]`、含 vision/qrot/mtpq） | 少/多某些组件 ⇒ **代码路径不同**（尤其 Engram 层数与 device 路径）；A3 这边跑不到现场那条路 |
| 2 | **硬件不同**：A2 = 8×910B3；A3 = 8×910C | 同一份 main，但 **KV 页几何 / 算子实现 / 静态内核**都不同 |

⇒ 因此下一步 = **在 A2(main) 上按手册 §3 跑四条**，把 4 份 `--out` JSON + 服务侧片段发回。
若 A2 上复现，按手册 §6 的**单变量顺序**切（`MAX_SEQS` → 关投机 → 关 Engram → 块大小 → 换模型）。

---

## 7. 未确认 / 遗留

| 项 | 说明 |
|---|---|
| 【未确认】现场"错误"的具体形态 | 目前只有"乱码"这一口径；需要 §3 的 `fp=U+FFFD/NUL` 计数与 `answer_repr` 原文才能定性 |
| 【未确认】是否与投机解码（spec）交互 | A3 这臂 spec 是开的（`SP_TOKENS=5`）且干净 ⇒ 单独不足以触发 |
| 【未确认】`v41-w4a8-flat` 与 qrot-mtpq 的**结构差** | 需要现场 `config.json`（尤其 `engram_layer_ids`、`quantization_config`、是否有 vision/mtpq 分片） |
| 【未做】A3 上 `ENGRAM=0` @520k | 需重启一次（约 7 min）；在 A2 数据回来之前优先级低 |
| 【未做】>600k 的更长 prefill | 现场是 520k；先对齐现场，再往上推 |
