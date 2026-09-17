# EXPECTED_PERF.md —— 性能预期（A3-node1 实测，供 A2 比对）·v6

> **v6 新增 §A2GAP**（最下面）：A2 首次运行后的 **A2 vs A3 差距拆解**（实测数据）。
> **先看那一节**——它决定了"接下来该攻什么"。

> 口径：quote 单流、**`max_tokens=256`**、`temperature=0`、`MAX_SEQS=1 PREFIX=0`、
> chips 8-15（A3-node1）、真权重、全补丁默认集合（MOE_AG / SP_TOKENS=5 / O_PROJ_2D /
> MOE_MASK / ROPE_IDXSEL / ENGRAM_JIT / QLI_NOCAND / LOCAL_OWNER=fast）+ CPython PGO。
> `ms/step = decode_s / steps_est`；`A = 1 + Σ posᵢ`；`tok/s = A × 1000 / ms_per_step`。
>
> **v4 的三大改动**：① 性能数字从"8 发中位"升级为 **163 发全量重算**（新增 §2 形态分类，
> 并更正了 v3 的"110 tok/s"叙述——它不是稳态，见 §2.3）；
> ② 新增 §7 **多 batch / 多轮对话**（生产口径 `MAX_SEQS=32 PREFIX=1`）；
> ③ `A` 的定位更新：**不许再用 A 的绝对值当绩效指标**（见 §3）。
>
> 数据来源：原始 jsonl（`logs_meta/samples/`）+ `reports/a-basin-and-acceptance-shape.md`（186 行）
> + `reports/session-attractor-and-clean-rate.md`（132 行）。

---

## 0. 一页结论（v4）

| | 128K 单流 **ms/step** | **A**（接受长度） | **tok/s** |
|---|---|---|---|
| 全补丁 + PGO（faB，8 发） | **30.23**（29.79–31.19） | 2.748（1.181–3.446） | 90.13 / 峰值 **110.5** |
| 全补丁，PGO 关（faA，8 发） | **31.63**（30.66–32.34） | 2.681（1.356–2.835） | 84.97 / 87.88 |
| **163 发全量**（p42 128K×256） | 见 §2（中位 ~31–34） | 见 §2.1（**不是单峰**） | **真正可交付的 ≥110 只有 1 发** |
| 最好单发（全量 163 发） | **26.9** | — | — |

**一句话**：**ms/step 线健康（中位 30.2–31.6，最好 26.9），但 tok/s ≥110 目前不可交付**
—— 163 发里只有 **1 发**可采纳的 ≥110（`faB_128k_r7`，A=3.446 @ ms=31.185 ⇒ **110.5**），
另 7 发放到 ≥110 的全部来自 `MOE_ZERO` 会话（已判定为"换吸引子的复读"，**不可采纳**）。

> **tok/s 口径（v4 修正）**：**以 `A × 1000 / ms = 110.5` 为准**（主 Agent 核过原始 jsonl）。
> v3 包 `EXPECTED_PERF.md` 曾写 **110.94**，那是 jsonl 的 `decode_tok_s` 字段
> （`gen_tokens/decode_window = 256/2.308`），**与 `A×1000/ms` 不是同一算法**。

⚠️ **v3 的一句话结论要改**：v3 写"A 一旦进优模式就能冲过 110 ⇒ A 决定一切"。
v4 的 163 发证据表明 **A 不是一个可优化的量，它是每发的抽签结果**（§2.2），
而且**脏会话的 A 反而更高**（§3）⇒ **A 只能当"装对没装对"的锚点，不能当绩效指标**。

---

## 1. 128K 单流逐发原始数据（faA/faB，8 发/臂）

### 1.1 faA（PGO 关）

| # | ms/step | A | tok/s | TTFT(s) | prefill(tok/s) |
|---|---|---|---|---|---|
| r1 | 31.482 | 2.695 | 85.60 | 24.67 | 5312 |
| r2 | 32.342 | 2.783 | 86.04 | 20.94 | 6260 |
| r3 | 31.596 | **1.356** | 43.10 | 20.87 | 6282 |
| r4 | 32.010 | 2.835 | **87.88** | 20.88 | 6276 |
| r5 | 31.439 | 2.677 | 84.82 | 20.90 | 6270 |
| r6 | 31.698 | 1.614 | 51.12 | 20.81 | 6300 |
| r7 | 31.655 | 2.684 | 85.13 | 20.81 | 6298 |
| r8 | **30.662** | 1.695 | 55.29 | 20.83 | 6293 |
| **中位** | **31.626** | **2.681** | **84.97** | 20.90 | 6273 |

文件：`logs_meta/samples/p42_t4_quote_131072_faA_128k_r{1..8}.jsonl`

### 1.2 faB（PGO 开）★推荐

| # | ms/step | A | tok/s | TTFT(s) | prefill(tok/s) |
|---|---|---|---|---|---|
| r1 | 30.103 | 2.824 | 92.72 | 24.71 | 5305 |
| r2 | 30.619 | 2.734 | 88.94 | 20.94 | 6259 |
| r3 | 29.792 | 1.656 | 55.80 | 20.89 | 6275 |
| r4 | 30.146 | 2.763 | 91.31 | 20.93 | 6263 |
| r5 | 30.066 | 2.763 | 91.55 | 20.91 | 6269 |
| r6 | 30.315 | 1.735 | 57.45 | 20.87 | 6280 |
| r7 | 31.185 | **3.446** | **110.5**‹ | 20.92 | 6266 |
| r8 | **30.649** | **1.181** | 38.67 | 20.97 | 6250 |
| **中位** | **30.230** | **2.748** | **90.13** | 20.92 | 6265 |

文件：`logs_meta/samples/p42_t4_quote_131072_faB_128k_r{1..8}.jsonl`
‹ r7 的 tok/s = `A×1000/ms` = **110.5**（权威口径）；同一条 jsonl 的 `decode_tok_s` 字段写 110.935（另一算法）。
其余各行的 tok/s 由同一公式复算，与 jsonl 字段一致到 0.1。
（PGO 服务侧收益 = 31.63 → 30.23，**−4.4%**，同会话 A/B 两臂各 8 发。）

---

## 2. ★ A 不是单峰：163 发全量形态分类（v4 新增，最重要的一节）

用 `reports/a-basin-and-acceptance-shape.md` §6 的方法对
`logs/perf/a21/p42_t4_quote_131072_*.jsonl` **全部 163 发**分类。

### 2.1 分类判据（与报告 §1/§6 逐字一致）

```
steep  = A ≥ 3.3  且  pos 严格单调  且  decay = (pos1 − pos4)/pos0 ≥ 0.45   ← 健康：逐字引用原文
flat   = A ≥ 3.3  但  pos 不单调                                          ← 健康度差：复读循环
shallow= A < 3.3                                                          ← 连贯但"没在抄原文"
```
（`posᵢ` = 第 i 个 draft token 的接受率；`A = 1 + Σ posᵢ`。`flat` 的 `decay ≤ 0.25`；
`tail = pos3 + pos4` 只作辅助，且**要求 steps ≥ 40**，短窗会系统性低估。）

### 2.2 163 发结果

| 类别 | n | 占比 | A 中位 | ms 中位 | 文本 |
|---|---|---|---|---|---|
| **steep**（健康） | **25** | **15.3%** | 3.45 | 34.3 | 连贯（182 条取证里 steep 0/14 复读） |
| **flat**（复读） | **12** | **7.4%** | 4.7 | 31.1 | **复读**（取证里 flat 6/6 复读） |
| **shallow** | **126** | **77.3%** | 2.75 | 34.5 | 多半连贯 |

**⇒ 结论：A 的分布是"三个吸引子"**，不是"一个优模式 + 噪声"：

* **steep 率 ≈ 15.3%**，且这是**每发抽签**，不是配置属性 —— 按时间排序，steep 在
  MoE AllGather **之前和之后都出现**（`fmc2b` 00:40、`nohot` 02:20、`lws128` 03:07、
  `base` 04:01、`rtcore3` 04:10、`hcclbuf2k` 04:23、`final` 06:07 / AG 之后：`sptok5` 08:40、
  `s5cap` 08:52、`f3b` 09:45、`faB` 18:09…）。同配置的 `rtcore3` 是 3/3、`ag_rtcore3` 是 0/3。
* **`flat` 的 A 比 `steep` 更高（4.7 vs 3.45），但文本是复读** ⇒ **A 高 ≠ 质量好**。

### 2.3 ≥110 tok/s 的真实分布（**v3 叙述的更正**）

| | n | 说明 |
|---|---|---|
| ≥110 tok/s（163 发中） | **8** | |
| 其中来自 `MOE_ZERO` 会话 | **7** | `MOE_ZERO` 已判**不采纳**（把系统推到另一个吸引子 + 非数值等价改动）⇒ 这 7 发不可采纳 |
| **真正可交付的 ≥110** | **1** | `faB_128k_r7`：A=3.446 @ ms=31.185 ⇒ **110.5 tok/s**（`A×1000/ms` 口径），pos 严格单调、decay=0.53、文本连贯 |

### 2.4 达标算术（下一步该攻什么）

`tok/s = A × 1000 / ms/step`：

| 组合 | tok/s | 是否已观测 |
|---|---|---|
| A=3.446 @ ms=31.18 | **110.5** | 是（**1 发**） |
| A=3.0 @ ms=27.3 | 110 | A≥3.0 有 40 发、ms≤28 有多次，**但从未同时出现** |
| A=2.75 @ ms=25.0 | 110 | ms 从未低于 **26.9** |

⇒ **两条可选路线**：① 把 **steep 率**从 15% 往上推；② 把 **ms 从 30.2 压到 ≤27.5**
（此时 A≥3.0 即可达标，而 A≥3.0 占 ~24%）。**②比追 steep 更现实。**

### 2.5 时间线否证了「某个补丁让 A 变差」

steep 在 MoE AllGather / SP_TOKENS / F3 等每个补丁的前后都出现过（§2.2 的时间线）
⇒ **A 的形态不是配置属性**。所以"某补丁 A/B 让 A 掉了 0.3"这类结论在 N<50 发时基本是噪声。

---

## 3. ⚠️ 用什么当绩效指标：**clean-rate，不是 A**

来源：`reports/session-attractor-and-clean-rate.md`（132 行）。

* **同一个配置、同一个 prompt、同一台机器，两次起服的"干净请求占比"差 2.3 倍**（37% vs 16%）；
* 而 **A 在脏会话里反而更高**（S1 clean 37% / A 中位 2.17；S2 clean 16% / A 中位 2.91）
  ⇒ **A 与质量反相关**；
* **clean 判据**：`pos0 ≥ 0.8`（`pos0` 观测上近双峰：干净 0.83–0.95 / 被污染 0.17–0.63，**没有中间值**）。

**⇒ v4 的测量纪律（写进验收判据）**：

1. **不许再用 A 的绝对值当绩效指标**。必须报 **`(clean-rate, ms/step)`** 或 **"clean 会话里的 A"**。
2. `ms/step` **不受污染影响**（27–29 ms 在干净/脏会话里都一样）⇒ **时延优化线照常**。
3. **真正要攻的是 clean-rate**：它跨起服波动 0→100%，是所有"结果不可复现"的根源。
4. 统计检验器（Fisher 等）**必须先用教科书用例自检**——本项目已经栽过两次工具 bug
   （`spread` 假性干净、`fisher()` 在相同表上给 0.0000）。本包附 `tools/fisher_recheck.py`。

---

## 4. 32K / 8K 单流

| 上下文 | 配置 | n | ms/step（中位） | A（中位） | tok/s（中位） | TTFT |
|---|---|---|---|---|---|---|
| **32K** | PGO 关（faA） | 2 | 32.159 | 1.661 | 51.73 | 4.88 s |
| **32K** | PGO 开（faB） | 2 | **30.512** | 1.836 | **60.38** | 4.88 s |
| **8K** | PGO 关（faA） | 2 | 31.803 | 1.633 | 51.61 | 1.35 s |
| **8K** | PGO 开（faB） | 2 | **28.892** | 1.672 | **58.04** | 1.33 s |
| 8K（warmup 发，PGO 开） | faB_w | 1 | 30.381 | 1.610 | 53.56 | 3.42 s |

> **A 在中短上下文明显更低**（8K 1.6–1.7、32K 1.7–1.8、128K 2.7），
> 这与"上下文越长、draft 越准"的直觉一致。⇒ A2 上 **128K 是 A 最好的场景**，
> 也最容易被误判：先跑 8K 会以为整套配置只有 ~52 tok/s。

---

## 5. `tok/s` 的算法（不要凭感觉估）

```
tok/s = A × 1000 / ms_per_step
```

| A \ ms | 29.0 | 30.2 | 31.6 | 33.0 |
|---|---|---|---|---|
| 1.2 | 41.4 | 39.7 | 38.0 | 36.4 |
| 1.7 | 58.6 | 56.3 | 53.8 | 51.5 |
| 2.7 | 93.1 | **89.4** | 85.4 | 81.8 |
| **3.45** | **119.0** | **114.2** | 109.2 | 104.5 |

⇒ 达标线（110 tok/s）在 `ms=30.2` 时要求 **A ≥ 3.32**；在 `ms=31.6` 时要求 **A ≥ 3.48**；
在 `ms=27.5` 时只要求 **A ≥ 3.03**。

---

## 6. 其他参考量（A2 上**必须现场重测**或按方法测）

| 项 | A3-node1 实测 | A2 期望 | 判定方式 |
|---|---|---|---|
| KV 池 tokens（GPU_UTIL=0.94，BF16 KV） | 3.39M → **4.16M**（开 MOE_AG 后） | 需现场测，**≥ 3,145,728（3Mi）** 才算过 | `grep -oE "GPU KV cache size: [0-9,]+ tokens" serve.log` |
| Engram 常驻 DRAM | ≈ **206 GiB** | **≈ 206 GiB（与平台无关）** | `docker stats` / 日志里的 host-resident 行 |
| Vision 23 例 | **23/23** | ≥ 19/23 | `results/*/vision.json` |
| GSM8K-200（chat） | **198/200、199/200、197/200**（三次） | 同量级（≥197） | `results/*/gsm8k.json`；原始 `A3-node2:logs/perf/gsm_{gate0,lo_on,qrot}.log` |
| 128K prefill | 6,250–6,300 tok/s | 会明显更低（910B） | jsonl 的 `prefill_tok_s` |
| static_kernel 降级检查 | 0 命中 | **必须 0** | `grep -ac "static_kernel.py:650" serve.log` |

### A2 与 A3-node1 的**平台差异**（不要把差异当成失败）

| 维度 | A3-node1（我们） | A2 |
|---|---|---|
| 芯片 | 8×910C（16 逻辑 die），1 TiB HBM | 8×910B3，512 GiB HBM |
| CPU | Kunpeng 640 核（宿主派发快） | **Kunpeng-920 192 核（宿主派发慢一个量级）** |
| 已知历史结果 | 无投机 ms/round ~27.5，draft ~1–4 ms | 无投机 **18.9 ms/round（更快！）**，draft **~26 ms/轮（慢 10×）** |

⇒ **A2 的 ms/step 主战场是 host 派发**，所以：
1. `PYTHON_PGO=1` 在 A2 上的相对收益**应当大于**我们的 −4.4%（我们 CPU 强，host 占比小）；
2. `DRAFT_GRAPH=1`（draft 入图）在 A2 上的潜在收益远大于我们（我们只有 −0.41 ms，A2 是 ~24 ms/轮）；
3. **A 与平台无关**，是"软件装对没装对"的最佳锚点 —— 与我们的中位差 ≤ ±15% 就说明装对了
   （但**只看 8 发中位**，不要看单发；且按 §3 同时报 clean-rate）。

---

## 7. ★ 多 batch / 多轮对话（v4 新增｜生产口径）

### 7.1 为什么这是 v4 最大的一块

**v3 及以前："并发"只覆盖 decode 并发，prefill 被故意串行化。**
历史 GSM8K / C-Eval 全部用 `--conc 4 --serialize-prefill 1`，而该开关的语义
（`acc_eval_p4s.py:173-174` 的 help 原文）是
**"hold a global lock until first token (avoid concurrent prefills)"**。
而且**我们验证的口径与 A2 生产不一致**：性能线是 `MAX_SEQS=1 + PREFIX=0`，
A2 生产是 **`--max-num-seqs 32` + prefix caching ON**。

⇒ **多轮对话（长历史 + 前缀复用）与"prefill+decode 真同时在跑"从未测过。**
本节的测量就是补这个空白。工具：`tests/multibatch/multibatch_gate.py`（三块 A/B/C），
启动器：`tests/multibatch/multibatch_session.sh`（`MAX_SEQS=32 PREFIX=1`）。

### 7.2 三块与判据

| 块 | 内容 | 判据 |
|---|---|---|
| **A 多轮对话** | 8 轮增长历史；第 1 轮埋 3 个随机 10 位串，之后每轮考一个；最后从全长历史再各问一次 | 逐字召回命中率 = 100%；每轮记录 `uniq2`（复读判据）与 `prompt_tokens` |
| **B 并发 batch** | **同一组 16 道算术题**，`conc=1` 与 `conc=8` 各跑一遍，**逐 item 比对** | **逐项不一致 = 0**（比"比准确率"灵敏得多） |
| **C 长短交错** | 1 条 128K 请求 + 2 s 后并发 6 条短请求（此时长请求被 chunked prefill 切成多个 chunk） | 短请求逐 item 正确性 vs 串行基线 ⇒ **逐项不一致 = 0** |

### 7.3 实测结果：生产口径 `MAX_SEQS=32 PREFIX=1` —— **三块全过**

> 2026-09-16 21:12 CST｜A3-node1 chips 8-15｜容器 `dsv41-a21-perf`｜真权重
> 权威报告（**逐字复制在本包**）：`reports/multibatch-and-mixed-load.md`（88 行）
> 原始日志：`A3-node1:/tmp/multibatch_session.log`、`A3-node1:$P/logs/perf/mbgP/`

**起服事实**：`MAX_SEQS=32 PREFIX=1` 起服 READY，`static_kernel.py:650` 降级 = **0**，
`CAPTURE_SIZES` 自动扩到 **15 个桶**（`1,2,3,4,6,8,12,16,20,24,32,40,48,96,192`）
—— 即 §7.4 的第 1 条风险（手写漏配 ⇒ 起不来）**在本包里由自动推导解决**。

| 块 | 臂 / 口径 | 结果 |
|---|---|---|
| **[A] 多轮对话**（8 轮增长历史 + 3 个随机针） | 轮内召回 | **7/7**（`turn2 k0 / turn3 k1 / turn4 k2 / turn5 k0 / turn6 k1 / turn7 k2 / turn8 k0` 全 OK） |
| | 末次**全长**历史再问 | **3/3**（`recall k0/k1/k2` 全 OK） |
| | 每轮 `uniq2`（复读判据） | **1.00**（**无复读**） |
| | 最终 `prompt_tokens` | **409** |
| | 每轮延迟 | **0.2–0.3 s** |
| **[B] 并发 batch**（16 题，`conc=1` vs `conc=8`，**逐 item 比对**） | `conc=1` 串行基线 | **16/16**（墙钟 3.7 s） |
| | `conc=8` | **16/16**（墙钟 8.2 s） |
| | **逐项不一致** | **0 项** ✅ |
| **[C] ★ prefill + decode 真正同时跑**（1×128K 先跑，**2 s 后**并发 6 条短请求） | 基线（短请求单独） | **6/6** |
| | 与 128K prefill **并发** | **6/6** |
| | **逐项不一致** | **0 项** ✅ |
| | 长请求本身 | 正常返回（**131,072 prompt tokens**，输出为连贯的红楼梦原文：`，什么没看过的戏，我不去。"凤姐道："他们那里凉快，两边又有楼…`） |

**判据（机器可读）**：`summary.json` 的
`{"A_needle": "PASS", "A_recall": "PASS", "B_itemwise": "PASS", "C_short_vs_long": "PASS"}`。

**⇒ 结论**：在"128K chunked prefill 与多条短请求 decode **同时进行**"这一**此前从未覆盖**的边界下，
未观察到任何正确性退化。

### 7.3.1 ⚠️ 局限（**必读，别把"全过"当成"无风险"**，抄自权威报告 §2）

1. **难度天花板**：B 块的 16 题是**简单算术**，两种并发度都 16/16 ⇒ 该测试能抓"整体性损坏"，
   **抓不到细微退化**。真要提灵敏度需换成"容易算错/需要长推理"的题，或加大样本。
2. **多轮历史偏短**：8 轮只到 **409 tokens**，**没有把长历史（几 K）+ 前缀缓存命中逼出来**。
3. **单次测量**：每块只跑 1 遍，**未重复** ⇒ 真实置信度有限。
4. **只测了 `PREFIX=1` 一臂**（生产对齐）。`PREFIX=0`（每请求都真 prefill、混合更凶）
   **尚未跑** —— 可用本包 `tests/multibatch/run_prod_both.sh` 跑（它会自动补这一臂）。
5. 只覆盖**功能性正确性**，**不涉及**"是否逐位确定"（那是正确性线的 `spread` 判据，
   见 `CORRECTNESS_STATUS.md` §2）。两个判据**不可互相代替**。
6. 复查（本包打包方补充）：**B 块 `conc=8` 的墙钟 8.2 s 反而比 `conc=1` 的 3.7 s 长**
   —— 这与"16 题简单算术、单题极短"有关（并发时每步 batch 变大但每题步数少，
   调度/前缀开销占主导）。**不要**从这一行推断并发吞吐，本块的目的是**逐项正确性**，不是吞吐。

### 7.3.2 这一节本身的价值：它覆盖的是**此前的空白**

历史 GSM8K / C-Eval 用的是 `--conc 4 --serialize-prefill 1`，而该开关的语义
（`acc_eval.py` 的 help 原文）是 **"hold a global lock until first token (avoid concurrent prefills)"**
⇒ **decode 并发、prefill 被故意串行化**。
而 A2 生产是 `max-num-seqs 32` + **prefix caching ON**，前缀缓存的本质就是
"让 decode 队列里随时插入新请求" ⇒ **该边界在生产中天然存在、却从未被测过**。
**[C] 块是它的首次覆盖** —— 这是 v4 相对 v3 最重要的交付价值。

### 7.3.3 **你自己怎么跑**（结果目录里会生成 `summary.json`；**A2 上第一件事就是复现这一节**）

```bash
# ① 一臂：生产对齐（MAX_SEQS=32 + PREFIX=1）—— 起服会比单流慢很多（15 个 capture 桶）
MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq MODE=prod bash scripts/run_test.sh
#   产出：results/<run_id>/{summary.json,multiturn.json,concurrency_c8.json,mixed_long_short.json}
#        + REPORT.md 的 §1b 表（make_report.sh 会自动读 summary.json）

# ② 两臂对照（PREFIX=1 与 PREFIX=0，各起一次服；脚本最后自动打一张对照表）
MODEL=/path/to/... bash tests/multibatch/run_prod_both.sh

# ③ 只跑其中一块 / 缩小规模（省时间）：
MODEL=... MODE=prod MBG_SKIP=A MBG_ROUNDS=4 MBG_LONG_CTX=32768 bash scripts/run_test.sh
```

**怎么判"过了"**：看 `summary.json` 的 `verdict`（机器可读，`multibatch_gate.py` 的退出码同源）：

```json
{"A_needle": "PASS", "A_recall": "PASS", "B_itemwise": "PASS", "C_short_vs_long": "PASS"}
```

* `A_*`：多轮对话逐字召回 **必须 100%**（轮内 + 从全长历史再问一次）；
* `B_itemwise`：`conc=1` 与 `conc=8` **逐 item 正确性不一致数必须为 0**；
* `C_short_vs_long`：与 128K 长请求并发时，6 条短请求的**逐 item 不一致数必须为 0**。

任何一项 FAIL，`results/<run_id>/*.json` 里都有**逐 item 的两侧原始输出**（`mismatch` 字段），可直接回报。

### 7.4 已知风险（**跑之前先知道**）

1. `MAX_SEQS=32` 要求 `CAPTURE_SIZES` 覆盖到 `32 × (1+SP_TOKENS) = 192`，否则大 batch 被
   padding 到"没有图"→ 报错或退回 eager。v4 的 `serve_a2.sh` 已自动推导
   （`...,40,48,96,192`，15 桶）；**手写漏掉会直接起不来**。
2. 生产口径的图捕获比单流慢得多（每桶 ~10–30 s，11 桶 → **15 桶**）⇒ 起服时间明显变长；
   我们观测到 `rejection_sampler` 的 Triton warmup 另外要 **~20 s/rank**。
3. **生产口径的 ms/A 与单流口径不可比**，不要拿 §1 的 30.2 ms 去对 §7 的数。

---

## 8. 与 `reports/` 里数字的差异说明（重要，别当成矛盾）

| 报告 | 它写的 | 实际 jsonl | 说明 |
|---|---|---|---|
| `reports/milestone-ms-target-met.md`（17:00） | 128K×8：med ms **31.392**、med A **2.657**、峰值 **109.93 tok/s** | 同名的 `faA_128k_r1..r8.jsonl` 在 17:57–18:00 被 PGO A/B 覆盖 | 报告的**结论没变**（ms 中位 ~31.4–31.6、A 中位 ~2.66–2.68、峰值 ~110），但**逐发数值对不上**（文件被覆盖） |
| `reports/cpython-pgo-verified.md`（16:06） | 宿主 micro-bench −16~23%，服务侧收益"未测" | — | 服务侧收益由本轮 faA/faB 补上：**−4.4%** |
| v3 自己的 `README.md` §0 | "A 的方差 2.9× ⇒ 跑 8 发看中位" | 163 发全量 | v4 更正为 **A 是三吸引子抽签**（§2）+ **A 不能当绩效指标**（§3） |

**本文件所有数字都直接来自 `logs_meta/samples/` 的原文件或上面点名的四份报告**
（`a-basin-and-acceptance-shape.md`、`session-attractor-and-clean-rate.md`、
`draft-graph-negative-control.md`、`multibatch-and-mixed-load.md`）；
`logs_meta/samples/` 的 19 个 jsonl 可用 `python3 tools/analyze_samples.py logs_meta/samples` 复算，
§2 的分类可用 `python3 tools/steep_summary.py` 在 A2 自己的 `results/` 上复现。

---

# §A2GAP —— A2 vs A3 的性能差距拆解（v6 新增）

> 数据来源：A2 首次成功起服后的长 decode 日志（`V41_ENGRAM_ROUTE_PROBE=1` + `[bneck]` 探针），
> 与 A3 同期同配置会话（`faB` / `mbgP`）逐字段对比。两侧都是 ~8K 上下文、单请求、S=5、真权重。
> **口径已对齐**，可比。

## A2GAP.1 原始对比（中位数）

| 字段 | A2 | A3 | 比值 |
|---|---|---|---|
| **`hp` = ms/step**（设备侧自测步钟） | **75.67** | **28.67** | **2.64×** |
| `total`（Engram host 全路径） | 9.22 | 2.97 | 3.10× |
| `d2h` | 5.04 | 1.00 | **5.04×** |
| `hash` | 0.165 | 0.070 | 2.36× |
| `route` | 1.85 | 0.79 | 2.35× |
| `pad` | 0.251 | 0.120 | 2.09× |
| **非 Engram 部分**（`hp − total`） | **66.5** | **25.7** | **2.59×** |

**`hp` 的含义**：`model.py` 的 `_BneckState.mark_step()` 在 `prepare_engram_inputs()` 开头调用，
记录相邻两步的 `perf_counter` 差 ⇒ **它就是 ms/step**，无需客户端即可读。
自校验：A2 `A=2.97 / tok/s=39.2` ⇒ `2.97×1000/39.2 = 75.8 ≈ hp 75.67` ✓

## A2GAP.2 ★ 差距分解：**87% 不在 Engram 里**

```
总差距 47.0 ms
├─ Engram host 贡献    6.2 ms  （13%）
└─ 其余                40.8 ms  （87%）   ← 真正的战场
```

**Engram host 那 9.22 ms 里能榨的顶多 ~2 ms** ⇒ **不要再往这个方向投入**。

## A2GAP.3 接受率反而更好 ⇒ 这是纯时延问题

| | A2 | A3 |
|---|---|---|
| `Mean acceptance length` | **2.77 / 2.97** | 2.74 |
| 逐位置 | `0.712 / 0.523 / 0.348 / 0.227 / 0.159` | — |
| 形态判定 | **严格单调，decay=0.51 ⇒ 健康族** | 健康 |

⇒ **tok/s 的差距 100% 来自 ms/step，一点都不是接受率的锅。**

## A2GAP.4 Engram 各相位均匀慢 2.1–2.6× ⇒ 这是纯 CPU 比值

`route` 内部 7 个相位（A2 53 行采样 vs A3 同窗口 n=1900）：

| 相位 | A2 | A3 | 比值 |
|---|---|---|---|
| `a2a` | 0.393 | 0.183 | 2.15× |
| `bcast` | 0.219 | 0.094 | 2.33× |
| `evt` | 0.193 | 0.089 | 2.17× |
| `h2d` | 0.123 | 0.058 | 2.12× |
| `lookup` | 0.557 | 0.222 | 2.51× |
| `plan` | 0.157 | 0.064 | 2.45× |
| `scatter` | 0.067 | 0.026 | 2.58× |
| **合计** | **1.709** | **0.728** | **2.35×** |

**全部落在 2.1–2.6×** —— 不是某一项特别差，而是**整体等比例慢**。

## A2GAP.5 CPU 差异（机制）

| | A2 | A3 |
|---|---|---|
| CPU part | Kunpeng-920（A76 级） | **`0xd02`**（TaiShan-v120 / Neoverse 级） |
| 逻辑核 | 192 | 640 |
| **容器绑核** | **144-167 = 24 逻辑 / 12 物理**（SMT2） | **320-639 = 320 逻辑 / 160 物理** |
| 物理核比 | **12** | **160** ⇒ **13.3×** |
| 实测负载 | **平均 60%**（**没有核饱和**） | — |
| `OMP_NUM_THREADS` | 1（镜像自带） | 1（镜像自带） |

**60% 这个数很关键**：它说明**没有核在打满**，所以：

* 不是"核数不够被榨干"，而是"**每个 host 操作都按 IPC 比值慢一点**"的全局效应
  （draft 派发受 GIL 限制是**主线程串行**，它的绝对延迟直接进 ms/step）
* ⇒ 单纯放宽绑核（`CPUSET=-1`）预期收益**有限**，但值得一试（超线程配对问题）

## A2GAP.6 `d2h` 是 5× —— 但它是**症状不是病因**

`d2h` 测的是（`model.py`）：

```python
_t0 = perf_counter()
ids_host = input_ids[:n].cpu().long()      # n=6，只有 48 字节
pos_host = positions[:n].cpu().long()
_bp.stat("d2h", _t0)
```

**`.cpu()` 在 torch_npu 上是阻塞的** —— 它要等设备把此前排队的工作做完。所以：

> **`d2h` 测的不是拷贝耗时，而是"Engram host 路径开始时，设备还剩多少活没干完"。**

A2 5.04ms vs A3 1.00ms ⇒ A2 走到这个点时设备队列里还压着 ~4ms。
这也解释了它的大抖动（A2 单 rank 0.59→9.28，A3 0.19→2.05）。
**⇒ 让 host 侧与设备侧重叠，最多回收 `hash+route ≈ 2ms`**（不是 5ms）。

## A2GAP.7 把 66.5ms 的非 Engram 部分劈开（**待做的实验**）

A2 历史锚点（`enable_engram:false`、S=7、MC2，py-spy 实测）：

```
53.7 ms = draft(eager dispatch) 26 + verify(主干) 18 + 编排 7
验证依据：无投机主干实测 18.9ms ≈ verify 18ms
```

现在（Engram on、S=5、AllGather）非 Engram = **66.5ms**，比历史三项之和（51.9）**多 14.6ms**。
三个候选，**必须实验区分**：

| 候选 | 说明 |
|---|---|
| ① draft 比历史更慢 | 历史 S=7 + MC2；现在 S=5 + AllGather，MoE 路径变了 |
| ② Engram **设备侧**注入 | `lookup=(2048,6144)` 注入 attention 第 1/14 层，**这部分不在 9.22ms 的 host total 里** |
| ③ 编排变慢 | 12 个物理核跑 8 worker + API server，采样/调度/图重放的 host 开销 |

### 实验 E2（**唯一能定方向的**，~12 min）

```bash
docker rm -f dsv41-a2
MODEL=... SPEC=0 bash scripts/serve_a2.sh     # 关投机
# 起服后打一个 8K 请求，等 20 步读 hp
curl -s http://127.0.0.1:8100/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41","prompt":"用一句话介绍你自己","max_tokens":128,"temperature":0}' >/dev/null
sleep 20; grep -oE "hp=[0-9.]+" results/*/serve.log | tail -1
```

| `hp`（SPEC=0） | 推论 | 下一步 |
|---|---|---|
| **≈19 ms** | draft ≈ 56ms ⇒ **draft 是主矛盾** | `DRAFT_GRAPH`（唯一大杠杆） |
| **≈35 ms** | draft ≈ 40ms，主干也多 16ms ⇒ **Engram 设备侧注入或主干本身** | 查 ② |

### 实验 E3（~12 min，可选）

```bash
MODEL=... CPUSET=-1 MEMS=-1 CPU_BIND=1 bash scripts/serve_a2.sh
```

让 **vLLM 自己**按 NUMA 拓扑精细绑（`enable_cpu_binding` 默认 True；它会绑
`acl_thread`/`release_thread` 并处理 **NPU 中断**，比我们"取整个 NUMA 节点"更细）。
**v6 已默认 `--ulimit memlock=-1`**；注意 `CPU_BIND` 在我们包里默认是 **0**，
所以**必须显式传 `CPU_BIND=1`**。

## A2GAP.8 目标可达性（**诚实结论**）

```
A2 现状：A=2.97 @ 75.67ms  →  39.2 tok/s
A3 现状：A=2.74 @ 28.67ms  →  95.6 tok/s      （A2 的 A 其实更好）
```

| 目标 | 需要 | 现实性 |
|---|---|---|
| 110 tok/s @ A=2.97 | **ms ≤ 27.0** | ❌ **A3 自己也只有 95.6** ⇒ 要比 A3 更快，不现实 |
| 消除 draft（历史 26ms → 0） | 75.67−26 = 49.7ms | → **60 tok/s** |
| 再把 Engram 压到 A3 水平（9.2→3.0） | 43.5ms | → **68 tok/s** |

⇒ **A2 的现实天花板约 60–68 tok/s**。建议把 A2 的目标重新定为
「**Engram 开 + 输出健康 的前提下，从 39.2 提到 ≥60 tok/s**」，
而不是 110（那已超出本平台架构）。

## A2GAP.9 一个需要复核的历史疑点

A2 历史生产值是 **82.5 tok/s @ A=4.43**。用**后来才建立**的 pos 形状判据看，
**A=4.43 超出观测到的健康上限（steep 最高 3.49）**，很可能落在 flat/退化族。

⚠️ **但这只是推测**：形状分类只在 A3 数据上做过，A2 历史的 pos 向量没留。
**在把 82.5 当基准之前，先确认那批输出是否健康。**
（判据：`tail = pos3+pos4`；健康 <0.5，退化 >1.5。见 §2 与 `reports/a-basin-and-acceptance-shape.md`。）
