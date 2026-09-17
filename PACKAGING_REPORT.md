# PACKAGING_REPORT.md —— 打包报告（a2_pkg_v5）

> 打包人：A2 交付测试包 v4 打包子代理｜打包时间：2026-09-16 20:5x–21:xx CST
> 起点：`a2_pkg_v3/`（153 文件，sha256 `18cb496bca7b40c29d6e16342499a1209c0831fd1ab25a76bd4bb9b68de935bf`）
> ⇒ `a2_pkg_v5/`（**增量**：新增 14 个文件、修改 7 个文件；**v3 原样保留**供对照）
> 纪律遵守：**未占卡、未起停任何容器、未 scp 到任何机器**。
> 只用了 `ssh -o BatchMode=yes A3-node1/A3-node2 '<cat/grep/ls/find 只读>'`（rsync 对端未安装 ⇒ 改用 `ssh cat`）
> 与本机文件读写 + 纯 CPU 校验（`bash -n` / `py_compile` / `DRY_RUN` 矩阵 / `tar` / `sha256sum`）。

---

## 1. 每个新数字的来源（**逐条到文件 + 行号**）

### 1.1 性能 / A 的形态（163 发全量）

| 数字 | 值 | **来源（文件:行）** |
|---|---|---|
| 128K ms/step 中位 | **30.2–31.6**（PGO 开/关），最好 **26.9** | `reports/a-basin-and-acceptance-shape.md:71` |
| A=3.446 ⇒ **110.5 tok/s**（1 发） | 110.5 | `reports/a-basin-and-acceptance-shape.md:74` |
| 163 发分类判据 | `steep = A≥3.3 且 pos 严格单调`；`flat = A≥3.3 但不单调`；`shallow = A<3.3` | 同上 `:88-89` |
| **steep = 25 发（15.3%）**，A 中位 3.45，ms 中位 34.3 | 25 / 15.3% | 同上 `:93` |
| **flat = 12 发（7.4%）**，A 中位 4.7，ms 中位 31.1 | 12 / 7.4% | 同上 `:94` |
| **shallow = 126 发（77.3%）**，A 中位 2.75，ms 中位 34.5 | 126 / 77.3% | 同上 `:95` |
| `≥110` 共 8 发，其中 **7 发来自 `MOE_ZERO` 会话**（不可采纳）⇒ **可交付的只有 1 发** | 8 / 7 / 1 | 同上 `:97-98` |
| `steep` 判据里的 `decay = (pos1−pos4)/pos0 ≥ 0.45`；`flat` 的 `decay ≤ 0.25` | — | 同上 `:33`（表）/ `:35` |
| "flat 6/6 复读、steep 0/14 复读"（182 条取证） | — | 同上 `:40` |
| 达标算术（A=3.0@27.3 从未出现；ms 从未低于 26.9） | — | 同上 `:106-108` |
| steep 在 AllGather **前后都出现**（时间线否证"某补丁让 A 变差"） | — | 同上 `:112-120` |
| 8 发逐发数据（faA/faB） | 31.626/2.681/84.97；30.230/2.748/90.13 | `logs_meta/samples/p42_t4_quote_131072_{faA,faB}_128k_r{1..8}.jsonl`（**原文件在包内**）+ v3 `EXPECTED_PERF.md` §1 |

### 1.2 clean-rate（为什么 A 不能当绩效指标）

| 数字 | 值 | **来源（文件:行）** |
|---|---|---|
| 同配置两次起服的 clean-rate 差 **2.3 倍** | 37% vs 16% | `reports/session-attractor-and-clean-rate.md:9` |
| S1（37%，A 中位 2.17）/ S2（16%，A 中位 **2.91**） | — | 同上 `:21-22` |
| **A 与质量反相关**（A 越高越可能是复读吸引子） | — | 同上 `:47-52` |
| 处置：必须报 **(clean-rate, ms/step)** | — | 同上 `:60-62` |
| clean 判据 `pos0 ≥ 0.8`（近双峰 0.83–0.95 / 0.17–0.63） | — | 同上 `:17`（双峰观察） |
| `MOE_ZERO` 交换引子：低峰消失，**高峰 0.86 → 0.73** | — | 同上 `:26-34`（§1.1 的逐发 `pos0` 序列 + 判读） |

### 1.3 4 个负结果（**本轮新出，写进 README §1.1**）

| 项 | 值 / 结论 | **来源（文件:行）** |
|---|---|---|
| `MOE_ZERO` 不采纳 | 低峰消灭但高峰 0.86→0.73；最常见 top-1 从 `</s>`(−0.776) 变 `《`(−4.201) ⇒ 非数值等价 | `reports/session-attractor-and-clean-rate.md:26-34`；A3-node2 `correctness-line.md:358-361` |
| `MOE_NONFINITE` 无差异 | clean **2/24 vs 2/24 = 8.3%**（2×2 `[[2,22],[2,22]]`） | `reports/session-attractor-and-clean-rate.md:124`（S4 修正表行，N=24）；原始 `/tmp/iab_s4.log`、`/tmp/iab.log` |
| `MOE_NONFINITE` 性能侧（128K 生产口径 3 发） | nf=1: A=2.024/1.536/1.977, ms=35.0/33.0/35.0；nf=0: A=2.336/1.401/2.265, ms=33.3/32.1/33.2 | **主 Agent 2026-09-16 直接提供**（本包未拿到原始 jsonl ⇒ 标为"二手"） |
| `LOCAL_OWNER=fast` vs `on` 无差异 | 2×2 `[[5,7],[4,8]]` → **修正后 p=1.0000**（旧 0.6843 作废） | `reports/session-attractor-and-clean-rate.md:121`（修正表行） |
| `HCCL_DET=true` 无差异 + **不能进交付** | `[[5,7],[3,9]]` → **p=0.6668**；**GSM8K 91/100** | 同上 `:123`；A3-node2 `correctness-line.md:139` / `:188` |
| **Fisher 工具 bug 与修正** | 相同表曾打印 0.0000；修正表见 CHANGELOG §4.1；已用教科书用例校验 | `reports/session-attractor-and-clean-rate.md:95-101`（§6 更正）+ `:119-130`（重算表）；主 Agent 21:1x 逐条确认 |

### 1.4 DSpark 入图（正控待验）

| 数字 | 值 | **来源** |
|---|---|---|
| 负控 A 恒 **1.000（8/8 发）** | r4–r8: 1.008/1.000/1.000/1.000/1.000 | `reports/draft-graph-negative-control.md:14-19` |
| `dspark-graph-capture` 打印次数 = **0** | 0 | 同上 `:22` |
| **ms/step 仍 30.2–30.8**（⇒ 单看时延发现不了） | 30.215–30.764 | 同上 `:16-19` / `:33` |
| 待办：正控未跑 | — | 同上 `:44-47` |

### 1.5 精度 / 数值确定性（新增 `CORRECTNESS_STATUS.md`）

| 数字 | 值 | **来源（文件:行，A3-node2）** |
|---|---|---|
| **不存在 ctx 阈值**：同一 ctx(21504) 三次重复 | **0.913 / 0.000 / 1.435** | `wt-graph/reports/correctness-line.md:295`（表行）/ `:303-304`（结论 1） |
| 16384 也会翻转（0.000 → 0.427） | — | 同上 `:293-299`（表：会话1 0.000 / 会话2 重扫 0.427）/ `:305-306`（结论 2） |
| 弱规律：spread=0 出现率 | 16384=2/3、20480=2/6、21504=1/6 | 同上 `:312` |
| clean 判据 | `uniq_top1==1 AND n_distinct_lp==1` | 同上 `:335`（§7 标题）/ `:341-346`（标准判据框） |
| **`SPEC=0` clean 0.50 / 0.50** | 5/10、5/10 | 同上 `:365-368` |
| **`SPEC=0` clean 0.5625（9/16）** | 9/16 | **只在 `/tmp/spec0_rep2.log`（20:41）**；报告 §9 写于 20:36 ⇒ 表内没有这一行（重要！） |
| **`SPEC=1` clean 0.10** | 2/20 | 同上 `:369` |
| Fisher 单侧 p | **0.0256**（表 `[[5,5],[2,18]]`，= 本包 `p_right`） | 同上 `:371` |
| "减少 forward 数"被否证 | 43/40=1.075，需 `1.075^k=5` ⇒ k≈22.4 | 同上 `:375-376` |
| clean × coherent **2×2 相同** | clean 4/1；dirty 4/1 | 同上 `:385-390`（探针 `clean_vs_quality.py`，原始 `/tmp/cvq2.log`） |
| `SPEC=0` 无 draft ⇒ ms/step 36.1–36.3 | — | 同上 `:393` |
| 精度矩阵：**Vision 23/23**、`HCCL_DET=true` **91/100**、`strict` **100/100** | — | 同上 `:135-140`（核心表）/ `:180-189`（§P1 矩阵） |
| GSM8K-200 三次 | **198/200、199/200、197/200** | `wt-graph/logs/perf/gsm_gate0.log:16`、`gsm_lo_on.log:16`、`gsm_qrot.log:16` |
| 128K 的 A 未收敛（中位 1.69–2.47） | — | `correctness-line.md:160-166`（§6 表） |

### 1.6 多 batch / 多轮 / prefill-decode 混合（v4 最大的一块）

**来源**：`reports/multibatch-and-mixed-load.md`（88 行，**已逐字复制进本包**；
md5 与本机 `A3-node1:reports/multibatch-and-mixed-load.md` **一致**：`81e5c713f14dfd1b3ecd780fdbd2ba62`）。
原始日志：`A3-node1:/tmp/multibatch_session.log`、`A3-node1:$P/logs/perf/mbgP/`。

| 数字 | 值 | **来源（文件:行）** |
|---|---|---|
| 起服口径 | `MAX_SEQS=32 PREFIX=1`，READY，`static_kernel.py:650` 降级 **0** | 报告 `:22-24`（§1 起服段） |
| `CAPTURE_SIZES` 15 桶 | `1,2,3,4,6,8,12,16,20,24,32,40,48,96,192` | 同上 |
| **[A] 轮内召回 7/7** | turn2 k0 / turn3 k1 / turn4 k2 / turn5 k0 / turn6 k1 / turn7 k2 / turn8 k0 | 报告 `:32-38`（表）+ `:40`（逐轮） |
| **[A] 末次全长召回 3/3** | k0/k1/k2 全 OK | 同上 |
| [A] 每轮 `uniq2` = **1.00**；最终 `prompt_tokens` = **409**；每轮 **0.2–0.3 s** | — | 报告 §1[A] 表 |
| **[B] `conc=1` 16/16（3.7 s）、`conc=8` 16/16（8.2 s）、逐项不一致 0** | — | 报告 `:52-59`（表） |
| **[C] 基线 6/6、与 128K 并发 6/6、逐项不一致 0** | — | 报告 `:63-72`（表） |
| [C] 长请求正常返回（131,072 prompt tokens，输出连贯） | 原文引用 `，什么没看过的戏，我不去。"凤姐道："他们那里凉快，两边又有楼…` | 报告 `:74-76` |
| **局限 5 条**（难度天花板 / 多轮只到 409 token / 单次测量 / `PREFIX=0` 未跑 / 只覆盖功能性正确性） | — | 报告 §2（`multibatch-and-mixed-load.md` 第 78–88 行） |
| 复现命令 | `TAG=mbgP MSEQS=32 PREFIX=1 CONC=8 ROUNDS=8 bash exp_tools/multibatch_session.sh` | 报告 §3 |

> **打包方补充的两点（不改原文，另标注）**：
> 1. **B 块 `conc=8` 墙钟 8.2 s > `conc=1` 3.7 s** —— 这是"16 题简单算术、调度/前缀开销占主导"
>    的正常现象，**不能**从这一行推断并发吞吐（本块目的是逐项正确性）。已在 `EXPECTED_PERF.md` §7.3.1 第 6 条写明。
> 2. `PREFIX=0`（压力臂）**未跑** ⇒ 本包 `tests/multibatch/run_prod_both.sh` 已备好该臂，
>    在 A2 上可一条命令补齐（它的价值是"每请求真 prefill、混合更凶"）。

### 1.7 环境 / 平台数字（沿用 v3，未变）

| 数字 | 来源 |
|---|---|
| A2 = 8×910B3 + Kunpeng-920、draft ~26 ms/轮、verify 18.9 ms/round | 本机 `$WORKSPACE/A2_复现报告.md`（§0、§2.4、§3.1）—— 与 v3 同 |
| KV 门槛 3,145,728（3Mi）、Engram 常驻 ≈206 GiB | `A2_PACKAGE_SPEC.md` §4 —— 与 v3 同 |
| `--conc 4 --serialize-prefill 1` 的语义原文 | A3-node1 `scripts/acc_eval_p4s.py:173-174`（help：**"1=hold a global lock until first token (avoid concurrent prefills)"**） |

---

## 2. ★ 与 brief/旧包**对不上**的地方（用户特别要求核对，我逐条 SSH 核过原文）

### 2.1 `SPEC` 的 Fisher p：**0.0256（同一会话，单侧）** 与 **0.0042（跨会话，双侧）** 都对，但口径不同

* brief 原文："`SPEC=0` clean 0.50/0.50/0.5625 vs `SPEC=1` 0.10，Fisher **p≈0.0026**"。
* **原文**（`correctness-line.md:371`）逐字是：`**Fisher 单侧 p = 0.0256**（反方向 p = 0.998）⇒ 显著。`
  该值对应表 `[[5,5],[2,18]]`（`SPEC=0` 第一次 5/10 vs `SPEC=1` 2/20），**两次测量都在同一会话内**。
* **主 Agent 21:1x 确认**并给出后续跨会话复现：第三次 `SPEC=0`（**全新容器**）**9/16 = 0.5625**，
  与 `SPEC=1` 的 2/20 双侧 Fisher **p=0.0042**（单侧约 0.0026）——
  原始证据 `reports/session-attractor-and-clean-rate.md` §6.4 的 `[[9,7],[2,18]] → 0.0042`。
* **本包处置**：两个口径**都写**并标明表与单侧/双侧（`CORRECTNESS_STATUS.md` §3.1/§3.1.1）：
  `[[5,5],[2,18]]`→单侧 0.0256；`[[9,7],[2,18]]`→双侧 0.0042 / 单侧约 0.0026。
  本机用 `tools/fisher_recheck.py` 复算：`[[9,7],[2,18]]` 的 `p_right`=0.0039、`p_two`=0.0042 ✅与原文一致。
* **结论不变**：`SPEC=0` 的 clean 率显著高于 `SPEC=1`（约 5 倍），三次测量 0.50 / 0.50 / 0.5625 一致。

### 2.1.1 峰值 tok/s：**以 110.5 为准**（不是 v3 包里的 110.94）

* `faB_128k_r7` 的 `A=3.446 @ ms=31.185` ⇒ **`A×1000/ms = 110.5`**（主 Agent 核过原始 jsonl，**以此为准**）。
* v3 包 `EXPECTED_PERF.md` §1.2 表里的 **110.94** 是 jsonl 的 `decode_tok_s` 字段
  （`gen_tokens/decode_window` 口径：256/2.308 = 110.9），与 `A×1000/ms` **不是同一个算法**。
  `reports/a-basin-and-acceptance-shape.md:74` 的"110.5"是修正后的权威值。
* **本包处置**：一律写 **110.5**（并注明 jsonl 另有 `decode_tok_s=110.935` 这个口径差异）；
  `EXPECTED_PERF.md` §0/§1.2/§2.3/§5 已同步。两者都 ≥110，**不影响"≥110 只有 1 发可交付"的结论**。

### 2.2 "GSM8K 198/200（两轮）"：我核到的是 **198 / 199 / 197 三次**

brief 写"GSM8K **198/200**（两轮）"。A3-node2 上实际有三次 200 题口径的日志：

| 日志 | 行号 | 结果 |
|---|---|---|
| `logs/perf/gsm_gate0.log` | `:16` | **198/200 = 99.0%** |
| `logs/perf/gsm_lo_on.log` | `:16` | **199/200 = 99.5%** |
| `logs/perf/gsm_qrot.log` | `:16` | **197/200 = 98.5%** |

`reports/cannbot-sweep-final-verdict.md:22` 的汇总口径写 "GSM8K 198/200 ✓"。
**本包按三次原始日志写（全部 ≥197）**，并保留 198/200 这个汇总值作为出处引用。

### 2.3 "≥110 的另 7 发"：原文说的是 **7 发来自 `MOE_ZERO` 会话**，不是 3 发

`reports/a-basin-and-acceptance-shape.md:97-98` 逐字："`>=110` 只有 **8 发**，其中 **7 发来自
`MOE_ZERO` 会话**（已判定为"换吸引子的复读"，不可采纳）=> **真正可交付的 >=110 只有 1 发**"。
本包按原文写（`EXPECTED_PERF.md` §2.3）。

### 2.4 `MOE_NONFINITE` 的性能侧数字：**二手**（主 Agent 提供，本包未拿到原始 jsonl）

`nf=1` → A=2.024/1.536/1.977、ms=35.0/33.0/35.0；`nf=0` → A=2.336/1.401/2.265、ms=33.3/32.1/33.2。
**我已标为"主 Agent 直接提供"**，未当成一手数据。若需入正式结论，请把原始 jsonl 发我补进包。

---

## 3. 我**核不了**的（明确标出）

| 项 | 为什么核不了 | 包内怎么处理 |
|---|---|---|
| **A2 上能否真跑起来**（镜像 build、起服、多 batch 三块） | 无 A2 访问权（纪律：不占卡、不启停容器） | ① 全部脚本过 `bash -n`；② `python3 -m py_compile`；③ **`DRY_RUN=1` 12 组合矩阵全过**（`tests/multibatch/verify_serve_flags.sh` ⇒ `pass=12 fail=0`）；④ `MODE=prod` 与 `run_prod_both.sh` 写清失败处置 |
| **`MODE=prod` 在 NPU 上的真实行为**（`MAX_SEQS=32` 图捕获、prefix caching） | 同上（需要 8 卡） | `CAPTURE_SIZES` 自动推导逻辑**在本机用纯 shell 验过**：`MAX_SEQS=1` → `1,2,3,4,6,8,12,16,20,24,32`（**与 v3 逐字节相同**）；`=32` → `+40,48,96,192`（15 桶） |
| `multibatch_gate.py` 的端到端跑通 | 需要已就绪的服务（我们不许起服） | ① `py_compile` 过；② 与 A3-node1 的 `exp_tools/multibatch_gate.py` **逐段对齐**（只改参数化 + 加 `verdict`/退出码）；③ 判据写成机器可读并进 `REPORT.md` §1b |
| **多 batch 的实测数字** | ✅ **已回填**（2026-09-16 21:12 CST，生产口径 `MAX_SEQS=32 PREFIX=1` 三块全过） | `EXPECTED_PERF.md` §7.3（结果）+ §7.3.1（**局限 5 条 + 我方补充 1 条**）+ §7.3.3（如何自己跑）；权威报告 `reports/multibatch-and-mixed-load.md` |
| `token_dispatcher_moennf.py` 与镜像内基线的兼容性 | 需要那台机的 A3-node1 运行时 | 只作**负结果复现**用、默认不挂；挂了会在起服时 `md5sum` 落到 `serve_cmd.txt` 便于对照 |
| `reports/` 里 55 份报告的交叉引用 | 时间有限 | 新增 4 份逐字复制的**权威报告**已在 `EXPECTED_PERF.md`/`CORRECTNESS_STATUS.md` 里点名为"以它们为准"（前 3 份的 md5 已对本机校验一致） |
| **`MODE=full` 的 GSM8K 段在 v3 里会直接失败**（`tests/t_gsm8k.py` 依赖的 `tests/acc_eval.py` 没打进 v3 包） | 打包时才发现（本机核对 v3 文件清单 + 对照 A3-node1 的调用关系） | v4 补入 `tests/acc_eval.py`（= A3-node1 `scripts/acc_eval_p4s.py`，`--serialize-prefill` 默认 1 与历史口径一致），并把硬编码 `ENC_DIR` 参数化；缺官方 `encoding` 目录时 `t_gsm8k.py` 会**明确跳过并写出 `skipped` 标记**，不再抛 `ModuleNotFoundError` |

---

## 4. v4 的**验证记录**（纯 CPU，可复跑）

| 检查 | 命令 | 结果 |
|---|---|---|
| 脚本语法 | `bash -n scripts/*.sh tests/*.sh tests/multibatch/*.sh` | 全过 |
| Python 语法 | `python3 -m py_compile tests/*.py tests/multibatch/*.py tools/*.py` | 全过 |
| **启动器开关矩阵** | `bash tests/multibatch/verify_serve_flags.sh` | **`[flags] pass=12 fail=0`** |
| **单流口径不漂移** | `DRY_RUN=1 bash scripts/serve_a2.sh \| grep CAPTURE_SIZES` | `1,2,3,4,6,8,12,16,20,24,32`（**与 v3 逐字节相同**） |
| **生产口径能覆盖到 192** | `DRY_RUN=1 MAX_SEQS=32 PREFIX=1 bash scripts/serve_a2.sh` | `…,40,48,96,192`（15 桶）+ `PREFIX=1` |
| Fisher 工具自检 | `python3 tools/fisher_recheck.py` | 4/4 用例 OK（含 `[[2,22],[2,22]] → 1.0`） |
| Fisher 修正表复算 | `python3 tools/interleave_ab.py` 的 `_fisher()`（走包内实现） | `[[2,22],[2,22]]→1.0000`、`[[5,7],[4,8]]→1.0000`、`[[5,7],[3,9]]→0.6668`，**与 §1.3 修正表逐位一致** |
| steep/flat/shallow 分类工具 | `python3 tools/steep_summary.py`（扫包内 16 个 jsonl 样本） | `total=16 steep=1 flat=0 shallow=15`，steep 那发正是 `faB_128k_r7`（A=3.446 @ ms=31.18）⇒ 与 `EXPECTED_PERF.md` §1.2/§2.3 自洽 |
| 包内自校验 | `sha256sum -c MANIFEST.sha256` | **167/167 全过**（在 `mktemp -d` 解包后重跑，见 §6） |
| **多 batch 结果回填** | `grep -c "7/7\|16/16\|6/6" EXPECTED_PERF.md` | 7 处命中（§7.3 的三块结果全在） |
| 权威报告 md5 | 与 `A3-node1:reports/multibatch-and-mixed-load.md` 对比 | **一致** `81e5c713f14dfd1b3ecd780fdbd2ba62`（88 行） |

---

## 5. 未验证 / 风险（**包内全部标了默认值与判据**）

| 项 | 状态 | 包内默认 | 用户该怎么用 |
|---|---|---|---|
| **多 batch 生产口径三块** | ✅ 生产臂（`PREFIX=1`）三块全过；⏳ **`PREFIX=0` 压力臂未跑**（我们这边仍在队列里） | `MODE=prod` 才跑 | 先跑 `MODEL=... MODE=prod bash scripts/run_test.sh`（一臂），再 `run_prod_both.sh`（补齐两臂） |
| `DRAFT_GRAPH` | ⚠️ 离线修复完成 + **负控已确认**，**正控待验** | **0（关）** | 开了之后**必须同时看 A（>1.5）与 `dspark-graph-capture`（>0）**；只报 ms 无效 |
| `MOE_ZERO` / `MOE_NF` / `LOCAL_OWNER=on` / `HCCL_DET=true` | ❌ **已判负结果** | 全关 | 只作复现用（`tests/multibatch/verify_serve_flags.sh` 覆盖了这些组合的解析） |
| PGO 在 A2 的实际收益 | ⚠️ 未在 A2 起服验证 | 开（缺产物自动降级） | 起服异常先 `PYTHON_PGO=0` |
| `MAX_LEN=1048576` | ⚠️ 只在 8K/128K 包络验证过 | 沿用 1M | 不稳就 `MAX_LEN=131072` |
| `Dockerfile` 未真正 build | ⚠️ 本机无该镜像、按纪律不起容器 | — | 若失败，最可能两点：① `patches/files/model.py` 与 A2 镜像差异过大；② 新增 sidecar 落位只读。兜底：`PATCH_MODE=mount` |
| `reports/` 里个别早期报告的结论已被后续覆盖 | ⚠️ | — | 冲突时以 `EXPECTED_PERF.md` §8 与三份权威报告为准 |

---

## 6. 交付物清单

```
$WORKSPACE/a2_pkg_v5.tar.gz        ← 待主 Agent 传输
$WORKSPACE/a2_pkg_v5.tar.gz.sha256 ← 权威校验口径
```

* **包根**：`a2_pkg_v5/`（解压即得到 `a2_pkg_v5/README.md`）
* **包内自校验**：`a2_pkg_v5/MANIFEST.sha256`（覆盖包内**全部**文件；条目数见文件首行）
* **补丁另有一套**：`patches/MD5SUMS`
* **校验命令**：
  ```bash
  sha256sum -c a2_pkg_v5.tar.gz.sha256
  tar -tzf a2_pkg_v5.tar.gz | head
  cd a2_pkg_v5 && sha256sum -c MANIFEST.sha256
  ```
* **大小 / sha256**：打包后写在**包外**并列文件里（`PACKAGING_REPORT` 不写自身 tar 的哈希 ——
  写进包内会造成自引用循环：写一次、重建一次、哈希就变了。v3 用同一约定）。

### 新增/修改清单（与 v3 的 diff）

| 类型 | 路径 |
|---|---|
| 新增 | `CORRECTNESS_STATUS.md`、`tests/multibatch/{multibatch_gate.py,multibatch_session.sh,run_prod_both.sh,verify_serve_flags.sh}`、`**tests/acc_eval.py**`（**v3 漏打的依赖**）、`tools/{fisher_recheck.py,interleave_ab.py,steep_summary.py}`、`patches/files/token_dispatcher_moennf.py`、`reports/{a-basin-and-acceptance-shape.md,session-attractor-and-clean-rate.md,draft-graph-negative-control.md}` |
| 修改 | `README.md`、`REPRO.md`、`CHANGELOG.md`、`EXPECTED_PERF.md`、`PACKAGING_REPORT.md`、`scripts/serve_a2.sh`、`scripts/run_test.sh`、`tests/make_report.sh`、`tests/t_gsm8k.py`、`Dockerfile`、`scripts/build_image.sh`（仅 tag v3→v4）、`MANIFEST.sha256` |
| 删除 | 无（v3 的文件一个没删） |

---

## 7. 已知缺口（**建议主 Agent 传输前知悉**）

1. **多 batch 只测了 `PREFIX=1` 一臂**（权威报告 §2 第 4 条自认）。`PREFIX=0`（每请求真 prefill、
   混合更凶）**仍在 A3-node1 队列里**；本包 `tests/multibatch/run_prod_both.sh` 可直接补齐两臂。
   ⇒ 拿到后我需要**再回填一次**（`EXPECTED_PERF.md` §7.3 会加一行两臂对照）并重打 tar。
2. **`MOE_NONFINITE` 的性能侧数字是二手的**（§2.4）；若需一手，请提供那 6 个 jsonl。
3. **`Dockerfile` 仍未真 build**（v3 已知缺口，未变；本机无该镜像且按纪律不起容器）。
4. **`SPEC=0` 跨容器复现只做了 1 次**（"三次含一次全新容器"）——`CORRECTNESS_STATUS.md` §3.2 已标为限制。
5. 多 batch 的**局限 5 条已逐字抄进 §7.3.1**（难度天花板 / 历史只到 409 token / 单次测量 /
   `PREFIX=0` 未跑 / 只覆盖功能性正确性）——**请勿只引用"三块全过"**。
6. 本次仍未做（时间优先级"能跑 > 数字准 > 文档全"下主动放弃）：
   `shellcheck`（未安装）、逐份核对 55 份报告的交叉引用、`quant/` 链在 A2 的端到端可跑性。
