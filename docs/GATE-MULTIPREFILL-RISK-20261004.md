# admission gate「多 prefill」改造 —— 风险取证（2026-10-04）

**执行**：`/root/tiny_fusion_ops` 子代理　**机器**：a3-21　**全程只读**（未启停服务、未改仓库文件，本报告除外）
**标注**：【实测】有原始出处；【推断】由实测推导；【未确认】缺证据；【未找到】搜过但不存在。

## 0. 结论速览
1. **gate 的原始动机不是崩溃，是 decode 饿死（延迟）** —— 出处是 gate 补丁自己的提交信息。【实测】
2. **"P2 hybrid crash/pollution" 找不到支撑文档**；仓库里唯一"P2 崩了"的记录是 **CED 连接器整池命中**的形状 bug，与调度无关。【实测+未找到】
3. **你要的改造已在工作区**（未提交，今天 13:38 改的），开关 `V41_GATE_MAX_PREFILL`（默认 1）。**但还缺 1 处 diff**，且**在跑的容器没生效**。【实测】

## 1. 原始 P2 hybrid 崩溃到底是什么（Q1）
### 1.1 gate 自己的提交信息：动机 = decode 饿死，不是崩溃【实测】
`patches/vllm/0001-feat-scheduler-admission-gate-to-protect-decode-from.patch:1-20`（`From 2e7db012…`，日期 `Thu, 17 Sep 2026 04:55:09 +0000`，标题 *"admission gate to protect decode from prefill starvation"*）原文：
```
长上下文场景下 prefill 会长期占住调度步，decode 被饿死（表现为
"prefill-only step #N ... deferred_decode_reqs=1" 连续上百步、首 token
之后长时间不出字）。admission gate 在同一个 step 里只放 prefill、把
decode 延后到下一个 step 之前的门控步，保证 decode 有稳定节拍。
```
⇒ 触发条件是 **"prefill 长期占住调度步"**（长上下文 chunked prefill），症状是**首 token 之后长时间不出字**；崩溃/乱码/污染一个字都没提。

### 1.2 "P2 hybrid crash/pollution" 只出现在代码注释里【实测+未找到】
全文只命中 2 处，都在 gate 补丁自身：`patches/admission_gate.patch:96`（"the boundary the P2 crash/pollution repro exercises"）与 `:142`（"The two request classes must never share a SchedulerOutput (P2 hybrid crash/pollution)"）。
**搜过但无任何文档解释它**：`docs/`、`reports/`、`CHANGELOG.md`、`CORRECTNESS_STATUS.md`、`~/projects/dsv41/reports/`、`a2/logs/*.md`、`upstream/`（含 `git log -S "P2 hybrid crash"`，只在打包提交命中，无更早来源）。⇒ 标【未找到】，**不编造**。

### 1.3 仓库里唯一"P2 崩了"的记录 = 连接器整池命中，与调度无关【实测】
`deploy/a3-ced-pd/payload/ced/mooncake_hybrid_connector.py:1619-1620`：
```
# （1M 那两条里 P1 少 63 个 token 所以是部分命中、没崩；
#   P2 正好对齐所以全命中、崩了。）
```
* 崩因：`RuntimeError: CED decoder expected 12 KV cache groups without DSpark`（同文件 `:1573-1600` 段注释）
* 触发条件：**整池前缀缓存命中** ⇒ `(prompt_len - 1) % 128 == 0` ⇒ `num_external_tokens = 0` ⇒ 上游 stock 给裸 `[]`，与 CED"每请求 12 个 group 列表"契约冲突 ⇒ 杀死 D 引擎。
* `P1/P2` = **语料位置/探针编号**（`docs/CED-PD-CACHE-HIT-PLAN-20260925.md:160`「P2（不同语料位置）」、`evidence/ced_prefix_hit_20260926/README.md:39`「P2（语料 offset len/2）」）；`hybrid` = **连接器名字**（`mooncake_hybrid_connector`），不是"prefill+decode 混合批"。

### 1.4 唯一有文档的"prefill 混进不该出现的步"事故【实测】
`docs/CED-DECODE-API-GUARD-20260927.md`：2026-09-27 00:01:48，CED **D 半边**被一条直连普通请求打死（8 个 TP worker 同时 raise ⇒ `EngineDeadError`，恢复 20 分钟）。崩因 `RuntimeError: CED decoder replay exceeded 128 tokens`，守卫在 `experimental/ced/dsa_v41.py`：`_CED_DECODE_ROLE and metadata.swa.num_prefills > 0 and max_query_len > 1`。
⇒ **触发条件 = 该 step 出现 prefill 且 query_len > 128，且实例是 decode 角色**。

### 1.5 反证：prefill+decode 真同时跑曾被实测通过【实测】
`reports/multibatch-and-mixed-load.md`（2026-09-16）[C] 块：1 条 128K prefill 与 6 条短请求并发 ⇒ 短请求 **6/6 正确、与串行基线逐项不一致 0 项**，长请求正常。（自述局限：样本小、只跑 1 遍、`PREFIX=0` 臂未跑。）

## 2. 这条改造是否触碰原始风险（Q2）
| 候选"原始风险" | 是否触碰 | 依据 |
|---|---|---|
| **decode 饿死**（gate 真实动机） | **不触碰，方向相反** | 多 prefill 让 prefill 步更少、decode 步更多 ⇒ decode 节拍更好【推断】 |
| 同一步混 prefill+decode 致崩溃/污染 | **不触碰**（硬不变量未变） | 只放宽"每步放几个 prefill"，`gate_prefill_step ⇒ 本步 decode 数=0` 的检查仍在（`admission_gate.patch:269-292`）【实测·代码】 |
| CED D 半边出现 prefill（§1.4） | **不触碰** | 多 prefill 不产生 decode-in-prefill-step，也不让 D 半边多出 prefill【推断】 |
| CED 连接器整池命中（§1.3） | 不触碰（本与调度无关） | 【实测·代码】 |
| **多 prefill 同一步导致崩溃** | **未找到任何反例** | 全仓搜索无记录；只说明"未发现"，**不能证明安全** ⇒ 需负控 |

**改造真正引入的新风险（有仓库先例，必须负控）**：
1. **一步内多次 `allocate_slots`**：原路径每步只走一次，多 prefill 后 KV 块预算/`max_num_running_reqs` 的交互**从未走过**。【未确认】
2. **prefill 步的 query_len 组合变多** —— **有实测崩溃先例**：`reports/a-basin-and-acceptance-shape.md:161-162`「桶与真实 query 数不一致」崩溃（`[6,2]` vs `[5,2]`）。多 prefill 步的 token 分布（1184×k）与单 prefill 步不同，**必须复验 capture 桶覆盖**。【实测·先例】
3. 单步 prefill token 从 ~1184 涨到 ~8192（`BAT_TOKENS` 上限）⇒ P→D 的 KV 传输/`num_external_tokens` 若含边界假设需复验（CED 形态）。【推断】

## 3. 最小 diff（Q3）—— **改造已在工作区，需补 1 处 + 1 处断言**
### 3.1 已存在的改动【实测】
`patches/admission_gate.patch` 工作区版本（**未提交**，mtime `2026-10-04 13:38:12`，`git diff` = **+27/−19**，注释标 `[GATE-MULTIPREFILL]`）已含三处：

| # | 位置（patch 行号） | 内容 |
|---|---|---|
| 1 | `:61-65` | `self._gate_max_prefill = max(1, int(os.environ.get("V41_GATE_MAX_PREFILL", "1")))` |
| 2 | `:190-195`（RUNNING 循环） | `gate_prefill_admitted += 1; if >= max or token_budget<=0: done; break` |
| 3 | `:243-248`（WAITING 循环）+ `:151` | 同款收口 + `gate_prefill_admitted = 0` 初始化 |

接线已就位：`scripts/serve_a2.sh:1526` 已透传 `V41_GATE_MAX_PREFILL`；`~/tmp/launch_gate8.sh` 设 `=8`。

### 3.2 ★ 必须补的 diff（否则判据被自己的日志污染）
`admission_gate.patch:293-299` 仍是**老的单 prefill 假设**：
```python
if gate_prefill_reqs > 1:
    logger.error("[admission_gate] invariant violated: step=%d scheduled %d "
                 "prefill requests (>1).", self.current_step, gate_prefill_reqs)
```
`max_prefill=8` 时**每个合法多 prefill 步都会打一条 `invariant violated` ERROR** ⇒ ops 误报 + 判据自污染。最小改法：`if gate_prefill_reqs > self._gate_max_prefill:`（原为 `> 1`）。

### 3.3 ★ 必须补的断言：live tree 判据区分不了新老版本【实测】
* 在跑的 `dsv41-tinyspark`（启动 **2026-10-01 13:56**，早于 patch 改动 3 天）：live `scheduler.py` md5 = `b959163e…` **= `patches/vllm/MD5SUMS` 记录值**，且 `grep -c _gate_max_prefill` = **0** ⇒ **多 prefill 没生效**；而容器内挂载的 `/opt/dsv41/admission_gate.patch` md5 = `b2529736…` **= 新版本**。
* `serve_a2.sh:1583` 的效果断言是 `grep -c admission_gate` —— **新旧都是 5 命中，区分不了**。
* ⇒ 验证第一步必须加专项断言：`docker exec <name> grep -c "_gate_max_prefill" /vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py` **必须 ≥ 3**（1 定义 + 2 使用）；否则整轮实验无效，rollback。

### 3.4 一致性提醒【实测】
`patches/vllm/0001-…patch`（上游投稿版）**不含** MULTIPREFILL（0 命中）；`patches/vllm/MD5SUMS` 记的仍是老 hash ⇒ 若跑 `tools/check_checksums.py`，按 AGENTS.md §3.10 用生成器刷新，别手改。

## 4. 验证清单（按"能否抓住 P2 hybrid 类问题"排序）
| # | 用例 | 命令/位置 | 抓什么 |
|---|---|---|---|
| 0 | **生效断言**（前置门） | `docker exec <name> grep -c _gate_max_prefill …/scheduler.py` ≥ 3 | 防"跑了 20 分钟却是老代码"【必需】 |
| 1 | **行为取证** | `grep -aE "\[admission_gate\]" results/<run>/serve.log` ⇒ 应出现 `prefill_reqs>1`，且**无** `invariant violated … (>1)` | 改造真的在同一步放了多个 prefill |
| 2 | **ramp 收益** | `tools/bench_concurrency.py --concurrency 1,4,8 --prompt-tokens 1024 --output-tokens 256 --repeats 3`（`~/tmp/verify_gate8.sh` 已封装） | ramp 8s→~2s；**必须报 `(ms/step, A, tok/s)` 三元组** |
| 3 | **回归** | `python3 ~/tmp/regress2.py 19210`（17×23 / count 2000+16000 / needle 904+8000 / 并发 8 一致） | 基础正确性 |
| 4 | **144K/1M 四针** | `~/tmp/20260926/dspark/tools/ced_pd_acceptance.py --mode all --context-tokens 144000,1000000 --out ~/tmp/<run>.json`（四针 A `ZQ7K-3341`/B `VX2M-8890`/C `HT4P-5527`/D `RB9N-6014`；产物样例 `~/tmp/dsv41_accept_1004_104559/{needle.json,needle_evidence/}`；1M 单跑 `~/tmp/20260924/ced_numeric/base1m_needle.sh`） | 长上下文 + 整池命中边界（§2 风险 3、§1.3 同族） |
| 5 | **混合负载负控** | `tests/multibatch/multibatch_gate.py`（[C] 块 = 真 prefill+decode 混跑） | **P2 hybrid 类问题最直接的用例** |
| 6 | **长 prefill 压 decode** | `~/tmp/dec_under_pf.py <port> <conc> <pf_ctx>`、`~/tmp/pf_mixed.py <port> <conc> <ctx>` | ramp 场景下 decode 是否被伤害（改造收益面） |
| 7 | **CED 形态冒烟** | `bash deploy/a3-ced-pd/launch/smoke.sh`（144K 四针最小判据） | 若改造要进 CED 交付档 |

**建议顺序**：0 → 1 → 3 → 2 → 5 →（进交付档时）4/7。**止损**：见 `invariant violated`/`EngineDeadError`/桶不匹配崩溃 ⇒ 立即回退 `V41_GATE_MAX_PREFILL=1`（= 历史行为逐字一致）。

## 5. 未找到 / 未确认
* 【未找到】"P2 hybrid crash/pollution" 的任何出处文档（搜索面见 §1.2）。建议把该注释改写成有出处的表述，否则后人会重复本次搜索。
* 【未确认】多 prefill 同一步是否安全 —— 只做到"未发现反例"，判据要等 §4 第 5 项。
* 【未确认】`max_prefill>1` 时单步 KV 块峰值是否触发 `_gate_force_decode_steps` 安全阀（未读块管理器内部实现）。
