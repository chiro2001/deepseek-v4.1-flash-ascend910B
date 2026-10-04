# admission gate 多-prefill：+11% N=8 吞吐（2026-10-04）

> 结论：把「一个 prefill-only 步只能放 1 个 prefill 请求」放宽到 **N 个**（`V41_GATE_MAX_PREFILL=8`），
> 在**保持"本步不含任何 decode"硬不变量**的前提下，N=8 交付吞吐 **+11.0%**、N=4 **+6.6%**、
> N=1 不变，TTFT **−29%**。144K 四针**全 PASS**。
> 根因：原实现的单-prefill 限制让 N 路冷启动的 ramp 变成 O(N) 个 prefill 步（子代理已量化：
> ramp 占窗口 39.6%、占 N=8 吞吐缺口 63%）。

## 1. 受控 A/B（同脚本、同参数、同机、3 rep，output=256）

| N | 基线（base_1004_135411） | **gate8（gate8_1004_133851）** | Δ |
|---:|---:|---:|---:|
| 1 | 107.0（单流中位） | 104.3 | −2.5%（噪声内） |
| 4 | 197.0 | **210.0** | **+6.6%** |
| 8 | 282.1 | **313.2** | **+11.0%** |
| TTFT(N=8) | 0.80 s | **0.57 s** | **−29%** |
| N=8 decode 窗口 | 13.15 s（历史） | **6.54 s** | — |

两臂配置逐行一致（`inner.sh` diff 仅 run_id/PROFILE_DIR）；唯一变量 = `V41_GATE_MAX_PREFILL`。

## 2. 行为取证（gate8 run 的 serve.log）

```
[admission_gate] prefill-only step #1 ... prefill_reqs=1 decode_reqs=0 total_tokens=1024
[admission_gate] invariant violated: step=4204 scheduled 8 prefill requests (>configured max)?  ← 已修
```
* 实测出现 `prefill_reqs=3/4/7/8` 的步（`total_tokens` ≤ 8064 ≤ BAT_TOKENS=8192）
  ⇒ **多 prefill 真的生效**，且 token 预算被正确钳位；
* `decode_reqs=0` 在所有 prefill-only 步上保持 ⇒ **硬不变量未破坏**。

## 3. 实现（`patches/admission_gate.patch`）

四处小改（**默认 `V41_GATE_MAX_PREFILL=1` = 与历史逐字一致**，零风险默认）：
1. `__init__`：`self._gate_max_prefill = max(1, int(os.environ.get("V41_GATE_MAX_PREFILL", "1")))`
2. 新增 `gate_prefill_admitted` 计数器
3. RUNNING 循环的封口：`>= max_prefill or token_budget <= 0` 才 break
4. WAITING 循环的封口：同上
5. （配套）invariant 检查从 `> 1` 改为 `> self._gate_max_prefill`，避免合法多 prefill 步被误报 ERROR

接线：`scripts/serve_a2.sh` 透传 `V41_GATE_MAX_PREFILL`；启动器 `~/tmp/launch_gate8.sh` 设 =8。

## 4. 正确性

| 用例 | 结果 |
|---|---|
| `regress2.py`（17×23 / 计数 2000+16000 / 针 904+8000） | **5/6 PASS**（唯一 FAIL = 并发一致性 6/8，与基线同） |
| **144K 四针 A/B/C/D** | **4/4 PASS**（逐字命中 ZQ7K-3341 / VX2M-8890 / HT4P-5527 / RB9N-6014） |
| 1M 四针 | 见 §5 |
| 并发 8 路空闲后重跑一致性 | 6/8（= 基线值；1/8 那次是测量期状态问题） |

## 5. 1M 与混布负控

* 1M 四针：**运行中**（本文件后续更新）。
* `tests/multibatch/multibatch_gate.py [C]`（128K 长请求 + 8 短请求真并发）：**待跑**。

## 6. 风险与边界

**不改的**：`gate_prefill_step ⇒ decode_reqs = 0`（这是 2026-09-17 加 gate 的原始目的——
防长 prefill 饿死 decode）。子代理取证结论：原始动机是**延迟**（首 token 后长时间不出字），
不是崩溃；"P2 hybrid crash/pollution" 这句话**只存在于 gate 自己的注释里**，全仓检索无出处。

**新引入的风险**（子代理列，均已考虑）：
1. 一步内多次 `allocate_slots`（KV 预算交互）—— 144K/1M 四针是这条的主要覆盖；
2. prefill 步的 batch 组合变多 —— 本仓有过"桶与真实 query 数不一致"的崩溃先例，
   但 prefill 路径**不走图捕获**（只有 decode 捕获），风险低，仍用 [C] 块复验；
3. 单步 prefill token 上限不变（受 `BAT_TOKENS=8192` 钳位）⇒ P→D KV 传输边界假设未变。

**止损**：`V41_GATE_MAX_PREFILL=1` 即刻回退到历史行为（无需改代码）。
