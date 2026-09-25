# 2026-09-25 验收矩阵：原始证据与两处必须知道的口径修正

## 1. 矩阵结果（CED-PD，P=前 20 层 + layer-20 全局源，D=128-token 重放 + 全 40 层）

服务口径：P/D 双 TP8、BF16 KV、`MAX_LEN=1048576`、`MAX_SEQS=4`、`BAT_TOKENS=8192`、
`SPEC=0 PREFIX=0 DRAFT_GRAPH=0`、`ENGRAM=1 ENGRAM_DEVICE_INDEX=0`、`CPU_BIND=0`、
D 侧 `MULTISTREAM=0 DSA_OVERLAP=0`、`num_blocks=29076`（4 GiB 上界，见
[`../../docs/CED-PD-BLOCK-BOUND-20260925.md`](../../docs/CED-PD-BLOCK-BOUND-20260925.md)）。

| 模式 | 144K | 1M |
|---|---|---|
| short（22 token） | PASS（0.598 s） | — |
| **needle（四针 A/B/C/D）** | **4/4 PASS**（11.0 s/条） | **4/4 PASS**（101.5–104.5 s/条） |
| **stream（流式）** | PASS（TTFT 10.952 s） | PASS（TTFT 101.217 s） |
| **multiturn（三轮）** | **3/3 PASS**（144,105–144,157 token，10.8 s/轮） | **3/3 PASS** |
| prefix | 2/2 PASS 但 `cached_tokens=0` | 2/2 PASS 但 `cached_tokens=0` |

总计 21 条请求全部 HTTP 200、判据全过。原始 JSON 与逐请求证据在 a3-21 的
`pkg_d7953e5_trace/results/ced_accept_20260925/`（含 `*.request.json` 与
`*.result.json`，逐条带 SHA-256）。

资源口径（同一次 1M needle 的 `/metrics` 增量）：P `prompt_tokens +3,997,758` /
`generation_tokens +4`；D `prompt_tokens +3,997,762` / `generation_tokens +30`
⇒ P 只产出交接标记（每请求 1 个），真正生成的是 D，角色划分在运行时可验。

## 2. ⚠️ 修正一：`multiturn` 之前是**静默缩水**的，两条都不算数

`tools/ced_pd_acceptance.py::run_multiturn` 原本把语料按 `target // 8` 构造，
于是：

| 声称 | 实际 `prompt_tokens`（旧） | 实际（修正后） |
|---|---|---|
| 144K 多轮 | **18,109 / 18,135 / 18,161** | **144,105 / 144,131 / 144,157** |
| 1M 多轮 | **125,105 / 125,131 / 125,157** | **999,510 / 999,536 / 999,562** |

判定依据就是证据里的 `usage.prompt_tokens`（旧记录本来也写着，只是当时没核对）。
单请求耗时同样能看出来：144K 多轮旧的是 **1.78 s**，修正后 **21.9 s**。

修正：改为按 `max_model_len` 反推每轮可用上下文

```
base_target = min(target, max_model_len − (轮数−1)×(max_tokens+64) − 512)
```

装不下时**打印警告**，并把每轮真实 `prompt_tokens` 写进证据。

**修正后两档都已在全 40 层基线上重跑并通过**：

| 目标 | 实际 `prompt_tokens`（三轮） | 每轮 wall | 判决 | 答案 |
|---|---|---|---|---|
| 144K | 144,105 / 144,131 / 144,157 | 21.93 s | **3/3 PASS** | `RB9N-6014` / `ZQ7K-3341` / `VX2M-8890` |
| 1M | **999,510 / 999,536 / 999,562** | 280.7 / 280.6 / 280.4 s | **3/3 PASS** | 同上 |

⇒ **两档多轮现在都名副其实**。1M 那三轮的 prompt 与 `max_model_len=1048576`
只差 4.7%，正好检验了新加的 reserve 算式（第 3 轮比第 1 轮长 52 token，仍在线内）。

**CED 臂也已按真实长度重跑通过**（2026-09-25 23:38–23:48）：

| 目标 | CED 三轮 `prompt_tokens` | CED 每轮 wall | 判决 |
|---|---|---|---|
| 144K | 144,105 / 144,131 / 144,157 | 10.80 / 10.85 / 10.84 s | **3/3 PASS** |
| 1M | **999,510 / 999,536 / 999,562** | 101.32 / 101.20 / 101.26 s | **3/3 PASS** |

两臂的 `prompt_tokens` 完全一致 ⇒ 干净的同口径对照。

## 3. ⚠️ 修正二：`prefix` 那两项确实没验证到缓存，但**基础设施是好的**

**先说当时的判定是对的**：那两次请求的 `prefix_cache_hits_total` 增量是 0
（`cached_tokens` 也恒为 0），因为 CED 启动器硬门 `PREFIX=0` —— 缓存根本没开，
所以是"同前缀各算一遍"。这一格当时**不该记通过**。

**但结论要更正一处**：2026-09-26 在**非 CED 基线**上开 `PREFIX=1` 实测，
缓存**可用且正确**：

| 步骤 | wall | P hits 增量 | D hits 增量 | 答案 |
|---|---:|---:|---:|---|
| 冷（新 prompt，带针） | 5.60 s | 0 | 0 | ✅ `RB9N-6014` |
| 同一 prompt 第 2 次 | 0.98 s | **144,000** | **144,000** | ✅ 与冷**相同** |
| 同一 prompt 第 3 次 | 1.00 s | **144,000** | **144,000** | ✅ 相同 |
| 换一个新 prompt | 5.78 s | 0 | 0 | ✅ |

两个不同 prompt 各 3 次全部一致 ⇒ **命中路径给出与冷路径逐字节相同的正确答案**，
端到端约 **5.7×** 加速。

**两个必须记住的坑**：

1. **响应里的 `cached_tokens` 恒为 0，即使命中也一样**（PD 代理不传递该字段）。
   判据必须用服务端 `vllm:prefix_cache_hits_total` /
   `vllm:prompt_tokens_by_source_total{source="local_cache_hit"}`。
2. 冷值在 5.6 s 与 22.9 s 之间波动（同机同长度），**尚未归因**。

⇒ 所以这一项现在的准确说法是：**基线口径已验证；CED 口径仍被硬门挡住
（待实现）**。详见 [`../../docs/CED-PD-CACHE-HIT-PLAN-20260925.md`](../../docs/CED-PD-CACHE-HIT-PLAN-20260925.md) §4。

## 4. 本矩阵能支持与不能支持的结论

**能支持**：CED-PD（P 20 层 + D 128-token 重放、BF16 KV）在 144K 与 1M 上的
needle 四针、流式、**以及三轮多轮**的**正确性**（多轮含 999,510–999,562 token
的近满上下文）；基线同口径对照同样全过。

**不能支持**：

* 两臂的 144K/1M 多轮**都已按真实长度重跑通过**（§2），不再是缩水版本。
* "缓存命中" —— **CED 口径**仍不可用（硬门 `PREFIX=0`）；**基线口径已验证可用且正确**（§3）。
* 吞吐 —— 本矩阵只记 wall/TTFT；吞吐口径见
  [`../../docs/CED-PD-PERF-20260925.md`](../../docs/CED-PD-PERF-20260925.md)。
