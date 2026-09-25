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
| **multiturn（三轮）** | 见 §2（修正后 3/3 PASS） | 见 §2（**未重跑**） |
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
| 1M 多轮 | **125,105 / 125,131 / 125,157** | 未重跑 |

判定依据就是证据里的 `usage.prompt_tokens`（旧记录本来也写着，只是当时没核对）。
单请求耗时同样能看出来：144K 多轮旧的是 **1.78 s**，修正后 **21.9 s**。

修正：改为按 `max_model_len` 反推每轮可用上下文

```
base_target = min(target, max_model_len − (轮数−1)×(max_tokens+64) − 512)
```

装不下时**打印警告**，并把每轮真实 `prompt_tokens` 写进证据。

**修正后的 144K 多轮已在全 40 层基线上重跑并通过**：
`prompt_tokens = 144,105 / 144,131 / 144,157`，三轮分别答对
`RB9N-6014` / `ZQ7K-3341` / `VX2M-8890`，各 21.9 s，**3/3 PASS**。

⇒ 结论：**"144K 多轮"现在名副其实；"1M 多轮"仍待重跑**（需重启 CED 的 P+D）。

## 3. ⚠️ 修正二：`prefix` 两项是**假通过**

`prefix` 模式两次请求的 `usage.prompt_tokens_details.cached_tokens` 都是 **0**，
`prefix_cache_hits_total` 增量也是 0 ⇒ 只是"同前缀各算一遍"，
**没有验证到任何缓存命中**。这不是判据太弱，是功能本身不可用：
CED 启动器硬门 `PREFIX=0`，且 D 侧预清零只覆盖 `get_unhashed_block_ids`。

目标里"缓存命中正确性验证"这一项**仍是代码级阻断**，要么单独做原型臂，
要么在验收表里明确标成未实现。

## 4. 本矩阵能支持与不能支持的结论

**能支持**：CED-PD（P 20 层 + D 128-token 重放、BF16 KV）在 144K 与 1M 上的
needle 四针、流式、以及（修正后）144K 三轮多轮的**正确性**。

**不能支持**：

* "1M 多轮" —— 未重跑（§2）。
* "缓存命中" —— 假通过（§3）。
* 吞吐 —— 本矩阵只记 wall/TTFT；吞吐口径见
  [`../../docs/CED-PD-PERF-20260925.md`](../../docs/CED-PD-PERF-20260925.md)。
