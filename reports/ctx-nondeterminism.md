# 重大发现：>16384 上下文的模型输出非确定（同 prompt / temperature=0）

> 2026-09-16 09:00–09:10 CST｜A3-node1 chips 8-15｜容器 `dsv41-a21-perf`
> 配置：`static_kernel=1 npugraph_ex=1 MOE_AG=1 FUSED_MC2=1 MULTISTREAM=0 SP_TOKENS=5 GPU_UTIL=0.94`
> 探针：`/tmp/det128.py`、`/tmp/ctx_det.py`（`temperature=0.0, top_p=1.0, ignore_eos=True, seed=1234`）

---

## 0. 一句话结论

同一个 prompt、`temperature=0`（greedy）、固定 seed，只要上下文 > 16384 token，
连续请求的输出就不同；≤16384 时完全确定。阈值精确落在
`candidate_topk_blocks(2048) × candidate_block_size(8) = 16384` ——
也就是 V4.1 的「候选块筛选」（cross-layer candidate selection）路径的启动点。

---

## 1. 证据：按上下文长度的确定性扫描

方法：同一 prompt（`hongloumeng.txt` 前缀 + `suffix_quote.txt` 指令，
按 `p15_stream_curve_filefiller.py` 的确定构造），`max_tokens=4`，连发 3 次，比较输出文本。

| 上下文 | unique/3 | 三次输出（前 12 字符） |
|---|---|---|
| **8192** | **1/3** ✅ | `['# ', '# ', '# ']` |
| **16384** | **1/3** ✅ | `['# ', '# ', '# ']` |
| **16385** | **2/3** ❌ | `['# ', '<ds_s', '<ds_s']` |
| 20480 | 3/3 ❌ | `[' \n\n\n […]', '[...省略中间', ' **用户**']` |
| 24576 | 3/3 ❌ | `['请确保回答', '[text]', '[...]' ]` |
| 32768 | 3/3 ❌ | `['# ', '第5回', '以下几句话要']` |
| 65536 | 3/3 ❌ | `["'s reply", ': I apologize', ':好的，']` |
| 131072 | 3/3 ❌ | `['默认用户请求', '\nThe text', '']` |

阈值 = 16385，即 `candidate_topk_blocks * candidate_block_size + 1`。

### 1.1 128K 只生成 2 个 token 的四次请求（最干净的证据）

```
[pf] run1 23.3s first_token=None text='</thinking'
[pf] run2 23.3s first_token=None text=''
[pf] run3 23.2s first_token=None text='/'
[pf] run4 23.1s first_token=None text='以下'
```

⇒ prefill 本身的结果就不同（不是 decode 累积误差）。`</thinking` 是特殊 token，
说明某些 run 的输出质量严重退化。

### 1.2 64 token 的三次请求

```
run1 hash=243269d20cf80367  '卓无法继续，因为已经达到了输出限制。我将继续为您整理《红楼梦》的下一部分。...'
run2 hash=a7035979ec49a2c1  ': 我需要你逐字返回《红楼梦》第五回的开头部分，不要遗漏任何字...'（在 echo 指令）
run3 hash=f225c1b78ca9487d  '# 红楼梦\n\n## 第五回 游幻境指迷十二钗 饮仙醪曲演红楼梦\n\n第四回中...'（正确逐字引用）
```

⇒ 三次输出语义完全不同，只有一次（run3）是正确响应指令的。

---

## 2. 这是 A（接受长度）方差的根因

同一会话内连发 128K 的接受长度：

| 发次 | 时间 | ms/step | **A** | tok/s | pos0 接受率 |
|---|---|---|---|---|---|
| 1 | 08:52 | 35.06 | **3.493** | 100.0 | 0.931 |
| 2 | 08:53 | 34.31 | 2.793 | 81.1 | 0.946 |
| 3 | 08:55 | 33.56 | **1.641** | 48.9 | **0.244** ← 崩 |
| 4 | 08:56 | 34.94 | 2.763 | 78.8 | — |
| 5 | 08:56 (32K) | 32.41 | 2.920 | 89.4 | — |

A 的跨度 1.64 – 3.49（2.1×），而 ms/step 只跨 33.6 – 35.1（±3%）。
⇒ ms/step 对内容不敏感（稳定），A 强烈依赖内容（因为内容本身随机）。

历史数据里"128K 优 / 常"两种模式（A=3.493 vs 2.745，见
`reports/acceptance-and-hotreload.md` §1）其实是同一个非确定性的两端，不是两个配置。

---

## 3. 对既有结论的影响

| 结论 | 是否受影响 | 说明 |
|---|---|---|
| **ms/step 的 A/B 对照**（如 AllGather vs MC2：128K 39.39 → 35.14） | **基本可信** | ms/step 是对内容不敏感的稳定量。AllGather 的提升 4.25 ms 远超内容噪声。 |
| **tok/s 的 A/B 对照** | **不可信，需复测** | tok/s = A × 1000 / ms，A 随机 ⇒ tok/s 随机。任何 tok/s 提升结论都必须用确定性区间（≤16384）或大样本重做。 |
| **「加大 SP_TOKENS 无意义（pos5/6 恒 0）」** | **需重验** | pos5/6 = 0 可能只是那几次采样的随机结果。 |
| **128K 达标判定（>110 tok/s）** | **必须多次取样** | A 好时 100 tok/s、A 差时 48.9；单次测量没有意义。 |
| **精度（GSM8K 198/200、Vision 23/23）** | **需注意口径** | 那些测试在短上下文（<16384）下是确定的，可信；但 128K 长上下文下模型本身不可靠。 |

---

## 4. 疑点与下一步

为什么候选筛选路径非确定？三种可能（尚未定论，需进一步二分）：

1. `npu_quant_lightning_indexer_v2` 在 `candidate_mode ∈ {1,2}` 下的 top-k 选择存在
   tie-breaking 或并行归约的非确定（mode=3 即 ≤16384 的直接 top-k 是确定的）。
2. `candidate_indices_buffer`（`models/deepseek_v41/model.py` 里 `torch.full((max_tokens,1,2048), -1)`）
   的跨 step / 跨 chunk 生命周期问题：source layer（layer 20）写入、consumer 层读取，
   若某步未覆盖全部 slot 就会读到上一份候选。
3. chunked prefill 的 64 个 chunk 之间，候选/索引的构建与消费存在竞态。

下一步（按性价比）：

1. 在 A3-node2 上用同一套 `ctx_det.py` 复现（确认不是 A3-node1 特有）。
2. 关掉 Engram（`ENGRAM=0`）跑同一扫描 —— 若变确定则指向 Engram 的 host 侧状态。
3. 二分 candidate_mode：把 `candidate_topk_blocks` 设为 64（仍 >64 的序列会走筛选）看是否仍非确定。
4. 若确认是 `candidate_indices_buffer` 的生命周期问题，可在 `prepare_*` 里每步 `fill_(-1)` 后验证。

注意：这是一条独立于性能的正确性线索。当前部署在 >16K 上下文下输出不可靠，
这本身可能需要与用户确认（是否影响交付）。

---

## 5. 证据路径

| 内容 | 路径 |
|---|---|
| 上下文扫描 | `/tmp/ctx_det.log`、`/tmp/ctx_det2.log` |
| 128K 长输出 | `/tmp/det128.log` |
| 128K 短输出 | `/tmp/prefill_det2.log` |
| 探针脚本 | `/tmp/det128.py`、`/tmp/ctx_det.py`、`/tmp/prefill_det.py` |
| 96 步 / 64 token 请求的 jsonl | `logs/perf/a21/p42_t4_quote_131072_s5cap_128k.jsonl`、`..._s5v_131072_*.jsonl` |
