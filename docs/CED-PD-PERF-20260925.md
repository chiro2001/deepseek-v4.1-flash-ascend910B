# CED-PD vs 全 40 层双 TP8 PD：受控性能对照（2026-09-25）

## 0. 结论

同一台 A3-21、同一份模型、同一批请求、同一端口与参数，**唯一变量是 P 执行的层数**
（CED 20 层 + layer-20 全局源 vs 基线 40 层）：

| 上下文 | CED-PD TTFT | 全 40 层 TTFT | 加速比 | CED 正确性 | 基线正确性 |
|---|---|---|---|---|---|
| 144K | **10.95 s** | **21.82 s** | **1.99×** | 4/4 PASS | 4/4 PASS※ |
| 1M | **101.22 s** | **282.16 s** | **2.79×** | 4/4 PASS | 1/1 PASS※ |

※ 基线只有在**关掉 D 侧多流**之后才是正确配置，见 §3。

144K 的 **1.99×** 正是"P 从 40 层降到 20 层"应有的比例，是机制的直接证据。
1M 的 2.80× 更大，超出层数比，说明长序列下 40 层路径还有额外开销（待归因）。

## 1. 口径

两臂都用 `needle` / `stream` 两种模式，走同一个 proxy（`127.0.0.1:18992`），
语料同一份 `data/hongloumeng.txt`，`temperature=0`、非流式/流式各测一次。

共同参数：`MAX_LEN=1048576`、`MAX_SEQS=4`、`BAT_TOKENS=8192`、`GPU_UTIL=0.92`、
`BLOCK=128`、`KV_DTYPE=bfloat16`、`ENGRAM=1 ENGRAM_DEVICE_INDEX=0`、`CPU_BIND=0`、
`SPEC=0 PREFIX=0 DRAFT_GRAPH=0`、`NPUGRAPH_EX=1`。
两侧 KV 池都被 `[CED-POOL-GUARD]` 钳到 `num_blocks=29076`（4 GiB 上界，见
[`CED-PD-BLOCK-BOUND-20260925.md`](CED-PD-BLOCK-BOUND-20260925.md)）。

| | CED 臂 | 基线臂 |
|---|---|---|
| P 执行层 | 0–19 + layer-20 全局源投影 | 0–39 |
| D 执行层 | 0–39 + 128-token 有界重放 | 0–39（无重放） |
| P 连接器 | `experimental/ced/mooncake_hybrid_connector.py`（CED 角色） | stock |
| D 连接器 | 同上（CED decode 角色） | stock |
| D 多流 | `MULTISTREAM=0 DSA_OVERLAP=0` | `MULTISTREAM=0 DSA_OVERLAP=0`（见 §3） |

## 2. 测量

### 2.1 TTFT（stream 模式，单请求，无并发）

| 上下文 | CED-PD | 全 40 层基线 | 比值 | 两臂是否都正确 |
|---|---|---|---|---|
| 144K | 10.952 s | 21.82 s | **1.99×** | 是（CED 4/4、基线 4/4） |
| 1M | 101.217 s | 282.162 s | **2.79×** | 是（CED 4/4、基线 1/1） |

### 2.2 P 侧 prefill 时长（独立复核，取自 P 日志的相邻 `Delaying free` 时间差）

同一批请求串行发出，两次 `Delaying free` 之间约等于"上一条的 D 解码 + 本条的 P prefill"，
而 D 解码只有 7 个 token，可忽略：

| 上下文 | CED P（20 层） | 基线 P（40 层） |
|---|---|---|
| 144K（~1137–1147 块） | 11–12 s | 24–27 s |
| 1M（~7820–7830 块） | 101–109 s | 286–291 s |

P 侧独立时间与 TTFT 一致 ⇒ 端到端时间基本就是 P 的 prefill，D 的交接与解码占比很小。

### 2.3 端到端（needle 模式，4 针串行）

| 上下文 | CED-PD | 全 40 层基线※ |
|---|---|---|
| 144K | 11.0 / 11.0 / 11.0 / 11.1 s | 26.1 / 22.2 / 22.1 / 22.1 s |
| 1M | 104.5 / 102.3 / 101.6 / 101.5 s | （未跑满，见 §4） |

※ 基线第一次跑是乱码版本；上表 144K 的 22 s 来自 §3 的修正配置。

### 2.3.1 多轮（修正口径后，2026-09-25 21:30–21:50 补测）
+
+`tools/ced_pd_acceptance.py` 的 `run_multiturn` 原来把语料按 `target//8` 构造，
+所以此前"144K 多轮"实际只有 18K、"1M 多轮"只有 125K（详见
+[`../evidence/ced_acceptance_20260925/README.md`](../evidence/ced_acceptance_20260925/README.md) §2）。
+修正后在全 40 层基线上重跑：
+
+| 目标 | 实际 `prompt_tokens`（三轮） | 每轮 wall | 判决 |
+|---|---|---|---|
+| 144K | 144,105 / 144,131 / 144,157 | 21.93 s | **3/3 PASS** |
+| 1M | **999,510 / 999,536 / 999,562** | 280.7 / 280.6 / 280.4 s | **3/3 PASS** |
+
+1M 三轮的 prompt 与 `max_model_len=1048576` 只差 4.7%，是最接近边界的一组用例。
+
+### 2.4 资源与机制证据（metrics 增量）

同一次 1M needle（4 条请求）期间，P 与 D 的 `/metrics` 增量：

| 端点 | prompt_tokens_total | generation_tokens_total |
|---|---|---|
| P（18990） | +3,997,758 | **+4** |
| D（18991） | +3,997,762 | **+30** |

P 只产出 4 个 token（每条请求一个交接标记），真正生成 30 个 token 的是 D
⇒ P 确实没有在做 decode，CED 的角色划分在运行时可验证。

## 3. **基线本身在长上下文下是坏的**（本轮新发现，且已单变量定位）

第一次跑基线（`serve_a3_pd.sh` 的默认 `MULTISTREAM=1 DSA_OVERLAP=1`）时，
144K 四针 **0/4 全错**，输出是典型乱码：

```
needle144k_off0_r0_A: '不以或少或无<｜box｜>这小子不看或少或无<｜box｜>要说点什么 elseignement ...'
needle144k_off0_r0_C: '朝夕的数字(numbers?)的回答(answer?) answers(answer[])<｜box｜>...'
```

特征是 HTTP 200、`completion_tokens=64`（打满 `max_tokens`）、无 `finish_reason`、
`u_fffd=0`（不是 UTF-8 解码问题）。P→D 的 KV 交接是正常的
（P 侧每条 144K 请求 `Delaying free of 1147 blocks`，D 侧 32 条 `KV cache transfer`）。

**只重启 D**、把 `MULTISTREAM=1 DSA_OVERLAP=1` 改成 `MULTISTREAM=0 DSA_OVERLAP=0`，
P 与其余参数一律不动，同一批 144K 四针立刻 **4/4 PASS**：

| D 配置 | 144K 结果 |
|---|---|
| `MULTISTREAM=1 DSA_OVERLAP=1`（默认） | **0/4**，全部乱码 |
| `MULTISTREAM=0 DSA_OVERLAP=0` | **4/4 PASS**，`ZQ7K-3341` / `VX2M-8890` / `HT4P-5527` / `RB9N-6014` 全对 |

⇒ **长上下文下 D 侧的多流/重叠路径会静默算错**。这与 2026-09-24 CED 排查中
"D 关多流 + prompt 尾步 eager + metadata 主流"的结论是同一类问题
（见 [`A3-CED-PD-HANDOVER-20260924.md`](A3-CED-PD-HANDOVER-20260924.md) §73–76）。
注意：这条**不是**本文档 §5.1.2 的 32 位回绕，两者独立。

> 影响：`docs/A3-PD-BF16.md` 记录的 2026-09-23 基线验收（142,426 token，乱码指纹 0）
> 用的是同一套默认多流配置。本轮的复现说明那个"通过"可能是
> **规模/时序相关的偶发**，需要在关多流的配置下重新确认。

## 4. 端到端性能：prefill token/s 与 decode ms/step（2026-09-25 19:32–19:42）

当前在跑的配置：**P = 全 40 层、`MULTISTREAM=1 DSA_OVERLAP=1`；
D = 全 40 层、`MULTISTREAM=0 DSA_OVERLAP=0`**（即 §3 里修好乱码后的基线）。

工具：`tools/ced_pd_bench.py`（流式逐块计时）。不要把
`ced_pd_acceptance.py` 的 `tpot_ms` 当 decode 速率——它对**非流式**请求等于
`wall / completion_tokens`，里面含了 prefill。本表用 `ignore_eos=true` 强制生成满
128 个 token，并按 **usage 里的真实 token 数**算平均（避免 vLLM 把多个 token
合进一个 SSE 块导致的"块间隔翻倍"）。

| 上下文 | prompt tokens | TTFT | **prefill tok/s** | decode ms/step（均值） | 中位 | p90 | **decode tok/s** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32K | 32,779 | 4.54 s | **7,219** | 25.4 / 25.5 | 25.6 | 51.2 | **39.3** |
| 144K | 143,989 | 21.82 s | **6,599** | 26.8 / 26.7 | 27.3 | 28.9–54.1 | **37.3** |
| 1M | 999,396 | 280.92 s | **3,557** | 37.4 / 37.6 | 40.4 | 80.5 | **26.7** |

两次重复的一致性：

* prefill tok/s：7215/7223、6603/6592、3557/3558（差 < 0.5%）
* decode ms/step：25.4/25.5、26.8/26.7、37.4/37.6

（毫秒/步按 `首块→末块总时长 ÷ (completion_tokens-1)` 算；括号里是两次重复值。
p90 里出现的 ~51/54/80 ms 是 vLLM 把 2 个 token 合进一个 SSE 块造成的，
不代表真实抖动。）

### 4.1 这两列怎么看

* **prefill token/s 随上下文下降**：7.2K → 6.6K → 3.6K。144K 到 1M 掉了 46%，
  比"线性变慢"更陡，说明长序列下 P 的注意力/访存效率明显退化，是后续优化最值钱的点。
* **decode ms/step 随上下文上升**：25.4 → 26.8 → 37.4 ms。1M 比 32K 慢 1.47×。
  层里有 4 个 group 需要全上下文、其余 10 个 group 只有 128-token 窗口，
  所以在 1M 上 decode 仍会被那 4 个 group 拖慢 —— 与 CED 论文"SWA 让 decode 成本
  与上下文近乎解耦"的预期还有差距。
* TTFT 基本就是 P 的 prefill：1M 的 280.9 s 与 P 日志里相邻 `Delaying free`
  时间差（286–291 s，含上一条的 D 解码）一致。

### 4.2 与 CED 臂的 prefill 对照

CED 臂还没跑这个基准（要重启 P+D，约 12 分钟），但用同一批请求的流式 TTFT 可以
直接折算 prefill token/s：

| 上下文 | CED-PD（P=20 层） | 基线（P=40 层） | 比值 |
|---|---:|---:|---:|
| 144K | 143,989 / 10.952 s = **13,147 tok/s** | 6,599 | **1.99×** |
| 1M | 999,396 / 101.217 s = **9,874 tok/s** | 3,557 | **2.78×** |

144K 的 1.99× 与层数比（40/20）完全吻合；1M 的 2.78× 更高，说明基线在长序列下的
退化比 CED 更严重（或者说 CED 省掉的正是最贵的那部分）。下表补齐后可以确认。

## 5. 尚未完成

1. 基线 1M 的**四针正确性**只跑到第一条（本轮用 stream 模式拿到 TTFT 与首答 PASS），
   完整 4/4 需要再约 19 分钟。
2. 1M 比值 2.79× 大于层数比 2.0×，**未归因**：需要 P 侧 profiler 对照
   （20 层 vs 40 层的算子/访存构成）。两侧 144K 的 profiler 已采集：

   | 臂 | 容器 | `torch_profiler_dir` | 大小 |
   |---|---|---|---|
   | CED P | `dsv41-ced-prof-p` | `/opt/dsv41/results/ced_prof_p_0925_171906/prof` | 4.3 G |
   | CED D | `dsv41-ced-prof-d` | `/opt/dsv41/results/ced_prof_d_0925_165058/prof` | 1.2 G |
   | 基线 P | `dsv41-base-p` | `/opt/dsv41/results/ced_base_p_0925_181150/prof` | 待测 |
   | 基线 D | `dsv41-base-d-ms0` | `/opt/dsv41/results/ced_base_d_ms0_0925_184318/prof` | 待测 |

   目录属主是容器内 root，宿主侧 `ls` 会 Permission denied，需
   `docker exec <容器> bash -lc 'ls /opt/dsv41/results/<run>/prof'`；
   分析用容器内的 `msprof`（CANn 9.1.0）。
3. D 侧多流在什么长度开始出错、以及具体是哪条流缺少依赖，尚未定位。

### 5.1 顺带的独立结论：缓存命中仍不可用

`prefix` 模式两臂都能答对（CED 144K 2/2、1M 2/2），但两次请求的
`usage.prompt_tokens_details.cached_tokens` **都是 0**，`prefix_cache_hits_total` 增量也是 0
⇒ 这两次只是"同前缀各算一遍"，**没有验证到缓存命中**。原因见
[`CED-PD-ACCEPTANCE.md`](CED-PD-ACCEPTANCE.md)：CED 启动器硬门 `PREFIX=0`，
且 D 侧预清零只覆盖 `get_unhashed_block_ids`。这一项仍是代码级阻断。

## 6. 复现命令

```bash
# CED 臂（P=20 层 + layer-20 全局源）
NAME=dsv41-ced-prof-p RUN_ID=ced_prof_p_$(date +%m%d_%H%M%S) PROFILE=1 \
  bash <pkg>/scripts/serve_a3_ced_pd.sh prefill      # chip0-7, 18990/19090
NAME=dsv41-ced-prof-d RUN_ID=ced_prof_d_$(date +%m%d_%H%M%S) PROFILE=1 \
  CED_EXPERIMENTAL_GRAPH=1 bash <pkg>/scripts/serve_a3_ced_pd.sh decode   # chip8-15, 18991/19091

# 全 40 层基线（P/D 都是完整模型；必须关 D 多流）
MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=8192 SPEC=0 PREFIX=0 \
  MULTISTREAM=0 DSA_OVERLAP=0 PROFILE=1 PORT=18990 KV_PORT=19090 DEVS="0 1 2 3 4 5 6 7" \
  NAME=dsv41-base-p bash <pkg>/scripts/serve_a3_pd.sh prefill
MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=8192 SPEC=0 PREFIX=0 \
  MULTISTREAM=0 DSA_OVERLAP=0 PROFILE=1 PORT=18991 KV_PORT=19091 DEVS="8 9 10 11 12 13 14 15" \
  NAME=dsv41-base-d bash <pkg>/scripts/serve_a3_pd.sh decode

# 测量
python3 tools/ced_pd_acceptance.py --base-url http://127.0.0.1:18992 \
  --tokenize-url http://127.0.0.1:18990 --model deepseek-v41-ced-pd \
  --corpus data/hongloumeng.txt --mode stream --context-tokens 144000 \
  --out results/<tag>.json
# P 侧 prefill 时长：取 P serve.log 里相邻两条 "Delaying free of N blocks" 的时间差
grep -a "Delaying free" <p>/serve.log
```
