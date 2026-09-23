# 长上下文 × Agent 精度问题 —— 复现手册（**A2 / main 分支**）

> 面向：现场反馈「一次 agent 的 context ≈520k 直接进入，完整做一次 prefill
> （可能还有其它流同时请求），然后出现错误」。
> 判据（三条一起看，缺一不可）：**答案原文 repr** ＋ **乱码指纹**（U+FFFD/NUL/C0 控制符/
> 孤立代理对）＋ **逐字判定**（`temperature=0`，重复时是否逐字相同）。

---

## 0. 为什么必须用这个探针（而不是"看着像乱码就报 bug"）

本仓踩过的坑（都写进 `a2/logs/`）：
* 题库 10 题的 prompt **只有几十 token** ⇒ 对长上下文**没有判别力**；
* replay vs fill 的 `sha`/逐字相同 —— 原文已写明"**对路径差异没有判别力**"；
* 计数器（`BlockStored`/`CPU_to_GPU`/`hits`）只说明"搬了字节"，不说明"搬对了内容"。

⇒ 所以用 `tools/ctx_agent_probe.py`：它在**长上下文**上问**有唯一答案**的问题，并逐字判。

---

## 1. 前置（一次）

```bash
cd <本仓>                      # A2 上是 release 仓的检出
git fetch origin main && git checkout main && git pull --ff-only
python3 tools/ctx_agent_probe.py --selfcheck     # ★ 先验探针自己的判据（应打印"失败 0 项"）
```

`--selfcheck` 不是花架子：它抓出过两个**判据自身**的缺陷（"问 B 却插了 A 的针"、
语料短于目标时切出空串、拿 91 token 冒充 520k）——判据错了比没有判据更危险。

---

## 2. 起服（**main 分支**原样口径）

```bash
MODEL=<模型目录> bash scripts/serve_a2.sh          # main 默认：PORT=8100 / MAX_LEN=1048576 / SERVED_NAME=deepseek-v41
# 等 /health=200（首次冷编译静态内核 15–20 min 属正常）
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8100/health
```

★ 需要 520k 的上下文 ⇒ **`MAX_LEN` 必须是 1M**（main 的默认就是 1048576）；
若你显式设过 `MAX_LEN=133120`，520k 会被服务端拒绝，那不是本次要查的问题。

---

## 3. 复现（按现场四种要素，逐个加）

```bash
BASE=http://127.0.0.1:8100

# ① 单发 520k 完整 prefill（现场"一次 agent 的 context 直接进入"）
python3 tools/ctx_agent_probe.py --base-url $BASE --model deepseek-v41 \
    --mode bigprefill --context-tokens 520000 --conc 1 --repeats 1 --out ~/r1_bigprefill.json

# ② 大 prefill 之后再走多轮（现场"完成 prefill 之后才出现错误"）
python3 tools/ctx_agent_probe.py --base-url $BASE --model deepseek-v41 \
    --mode biggrow --context-tokens 520000 --followups 3 --out ~/r2_biggrow.json

# ③ ★ 混合负载：大 prefill 在飞 + 其它流同时打短请求（现场"可能还有其他的流同时请求"）
python3 tools/ctx_agent_probe.py --base-url $BASE --model deepseek-v41 \
    --mode mixed --context-tokens 520000 --conc 4 --out ~/r3_mixed.json

# ④ ★ 真流式（SSE）＋ 工具定义：真实 Agent 客户端就是这么发的（不是 stream:false！）
python3 tools/ctx_agent_probe.py --base-url $BASE --model deepseek-v41 \
    --mode stream --context-tokens 520000 --conc 8 --out ~/r5_stream.json

# ⑤ 并发大 prefill（多路各自 520k）
python3 tools/ctx_agent_probe.py --base-url $BASE --model deepseek-v41 \
    --mode bigprefill --context-tokens 520000 --conc 2 --repeats 2 --out ~/r6_conc.json
python3 tools/ctx_agent_probe.py --base-url $BASE --model deepseek-v41 \
    --mode bigprefill --context-tokens 520000 --conc 2 --repeats 2 --out ~/r4_conc.json
```

每条的判读：
* `PASS/FAIL` 是**逐字**判定（不是"看着像"）；
* `fp=U+FFFD/NUL` 是乱码指纹计数（**非 0 就是实锤的乱码**）；
* `★ 汇总` 里的 **`bad_ctx`** 必须为 0 —— 非 0 表示**这次读数无效**
  （上下文没到目标长度），**不能**当成"通过"或"失败"；
* 每条都会写 `--out` JSON（可离线复算，直接发回来即可）。

---

## 4. 同时把服务侧证据留下（**服务还活着时**）

```bash
# 语言模型自己报的接受长度（本仓最灵敏的行为判据：对亚文本级污染敏感）
curl -s http://127.0.0.1:8100/metrics | grep -E 'SpecDecoding|prefix_cache|kv_cache_usage' | tail -20

# 引擎有没有报错/异常
grep -aE 'ERROR|Traceback|KeyError|AssertionError|illegal|out of range' <serve.log> | tail -30

# 关键容量与几何（确认这臂确实是 1M 口径）
grep -a 'GPU KV cache size' <serve.log> | tail -1
```

---

## 5. 我们已经排除过的（**别重复走**）

在 **A3-21（main 检出 + 官方 a3 镜像 + 1M + `CPU_BIND=0`）** 上，下列全部**干净**（逐字命中、乱码指纹全 0、`bad_ctx=0`）：

| 臂 | 规模 | 结果 |
|---|---|---|
| `bigprefill` 单发 | 520,852 token prefill，111 s | PASS |
| `bigprefill` + 2 路并发（各 520k） | 6 次请求 / 662 s | 全 PASS |
| `biggrow` 520k → 同会话追问 3 轮 | 4 次请求 | 全 PASS |
| `mixed`：1 路 520k prefill ＋ 4 路短流并发 | 9 次请求 | 全 PASS（含空闲基线对照） |
| `stream`：520k **真流式（SSE）** ＋ 4 路并发短流 ＋ 流式工具参数 | 6 次请求 / 217 s | 全 PASS |

★ `stream` 臂的一个**重要读数**：4 路短流的**首字延迟都在 106.6 s**（= 大 prefill 结束那一刻）
⇒ 在 `MAX_SEQS=4` 下它们被**推迟**了、并没有与 520k prefill 真正交错；
而 **main 的默认是 `MAX_SEQS=32`** ⇒ 现场若用 main 默认，交错行为会不同。
（本表后续会补 `MAX_SEQS=32` 的读数。）
| `needle`/`grow`/`reuse`/`toolargs`/`evict`/`conc` | 8K / 32K / 131K | 全 PASS |

⇒ **【实测】A3 + main 上，仅"520k 大 prefill + 并发/追问/混合流"不足以触发**。
仍未覆盖、且与现场不同的两条**结构性差异**：

1. **模型不同**：现场用的是 `v41-w4a8-flat`，A3 上只有
   `v41-w4a8-engram-dr-vision-qrot-mtpq`（Engram 层 `[1,14]`、含 vision/qrot/mtpq）。
   ⇒ 若现场模型少了某几个组件（例如 vision/qrot/mtpq 或不同的 `engram_layer_ids`），
   **代码路径不同**，A3 这边跑不到那条路。
2. **硬件不同**：A2 = 8×910B3，A3 = 8×910C（同一份 main、不同 SoC/不同 KV 页几何）。

⇒ 所以本轮最该做的是：**在 A2（main）上按 §3 跑**，把 `--out` 的 4 份 JSON + §4 的日志片段发回。

---

## 6. 如果 A2 上复现出来了，下一刀怎么切（单变量）

按**代价从低到高**，每次只改一个变量：

| 顺序 | 改什么 | 判什么 |
|---|---|---|
| 1 | `MAX_LEN` 从 1M 降到 520k 以下（此时 520k 请求会被拒） | 确认"必须真的走满 520k"才算复现 |
| 2 | `MAX_SEQS` 1 ↔ 32 | 并发调度是否是触发条件 |
| 3 | **关掉投机解码**（`SP_TOKENS=0` 或等价的关闭开关） | prefill/decode 边界与 spec 的交互 |
| 4 | 关 Engram（`ENGRAM=0`） | Engram 是否参与 |
| 5 | `--block-size` / `--max-num-batched-tokens`（8192 → 16384） | 分块 prefill 的块边界 |
| 6 | 换模型（用 A3 上那套 `…qrot-mtpq` 或任何已知良好的权重） | 模型本身 vs 引擎 |

★ 纪律（本仓反复栽过）：**"我传了这个变量" ≠ "这个变量生效了"** ——
每条臂都要从**干跑日志/容器内 md5/metrics** 上核对它真的生效了。
