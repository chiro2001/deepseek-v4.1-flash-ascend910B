# 数据缺口总表 —— 上游材料还差哪些测量（2026-09-21）

> **用途**：在把材料交给上游之前，先回答一个问题：**每份草稿里写着的 "not measured / not yet run / TODO"，哪些能补、哪些补不了、补不了的怎么说。**
> 目录里每一份 `pr/*.md` 都有一节 "What is still missing"，本表把它们汇总成一张可执行的清单。
>
> 状态口径：【已补】= 本次（2026-09-21 下午）新增实测；【在补】= A3 上正在跑；【不可补】= 缺环境，只能如实标注。

---

## 0. 一句话结论

* **能补的 8 项里，6 项本轮在 A3（A3-node1，4 张空闲 die）上补完或正在补**；
* **补不了的 5 项全部是"环境不存在"**（W8A8、A5、multi-node、8 卡全空闲、A2 直连），不是"没跑"；
* 因此材料里对应的 "not measured" **不用删**，改成 "not measured on this hardware, and why" 即可 —— 这比含糊其辞更符合 RFC 的 "reproducible comparisons" 要求。

---

## 1. 能补、且本轮已补/在补（A3 单卡 + 空闲 die）

| # | 缺口（原文出处） | 上游要求 | 怎么补 | 状态 |
|---|---|---|---|---|
| G1 | `PR-rope-index-select.md`："**non-contiguous / empty / padded 的形状矩阵**由 `pr/rope_edge_cases.py` 补测，结果回填后再发" | RFC **[91]** *"Validate numerical accuracy, **non-contiguous cache strides, empty/padded batches**, and prefill/decode shapes for each fusion. Benchmark both individual kernels and the full pipeline"* | A3 上跑 `rope_edge_cases.py` 全量（shapes / draft / seq / graph / profile 五个 phase） | 【在补】→ `logs/35` |
| G2 | `pr/RFC-comment.md` **[91]** 段："**Not measured:** a systematic matrix over non-contiguous strides, empty/padded batches and prefill/decode shapes **per fusion**" | 同上 | 同 G1；本轮至少把 **RoPE 这一项融合**的矩阵补齐，措辞改为"per-fusion 只覆盖 RoPE，其余未测" | 【在补】→ `logs/35` |
| G3 | `RFC-16375-CONTRIBUTION.md` §3.1：**Control arm**（chunking gate = 0 的第三臂）"not in the script yet, listed as a TODO" | RFC **[97]** *"reproducible comparisons"* | 脚本里其实已有该臂；本轮**重跑并回填**，顺带与 2026-09-21 单卡机的旧结果做一次独立复现 | 【在补】→ `logs/36` |
| G4 | `RFC-16375-CONTRIBUTION.md` §3.3 第一行：host table registration **"not yet run"** | RFC **[46]/[47]**、上游 issue **#16828** 的分岔点 | A3 上跑 `probe_engram_hostmap.py`：`aclrtHostRegister` 与 `aclrtHostRegisterV2` 在 MAPPED / PINNED / MAPPED\|PINNED / 无 flag 下的 **ret 码 + 设备侧读回** | 【在补】→ `logs/37` |
| G5 | `RFC-16375-CONTRIBUTION.md` §3.3 第二行：token history update **"not yet run"** | RFC **[47]** *"lookup latency"* | A3 上跑 `bench_ngram_history.py`（纯 host 侧，5~6 臂） | 【在补】→ `logs/37` |
| G6 | `RFC-comment.md` **[47]** 段："**NUMA/bandwidth sensitivity as such is not measured**" | RFC **[47]** 五项里的最后一项 | 新脚本 `bench_host_dram_bw.py`：设备侧读 host-mapped DRAM 的**连续 / 随机行 gather** 带宽、**1 die vs 3 die 并发**、**跨 NUMA node**（A3 实测 8 个 node） | 【在补】→ `logs/38` |
| G7 | `issue-track-C.md` §3.4："**No per-component `npugraph_ex` attribution**" | RFC **[75]** | 需要 8 卡服务 + profiler 分组件 A/B ⇒ **A3 空闲 die 不够**（见 §3），本轮只能保持"未测"并写清原因 | 【不可补】 |
| G8 | `issue-track-C.md` §3.6："**No trace evidence for the overlap claim**"（RFC line 104 要求 overlap 条目必须给 trace） | RFC line 104 | 同上：需要 8 卡服务 + `PROFILE=1` 重启。A3 的 8020 服务**没有** `/start_profile`（实测 404），重启代价 ≈13 min 且会打断用户正在用的 codex 链路 | 【不可补·本轮】 |

---

## 2. 需要 8 卡整机的（A3 当前空缺）

| # | 缺口 | 为什么需要 8 卡 | 备注 |
|---|---|---|---|
| G9 | `RFC-16375-CONTRIBUTION.md` §5：**单会话 ablation**（一个 session 里逐 gate 开关，A1 harness） | `MODEL=... MODE=full bash scripts/run_test.sh` 走的是 TP8/EP8 服务；每换一次 gate 就要重启一次服务 | A3 上 die 0–5 被别人的服务占着（`VLLMEngineCore` / SGLang / 其他容器），**只剩 die 3 / 6 / 7 三张空闲**，起不了 8 卡服务 |
| G10 | G7（`npugraph_ex` 分组件归因） | 同上 | —— |
| G11 | G8（overlap 的 trace artifact） | 同上 | 若日后 A3 能整机空出，命令已写在 `RFC-16375-CONTRIBUTION.md` C16 |

> **可替代的次优做法**（如果用户希望本轮就动）：停掉 A3 上别人的容器后用 die 0–7 起一个 8 卡服务 —— 这需要用户确认，主代理**没有**擅自做。

---

## 3. 需要 A2（生产目标机）的

> A2 我这边连不上（只有用户粘贴的 log）。以下命令都设计成**只读、不影响线上服务**，用户可以随时粘一段回来。

| # | 缺口 | 一行命令 | 期望能得到 |
|---|---|---|---|
| A1 | PREFIX on/off 两臂对照（当前 A2 只测了 PREFIX=1 的一臂） | 见 §4 | TTFT 与 prefill tok/s 的差值 = 前缀缓存的实际收益 |
| A2 | 延迟分位数（当前只有中位数） | `curl -s localhost:8077/metrics \| grep -E "ttft\|e2e\|inter_token"` | p50/p90/p99，`/metrics` 自带 histogram |
| A3 | `/metrics` 累计值（当前贴的是瞬时值） | `curl -s localhost:8077/metrics > /tmp/m.txt; wc -l /tmp/m.txt` | 累计 token 数、prefix 命中率、spec 接受率 |
| A4 | 并发上限对照（当前只到 C4，上限 32） | `python3 tools/bench_concurrency.py --base-url http://127.0.0.1:8077 --model deepseek-v4-flash --concurrency 1,4,8,16,32 --prompt-tokens 1024 --output-tokens 256` | 与 A3 的 C1–C64 曲线同口径对比 |

---

## 4. A2 上可直接粘贴的命令块（只读，不改服务）

```bash
# ① 当前服务的启动参数（确认 prefix / spec / dspark 的真实取值）
ps -eo args | grep -m1 "[v]llm serve" | tr ' ' '\n' | grep -E "prefix|spec|dspark|max-num-seqs|util" 

# ② 延迟直方图 + 累计量（重启后才会清零，所以这是"自起步以来"的累计）
curl -s localhost:8077/metrics | grep -E "^vllm:(time_to_first_token|e2e_request_latency|inter_token_latency|request_success|prefix_cache)" | head -40

# ③ 前缀缓存两臂对照：同一段 prompt 连发两次，看第二次的 TTFT
for i in 1 2; do
  curl -s -o /dev/null -w "run$i ttft=%{time_starttransfer}s total=%{time_total}s\n" \
    -X POST localhost:8077/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-flash","max_tokens":1,"messages":[{"role":"user","content":"'"$(head -c 6000 /dev/urandom | base64 | head -c 6000)"'"}]}'
done

# ④ 并发曲线（若仓库里没有 tools/bench_concurrency.py，跳过这条）
python3 tools/bench_concurrency.py --base-url http://127.0.0.1:8077 --model deepseek-v4-flash \
  --concurrency 1,4,8,16,32 --prompt-tokens 1024 --output-tokens 256 2>&1 | tail -20
```

---

## 5. 承认补不了的（环境不存在，而不是没跑）

| 项 | 属于哪条 RFC / 草稿 | 为什么补不了 |
|---|---|---|
| **W8A8** 的任何数字 | RFC [63] 硬件矩阵（A2/A3 = W8A8）、多份草稿的 "Not measured: we have no W8A8 run" | 我们手上只有 W4A8 权重；W8A8 需要重新量化 + 重跑验收 |
| **A5** 平台 | RFC [96] | 没有 A5 机器 |
| **EP=16/32、multi-node、DP composition** | `RFC-comment.md` [63]/[65] 段 | 需要两台以上 A3 节点 |
| **`fullmesh_v2` 对比** | `RFC-comment.md` [65] 段 | 上游实现未公开到可复现的程度 |
| **SP / DCP / PD** | `issue-track-C.md` §3.2、`RFC-comment.md` [73] 段 | 我们的部署是单节点 TP8/EP8，没有开 SP/DCP/PD |

> 对这类项，材料的写法统一为：**"not measured — no <X> in our environment; the claim is scoped to <what we did measure>"**，绝不用相邻数字替代（这是 `RFC-16375-CONTRIBUTION.md` Appendix B 第 5 条自己定的规矩）。

---

## 6. 本轮补完后，材料里会变化的位置

| 文件 | 变化 |
|---|---|
| `pr/PR-rope-index-select.md` | 新增边界矩阵一节（G1/G2 的 RoPE 部分） |
| `pr/RFC-16375-CONTRIBUTION.md` | §2.1（NUMA/带宽行）、§3.1（control arm 去掉 TODO）、§3.3（两行改成实测） |
| `pr/RFC-comment.md` | [91] 段的 "Not measured" 收窄为"per-fusion 覆盖了 RoPE"；[47] 段补 NUMA/带宽 |
| `logs/35…38` | 四份新日志（本轮新增实测） |

---

## 7. 复现入口

| 想要 | 命令 / 路径 |
|---|---|
| A3 上的三个独占槽位 | `bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh <c0\|c1\|c2> --name <名> -- <命令>`（c0=die3、c1=die6、c2=die7；退出码 75 = 没抢到锁） |
| 起/查容器 | `bash ~/projects/dsv41-upstream-pr/tools/a3_up.sh [--status]` |
| 远端工作区 | `ssh A3-node1`，`~/projects/dsv41-upstream-pr/`（`bench/` 脚本、`agents/` 各自产物、`locks/` 锁） |
| 本机日志索引 | [`logs/README.md`](logs/README.md) |
