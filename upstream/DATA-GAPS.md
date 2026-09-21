# 数据缺口总表 —— 上游材料还差哪些测量（2026-09-21）

> **用途**：在把材料交给上游之前，先回答一个问题：**每份草稿里写着的 "not measured / not yet run / TODO"，哪些能补、哪些补不了、补不了的怎么说。**
> 目录里每一份 `pr/*.md` 都有一节 "What is still missing"，本表把它们汇总成一张可执行的清单。
>
> 状态口径：【已补】= 新增实测；【在补】= A3 上正在跑；【不可补】= 缺环境，只能如实标注。
> **两轮补测**：第一轮 12:5x–13:1x（G1–G6），第二轮 13:3x 起（G12–G15，见 §1.2）。

---

## 0. 一句话结论（2026-09-21 15:5x 更新）

* **已补 21 项**（G1–G6、G12–G18、G20–G22）；**本机再无排期的缺口**。
* **需要 8 卡整机 3 项**（G9–G11，§2）；**需要 A2 现场 4 项**（A1–A4，§3）
  —— **用户已明确当前无法使用 A2**，这四项只能挂着；
  **环境不存在 5 项**（§5）。
* **补不了的都写清"为什么"**，绝不用相邻数字顶替 —— 这比含糊其辞更符合 RFC 的
  *"reproducible comparisons"* 要求（`RFC-16375-CONTRIBUTION.md` Appendix B 第 5 条）。
* ★ **最硬的一条已闭合**：host offload 的判据原来只有 **512 MiB 合成表 / 1–3 die**，
  现已补 **真实 206 GiB 表 / 3 die 并发**（= 8 rank 的 3/8 代理）—— 见 §1.2 G12 与 `logs/40`。
* ★★ **最后一项本机可做的（G18）在 `logs/43` 闭合，而且它更正了一个会误导上游的数字**：
  "ceiling=512 只比上游慢 1.19–1.38×" 是 **eager 假象**，进图后上游臂塌到 0.099 ms（n=1），
  真实比值是 **1.41×（512）~ 6.84×（1）**；推荐值不变，但引用口径必须改成图内。

---

## 1. 能补、且本轮已补/在补（A3 单卡 + 空闲 die）

| # | 缺口（原文出处） | 上游要求 | 怎么补 | 状态 |
|---|---|---|---|---|
| G1 | `PR-rope-index-select.md`："**non-contiguous / empty / padded 的形状矩阵**由 `pr/rope_edge_cases.py` 补测，结果回填后再发" | RFC **[91]** *"Validate numerical accuracy, **non-contiguous cache strides, empty/padded batches**, and prefill/decode shapes for each fusion. Benchmark both individual kernels and the full pipeline"* | A3 上跑 `rope_edge_cases.py` 全量（shapes / draft / seq / graph / profile 五个 phase） | **【已补】** → `logs/35`：**32/32 + 5/5 checks**；`n=0` 空批逐位一致、`n=4096` 图内外 −376/−384 µs、6→2 kernel |
| G2 | `pr/RFC-comment.md` **[91]** 段："**Not measured:** a systematic matrix over non-contiguous strides, empty/padded batches and prefill/decode shapes **per fusion**" | 同上 | 同 G1；本轮至少把 **RoPE 这一项融合**的矩阵补齐，措辞改为"per-fusion 只覆盖 RoPE，其余未测" | **【已补】** → `logs/35`；`pr/RFC-comment.md` 的 [91] 段已改写为"RoPE 已测、其余融合仍未测"（正文里明说还差哪几项） |
| G3 | `RFC-16375-CONTRIBUTION.md` §3.1：**Control arm**（chunking gate = 0 的第三臂）"not in the script yet, listed as a TODO" | RFC **[97]** *"reproducible comparisons"* | 脚本里其实已有该臂；本轮**重跑并回填**，顺带与 2026-09-21 单卡机的旧结果做一次独立复现 | **【已补】** → `logs/36`：时间 1.00–1.06×（换文件在时间上免费）、**显存恒定 1.20×**；三次独立运行同向 |
| G4 | `RFC-16375-CONTRIBUTION.md` §3.3 第一行：host table registration **"not yet run"** | RFC **[46]/[47]**、上游 issue **#16828** 的分岔点 | A3 上跑 `probe_engram_hostmap.py`：`aclrtHostRegister` 与 `aclrtHostRegisterV2` 在 MAPPED / PINNED / MAPPED\|PINNED / 无 flag 下的 **ret 码 + 设备侧读回** | **【已补】** → `logs/37`：两条 API **ret=0 且设备侧逐字节一致** ⇒ 本机这一档没有分岔；206 GiB 满表另见 G12 |
| G5 | `RFC-16375-CONTRIBUTION.md` §3.3 第二行：token history update **"not yet run"** | RFC **[47]** *"lookup latency"* | A3 上跑 `bench_ngram_history.py`（纯 host 侧，5~6 臂） | **【已补】** → `logs/37`：生产 decode `n=128` **1.68 → 0.074 ms（22.8×）**，per-token 走法 **1312×**，`torch.equal` 14/14 |
| G6 | `RFC-comment.md` **[47]** 段："**NUMA/bandwidth sensitivity as such is not measured**" | RFC **[47]** 五项里的最后一项 | 新脚本 `bench_host_dram_bw.py`：设备侧读 host-mapped DRAM 的**连续 / 随机行 gather** 带宽、**1 die vs 3 die 并发**、**跨 NUMA node**（A3 实测 8 个 node） | **【已补】** → `logs/38`：连续 **107 GB/s**、随机 gather **96 GB/s**；并发**按 CPU socket 封顶 ≈115 GB/s**（不是摊薄） |
| G7 | `issue-track-C.md` §3.4："**No per-component `npugraph_ex` attribution**" | RFC **[75]** | 需要 8 卡服务 + profiler 分组件 A/B ⇒ **A3 空闲 die 不够**（见 §3），本轮只能保持"未测"并写清原因 | 【不可补】 |
| G8 | `issue-track-C.md` §3.6："**No trace evidence for the overlap claim**"（RFC line 104 要求 overlap 条目必须给 trace） | RFC line 104 | 同上：需要 8 卡服务 + `PROFILE=1` 重启。A3 的 8020 服务**没有** `/start_profile`（实测 404），重启代价 ≈13 min 且会打断用户正在用的 codex 链路 | 【不可补·本轮】 |

### 1.2 第二轮新识别（读文档时才发现口径不对）

| # | 缺口（原文出处） | 上游要求 | 怎么补 | 状态 |
|---|---|---|---|---|
| **G12** ★ | `RFC-16375-CONTRIBUTION.md` §2.1 `Missing from [47]`：*"on a **synthetic** 512 MiB / 2 GiB table … **not** on the 206 GiB production table and not at 8 ranks. Still unmeasured: **bandwidth at the real table size and rank count**; whether the 8-rank bring-up's **concurrent registration perturbs steady-state bandwidth**; and **any real hot-row cache**" | RFC **[47]** "NUMA/bandwidth sensitivity **under realistic concurrency**" | 用**真实 206 GiB 表本体**（A3 上就有，`stat %b` 已确认非稀疏）+ **3 die 并发**（8 rank 的 3/8 代理）：并发满表注册的 ret 码 / 有无 `207001`·`507011`、真实表尺寸下的连续读与随机 gather 带宽、热行 skew 的收益、"注册中"对其它 die 稳态带宽的扰动 | 【在补·收尾】→ `logs/40`：**单 die 满表与 3 die 并发都已跑完、锁已释放**，子代理正在写报告并回填 §2.1 |
| **G13** | `RFC-16375-CONTRIBUTION.md` §3.2.1：*"even the ablation still pads to 512 rows, so its 1.3–1.5× is pad work as well — **a ceiling closer to the real batch size is the next knob, and it is not measured here**"* | RFC **[90]** "Optimize the Engram gather/dequantization/gating fusion" | 固定 `CHUNK=512`，扫 padding 天花板 256/512/1024/2048/4096 × 多个真实 batch size，给出"天花板 vs 时间/显存"曲线与拐点 ⇒ 把"tunable"量化成"调到多少、代价多少" | **【已补】** → `logs/41`：**t ≈ 0.1 + 0.69×(MAX/512) ms，无拐点**；生产约束下 2048 已最优，小 batch 图的最优是 512（差 3.4–3.6×，只能靠 per-capture-size 分桶拿掉）；另发现 `MAX=256` 被静默抬到 4096 的坑 |
| **G14** | `logs/35` §5：**小表（8K）下 PR 臂的 kernel 计数**未测（probe 只 profile 了 stock 臂） | RFC **[91]** | 让 PR 臂也被 profiler 覆盖，确认小表下同样 3→1 kernel/次 | **【已补】** → `logs/41` §6.1：8K 表下 **12.00 → 2.00（6× 少）**，`Transpose` 全部消失 |
| **G15** | `logs/35` §5：int32 eager 两格回退（+20.7 / +28.9 µs）**只有计数级归因，无逐 op 时序** | RFC **[91]** | 补逐 op 时序分解，把归因从【推断】升成【实测】 | **【已补】→ 并且修掉了** → `logs/41` §6.2/§6.3 + `logs/42`：归因是 host 侧"每方向多 build 一次 index"；把 index 提到每次调用一次后 **+20.7/+28.9 变成 −17.9/−17.7 µs**，kernel 数 4.00→3.00 |

### 1.3 第二轮补测后又新暴露的两项

| # | 缺口 | 为什么重要 | 状态 |
|---|---|---|---|
| **G16** | `perf/rope-index-select-on-16285` **组合分支的行为不变性**（改 hoist 后重建了组合版） | PR 正文的 "Overlap with #16285" 一节直接引用它 | **【已补】** → `logs/42` §4 + `logs/raw/42-rope-composition-check-a3.txt`：6 组配置全部 `base==merged` 且等于表查表结果，组合版独有的 `cached_output_len` 六项语义全 True ⇒ `ALL CHECKS PASSED`。（第一次尝试因三卡被 G12 占满而超时，槽位一空即补跑） |
| **G17** | 新 head（`ed5b928c`）上的 **13 个单元测试重跑**：改 hoist 时动了 `_rope_gather_rows` 的两个调用点 | 上游 CI 会跑，但不能凭空声称"本地已跑" | **【已补】** → `logs/raw/42-rope-ut-standalone-a3.log`：**13 passed / 0 failed**。做法是绕开损坏的 conftest（容器 vllm 比分支旧 ⇒ `adapt_patch()` 就死），用 `pr/run_rope_ut_standalone.py` 按路径 import 测试模块、自带两个 fixture 逐条跑。**不覆盖 collection/参数化/conftest 桩**，CI 仍是正式口径 |

### 1.4 本轮之外的既有缺口（未变）

见 §2（8 卡整机）与 §5（环境不存在）。这两节的内容本轮没有变化，只是 A3 的 8 卡服务现在**又少了一张空闲 die**（die 3/6/7 被我们自己的槽位占用，die 0–5/8–15 是别人的服务）—— 所以 G9/G10/G11 仍然只能等整机空闲。

### 1.5 第三轮：本机（A3 空闲 die）能做的最后几项 —— **全部已补**

> 2026-09-21 15:5x：下面 5 项里 4 项已补（G18/G20/G21/G22），**只剩 G19 需要 8 卡服务**
> （⇒ 归入 §2 的 G9）。**本机已无排期的数据缺口。**

| # | 缺口 | 为什么值得补 | 难度 |
|---|---|---|---|
| **G18** ★ | **图内（ACLGraph）的 ceiling 扫描** | `logs/41` 的 ceiling 曲线是 **eager** 口径，而**生产帧是图内** —— `V41_ENGRAM_GATE_MAX_TOKENS` 本来就是为 capture 设的常数 | **【已补】** → `logs/43`：曲线**仍是直线**（图内斜率 0.639–0.691 ms/512 行，eager 0.648–0.710），`MAX=512` vs `2048` 的绝对差几乎不变（−1.92…−2.10 vs eager −2.02…−2.18）⇒ **推荐值不变**。但**「只慢 1.19–1.38×」是 eager 假象**：上游进图后 n=1 从 0.546 → **0.099 ms**，图内真实比值 **1.41×(512) ~ 6.84×(1)**、shipped 2048 是 **5.1× ~ 26.1×**。126/126 格逐位一致 + 活性检查通过 |
| **G19** | engram gate 的**端到端（整 step）放大** | `logs/41` §8 明确标注：函数级 × 层数只是【推断】，**没有实测**。RFC [90] 的收益最终要在 step 上看 | 高：要 8 卡服务 ⇒ 实际落到 G9 |
| **G20** | 8K **小表的时间**（现在只有 kernel 计数） | `logs/41` §6.1 只量了 kernel 数；小表在 wall clock 上的收益没测 | **【已补】** → `logs/43`：eager n=192 单调用 **213.7 → 83.9 µs**、图内 40 层 **148.7 → 18.7 µs（7.9×）**；顺带确认"小表下 stock 反而更慢"（1M 48.4 µs vs 8K 147.6 µs，PR 两格都 18.7） |
| **G21** | 注册耗时的**双峰/散布成因**（0.0014–0.470 ms/MiB） | `logs/38` §3.3 把它列为未解 | **【已补】→ 并且更正了旧结论** → `logs/40` §2.2：**否证 `logs/29` 的"65× 是缓存冷热"** —— 两片 mincore 都 1.00，单独注册仍复现 71×，快/慢是**该文件范围内页的固有属性**；页缓存最多解释 7–22×（冷/热 7.1×、真从盘读 22.4×）。根因仍【未确认】（候选 THP 覆盖 / NUMA 落点，无驱动源码无法直验） |
| **G22** | n > 4096 / 多于 40 层、**跨 CANN·驱动版本稳定性** | `logs/35` §5 挂着的两条 | **【已补（仅 n>4096）】** → `logs/43`：n=8192 图内 **12.77 vs 上游 13.26 ms（0.96×）**、显存 1440 vs 3520 MB；**超合同探测，不作生产口径**。跨 CANN/驱动版本**本机做不了** ⇒ 归入 §5 |

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
