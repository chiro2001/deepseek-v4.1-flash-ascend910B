# DeepSeek-V4.1-Flash W4A8 优化版推理服务

在 **Ascend 910B（A2，8×910B3）** 与 **Ascend 910C（A3，8×910C）** 上拉起
DeepSeek-V4.1-Flash 的 W4A8 量化推理服务，含完整补丁、一键起服、自检与验收。

**本包的全部优化工作（算子分析、补丁编写、性能调优、文档）均由 `deepseek-v4.1-flash`
模型自主完成**，未经人工逐行改写。

---

## 1. 这个包解决什么问题

官方镜像里的 vllm-ascend 能跑起 DeepSeek-V4.1-Flash，但在长上下文与高并发下有几处
明确的性能瓶颈。本包提供 11 个补丁（8 个性能 + 1 个调度 + 2 个量化适配）与配套脚本，
把这些瓶颈逐个消掉，并给出**可复现的验收口径**。

约束（交付形态必须成立）：

* Engram-int8 常驻 DRAM
* DSpark 投机解码开启（5 tokens）
* KV cache 在 HBM 且 **> 3M tokens**（实测详见 §2.5 的取舍说明）
* Vision 23/23、GSM8K ≈198/200
* 静态内核不得静默降级（`static_kernel.py:650` 命中数必须为 0）

## 2. 一键起服

### 2.1 A3（8×910C）

```bash
# ① 先看哪些卡空着（只读，打印每张卡的占用与进程属主）
bash tools/list_chips.sh

# ② 指定要用的 8 张卡（DEVS 必填，脚本不替你选）
DEVS="8 9 10 11 12 13 14 15" \
  MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq \
  bash scripts/serve_a3.sh
```

`DEVS` 是**用户输入**：一台机器上哪 8 张能用取决于当前谁在跑什么，脚本无法替你判断。
默认会**拒绝已被占用的卡**并打印占用进程（确实要带占用起服务才加 `ALLOW_BUSY=1`）。

### 2.2 A2（8×910B3）

```bash
bash scripts/build_image.sh                 # 烘焙补丁，产出 dsv41-a2:v6（约 10–20 min）
MODEL=/path/to/v41-w4a8-engram-dr-vision-qrot-mtpq bash scripts/serve_a2.sh
```

### 2.3 起服前/后

```bash
# 起服前（30 秒，不起容器）：9 组自检
MODEL=/path/to/model bash tools/preflight_a2.sh

# 干跑：只打印将执行的命令，不碰 docker
DRY_RUN=1 ... bash scripts/serve_a3.sh

# 服务已在跑时：附着自检（不起容器、不删容器）
PORT=8020 bash tools/attach_test.sh

# 完整验收：8K/32K/128K + Vision + GSM8K
MODEL=/path/to/model MODE=full bash scripts/run_test.sh
```

### 2.4 两个默认值

| 开关 | 默认 | 说明 |
|---|---|---|
| `PREFIX` | **1（开）** | prefix caching。面向用户的默认是生产形态。要测**无前缀缓存**的性能口径用 `NO_PREFIX=1`（等价 `PREFIX=0`）。两种口径的 ms/step 与接受长度**不可直接混比**，报数字时必须写明。 |
| `MAX_SEQS` | **32** | 并发上限。它会决定 CUDA graph 的捕获桶数，越大启动时捕获越久（首次多 1–2 min）。 |
| `BAT_TOKENS` | **8192** | `--max-num-batched-tokens`。**正确性关键参数**，见 §2.5 —— 不要随手调小。 |

### 2.5 ★ `BAT_TOKENS`：长上下文正确率的开关

chunked prefill 把长 prompt 切成 `ceil(prompt / BAT_TOKENS)` 段依次前向。
**每段都有一次独立的"偏离"机会，且误差沿后续 chunk 累积** ——
实测偏离率约 **2%/chunk**，因此 **chunk 数越多、长上下文正确率越低**，
而且是**平滑下滑**（不是"超过某个长度就崩"）。

用"把唯一事实埋在长文档中段、只问一个答案唯一的问题"做探针，
每档 10 个不同样本（内容不同）：

| prompt_tokens | `BAT=2048`（10 chunk/20K tok） | `BAT=8192` |
|---:|---:|---:|
| 10,394 | 10/10 | **10/10** |
| 20,318 | 8/10 | **10/10** |
| 40,163 | 7/10 | **10/10** |
| 60,012 | 5/10 | **10/10** |
| 79,855 | 3/10 | **10/10** |
| 149,986 | ~0% | **6/6** |
| 259,985 | ~0% | **6/6** |

⇒ **默认 `BAT_TOKENS=8192` 把这七档全部拉到 100%**，同一 prompt 重复 10 次
输出逐字节一致。

**代价**：activation 峰值从 0.79 涨到 **3.21 GiB**，KV cache 从
**4,145,957 → 2,823,080 tokens**（8×910C、默认 `GPU_UTIL=0.92` 实测；
若用 0.94 则是 3,088,412，但那样 prefill 会慢 6~7×，见下）。
若你的场景更看重 KV 容量、且上下文主要在 <20K，可以显式 `BAT_TOKENS=2048`。

> ⚠️ **`GPU_UTIL` 是本包最重要的性能开关** —— 0.94 会让长 prompt 的 prefill 慢 6~7×：
>
> | `GPU_UTIL` | 设备余量 | 8K prompt 首 token | KV tokens |
> |---:|---:|---:|---:|
> | **0.92（默认）** | **7.36 GiB** | **1.14 s** | **2,823,080** |
> | 0.94 | 6.11 GiB | **8.0 s** | 3,088,412 |
> | 0.88 | 9.80 GiB | 1.28 s | 更少 |
>
> 原因是**真实请求的 activation 峰值远高于启动 profiling 报告的值**：
> profiling 在 KV cache 分配**之前**量到 3.21 GiB，而真实 8K prefill 需要
> **约 6 GiB**（差额 2.8 GiB）。余量不足时分配器要反复向驱动申请/归还，
> `forward` 慢 2.1×，新请求还要多等 ~6 s 把池子撑大。
> **详细机制与全部证据见 [`docs/prefill-memory-headroom.md`](docs/prefill-memory-headroom.md)。**
>
> ⚠️ 也**不要往 0.95 及以上调**：实测首个真实 prefill 就 OOM；
> `BAT=6144` + `GPU_UTIL=0.94` 会 ACL graph 重放 OOM。
> 若必须满足更高的 KV 门槛，应评估裁剪 `CAPTURE_SIZES`
> （代价：高并发 decode 退回 eager），而不是调 `GPU_UTIL`。

> **自查方法**：本包提供 `tests/agent_trace/longctx_retrieval.py`，
> 用 `--tokens 60000 --reps 10` 跑一次，正常应 10/10。
>
> 详细分析（含机制、消融、方法论教训）：[`reports/longctx-accuracy-fix.md`](reports/longctx-accuracy-fix.md)

### 2.6 绑核

**外部不计算绑核位置**：容器默认不设 `--cpuset-cpus/--cpuset-mems`，
由 vllm-ascend 内部的 `cpu_binding` 按 NPU 拓扑给每个 rank 自己绑
（`--additional-config` 的 `"enable_cpu_binding": true`）。起服日志里能看到它的决策：

```
[cpu_binding.py] [cpu_bind_mode] mode=topo_affinity rank=0 visible_npus=[...]
[cpu_binding.py] NPU8: main=[322..357] acl=[358] release=[[359]]
[cpu_binding.py] [migrate] NPU:15 -> NUMA [7]
```

> 为什么不在外部绑：外部先圈定 NUMA 等于替内部做了决定，一旦选卡组合与外部区间对不上
> （多机、多租户、混合选卡），内部再绑也回不到正确节点。位置交给知道拓扑的那一方。

`CPUSET=`/`MEMS=` 保留为逃生口，显式给了才会透传给 docker。

## 3. 性能数据

### 3.1 单流延迟（128K 上下文）

**A3（8×910C）实测**，128K 上下文、单流独占：

| 指标 | 值 |
|---|---|
| ms/step | 中位 **30.2 – 31.6**，最好 26.9 |
| 单流 tok/s | 中位 **85 – 90** |
| KV 池 | **3,088,412 tokens**（`BAT_TOKENS=8192`，见 §2.5） |

> 语料为《红楼梦》全本按目标 token 数精确截取 + 指令后缀（quote 口径）。
> 口径：**unprofiled 客户端墙钟**，单流独占。
>
> ⚠️ 不要用接受长度当绩效指标 —— 它与文本质量**反相关**（脏会话里反而更高）。
> 报数字请用 `(clean-rate, ms/step)` 或上面这种 `(ms/step, tok/s)` 组合。

### 3.2 并发吞吐

![concurrency](docs/img/conc_dihuo_en.png)

**A3（8×910C，TP8+EP8）实测**，`MAX_SEQS=64` + `PREFIX=1`（生产口径），
prompt **每条精确 1024 token** / 输出 256 token（`ignore_eos` 强制生成满），
**每档都跑完同一批 64 条请求**，2 次取中位数：

| 并发 | 单流吞吐 (tok/s) | 总吞吐 (tok/s) | 加速比 | 单流效率 | 接受长度 | TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **96.3** | **94.8** | 1.00× | 100.0% | 2.79 | 0.27 s |
| 2 | **88.7** | **143.5** | 1.51× | 92.1% | 2.73 | 0.51 s |
| 4 | **71.6** | **211.5** | 2.23× | 74.3% | 2.68 | 0.77 s |
| 8 | **54.7** | **294.7** | 3.11× | 56.7% | 2.78 | 1.30 s |
| 16 | **41.0** | **405.2** | 4.27× | 42.5% | 2.69 | 2.55 s |
| 32 | **26.5** | **510.5** | 5.39× | 27.5% | 2.61 | 4.69 s |
| 64 | **17.4** | **595.8** | 6.29× | 18.0% | 2.63 | 9.09 s |

* **单流吞吐** = 每个请求自身的 decode 速率，取所有请求的中位数。
  它回答"我自己发一条能有多快"。
* **总吞吐** = 全部请求输出 token 之和 ÷ decode 窗口墙钟。它回答"服务整体每秒吐多少 token"。
* **加速比** = 该并发总吞吐 ÷ 并发 1 总吞吐；**单流效率** = 该并发单流吞吐 ÷ 并发 1 单流吞吐。

**怎么读这张表**：总吞吐在 64 并发时达到 **595.8 tok/s**（单流的 6.29×），但单流已被压到
17.4 tok/s（效率 18.0%）。交互式场景（要低延迟）**并发 2–4 是甜点区**（单流仍有 72–89 tok/s）；
离线批量（要总吞吐）拉到 32–64。

口径说明：

* 语料为**现代中文白话小说**，每个请求取**不同位置的正文切片** + **轮换 5 个不同任务**
  （摘录/问答/续写/词条抽取/复述），每条 prompt 用服务端 `/tokenize` **精确校准到 1024 token**
  （64/64 命中，偏差 0）；切片起点自适应，**完全互不重叠**。
* 每个请求开头带唯一 id ⇒ 绕开 prefix cache，测的是**冷 prefill**。
* 只计 **decode 阶段**（首 token → 末 token），不含 prefill；TTFT 单独列出。
* **接受长度全程 2.56–2.86**（正常范围）。这一点必须看：接受长度异常高（> 3.5）通常意味着
  模型掉进复读循环，吞吐会被**虚高到 1.5×以上**。详见
  [`docs/BENCH-METHODOLOGY.md`](docs/BENCH-METHODOLOGY.md)。
* 客户端与服务同机（`127.0.0.1`），网络开销可忽略；宿主内存 2 TB，全程无换页。

复现命令：

```bash
python3 tools/bench_concurrency.py \
  --base-url http://127.0.0.1:8020 --model deepseek-v41 \
  --concurrency 1,2,4,8,16,32,64 \
  --prompt-tokens 1024 --output-tokens 256 --repeats 3 \
  --corpus-file data/dihuo.txt --suffix-dir data/dihuo_local

# 出图（中英双版 + CSV）
python3 tools/plot_concurrency.py results/bench/*.json -o docs/img --prefix conc
```

> ⚠️ `MAX_SEQS` 决定 CUDA graph 的捕获桶：取 64 会多捕获 192/384 两个大形状，
> **首次起服需冷编译静态内核，engine init 约 9–10 分钟**（vs `MAX_SEQS=4` 的约 4 分钟）。
> 编译完成后缓存复用，后续起服回到分钟级。

### 3.3 长 prompt 的首 token 延迟（prefill）

上面两节量的是 **decode**。长 prompt 的首 token 延迟（≈ prefill 时间）单独列在这里，
因为**它由 `GPU_UTIL` 主导**（见 §2.5），而不是由 decode 的优化决定。

**A3（8×910C，TP8+EP8）实测**，**发布默认配置**（`STATIC_KERNEL=1` + `PREFIX=1` +
`GPU_UTIL=0.92`），单请求、真实语料切片（各请求 token 区间互不重叠，
服务端 prefix cache 命中率全程 0.0%）：

| prompt tokens | chunk 数 | 旧默认 `GPU_UTIL=0.94` | **默认 `GPU_UTIL=0.92`** | 改善 |
|---:|---:|---:|---:|---:|
| 8 192 | 1 | 8.0 – 8.6 s | **1.14 / 1.16 / 1.17 s** | 7.0× |
| 32 768 | 4 | 29.1 s | **4.22 s** | 6.9× |
| 131 072 | 16 | 102.3 s | **18.17 s** | 5.6× |

换算：**prefill 吞吐 ≈ 7.2 K token/s，且与序列长度无关** ——
每 8192-token chunk 的成本恒定在 **1.03–1.18 s**。

> 为什么恒定：稀疏注意力的 `index_topk=512` 是固定的，每个 chunk 的注意力代价
> 不随既有上下文长度增长。所以"长 prompt 慢"来自 chunk 数，不来自单 chunk 变慢。

**口径**：单请求独占、客户端与服务同机（`127.0.0.1`）、首 token 墙钟；
prompt 用目标模型的 tokenizer 精确切到目标 token 数（不按字符截）；
关前缀缓存以保证每个 prompt 都真跑 prefill。并发场景下 TTFT 见 §3.2 的 TTFT 列。

> ⚠️ §3.2 的 TTFT 看起来小得多（1024 token 只要 0.27 s），因为那里 prompt 只有 1K ——
> prefill 的 activation 与耗时都随 prompt 长度增长，**1K 的 prompt 碰不到显存余量问题**。
> 这也是此前没发现 `GPU_UTIL=0.94` 问题的原因。

### 3.4 精度

| 项目 | 结果 |
|---|---|
| Vision | **23/23**（BAT=8192 下复测） |
| GSM8K-200 | **199/200**（BAT=8192 下复测）；历史三次 198 / 199 / 197 |
| 静态内核降级 | **0 次** |
| 长上下文检索（8K/32K/128K） | **10/10、10/10、10/10** |
| 真实 agent 轨迹（27 工具、多轮） | **10/10、10/10** |

完整验收记录（含前后对比与官方 API 对照）：[`reports/longctx-verification.md`](reports/longctx-verification.md)

## 3.5 起服后必查（5 项）

服务 READY 后**必须**确认这四项，否则后面的数字都不可信：

```bash
R=results/<run_id>              # 起服脚本会打印这个目录

# ① 静态内核没有被静默降级 —— 必须输出 0
grep -ac "static_kernel.py:650" $R/serve.log

# ② KV 容量 —— 默认 GPU_UTIL=0.92 时实测 2,823,080 tokens
#     （0.94 时 3.09M 但 prefill 慢 6~7×；两者取舍见 §2.5 与
#      docs/prefill-memory-headroom.md）
grep -oE "GPU KV cache size: [0-9,]+ tokens" $R/serve.log | tail -1

# ②b 长 prompt 的首 token 延迟 —— 8K prompt 应 ~1.1 s（0.94 时会是 8 s）
#     用 §3.3 的口径实测一次；这是判断显存余量是否足够的直接指标

# ③ 口径对不对（性能口径 vs 生产口径不可混比）
grep -E "MAX_SEQS|PREFIX" $R/serve_cmd.txt

# ④ Engram local-owner 是否切到 fast 路径
grep -E "VALIDATE OK|local-owner" $R/serve.log | tail -2
```

一条命令替代：

```bash
PORT=8020 NAME=<容器名> SLOG=$R/serve.log bash tools/attach_test.sh
```

`tools/attach_test.sh` 是**附着式**的：服务已在跑时用它，不会起容器也不会删容器。

## 4. 补丁清单

### 4.1 做了哪些优化（按瓶颈分类）

优化集中在四类瓶颈。每项都**独立门控**、可单独关掉做 A/B；下面的收益是同会话配对实测值。

**① 通信 / 调度：让通信不再白占时间**

| 优化 | 做法 | 收益 |
|---|---|---|
| **MoE 走 AllGather** | TP=EP 时改用 AllGather 路径，把标量开销按 token 数摊薄 | 128K **−4.25 ms**、32K −1.35、8K −1.23；**KV 池 3.39M→4.16M** |
| **admission gate** | 调度器里给 prefill 加门控，不再让长 prefill 饿死 decode | 首 token 后不再长时间不出字（`[admission_gate]` 日志可验证） |

**② Engram（V4.1 的记忆模块）：把 host 侧开销压到最低**

| 优化 | 做法 | 收益 |
|---|---|---|
| **INT8 表 host 常驻 + local-owner** | 表放 DRAM，走 local-owner 快路径，省一次 metadata all_gather 与 ids all_to_all | 释放 HBM 给 KV；route 步明显变短 |
| **hash / plan 的 numba JIT** | host 侧两条热路径改 JIT（带磁盘缓存） | hash 0.427→**0.076 ms**、plan 0.261→**0.068 ms** |
| **gate 分块** | 按 chunk 计算，去掉固定 2048 行 padding | 8K **−1.56 ms**，KV 反而更省 |

**③ 算子 / 访存：消掉列式访问与小 kernel**

| 优化 | 做法 | 收益 |
|---|---|---|
| **QLI 无候选快速路径** | 无候选时短路掉去重/排序链 | 单算子 99.3→50.3 µs ⇒ **−0.49 ms** |
| **rope 取表融合** | cos/sin 取表链 6 kernel → 2 | **−0.45 ~ 0.62 ms** |
| **`wo_a` 2D matmul** | 退化的 batch matmul 改 2D（并缓存转置，避免每次重转） | **−0.31 ~ 0.76 ms** |
| **expert mask 范围比较** | `expert_map[topk_ids] != -1` 换成区间比较，省掉 Index + IndexCheck 两个大 kernel（掩码本身保留，否则 unpermute 会读未写入的行） | **−0.51 ms** ／ GSM8K 100/100、Vision 23/23 |

**④ 量化适配（仅在重新量化时需要）**

msmodelslim 侧的 V4.1 W4A8 支持，含 hiaux 变体配方。

> 所有优化**默认关闭**，由起服脚本显式打开 —— 这样任何一项出问题都能单独关掉定位。
>
> **累计效果**（128K 单流、同口径真权重、8 发中位数）：
>
> | 阶段 | ms/step | 来源 |
> |---|---:|---|
> | 优化前基线 | 39.10 | `reports/optimization-headroom-estimate.md` |
> | + MoE AllGather | 35.14 | `reports/moe-allgather-breakthrough.md` |
> | + 其余 7 个补丁 | 32.74 | `reports/consolidated-6patch-result.md` |
> | **+ Engram JIT 等（全补丁）** | **31.39** | `reports/milestone-ms-target-met.md` |
>
> 合计 **−7.71 ms/step（−19.7%）**；多轮实测中位区间 **30.2–31.6 ms/step**，最好 26.9。

### 4.2 补丁明细

11 个补丁，全部**默认关闭**（由起服脚本显式打开），可单独摘出来做 A/B。

| # | 补丁 | 门控 env | 实测收益 |
|---|---|---|---|
| 1 | MoE dispatch/combine 走 AllGather | `V41_MOE_COMM_ALLGATHER=1` | 128K **−4.25 ms**、32K −1.35、8K −1.23；KV 3.39M→**4.16M** |
| 2 | expert mask 范围比较 | `V41_MOE_MASK_RANGE=1` | **−0.51 ms** |
| 3 | rope cos/sin 取表融合 | `V41_ROPE_IDXSEL=1` | **−0.45 ~ 0.62 ms** |
| 4 | QLI 无候选快速路径 | `V41_QLI_NO_CANDIDATE=1` | 99.3→50.3 µs ⇒ **−0.49 ms** |
| 5 | `wo_a` 2D matmul | `V41_O_PROJ_2D=1` | **−0.31 ~ 0.76 ms** |
| 6 | Engram gate 分块 | `V41_ENGRAM_GATE_CHUNK=<int>` | 8K **−1.56 ms** |
| 7 | Engram host 常驻 + local-owner | `V41_ENGRAM_HOST_RESIDENT=1` | 省一次 metadata all_gather + ids all_to_all |
| 8 | hash/plan numba JIT | `V41_ENGRAM_JIT=1` | hash 0.427→**0.076 ms**；plan 0.261→**0.068 ms** |
| 9 | admission gate（vllm core） | `VLLM_ADMISSION_GATE=1` | prefill 不再饿死 decode |
| 10-11 | msmodelslim 量化适配 | — | 仅在**重新量化**时需要 |

补丁同时提供两种形态，内容经机器验证**逐字节等价**：

| 形态 | 路径 | 用途 |
|---|---|---|
| git 历史系列 | `patches/{vllm-ascend,vllm,msmodelslim}/` | `git am` 可复现、评审、向上游提交 |
| 逐字节整文件 | `patches/files/` | 烘焙进镜像 / bind-mount |

基线 commit、`git am` 方法与依赖顺序见 [`patches/README.md`](patches/README.md)。

## 5. 已知限制

| 项 | 状态 |
|---|---|
| 110 tok/s | **不可交付**：设备 busy 本身 30.9 ms > 达标所需的 25.1 ms |
| 接受长度 A | **不能当绩效指标**；必须报 `(clean-rate, ms/step)` |
| `DRAFT_GRAPH=1` | 未采纳：缺 `DSPARK_GRAPH_CAPTURE_METADATA=1` 时会静默失效（A 恒 1.0 但 ms 看着正常） |
| `V41_MOE_ZERO_INVALID` / `MOE_NF` | 实验项/负结果，默认关 |
| 128K 以上长文 | **已定位并修复**：chunked prefill 的 chunk 数决定偏离率（~2%/chunk）。默认 `BAT_TOKENS=8192` 后 260K token 档实测 6/6。见 §2.5 |
| `BAT_TOKENS` 的 KV 代价 | 提到 8192 会让 KV cache 从 4.15M 降到 **2,823,080** tokens（activation 峰值 0.79→3.21 GiB，且默认 `GPU_UTIL` 为 0.92）。若改用 0.94 则是 3,088,412，但 prefill 会慢 6~7× —— 取舍见 §2.5 |
| **KV 门槛 3Mi（3,145,728）** | **默认配置不再满足**：0.92 下 2,823,080（< 3M），0.94 下 3,088,412（> 3M 但 < 3Mi）。**这是用 KV 容量换 prefill 速度的主动取舍**；需要 3Mi 的场景应显式 `GPU_UTIL=0.94` 并接受 8 s 级首 token 延迟，或评估裁剪 `CAPTURE_SIZES` |
| A2 与 A3 的性能差 | 硬件（含 HBM 带宽，两边同为 1600 GB/s/die）只能解释 ~15%，其余在 host 侧 |

细节与原始数据见 [`EXPECTED_PERF.md`](EXPECTED_PERF.md)、[`CORRECTNESS_STATUS.md`](CORRECTNESS_STATUS.md)、
[`reports/`](reports/)。

## 6. 目录结构

```
README.md  REPRO.md  LICENSE  NOTICE
├── patches/       补丁系列（两种形态）+ 基线 commit 说明
├── scripts/       起服（A2/A3）、镜像构建、验收（run_test / attach_test）
├── tools/         选卡、模型目录自检、附着测试、**并发压测**、**出图**、负控
│   ├── bench_concurrency.py   ← 并发扫描（单流 + 总吞吐 + 接受长度）
│   └── plot_concurrency.py    ← 出图（中英双版 + CSV）
├── data/          测试语料
│   ├── hongloumeng.txt        ← 全本（128K 单流口径）
│   ├── dihuo.txt              ← 现代白话小说（并发口径）
│   └── */_local/              ← 各语料的短切片专用问题
├── build_scripts/ CPython PGO 目标机编译（产物不入库）
├── quant/         W4A8 / Engram-int8 / DSpark 量化复现
├── tests/         单流 / 视觉 / GSM8K / 多 batch 验收
│   └── agent_trace/           ← ★ agent 形态精度门
│       ├── longctx_retrieval.py  长上下文检索探针（60K token 即可测出退化）
│       └── accuracy_gate.py      工具调用门（5 长度档 + Wilson CI）
├── reports/       实验记录（每项结论的原始依据）
│   ├── longctx-accuracy-fix.md   ← ★ BAT_TOKENS 修复的完整分析
│   └── probe/                    稀疏状态插针 + 设计文档（PROBE=1 启用）
├── docs/
│   ├── BENCH-METHODOLOGY.md   ← ★ 为什么"单流 tok/s"必须配上下文看
│   ├── img/                   ← 曲线图 + CSV
│   ├── A3-SELFTEST-20260917.md
│   ├── RELEASE-NOTES.md
│   └── LEGACY-DELIVERY-NOTES.md  ← 早期 A2 交付流程（仍有参考价值）
└── optim/pgo/     PGO 构建入口（产物在目标机生成，不随包分发）
```

## 7. 许可与来源

本包以 **Apache License 2.0** 发布（见 [`LICENSE`](LICENSE)），派生自以下项目（均为 Apache-2.0）：

| 项目 | 用途 | 基线 commit |
|---|---|---|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | `patches/vllm/` | `6e448d0e` |
| [GDzhu01/vllm-ascend-v41-private](https://github.com/GDzhu01/vllm-ascend-v41-private) | `patches/vllm-ascend/` | `46856f89e` |
| [Ascend/msmodelslim](https://gitcode.com/Ascend/msmodelslim) | `patches/msmodelslim/` | `92e219fa` |

> vllm-ascend 为什么不用 `vllm-project/vllm-ascend`：V4.1 的模型代码
> （`models/deepseek_v41/`、Engram、DSpark）不在官方仓，详见 [`patches/README.md`](patches/README.md)。

**本包不含**：模型权重、tokenizer 文件、容器镜像、任何预编译二进制
（PGO 产物由 `build_scripts/00_ensure_pgo.sh` 在目标机编译生成）。
详见 [`NOTICE`](NOTICE)。

### 测试语料说明

| 语料 | 用途 | 在仓库里？ |
|---|---|---|
| 《红楼梦》（`data/hongloumeng.txt`） | 128K 单流延迟口径 | ✅ 有（公共领域） |
| 现代中文白话小说（`data/dihuo.txt`） | 并发吞吐口径 | ❌ **需自行下载** |

并发吞吐用的那份是当代文学作品，**不属于本项目的 Apache-2.0 许可范围**，
所以仓库只提供来源链接与校验和，不附带原文：

```bash
bash tools/fetch_corpus.sh          # 下载 → 校验 sha256 → 清洗 → 再校验
bash tools/fetch_corpus.sh --check  # 只校验
```

来源与两个阶段的哈希见 [`data/dihuo_local/README.md`](data/dihuo_local/README.md)。
脚本任一步哈希不符就**拒绝继续**，这样发布出去的性能数字可以逐字节复现。

> 想换别的文本也行：`bench_concurrency.py` 只要求"一段足够长的现代中文散文"，
> 加 `--corpus-file` / `--suffix-dir` 即可。接受长度会随文本而变
> （见 [`docs/BENCH-METHODOLOGY.md`](docs/BENCH-METHODOLOGY.md)）。
