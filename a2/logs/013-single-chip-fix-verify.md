# 013 · 单卡 tiny 验证 D2 的卸载修复 + 量化「池子要多大才够」

**日期**：2026-09-22 00:19 – 00:59（A3 本地时钟；共 **22 次运行 = 20 条判据臂 + 2 次起服失败**）
　**执行**：子代理 `V1_verify`
**机器**：A3（A3-node1），**单张卡**（槽位 `c1` = Phy-ID 6，容器 `prbench-c1`，TP1）
**上游/镜像**：与 `010`（L1_dummy）、`009`（D2_offload）同一套：vLLM `0.27.1` +
vllm-ascend `e43cf1e9f`（共享容器 `prbench-c1`，**未改镜像内任何源码**）
**标记约定**：【实测】= 本机跑出来的原始数据；【推断】= 代码/日志推出来但没直接测；
【未确认】= 没跑到。

---

## 0. 一句话结论

| 问题 | 结论 |
|---|---|
| **Q1：D2 的修复在单卡 tiny 上成立吗？** | **【实测·成立】** 四条判据全中（臂 `v1-d2-4g`）：`BlockStored(CPU)=704`、`CPU_to_GPU=273 MB`、`external_prefix_cache_hits=65,520`、**replay TTFT 50.9 ms vs fill 465.9 ms = 9.2×**。⇒ **"2 min/臂"的快迭代通道对"修复类"改动同样成立**。 |
| **快迭代通道的实测代价** | **【实测】单臂 wall（锁内 start→end）中位 1.22 min、均值 1.48 min、范围 1.02–2.53 min**（n=21 次计时）：16 请求的臂 **1.0–1.3 min**，96 请求的臂 1.9–2.5 min（起服 ≈75 s 是固定成本）。 |
| **同一条修复，池子不配平时呢？** | **【实测】仍然 0 取回**（臂 `v1-n16-640m`：`CPU_to_GPU=0`、`hits=0`）⇒ 修复必要但**不充分**，必须同时把池子配到 ≥ 一轮工作集。 |
| **对照：只挂 L1 的 unwrap 补丁（不打 D2 的参与位修复），池子放到 4 GiB？** | **【实测】`hits=0`、`CPU_to_GPU=0`、replay 451.9 vs fill 460.4 ms**（臂 `v1-l1ct-4g`）⇒ **决定性的那一项是 D2 的修复（`state` 组不参与存/查），不是池子大小**。 |
| **Q2：池子要多大才够？** | **【实测】`num_cpu_blocks ≥ 一轮待取回的 (group, chunk) 条目数`**，在三个尺度上都是硬边界：704（16 请求）、1408（32 请求）、4224（96 请求）块各对应 1.000× 通过、0.91–0.97× 归零。 |
| **★ 池子略小于工作集：断崖还是部分命中？** | **【实测】同序重放时是断崖**（0.909× → 0/16 命中），但**不是"数据不在池子里"**：同一个 640 MiB 池，把 replay 轮改成**倒序**，命中立刻变成 **14/16**（`hits=57,330`）⇒ 归零是 **LRU 从头部淘汰 + `_maximal_prefix_lookup` 要求从第 0 块连续命中**共同造成的**级联**。 |

---

## 1. 环境与做法（一条臂 ≈ 2.5 min）

| 项 | 值 |
|---|---|
| 卡 | **c1（Phy-ID 6）**，全程走 `tools/a3_chip.sh c1` 锁；**没碰** c0/c2、Phy-ID 8–15、`dsv41-a3`、`mooncake-master` |
| 模型 | `agents/L1_dummy/models/model-tiny`（6.4 MB，dummy 权重，**保留 40 层 KV 结构**） |
| 起服 | TP1、`--load-format dummy`、`--block-size 128`、`--enable-prefix-caching`、`--prefix-match-unit 32`、`--kv-cache-memory-bytes 1 GiB`、`ENGRAM=0` |
| 用法 | 只改**池子**（`OFFLOAD_GB` / 精确的 `OFFLOAD_BYTES`）与**请求数 / 长度**；其余参数逐字沿用 `010` |
| 每臂流程 | 起服 ≈80 s → `/health` → 起 ZMQ KV 事件探针 → fill 轮 → `POST /reset_prefix_cache` → replay 轮 → 收 `/metrics` → 停服 |
| 单臂产物 | `out/<tag>.{server.log,client.json,kv_events.log,metrics_before.txt,metrics_after.txt,meta.txt,server_args.txt}` |

### 1.1 D2 的补丁是怎么挂上的（**整文件替换**，不是重写一遍）

D2 的修复是 7 处改动（`009` §4），逐处复刻容易漏，所以直接挂 **D2 在 8 卡上验证通过的那一份文件**：

* 文件：`agents/D2_offload/patches/offload_dsv41/scheduler.py`，**md5 `0302fab4c68c3adc7d2c4a135c7c4289`**（A3 上的副本 `agents/V1_verify/patch/d2_scheduler.py`，md5 一致）；
* 机制：`agents/V1_verify/patch/sitecustomize.py` 用 `sys.meta_path` 钩子，在**第一次 import**
  `vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler` 时，用 D2 的文件作为模块源
  （等价于 `serve_a2.sh` 的 `OFFLOAD_SCHED_PATCH=1` 挂载，但只影响我自己起的进程、**不写镜像**）；
* **开/关可验证**：`V1_D2_PATCH=1` 时 `module.__file__ = /work/agents/V1_verify/patch/d2_scheduler.py` 且
  `_offload_participates` 存在；`V1_D2_PATCH=0` 时是镜像内的原文件、无该符号（探针见 §6）。
* 顺带保留 `010` 的 `wo_a` dummy 适配（`--load-format dummy` 不调用 `weight_loader`，必须补），
  以及一段**只读**日志：把 CPU 卸载池的真实块数打出来（`[V1_verify] CPU 卸载池: num_blocks=…`）。

### 1.2 池子的真实块数（**第一手**，不再是公式推算）

```
[V1_verify] CPU 卸载池: num_blocks=4096 kv_bytes_per_chunk=1048576
   cpu_page_size_per_worker=1048576 replicated_layout=False blocks_per_chunk=8
   cpu_bytes_to_use=4294967296 worker_kv_bytes_per_block=131072 world_size=1
```

⇒ 本机 tiny/TP1 配置下：**1 个 chunk 条目 = 1 MiB 池**（`131072 B × 8 blocks/chunk × 1 副本`），
即 **1 GiB 池 = 1024 个条目**（与 `cpu_bytes_to_use // round_up(kv_bytes_per_chunk, ALIGN)` 逐字吻合）。
（⚠️ 与 8 卡不同：那里 `num_copies = world_size = 8` ⇒ 一格 34.6 MB。）

### 1.3 臂清单

| 臂 | 池 | 请求数 × 长度 | D2 补丁 | 目的 |
|---|---|---|---|---|
| `v1-d2-4g` | 4 GiB（4096 块） | 16 × 4096 | ✅ | **Q1 主臂** |
| `v1-l1ct-4g` | 4 GiB | 16 × 4096 | ❌（只挂 L1 的 unwrap 补丁） | **Q1 对照**：没有 D2 的修复时，池子再大也没用 |
| `v1-n16-640m` | 640 MiB（640 块） | 16 × 4096 | ✅ | Q2：0.909× 工作集 |
| `v1-n16-672m/673m/674m/688m/703m/704m/768m` | 672–768 MiB | 16 × 4096 | ✅ | Q2：把拐点夹到 1 块之内 |
| `v1-n32-1408m` | 1408 MiB（1408 块） | 32 × 4096 | ✅ | Q2：另一尺度的 1.000× |
| `v1-n96-{1g,2g,3840m,4g}` | 1024/2048/3840/4096 块 | 96 × 4096 | ✅ | Q2：**1/2/4/8 GiB 扫描**（工作集 4224 条，拐点落在扫描区间内） |
| `v1-n96-{4224m,8g}` | 4224/8192 块 | 96 × 4096 | ✅ | Q2：1.000× 与 2× |
| `v1-n16-640m-rev` | 640 MiB | 16 × 4096，**replay 倒序** | ✅ | ★ 机制判别：断崖是"数据不在"还是"LRU 级联" |
| `v1-nod2-4g` | 4 GiB | 16 × 4096 | ❌ **完全不挂补丁** | 撞 `assert isinstance(kv_cache_spec, FullAttentionSpec)`（复现 `010` §3.2），**不算判据臂** |

**口径说明（不许用相邻数字顶替缺的那格）**：`replay p50` 一律指 `client.json` 里
`rounds[tag=replay1].ttft.p50_ms`；`hits/queries/CPU_to_GPU` 一律取**两轮跑完后的
`metrics_after.txt`**（fill 轮全冷启动、不可能有外部命中，`010` 已实测 fill 轮 hits=0）。
每臂有 **1 个** prompt 的 TTFT 取不到（`max_tokens=1` 时该请求没有可计时的首 token，两轮同样缺，
所以是 n=15/16 或 n=95/96，`requests_failed=0`）—— 这一格**照实标 `—`**，没有拿相邻数字顶替。

---

## 2. Q1：D2 的修复在单卡 tiny 上**成立**【实测】

四条判据（口径与 `001`/`009`/`010` 一致，臂 `v1-d2-4g`：4 GiB 池、16 × 4096 token、HBM KV 1 GiB）：

| # | 判据 | 8 卡（`009` 臂 `d2-dram32-2p`） | **单卡 tiny（`v1-d2-4g`）** | 通过？ |
|---|---|---|---|---|
| ① | `BlockStored(medium="CPU") > 0` | 768 | **704**（= 16 请求 × 44 条目） | ✅ |
| ② | `kv_offload_total_bytes{CPU_to_GPU} > 0` | 1.56 GB | **272,957,440 B ≈ 273 MB**（16 个 load job） | ✅ |
| ③ | `external_prefix_cache_hits > 0` | 63,488 | **65,520**（queries 131,328；**replay 轮 99.8%**） | ✅ |
| ④ | replay TTFT ≪ fill TTFT | 298.2 vs 4424.6 ms（14.8×） | **50.9 vs 465.9 ms（9.2×）** | ✅ |

**独立于计数器的取回证据**（D2 补丁自带的 load job 日志）：

```
[D2_offload] load job req=cmpl-… keys=14 group_sizes=[32, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] src_blocks=14 dst_blocks=42
```

* `group_sizes[1] = 0` ⇒ **`state` 组不参与取回**（`009` §5.1 的语义在单卡上逐字复现）；
* `group_sizes[0] = 32`（4 个 chunk × 8 block）+ 10 个 SWA 组各 1 ⇒ `keys=14`；
* 16 个请求各一条 load job ⇒ **真的是从 DRAM 搬回来的**，不是"少算了几块 token"。

### 2.1 对照臂：没有 D2 的参与位修复，4 GiB 池也是 0 取回【实测】

`v1-l1ct-4g`（挂 `010` 的 3 处 unwrap 补丁、**不挂** D2 的参与位修复；池子 4 GiB = 4096 块 =
工作集的 **5.8×**）：

| 观测 | 值 |
|---|---|
| `BlockStored(CPU)` / `BlockRemoved(CPU)` | 704 / 0（**存得下、一块没淘汰**） |
| `CPU_to_GPU` | **0.0** |
| `external_prefix_cache_hits` | **0 / 131,328** |
| fill p50 → replay p50 | 460.4 → **451.9 ms（−1.8%，纯噪声）** |

⇒ 池子放大 5.8 倍也救不回来；**把 `state` 组从"存/查"里摘掉这一步是决定性的**（与 `009` §2 一致）。

（同一天还跑了一条 `v1-nod2-4g`：**完全不挂**任何补丁 ⇒ 起服就撞
`assert isinstance(kv_cache_spec, FullAttentionSpec)`，复现 `010` §3.2；这条臂**不算判据臂**。）

---

## 3. Q2：池子要多大才够（**拐点与公式**）

### 3.1 全部臂的原始数据（一行一臂）

「工作集」定义：**一轮每个请求要存进池子的 (group, chunk) 条目数**。
本机 tiny（4096 token/请求、`blocks_per_chunk=8`、`block_size=128` ⇒ 每 chunk 1024 token）：
**44 条/请求 = 4（full 组）+ 40（10 个 SWA 组 × 4）**，所以 16/32/96 请求的工作集分别是
**704 / 1408 / 4224 条**。（这 44 条里只有 **14 条**是取回时真正要用的，见 §4.3。）

| 臂 | 请求数 | 池 (MiB) | `num_blocks` | 池/工作集 | `BlockStored:CPU` | `BlockRemoved:CPU` | `CPU→GPU` (MB) | load job | `external hits` | `queries` | fill p50 (ms) | **replay p50 (ms)** | 加速 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v1-d2-4g` ★Q1 | 16 | 4096 | 4096 | **5.82×** | 704 | — | 273.0 | 16 | **65,520** | 131,328 | 465.9 | **50.9** | **9.2×** |
| `v1-l1ct-4g` ★对照 | 16 | 4096 | (4096)【推断】 | 5.82× | 704 | — | **0** | 0 | **0** | 131,328 | 460.4 | **451.9** | 1.0× |
| `v1-n16-640m` | 16 | 640 | 640 | 0.909× | 1,408 | 768 | **0** | 0 | **0** | 131,328 | 460.8 | 453.4 | 1.0× |
| `v1-n16-672m` | 16 | 672 | 672 | 0.955× | 1,216 | 544 | **0** | 0 | **0** | 131,328 | 460.5 | 452.9 | 1.0× |
| `v1-n16-674m` | 16 | 674 | 674 | 0.957× | 1,184 | 510 | **0** | 0 | **0** | 131,328 | 461.4 | 452.4 | 1.0× |
| `v1-n16-688m` | 16 | 688 | 688 | 0.977× | 960 | 272 | **0** | 0 | **0** | 131,328 | 460.6 | 451.9 | 1.0× |
| `v1-n16-696m` | 16 | 696 | 696 | 0.989× | 709 | 13 | **267.0** | 16 | **63,473** | 131,328 | 461.6 | **47.4**（1 个请求 261.0） | 9.7× |
| `v1-n16-700m` | 16 | 700 | 700 | 0.994× | 704 | 4 | 273.0 | 16 | **65,520** | 131,328 | 460.0 | **47.8** | 9.6× |
| `v1-n16-703m` | 16 | 703 | 703 | 0.999× | 704 | 1 | 273.0 | 16 | **65,520** | 131,328 | 459.9 | **46.4** | 9.9× |
| `v1-n16-704m` | 16 | 704 | 704 | **1.000×** | 704 | 0 | 273.0 | 16 | **65,520** | 131,328 | 459.0 | **47.1** | 9.7× |
| `v1-n16-768m` | 16 | 768 | 768 | 1.091× | 704 | 0 | 273.0 | 16 | **65,520** | 131,328 | 460.3 | **46.9** | 9.8× |
| ★`v1-n16-640m-rev`（**replay 倒序**） | 16 | 640 | 640 | 0.909× | 768 | 128 | 238.8 | 14 | **57,330** | 131,328 | 460.7 | **47.0**（2 个请求 450.2 / 450.7） | 9.8× |
| `v1-n32-1408m` | 32 | 1408 | 1408 | **1.000×** | 1,408 | 0 | 545.9 | 32 | **131,040** | 262,400 | 458.1 | **47.0** | 9.7× |
| `v1-n96-1g` | 96 | 1024 | 1024 | 0.242× | 8,448 | 7,424 | **0** | 0 | **0** | 786,688 | 447.2 | 445.7 | 1.0× |
| `v1-n96-2g` | 96 | 2048 | 2048 | 0.485× | 8,448 | 6,400 | **0** | 0 | **0** | 786,688 | 447.2 | 445.3 | 1.0× |
| `v1-n96-3840m` | 96 | 3840 | 3840 | 0.909× | 8,448 | 4,608 | **0** | 0 | **0** | 786,688 | 446.6 | 444.7 | 1.0× |
| `v1-n96-4g` | 96 | 4096 | 4096 | 0.970× | 8,448 | 4,352 | **0** | 0 | **0** | 786,688 | 447.3 | 445.5 | 1.0× |
| `v1-n96-4224m` | 96 | 4224 | 4224 | **1.000×** | 4,224 | 0 | 1,637.7 | 96 | **393,120** | 786,688 | 447.8 | **46.9** | 9.5× |
| `v1-n96-8g` | 96 | 8192 | 8192 | 1.939× | 4,224 | 0 | 1,637.7 | 96 | **393,120** | 786,688 | 446.9 | **47.1** | 9.5× |

**读表要点**

* `BlockRemoved:CPU` 与 `BlockStored:CPU`、`num_blocks` **精确闭合**（`removed = stored − num_blocks`）：
  640 → 1408−640=768 ✅、672 → 544 ✅、688 → 272 ✅、1 GiB → 8448−1024=7424 ✅ …… ⇒ 池子**每一档都被塞满并持续淘汰**。
* 未命中那一侧的 `BlockStored:CPU` 是 **2 倍工作集**（如 96 请求：8448 = 2×4224）⇒ **回放轮把每个请求又重存了一遍**，
  这就是"整轮零命中"的独立证据（命中就不会再 store）。
* `hits` 在满命中臂里 = `请求数 × 4095`（4096−1，尾 token 的既有口径）；部分命中臂 `696m` = `15×4095 + 2048`。
* `v1-l1ct-4g` 的 `num_blocks` 那一格：L1 的补丁没有池子日志，**没有实测值**，
  按同一模型/同一 `blocks_per_chunk` 的同构臂（`v1-d2-4g` = 4096）标【推断】，**没有用相邻数字顶替**。

### 3.2 拐点

把池子从 640 块一路调到 704 块（工作集 = 704 条），拐点**极窄**：

| 池/工作集 | 短多少条 | 结果（16 请求） |
|---|---|---|
| **1.000×**（704）、0.999×（703）、0.994×（700） | 0 / 1 / 4 | **16/16 全中**，replay p50 46.4–47.8 ms |
| **0.989×**（696） | 8 | **15/16 全中 + 1 个请求部分命中**：`hits=63,473`（缺 2,047 = 4,095−2,048 token ⇒ 该请求只命中 2/4 个 full chunk），它的 TTFT = **261.0 ms**（其余 15 个 ~47 ms） |
| **0.977×**（688） | 16 | **0/16 全部退化为重算**（replay p50 451.9 ≈ fill 460.6） |
| 0.957×（674）、0.955×（672）、0.909×（640） | 30 / 32 / 64 | **0/16** |
| 0.970×（96 请求）、0.909×、0.485×、0.242× | 128 / 384 / 2176 / 3200 | **0/96**（同一规律的另一个尺度） |

**⇒ 断崖还是部分命中：两种都出现了，取决于越过边界多少【实测】**

1. **略欠一点（≈1%）⇒ 部分命中**：696 块时那一个请求命中了**前缀的前 2 个 chunk**（TTFT 261 ms 正好落在
   47 ms（全中）与 451 ms（全重算）之间）⇒ `_maximal_prefix_lookup` **确实是从第 0 块起"能连续多少算多少"**，
   不是全有或全无；
2. **再欠一点（≥2.3%）⇒ 整轮归零**（断崖）：688 块起，**每个**请求的第 0 块都被 LRU 前沿吃掉，
   而回放又按同一顺序逐个重放 ⇒ 每次未命中触发的重存**刚好**把"下一个请求的头部"挤掉（相位锁定的级联）。

**★ 级联（而不是"数据不在池子里"）的直接证据【实测】**：把 `v1-n16-640m`
（0.909×，**0/16 命中**）的回放顺序改成**倒序**，同一个池子立刻变成 **14/16 命中**
（`hits=57,330`、`CPU_to_GPU=238.8 MB`、14 个请求 ~47 ms，只有 2 个 450 ms —— 正是原来排在
最前面的那两个 prompt）。**数据在池子里，是被"回放顺序 + 头部优先淘汰"错杀的。**

### 3.3 可外推的公式

```
① 先算"条目数"，不是字节数：
   E_need = N_resident × Σ_{g ∈ 参与组} ceil(L_min_g / tokens_per_chunk_g)
            其中 tokens_per_chunk_g = tokens_per_block_g × blocks_per_chunk

② 再换成字节：
   pool_bytes ≥ E_need × round_up(worker_kv_bytes_per_block × blocks_per_chunk × num_copies, ALIGN)
                num_copies = 1（replicated_layout，cuda-alike）否则 = world_size

③ 余量：E_need 按 **≥1.2×** 留（见下）
   ★ 注意：E_need 是"**同一批要被再次取回的请求**"，不是"唯一 KV 字节"，也不是"并发数"本身
```

**本机 tiny/TP1 的实测标定**【实测 / 实测·算术闭合】：

| 量 | 值 | 出处 |
|---|---|---|
| `worker_kv_bytes_per_block` | **131,072 B** | `[V1_verify] CPU 卸载池:` 行 |
| `kv_bytes_per_chunk`（= 1 个条目占的池子） | **1,048,576 B = 1 MiB** | 同上（`131072×8×1`，无 padding） |
| 每 GiB 池 = 条目数 | **1024** | `num_blocks` 实测 640/672/…/4096/8192 与池字节数逐个吻合 |
| 16 请求 × 4096 token 的工作集 | **704 条 = 704 MiB** | 实测：703 条全中、688 条归零 |

**8 卡真权重（`009` 的配置）代入**【推断，但与 `009` 的实测一致】：
`541,198 B × 8 × 8 = 34.6 MB/条目` ⇒ 2 请求 × 12 组 × 32 chunk = **768 条 ≈ 27 GiB**（32 GiB 池 ⇒ 963 条 ⇒ 0.80×，实测通过）；
16 请求 ⇒ **6144 条 ≈ 213 GiB**（32 GiB 池 ⇒ 0.16×，实测 0 命中）。

**"参与组数"这一项怎么读**：DSV4.1 有 **11 组（tiny，无 dspark）/ 12 组（8 卡，含 dspark）**都参与存，
而池子是按 `(group, chunk)` 计费的 ⇒ 同样的"会话 token 数"要多花 ~11–12 倍的池子
（这正是 `009` §3.1 的 12× 根因；本轮在单卡上**逐字复现**：44 条/请求 vs 唯一 KV 只有 4 个 full chunk 的 145 MB 量级）。

**余量给多少**【推断·工程建议，依据是本轮的实测边界】：
实测"短 0.6% 全中 → 短 1.1% 开始掉 → 短 2.3% 整轮归零"，所以**不要**贴着 1.00× 配；
建议 `pool_bytes ≥ 1.2 × E_need`（20% 余量），并**按条目数记账**（不是按唯一 KV 字节）。
另外：池子上限在本机单卡上是软约束 —— 单卡一次 pin **8 GiB 成功**【实测】，
而 8 卡 × 8 worker 各 8 GiB 会撞 `207001`（`009` §3.2）⇒ 容量规划要同时看 pin 的**形态**。

---

## 4. 机制（为什么是断崖、为什么倒序能救）

### 4.1 命中必须"从第 0 块起连续"【实测·代码 + 数据】

`_maximal_prefix_lookup()`（D2 补丁文件 `scheduler.py:660-687`，上游同处）逐 key 查，
`case LookupResult.MISS: break` ⇒ **返回的是"从头开始连续命中"的块数**。
`696m` 臂的那个请求拿到 2,048 token（2/4 chunk）就是这条语义的正面证据：
**能连续多少算多少**；反过来，第 0 块丢了就**一条都不算**。

### 4.2 淘汰是"从头吃"，回放又是同序 ⇒ 相位锁定【实测·代码 + 数据】

`vllm/v1/kv_offload/cpu/policies/lru.py`：`evict()` 从 `evictable_blocks` 的**头部**取候选
（= 完成时间最早的那些），`insert()` / `mark_evictable()` 追加到**尾部**。
回放顺序与 fill 相同时，"最老的一批"恰好是"马上要回放的那一批" ⇒
第一次未命中触发的重存会挤掉**下一个**请求的头部，级联到整轮。

### 4.3 存了但永远用不上：44 条里有 30 条【实测】

每个请求存 **44 条**，但 load job 只需要 **14 条**（`group_sizes=[32,0,1×10]`、`keys=14`）：
10 个 SWA 组各自只有**窗口那 1 个 chunk**会被取回，另外 3 个 chunk × 10 组 = **30 条是纯占位**。
⇒ 有效容量再被稀释 ~3.1×（叠在 11–12 组的倍率之上）。
**给上游的优化建议**：SWA 组只存窗口 chunk（`is_store_reachable_swa_chunk` 已经有这个钩子，
但本轮 tiny 的 `alignment_chunk_count` 退化成 1 ⇒ 全存），可把 `E_need` 从 44N 降到 14N。

---

## 5. 未确认 / 风险

| 项 | 状态 |
|---|---|
| 数值正确性 | 【未确认】本轮只跑 TTFT/命中判据，**没做**精度/logprob 对比（`state` 组不参与卸载的语义与 GPU 前缀缓存一致，仍是【推断】，同 `009` §2.5） |
| 部分命中的 TTFT | 【实测·单样本】只有 `696m` 一个请求（261.0 ms）；不能当分布用 |
| `concurrency > 1` | 【未确认】本轮全部 `concurrency=1`。并发交错会改变 fill/回放顺序，级联形态可能变化 ⇒ 余量建议 1.2× 是**保守推断**，A2 上线前应在真机复核 |
| 8 卡上的拐点位置 | 【推断】用公式 + `009` 的 2 请求/16 请求两点外推；本轮**没有**在 8 卡上重扫 |
| tiny 与真权重的差异 | 【实测·已排除一条】候选差异①"tiny 少 MTP/dspark ⇒ 参与组 11 而非 12"**不构成单卡/8 卡差异**：修复在本机成立；差异②`num_cpu_blocks` 公式的 `num_copies`（TP1=1 vs TP8=8）只改**每条目字节数**，不改结论 |
| 池子上限 | 【实测】单卡一次 pin 8 GiB 成功（`v1-n96-8g`）；8 卡 8 GiB/worker 失败（`009`）⇒ 单卡通道能验证到 8 GiB 级别 |

---

## 6. 复现（单臂 1.4–2.6 min，实测 22 条臂）

```bash
# 0) 一次性：把 D2 的补丁文件复制到自己的目录（逐字节副本，md5 0302fab4…）
cp agents/D2_offload/patches/offload_dsv41/scheduler.py agents/V1_verify/patch/d2_scheduler.py

# 1) 一条臂（c1 槽位锁内；退出码 75 = 没抢到锁，重试即可）
cd ~/projects/dsv41-upstream-pr
bash tools/a3_chip.sh c1 --timeout 900 --name v1-x -- \
  env TAG=v1-x PROMPTS=16 OFFLOAD_BYTES=738197504 \
  bash /work/agents/V1_verify/scripts/run_arm.sh
#    → out/v1-x.{server.log,client.json,kv_events.log,metrics_after.txt,meta.txt}

# 2) 观察点
#    meta.txt        → [V1_verify] CPU 卸载池: num_blocks=…（池子真实块数）
#    server.log      → [D2_offload] 参与卸载的组 / load job / miss-scan
#    metrics_after   → kv_offload_total_bytes{CPU_to_GPU} / external_prefix_cache_hits
#    client.json     → rounds[].ttft.p50_ms 与 per-prompt 明细

# 3) 压成一行：python3 agents/V1_verify/scripts/summarize_arms.py --dir <产物目录> <tag…>

# 4) 挂/不挂补丁的自检（不占卡）：
#    V1_D2_PATCH=1 → module.__file__ = .../patch/d2_scheduler.py，`_offload_participates` 存在
#    V1_D2_PATCH=0 → 镜像内原文件，无该符号
```

---

## 7. 产物清单

| 文件 | 作用 |
|---|---|
| `a2/logs/013-20260922-single-chip-fix-verify.md` | 本日志 |
| `a2/agents/V1_verify/patch/sitecustomize.py` | 进程内挂 D2 补丁（meta_path 整文件替换）+ `wo_a` dummy 适配 + **只读**池子日志 |
| `a2/agents/V1_verify/patch/d2_scheduler.py` | D2 补丁的逐字节副本（md5 `0302fab4c68c3adc7d2c4a135c7c4289`） |
| `a2/agents/V1_verify/scripts/run_arm.sh` | 单卡单臂运行器（支持 `OFFLOAD_GB` / 精确 `OFFLOAD_BYTES` / `CLIENT_ARGS`） |
| `a2/agents/V1_verify/scripts/summarize_arms.py` | 把一臂的产物压成一行判据（只读） |
| `a2/agents/V1_verify/bench/{kv_offload_client.py,kv_events_probe.py}` | L1 压测端的副本；`kv_offload_client.py` 多一个 `--reverse-replay`（★ 机制判别实验） |
| `a2/logs/raw/013-single-chip-fix-verify/` | **22 条臂的原始产物**（227 个文件，6.7 MB） |
| `a2/logs/raw/013-raw-arms.tar.gz` | 同上 + 脚本 + 补丁的打包（697 KB，来自 A3 `agents/V1_verify/tmp/`） |
| `a2/logs/raw/013-{batchA,B,C,D,E,rev,703m,v1-d2-4g}.out` | 22 条臂的完整运行日志（含一次 `warmup` 参数漏传导致起服失败的重跑） |

---

## 8. 红线遵守

* **只用 c1（Phy-ID 6）**：全程 `tools/a3_chip.sh c1` 锁；**没用** c0/c2、**没碰** Phy-ID 8–15 / `dsv41-a3` / `mooncake-master`；
* **不手设** `ASCEND_RT_VISIBLE_DEVICES`（由锁脚本注入）；
* **不用 `/tmp`**：本机临时区 `~/tmp/20260922/v1_verify/`（`source a2/scripts/tmpdir.sh v1_verify`），A3 侧产物只落 `agents/V1_verify/`；
* **没写** `upstream-v41/`（只读）；**没改**镜像内任何源码（全部走 `PYTHONPATH` 进程内补丁）；
* 跨机传文件走 **coscli**（`cos-xfer.sh`），SSH 只跑命令；ssh 一律 `-o ControlPath=none`；
* 每臂收尾由 `run_arm.sh` 的 trap 停掉自己起的 `vllm serve`（共享容器本体保留）；
* 结论全部标 **【实测】/【推断】/【未确认】**；缺的格子标 `—`，**没有用相邻数字顶替**。
