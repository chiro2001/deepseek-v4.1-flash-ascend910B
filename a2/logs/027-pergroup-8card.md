# 027 · per-group `blocks_per_chunk` 的 **8 卡真权重**终验（A2 上线的最后一格）

**日期**：2026-09-22 01:57 – 04:3x（A3 本地时钟）
　**执行**：子代理 `L3_8card`
**机器**：A3（A3-node1）**Phy-ID 8–15**（8 张；`locks/c0.lock` 全程持锁）
**镜像/模型**：`quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`、`~/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`（**真权重**）
**起点**：`logs/021`（单卡 tiny 的 per-group bpc）、`logs/016`/`022`（8 卡 L2 终验与拐点）、`logs/019`（池子账）
**标记约定**：【实测】= 本机跑出来的原始数据；【推断】= 代码/算式推出来但没直接测；【未确认】= 没跑到
**原始数据**：`logs/raw/027-l3-8card/`；脚本与补丁：`agents/L3_8card/`

---

## 0. 一句话结论（先给判断）

| 任务书问题 | 结果 |
|---|---|
| **① 三个补丁能否共存（同一份 `scheduler.py`）？** | **【实测·能】**D2 的 `offload_participating` 组排除 + P1 的 `cpu_npu.py` registered 池 + per-group 的 unit 展开**同时生效**：`P1_pinned ret=0` **128 行 / 8 个 rank** 全中、`[D2_offload]` 组清单与排除组都在、`unit 模式：cache.blocks_per_chunk=1`、`PerGroupBPCManager` 都打出来了（§2）。 |
| **② per-group 在 8 卡真权重上有效吗？** | **【实测·有效】**`per_group={0:8, 1:8, 2..12:1}`，池的格子从"8 MiB 的 chunk"变成 **"1 个 GPU block = 1 MiB 记账"**（`kv_bytes_per_unit=1048576`）。 |
| **③ ★ 128K × 16 并发四条判据？** | **【实测·全中】**`BlockStored:CPU=30,605`、`CPU→GPU=18.83 GB`（112 job）、`hits=788,480`、**replay 1,420.6 ms vs fill 18,202.6 ms = 12.81×**（§4 臂 B）。 |
| **④ 宿主实占与 `021 §6` 的 264 GiB 预测吻合吗？** | **【实测·不吻合】**实测 **277.77 GiB**（我给的是 021 口径的 1.000×，即 40,960 unit）。根因不是"少算了 draft 组"那么简单 —— 真正的需求是 §3 反解出来的 **23.5 unit / 1024 token / 请求**（021 的模型是 19、我中途的修正模型是 20，**都低 ~18%**）。 |
| **⑤ `×6.945` 在 8 卡新口径下还成立吗？** | **【实测·成立，逐字节】**9,728 unit → `70,836,027,392 B = 65.97 GiB`；40,960 unit → `298,256,957,440 B = 277.77 GiB`；3,072 unit → `22,369,271,808 B = 20.83 GiB`（= units × **7,281,664 B**）。 |
| **⑥ ★★ 池子到底要多大？** | **【实测·反解】**每请求：**32K → 758 unit**、**128K → 3,004 unit**；即 **≈ 23.5 unit / 1024 token / 请求**（full 8 + 10 个 SWA × ~1.125 条 + draft × 2×SWA 条数）（§3）。 |
| **⑦ ★ 欠配的表现是什么？** | **【实测·级联归零，没有中间态】**池 0.802×（A）/0.844×（A'）/0.852×（B 的 fill）都表现为**全场 0 命中**，`miss-scan` 显示**整条前缀链 `present=0`**（§3.3）。**运维判据见 §3.3。** |
| **⑧ 变长前缀安全（8 卡复核）？** | **【实测·安全口径成立】**fill 128K → replay **64K**：`hits=788,480`、12.81×；同一池子换 `SWA_TRIM=window`：`hits=258,048`、**2.79×**（§5）。 |
| **⑨ A2 的最终参数** | **【实测+推断】**见 §6（含"只挂 L5 / L5+L1 / L5+L1+KV8"三档对照表）。 |
| **⑩ ★ 生产候选点（×1.2）实测了吗？** | **【实测·全中，`BlockRemoved:CPU = 0`】**臂 **B2**（57,856 unit = 56.5 GiB 记账 = **392.35 GiB 宿主**）：`BlockStored:CPU=29,436`、**`BlockRemoved:CPU=—（0 次淘汰）`**、`CPU→GPU=25.69 GB`（208 job）、`hits=1,062,400`、**利用率 0.757**、replay **1,406.4 ms**（14.31×）。 |
| **⑪ ★ 服务在 `temperature=0` 下确定吗？** | **【实测·不确定】**同 prompt、**每次清缓存**、连发 4 次 `max_tokens=1` 仍有 **2 个不同的首 token** ⇒ 与 KV 卸载**无关**的独立质量问题，已单独成文 **`logs/037-20260922-nondeterminism.md`**。 |

---

## 1. 第 1 步：把 per-group 补丁接到 8 卡的挂载链上（不占卡）

### 1.1 换挂载方式：`sitecustomize` → **docker -v 文件挂载**

021 在单卡 c2 上是 `PYTHONPATH` + `sitecustomize.py`（`sys.meta_path` 整文件替换 + 3 个函数钩子）。
8 卡的挂载链（`shadow-pkg/scripts/serve_a2.sh`）走的是 **`docker -v` 单文件覆盖**（D2 的
`scheduler.py`、P1 的 `cpu_npu.py` 都是这个机制）⇒ 我把同一套逻辑落成 **4 个文件**（逻辑逐字不变）：

| 容器内路径 | 来源 | 作用 |
|---|---|---|
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | **D2 产物（md5 `0302fab4…`）+ 021 的 per-group 改动** | ★ **同一份文件**里同时有 D2 的 `offload_participating` 组排除 **和** per-group 的 unit 展开 |
| `vllm/distributed/.../offloading/config.py` | 镜像原文 + 15 行 | `blocks_per_chunk` 支持 **dict**，解析结果写进 `extra_config["blocks_per_chunk_by_group"]` |
| `vllm/v1/kv_offload/cpu/spec.py` | 镜像原文 + `get_manager()` 分支 | unit 模式 ⇒ `PerGroupBPCManager`；否则逐字回退 |
| `vllm/v1/kv_offload/cpu/pgp_manager.py` | 021 的 `pgp_manager.py` 逐字 + 3 个解析函数 | 池的"一格 = 1 个 GPU block"、一个 key 占 `bpc_g` 个 unit、**淘汰按 unit 记账** |

**为什么不能挂两次 `scheduler.py`**：docker 对同一目标路径挂两次的行为不保证 ⇒ 不能"D2 挂一次、
per-group 再挂一次"。合并版由 `agents/L3_8card/patch/build_patch.py` **生成**（脚本里对 D2 与 021 的
产物 md5 都有指纹断言，任一变就报错）。

**一个不在计划内的发现**：P1 的 `native/cpu_npu.py` 挂载点**嵌在 `OFFLOAD_SCHED_PATCH` 那个 `if` 里**
（影子包现有形态，不是我们改的）⇒ `OFFLOAD_SCHED_PATCH=1` 必须保持为 1 才能同时挂上 P1。

### 1.2 补丁生效自检（★ 起服后、压测前；不满足 `exit 9`）

`run_arm_l3.sh` 在就绪后立刻做 8 项检查（任一不过 ⇒ 删容器 + exit 9，不浪费 20 分钟压测）：

```bash
grep -c "P1_pinned.*ret=0"     serve.log   # 期望 >=8（真权重实际 128 = 16 张 canonical 张量 × 8 rank）
grep -a "P1_pinned.*ret=0" | grep -oE "Worker_TP[0-9]+" | sort -u | wc -l   # 期望 8（★ 行数会被张量数放大）
grep -c "\[D2_offload\]"       serve.log   # 期望 >0
grep -c "unit 模式：cache.blocks_per_chunk"  # per-group 生效
grep -c "PerGroupBPCManager 生效"            # manager 换上了
grep -c "alignment_tokens=1024|alignment_chunk_count.*8"   # 裁剪谓词活了
grep -c "CPU 卸载池: num_units="             # 池子日志（第一手数字）
docker inspect -f '{{.Mounts}}' | grep -c L3_8card/patched  # 挂载真的进去了（期望 4）
```

**臂 B 的自检输出（【实测】原样）**：

```
[l3][self-check] startup_seen=1
[l3][self-check] P1_pinned ret=0 行数=128（期望>=8，rank 数=8 期望 8；回落 pinned=0 期望 0）
[l3][self-check] D2_offload 行数=2（期望>0）
[l3][self-check] per-group 生效行数=10（期望>0）PerGroupBPCManager=1（期望>0）
[l3][self-check] alignment/unit 标志行数=2（期望>0）池子日志行数=9（期望>0）
[l3][self-check] L3_8card 挂载源数=4（期望>=3：config/spec/pgp_manager）
```

### 1.3 不占卡探针（`agents/L3_8card/scripts/probe_l3.py`，c0 探针锁 ~10 s）

把 021 的 `probe_unit_mode.py` 原样跑在**挂载版** `pgp_manager.py` 上（只改 patch 目录与解析函数来源）：
解析规则、上游裁剪谓词、unit 分配/淘汰、调度侧展开一致性**全部通过**。

### 1.4 ★ 我自己的三个坑（记下来，别再踩）

| # | 坑 | 症状 | 代价 |
|---|---|---|---|
| 1 | 就绪标志等错了文件：`Application startup complete` 在**引擎自己的** stdout（`shadow-pkg/results/<RID>/serve.log`），不在 `serve_a2.sh` 的外层 stdout | 白等 27 min → 自检判 `startup_not_seen` → exit 9 | 1 条臂 |
| 2 | 那个路径**起服几秒后才创建**，一次性 `[ -f ]` 解析会永久指向错文件 | 白等 6.5 min → 同样 exit 9 | 1 条臂 |
| 3 | 串行链脚本里 `WAIT` 用了默认 240 s，撞上前一条臂的锁 ⇒ 抢锁失败 | 发现后改 `WAIT=1800`，没真跑起来 | 0 |

**修法**：每轮**重新**检查引擎日志 + 外层日志 + `/health` 三者任一命中；`WAIT` 默认改 1800 s。
（自检本身是有效的——它抓到了两次"补丁没生效"的假阳性，避免了 2 × 20 分钟的无效压测。）

---

## 2. 三个补丁共存的**第一手证据**（【实测】，臂 B 的 serve.log）

```
(Worker_TP0_EP0 pid=1197) [P1_pinned] CPU pool backend = registered (NPU_OFFLOAD_HOST_MEM)
(Worker_TP0_EP0 pid=1197) [P1_pinned] CPU pool 40960 x 524288 (20.00 GiB): registered dev=0x... ret=0
(Worker_TP0_EP0 pid=1197) [P1_pinned] CPU pool 40960 x  65536 ( 2.50 GiB): registered dev=0x... ret=0
...（16 张 canonical 张量 × 8 rank = 128 行，全部 ret=0，无一条回落 pageable/pinned）
(EngineCore pid=1170) [D2_offload] KV 卸载 group 清单 n=13: [(0,'DeepseekV41FullSpec',128,8,...,True),
   (1,'DeepseekV41CompressorStateSpec',32,3,...,False), (2..11,'DeepseekV41SWASpec',128,4,...,True),
   (12,'DeepseekV41DraftSWASpec',128,3,...,True)]
(EngineCore pid=1170) [D2_offload] 参与卸载的组：full_attention=[0] sliding_window=[2..12]；被排除的组=[1]
(Worker_TP0_EP0 pid=1197) [L3_8card] unit 模式：cache.blocks_per_chunk=1 per-group={0: 8, 1: 8, 2: 1, ..., 12: 1}
(Worker_TP0_EP0 pid=1197) [L3_8card] CPU 卸载池: num_units=40960 kv_bytes_per_unit=1048576
   cpu_page_size_per_worker=131072 replicated_layout=False num_copies=8 blocks_per_chunk=1
   per_group={0: 8, 1: 8, 2..12: 1} cpu_bytes_to_use=42949672960
   worker_kv_bytes_per_block=131072 world_size=8
[SWA_trim] alignment: blocks_per_chunk=1 full_attn_tokens_per_chunk=[1024] alignment_tokens=1024
[SWA_trim] SWA_TRIM=off group 表 (idx, tpb, tpc, bpc, sw_chunks, alignment_chunk_count, is_eagle, participates):
   [(0,128,1024,8,None,None,False,True), (1,32,256,8,None,None,False,False),
    (2..11,128,128,1,1,8,False,True), (12,128,128,1,1,8,True,True)]
```

**★ 取回不是"重算侥幸"的第一手证据**：

```
[D2_offload] load job req=cmpl-… keys=66 group_sizes=[440, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] src_blocks=451 dst_blocks=451
（14 条；group_sizes[1]=0 = state 组占位；★ src_blocks == dst_blocks ⇒ per-group 的 unit 展开与 GPU block 逐项对齐）
```

> `022` 的**标量**臂这一行是 `src_blocks=42 dst_blocks=42`（CPU 侧一个 id = 一个 8-MiB chunk）；
> 本轮是 **`src_blocks=451` = 一个 GPU block 一个 id**。搬多少字节由 `CPU→GPU` 指标给出，
> 两侧逐项对齐由 `assert len(cpu_ids) == len(gpu_blocks)` 兜底（对不齐就 fail-fast）。

---

## 3. ★★ 本轮最有价值的数字：**真实的 unit 需求是 23.5 / 1024 token / 请求**

### 3.1 怎么反解出来的（第一手，不依赖任何模型）

D2 的诊断计数器（`patched/scheduler.py:1522`）里，每落一个 store key 就 `_dbg_tally[group] += 1`；
而 `lookup-summary` 每次打一行**累计值** ⇒ **相邻两行的差 = 恰好一个请求的 key 数**。

| 臂 | 每请求 key 数（full / 单个 SWA / draft） | **每请求 unit 数** | 16 请求合计 | 我给的池 | 倍率 |
|---|---|---:|---:|---:|---:|
| A'（16×32K） | 32 / **36** / **71** | 32×8 + 10×36 + 71×2 = **758** | **12,128** | 10,240 | **0.844×** ⛔ |
| B（16×128K，fill 轮） | 128 / **142** / **280** | 128×8 + 10×142 + 280×2 = **3,004** | **48,064** | 40,960 | **0.852×** ⛔ |
| B（16×64K，replay 轮） | 64 / **71** / **140** | 64×8 + 10×71 + 140×2 = **1,502** | **24,032** | 40,960 | 1.70× ✅ |

```
32K : 758 unit / 32 段 = 23.7 unit/段      128K: 3,004 / 128 段 = 23.5 unit/段
⇒ ★ 点估计：23.5 unit / 1024 token / 请求（线性，两档残差 0.7%）
   组成：full 8 + 10 个 SWA × ~1.125 条/段 + draft × (2 × SWA 条数)
   021 的模型 = 19（每 SWA 组每段 1 条）、我中途的"draft 修正"模型 = 20 —— 都低 ~18%
   多出来的 ~3.5 unit/段 = 批次边界上"尾部半满段"被当作 reachable tail 留下（机理【推断】，数字【实测】）
```

### 3.2 为什么两个模型都低

* `is_store_reachable_swa_chunk()` 的判据是"每个对齐段留 `sliding_window_chunks + is_eagle` 条尾部 chunk"，
  但它还有一条 `actual_segment_length = min(alignment, storable - segment_start)`：
  **当一个 store 批次在段中间结束时，"当前段"会被当成一个更短的段**，于是它的尾部 1（draft 2）条也被留下。
  一个 128K 请求要分 16 个批次（`max_num_batched_tokens=8192` = 8 段/批），每个批次边界都会多留 ⇒ 每段平均多 ~0.125 条；
  draft 组乘 2（eagle）⇒ 正好是观测到的 +3.5。
* **这个成分与上下文长度无关**（只与批次边界有关）⇒ 23.5 在 32K/128K 上都是 23.5，与实测一致。

### 3.3 ★★ 欠配的表现：**级联归零，没有中间态**（A/A'/B 三臂独立证明）

`miss-scan` 第一手（A' 的回放轮，24 条里前 4 条，省略号是我加的）：

```
[D2_offload] miss-scan req=cmpl-9cb6… group=0 scanned=32 present=0 head_key=7de64db3fb8c534d chunk_bytes=1024
[D2_offload] miss-scan req=cmpl-9b6b… group=0 scanned=32 present=0 head_key=40cbfd53d7a76f82 chunk_bytes=1024
[D2_offload] miss-scan req=cmpl-aa4e… group=0 scanned=32 present=0 head_key=fe02b6443b497f45 chunk_bytes=1024
[D2_offload] miss-scan req=cmpl-90e7… group=0 scanned=32 present=0 head_key=37d95db0753af5b6 chunk_bytes=1024
（回放轮里前面请求的 head_key 与填充轮**逐字相同**，但池里**一条都不剩**）
```

机理（【实测】支持）：

```
填满轮就超配（0.844×）⇒ 回放轮里前面请求命中不了 ⇒ 重算 ⇒ 重新 store
⇒ 用**更新的** LRU 条目把"还没轮到回放的后面请求"的链挤掉 ⇒ 级联 ⇒ 全场 0 命中
```

**⇒ 两条给 A2 的硬结论（运维判据）**：
1. **不能靠"观察命中率是否下降"判断池子够不够** —— 一欠配就是 **0**，没有中间态；
2. **上线后要监测 `vllm:kv_offload_block_removed_total{medium="CPU"}` 是否为 0**（它是"曾经撑爆"的直接证据）：
   A 在 0.802× 下 `BlockRemoved:CPU = 8,271`、A' 在 0.844× 下 = 7,892、B 在 0.852×（fill 轮）下 = 3,038。

---

## 4. 第 2 步：8 卡验证（臂清单与结果）

### 4.1 臂清单

固定量：`ENGRAM=0`、`PREFIX_MATCH_UNIT=32`、`MAX_SEQS=32`、`BAT_TOKENS=8192`、`concurrency=1`、
`PROMPT_SALT=20260922`、`speculative_config=SpeculativeConfig(method='dspark', num_spec_tokens=5)`
（**与 `016`/`022` 的 L2 基线逐字相同**；各臂的差异只有 `PROMPT_TOKENS`/`REPLAY_PROMPT_TOKENS`/
`OFFLOAD_BYTES`/`SWA_TRIM`/`SHARED_PREFIX`/`MAX_LEN`/`KV_MEM_BYTES`）。

| 臂 | 池（unit / 记账） | 请求 × 上下文 | 回放 | 存侧规则 | 相对**实测需求**的倍率 |
|---|---:|---|---:|---|---|
| **A** | 9,728 / 9.3 GiB | 16 × 32K | 32K | `off` | 021 口径的 1.000×（实测需求 **0.802×**） |
| **A'** | 10,240 / 10.0 GiB | 16 × 32K | 32K | `off` | 我中途的修正口径 1.000×（实测需求 **0.844×**） |
| **B** | 40,960 / 40.0 GiB | 16 × 128K | **64K** | `off` | fill 轮 **0.852×** / replay 轮 1.70× |
| **C** | 40,960 / 40.0 GiB | 16 × 128K | 64K | **`window`** | 对照 |
| **D** | 3,072 / 3.0 GiB | 16 × 128K（**共享同一前缀**） | 128K | `off` | 共享前缀的账（0.874×） |
| **B2** | **57,856 / 56.5 GiB** | 16 × 128K | 64K | `off` | **实测需求的 ×1.2** |
| **COLD1** | 40,960 | 16 × 64K（单轮冷算） | — | `off` | B/C 回放的冷算参考（§8） |
| COLD2 | 40,960 | 16 × 64K（`max_tokens=8` 单轮） | — | `off` | 被 COLD1 的 token 级证据提前定性 ⇒ **只作旁证** |
| **det（格 3）** | 1 GiB | 1 × 32K + 1 × 512（`SPEC_ON=0`） | — | `off` | **响应本身非确定** ⇒ 见 `logs/037-20260922-nondeterminism.md` |

### 4.2 结果表（【实测】）

| 臂 | 池 / 需求 | 池（宿主实占） | `BlockStored:CPU` | `BlockRemoved:CPU` | `CPU→GPU` | load job | `hits` | 利用率 | fill p50 | **replay p50** | 加速 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.802× | 65.97 GiB | 14,828 | 8,271 | **0** | 0 | **0** | 1.00 | 4,029.7 ms | 4,008.4 ms | **1.01×** ⛔ |
| A' | 0.844× | 69.44 GiB | 14,828 | 7,892 | **0** | 0 | **0** | 0.999 | 4,028.9 ms | 4,007.4 ms | **1.01×** ⛔ |
| **B** | 0.852×（fill）/1.70×（replay） | **277.77 GiB** | **30,605** | 3,038 | **18.83 GB** | **112** | **788,480** / 3,145,984 | 1.00 | 18,202.6 ms | **1,420.6 ms** | **12.81×** ✅ |
| **B2 ★** | **1.193×** | **392.35 GiB** | **29,436** | **—（0）** | **25.69 GB** | **208** | **1,062,400** / 3,474,176 | **0.757** | 20,126.3 ms | **1,406.4 ms** | **14.31×** ✅ |
| C | 同上 | 277.77 GiB | 9,420 | 0 | 6.68 GB | 128 | 258,048 / 3,145,984 | 0.58 | 18,205.8 ms | 6,527.5 ms | 2.79× |
| D | 3,072 / 共享 = 0.874× | 20.83 GiB | 1,851 | 0 | **2.83 GB** | 8 | 120,832 / 270,080 | 0.894 | —（单轮冷算） | 319.5 ms | — |

#### 4.2b ★ B2（生产候选点）单独读法

| 判据 | B2 的值 | 含义 |
|---|---|---|
| `BlockRemoved:CPU` | **`—`（0）** | ★ **池子从来没有被撑爆过** ⇒ 这是"够用"的**直接证据**（§3.3 的监测判据） |
| 利用率 | **0.757** | ★ **不能**反推"池子可以砍到 0.757×" —— 见下方「★ 别误读」 |
| `CPU→GPU` / load job | 25.69 GB / 208 | 比 B（18.83 GB / 112）**多 36%** ⇒ 多出来的命中是 B2 用余量换来的 |
| replay p50 | **1,406.4 ms** | 比 B 的 1,420.6 ms 略快；加速 12.81× → **14.31×** |

> ★ **"别误读 0.757"**：**峰值需求 ≈ 需求均值 × 1.19，而均值利用率只有 0.757**
> （回放轮的 churn 会瞬时抬高占用）。实测三个点已经把这条钉死：
> **0.852× 归零、1.000×（A'）归零、1.193×（B2）才 `removed=0`**。
> ⇒ **绝对不能**用"利用率 0.757"去反推池子可以再小。

### 4.3 臂 B 的四条判据（任务书口径）

| 判据 | 值 | 门槛 | 结果 |
|---|---|---|:--:|
| `BlockStored{medium="CPU"} > 0` | **30,605** | >0 | ✅ |
| `kv_offload_total_bytes{CPU_to_GPU} > 0` | **18.83 GB**（112 个 load job） | >0 | ✅ |
| `external_prefix_cache_hits > 0` | **788,480** / 3,145,984 | >0 | ✅ |
| `replay ≪ fill` | **1,420.6 vs 18,202.6 ms** | ≪ | ✅ **12.81×** |

### 4.4 宿主实占：`×6.945` 在 8 卡上**逐字节成立**（三条臂）

| 臂 | units | 预测宿主（units × 7,281,664 B） | **实测**（P1 每张 canonical 张量的 N × M 求和） |
|---|---:|---:|---:|
| A | 9,728 | 70,836,027,392 B = 65.97 GiB | **70,836,027,392 B = 65.97 GiB** ✅ |
| B/C | 40,960 | 298,256,957,440 B = 277.77 GiB | **298,256,957,440 B = 277.77 GiB** ✅ |
| D | 3,072 | 22,369,271,808 B = 20.83 GiB | **22,369,271,808 B = 20.83 GiB** ✅ |

（每 rank 16 张 canonical 张量；臂 B 每 rank `37,282,119,680 B = 34.72 GiB`。
第一手输出见 `logs/raw/027-l3-8card/<tag>.pool_bytes.txt`。）

### 4.5 任务书里那个 `BlockStored:CPU ≈ 9,728` 的口径要拆成两个数

任务书写"`BlockStored:CPU ≈ 16 × (32×8 + 352×1) = 16×608 = 9,728`"。【实测】这两个数不是同一个东西：

* **9,728 是 unit 数**（= **池的占用量**）—— 臂 A 的池正好 9,728 unit 且 `cpu_cache_usage_perc = 1.00`；
* `BlockStored:CPU` / `kv_offload_cpu_allocation_size_sum` 数的是 **key（= 1 个 chunk）**：
  A' 的 `allocation_size_sum = 14,828`（含两轮的 store）；
* ⇒ "每请求 unit"必须用 §3.1 的 **tally 增量**反解，不能用 `BlockStored` 直接除。

---

## 5. 第 3 步：变长前缀安全（8 卡复核，**与 021 的 tiny 结果不同**）

### 5.1 两臂同口径对照（fill 16 × 128K → replay **64K**，同一个池 40,960 unit）

| | **B：`SWA_TRIM=off`（上游规则，本轮正解）** | **C：`SWA_TRIM=window`（017 那条）** |
|---|---|---|
| `hits` | **788,480** ✅ | **258,048** |
| `CPU→GPU` | **18.83 GB**（112 job） | 6.68 GB（128 job） |
| replay p50 | **1,420.6 ms** | 6,527.5 ms |
| 加速 | **12.81×** | **2.79×** |
| `BlockStored:CPU` | 30,605 | 9,420 |
| 池利用率 / `BlockRemoved:CPU` | 1.00 / 3,038 | 0.58 / **0** |

### 5.2 结论（【实测】）

**变长前缀在 8 卡真权重上是安全的**：同样的池、同样的 workload，把回放从 128K 换成 **64K**，
仍 `hits=788,480`、`CPU→GPU=18.83 GB`、12.81× ⇒ **短前缀回放不会归零**。
`SWA_TRIM=window` **严格更差**：加速比 **12.81× → 2.79×**（replay 慢 4.6×），且只取回 1/3 的字节。

### 5.3 ★ 但 021 的"`window` 归零"在 8 卡上**没有复现** —— 本轮第三条修正

021 在单卡 tiny（4096 → 2048）上测得 `window` 规则 `hits=0`、加速 1.9×（= 完全重算）。
本轮 8 卡 128K → 64K 上它**没有归零**（`hits=258,048`、2.79×）。原因【推断·强，可由日志数字复核】：

| | 021 的 tiny 口径 | 本轮 8 卡口径 |
|---|---|---|
| 一次 store 批次覆盖多少 token | 批次末尾 chunk ≠ 对齐段末尾 | **`max_num_batched_tokens=8192` = 64 个 SWA chunk = 正好 8 个对齐段** ⇒ `window` 留下的"批次末尾 chunk"**恰好就是每个对齐段的尾部 chunk** |

⇒ `window` 规则在**这个配置下与上游规则留的 SWA chunk 高度重合**（这是它 `hits>0` 的原因），
只是留下的量少（9,420 vs 30,605 key）⇒ 部分命中、加速比掉到 2.79×。

**★ 判据陷阱（必须写清）**：`hits>0` **不等于**"取回的数据是对的" ——
`_maximal_prefix_lookup` 只要求从第 0 块开始**连续命中**前 N 块；C 臂命中的是"前 258,048 token"这一段，
**被裁掉的更长的前缀它根本没去查**。⇒ **"不归零"不等于"window 规则安全"**；
真正的安全判据是"冷算 vs 取回逐 token 相同"（§8）。

---

## 6. 第 4 步：A2 的最终参数（【实测】+【推断】）

### 6.1 需求（用 §3 的**实测** 23.5 unit/1024 token/请求；1 unit 记账 = 1 MiB、1 unit 宿主 = 6.944 MiB）

| 场景 | unit/请求 | 16 并发池（记账） | **宿主实占** | A2 余量 442 GiB |
|---|---:|---:|---:|---|
| 16 × 32K | 758 | 11,846 unit = 11.6 GiB | **80.3 GiB** | ✅ 很宽松 |
| **16 × 128K** | **3,004** | **46,937 unit = 45.8 GiB** | **318.2 GiB** | ✅（占 72%） |
| **16 × 128K ×1.2** | 3,605 | **56,325 unit = 55.0 GiB** | **381.8 GiB** | ⚠️ 可行但紧（占 86%） |
| 32 × 128K | 3,004 | 93,875 unit = 91.6 GiB | **636.4 GiB** | ❌ 超余量 |
| **64 × 128K** | 3,004 | **187,750 unit = 183.4 GiB** | **1,272.9 GiB** | ❌❌ **不可行（超 2.9×）** |

**⇒ ★ "64 并发 × 128K 全量驻留"在 A2 上不可行** —— 不是"没测"，而是**反解 + 实测都指向不可行**
（B 臂在 0.852× 处实测宿主 277.77 GiB；线性外推 64 请求 = 1,272.9 GiB，A2 只有 442 GiB）。
per-group bpc 相对现状（`019` §6 的 1,333 GiB）已降 **4.2×**，但离 442 GiB 还差 2.9×。
要跑 64×128K，只能叠加低精（`020` 的 KV8）或**共享前缀**（§6.4）。

### 6.2 ★ A2 上线对照表（与其它线叠加；数字全部用 §3 的 23.5 与 ×1.2）

| 组合 | 16×128K 需求（记账） | **宿主实占** | A2 余量 442 GiB |
|---|---:|---:|---|
| **只挂 L5（per-group bpc）** | 45.8 GiB / **55.0 GiB**（×1.2） | **318 / 382 GiB** | ⚠️ ×1.2 占 86%，**没有余量给别的租户** |
| **L5 + L1**（L1 实测 1.96× 降幅） | 23.4 / 28.1 GiB | **162 / 195 GiB** | ✅ 宽裕 |
| **L5 + L1 + KV8 + ring16**（3.75×） | 12.2 / 14.7 GiB | **85 / 102 GiB** | ✅✅ 很宽裕 |

> **⇒ 建议：A2 长上下文按 "L5 + L1 起步、KV8 作为容量加倍项"，而不是 "L5 alone 硬扛"。**
> （L5 alone 的 ×1.2 = 382 GiB 能用，但它把 86% 的宿主余量吃光，而 A2 上还有别的租户。）

### 6.3 建议参数（A2，**只挂 L5** 的那一档）

| 参数 | 值 | 依据 |
|---|---|---|
| `kv_connector_extra_config.blocks_per_chunk` | `{"default":8,"swa":1}` | 【实测】8 卡 per-group 生效（§2） |
| `cpu_bytes_to_use`（= `OFFLOAD_GB`） | **56 GiB**（= 57,344 unit） | = 16×128K 实测需求的 **1.193×**；宿主 **380.4 GiB**（【实测】×6.944） |
| `--max-model-len` | **131072** | 长上下文目标场景 |
| `--max-num-seqs` | **16** | 池按 16 并发配；>16 并发会级联归零（§3.3） |
| 服务器 | 必须挂 **合并版 `scheduler.py`** + `config.py`/`spec.py`/`pgp_manager.py`（§1.1） | 三补丁共存【实测】 |
| **上线监测** | `vllm:kv_offload_block_removed_total{medium="CPU"}` **必须为 0** | 【实测】欠配的证据是"淘汰 > 0 + 命中 0"（§3.3） |

#### 6.3b ★ `56` 这个数的**推导起点**（三个数并排，供复核）

```
OFFLOAD_GB     = 56 GiB ÷ 1 MiB/unit = 57,344 unit          ← 配置值
需求（实测）    = 16 请求 × 3,004 unit = 48,064 unit          ← §3.1 反解（23.5 unit/1024 token）
余量           = 57,344 / 48,064 = 1.193× ≈ 1.2×             ← 唯一要抄的数字
宿主实占       = 57,344 unit × 7,281,664 B = 417.6 GB = 389.0 GiB（【实测】系数）
（同口径实测臂 B2 = 57,856 unit ⇒ 392.35 GiB，逐字节一致）
```

#### 6.3c ★★ A2 的两个档位：**满配 56 vs 保守 48**（**保守档我给不出实测支持，必须说明**）

| 档 | `OFFLOAD_GB` | unit | 相对实测需求 | 宿主实占 | 占 A2 余量 442 GiB | 实测状态 |
|---|---:|---:|---:|---:|---:|---|
| **满配（推荐）** | **56** | 57,344 | **1.193×** | **389.0 GiB** | **88%** ⚠️ | **【实测】同口径臂 B2 `removed=0`、14.31×** |
| ~~保守~~ | ~~48~~ | 49,152 | **1.023×** | 333.4 GiB | 75% | ⛔ **【实测·不支持】**：1.000×（臂 A'）**归零**，1.023× 与它只差 2.3% ⇒ **不能推荐** |

⇒ **结论**：`48 GiB` 这个"保守档"**与本轮实测冲突**（同口径的 1.000× 臂 A' 是全场归零的），
所以我**不推荐**它。若上线方必须把宿主压到 75%，**正确做法是先跑一条 52 GiB（1.082×）的臂**确认安全边界，
而不是直接抄 48。**"更小池子"的代价本轮有实测锚点**：

```
0.852×  ⇒ 0×（全场归零）            B 臂
1.000×  ⇒ 0×（全场归零）            A' 臂
1.193×  ⇒ 14.31×、removed=0         B2 臂  ← 目前唯一被实测证明安全且有余量的点
```

**若只要 32K**：`OFFLOAD_GB=15`（15 GiB 记账 ≈ 104 GiB 宿主，需求 11.6 GiB ⇒ **1.29×**），`MAX_SEQS=16`。

### 6.4 ★★ 限定：以上都是**最坏情况**（每请求前缀互不相同）

本节所有池子都按 **16 个请求各有各的 128K 前缀**算。真实 agent 流量是"**共享 system prompt + 变长历史**"，
那时池子要覆盖的是**唯一前缀的总量**，不是 Σ 请求长度。**臂 D 就是这一格的实测**：

| | 16 请求各不同前缀（臂 B） | **16 请求共享同一 128K 前缀（臂 D）** |
|---|---:|---:|
| 每请求 key / unit | 128 / 3,004 | **3,004（唯一前缀只有 1 份）** |
| 1.000× 池 | 48,064 unit = 45.8 GiB | **3,517 unit = 3.35 GiB**（实测 3,072 unit 时 `usage=0.894`） |
| 宿主实占 | 318 GiB | **20.83 GiB**（实测） |
| 比值 | 1× | **≈13.7× 小** |

⇒ **共享前缀把 1.000× 池从 318 GiB 压到 23.8 GiB（≈13.4×）** ⇒ 真实 agent 场景在 A2 上**非常宽裕**。

**★ 这条限定必须随数字一起引用**：本节的 1.000× 池是**最坏情况**（每请求前缀互不相同）；
真实共享前缀场景下，池子需求 ≈ **唯一前缀总量**，会显著更小。

### 6.5 ★ 关于臂 D（**不是 L6/`P3_sharedregion` 的判据**）

臂 D 的 `replicated_layout=False / num_copies=8`（与所有其它臂一样）⇒ 它**不是 L6 臂**，
它的 20.83 GiB **不是**"8 份副本被消掉"的效应，而是"**16 请求共享同一前缀 + 小池**"的效应。
它与 `logs/032`（`P3_sharedregion`：L6 暂不推进）**不矛盾，也不构成支持或反对**；
它的价值只在"共享前缀的池子账"这一格。

---

## 7. 第 5 步：cannbot 对照（`a2/AGENTS.md` §6 强制）

本轮**没有写算子/kernel、也没做量化数值验证**（`ops/*` 与 `model-infer-quantization/*` 不适用），
相关的是 **`model/model-infer-kvcache/SKILL.md`**（与 `021` §7 同一节；索引见
`layer_bench/report/cannbot-layer-guide.md` 与 `engram_ref/wtgraph/docs/CANNBOT_TUNING_NOTES.md`）：

| 问题 | cannbot 的回答 | 本轮处置 |
|---|---|---|
| 有没有 per-group `block_size`？ | 没有。`SKILL.md:102-110` 把分页注意力写成**单一** `block_size`，物理 slot = `block_table[..] × block_size + seq_pos % block_size`；§1.1（`:38-52`）只按 `attn_type` 分组。 | **不冲突**：GPU 侧 `block_size`/`block_table`/slot 映射一个字节没动；改的是**卸载池里"一个池条目覆盖几个 GPU block"**，属 vLLM `kv_offload` 层。 |
| 滑窗正确性谁负责？ | `SKILL.md:241`：长序列 `KV_len > sliding_window` 的正确性**必须靠模型层**（环形 buffer 或截断 `actual_seq_lengths_kv`），**不是 op 层**。 | 与 `017` §7.1 一致：DSV4.1 的 SWA 是"**整序列 KV + 窗口化注意力**"⇒ 每请求会产出 32/128 个 SWA chunk——这正是"细粒度后只有 1/N 有用"的由来，也是 `is_store_reachable_swa_chunk()` 存在的原因。 |
| 推荐的替代布局/多级 block？ | 只有一句高阶提示（`SKILL.md:426`）：CPU-GPU Offload 用 `torch_npu.empty_with_swapped_memory` + 异步双流。 | **没采纳**（A2 侧已用 `aclrtHostRegister` 走通，见 `logs/014`）。它印证"host 池 + 异步双流"是 CANN 常规形态。 |

**汇总结论**：cannbot 的 `model-infer-kvcache` **既不支持也不禁止**本改动（它讲模型/算子层的 PA 布局，
不讲卸载池的保留策略）；本轮**没有采纳任何写法改动**。

---

## 8. ★ `sha256(fill) vs sha256(replay)` 在"变长回放"下**没有判别力**

### 8.1 现象

臂 B/C/D 的 `replay_matches_fill_sha256 = False`、`sha256_mismatched_prompts = 16/16`。

### 8.2 为什么这是**必然的，不是 bug**

B/C 的回放只喂了**前 65536 token**（`--replay-prompt-tokens 65536`），模型看到的上下文短了一半，
**下一个 token 本来就该不同** ⇒ 这个比较在"变长回放"口径下**没有判别力**。

### 8.3 正确的判据（`021` 用的就是这个）

```
replay(65536 前缀, 走 DRAM 命中)  vs  cold(65536 前缀, 冷算)
   两者 sha256 相同 ⇒ 语义正确；不同 ⇒ 集成/bug
```

### 8.4 ★ 另一个必须报的信号：**`空文本` vs `空格`**

臂 D（16 请求共享同一 128K 前缀）：

| 轮 | 逐 prompt 输出 | 说明 |
|---|---|---|
| fill（冷算，`max_tokens=1`） | **16/16 空文本**（`out_sha256 = e3b0c442…` = sha256("")） | ttft 也为空（客户端只在拿到非空 text 时记 ttft） |
| replay（DRAM 取回） | **16/16 `" "`**（`d363ad4c…` = sha256(" ")） | |

### 8.5 ★★ 判别结果（【实测】）：**这个服务本身不确定** ⇒ 本节所有 sha 判据**整体失效**

`COLD1`（16×64K、单轮冷算、`max_tokens=1`）落地的 **token 级证据**（`out/l3-cold1-65536/l3-cold1-65536.diag.json`）：

```
同一服务、同一 prompt（65536 token）、背靠背：
  mt1_a : text_repr=' '   sha=36a9e7f1…   finish_reason=length
  mt8   : text_repr=' ~~~~~_~_~_'
  mt1_b : text_repr='_'   sha=d2e2adf7…   finish_reason=length   ← ★ 同一个 prompt，紧接第二次
```

* **不是 EOS 渲染口径**：`finish_reason` 一律 `length`，两次都是 1 个 token，**就是 token 不同**（`' '` → `'_'`）；
* 进一步的 2×2/三格实验（`logs/037`）表明：**即使每次先 `/reset_prefix_cache`**，32K 4 次里仍有 2 个不同、
  512 短 prompt 4 次里有 **3 个不同** ⇒ **连"冷算 vs 冷算"都不可复现**。

⇒ **结论**：`fill vs replay`、`cold vs replay`、**甚至 `cold vs cold`** 的 sha 比较在本服务上**都不可用**。
⇒ **"某条臂 sha 相同"不能证明取回正确，"sha 不同"也不能证明取回错误。**
⇒ `DRAM 取回数值正确性` 因此**仍是【未确认】**（既没被推翻、也没被洗清）；
要判它必须换**不经过模型**的判据：★ **KV 级逐字节比对**
（`agents/L3_8card/scripts/kv_bytecheck.py`：把池里那一行与 GPU 侧对应 block 在 store 时刻的内容逐字节比）。
**本轮没跑成（卡时用尽）**，标【未确认 → 下一步】。

---

## 9. 未确认 / 风险

| 项 | 状态 |
|---|---|
| **DRAM 取回的数值正确性** | **【未确认·最高优先级】**见 §8.4。 |
| 交叉路径的确定性 | **【实测·不确定】**见 §8.5 与 `logs/037`：清缓存后仍抖（32K 2/4、512 **3/4**）⇒ sha 判据全部失效。 |
| **KV 级逐字节判据** | **【未确认·下一步】**`agents/L3_8card/scripts/kv_bytecheck.py` 已写好（挂 `sitecustomize`、hook worker 的 store/load、分块采样指纹），**本轮没跑成**（卡时用尽）。 |
| 23.5 unit/1024tok 的适用范围 | 【实测·两点外推】32K（758/请求）与 128K（3,004/请求）**两档**；**其它长度、其它 `max_num_batched_tokens`、其它 batch 大小下没测**（模型预测它与批次边界有关）。 |
| 共享前缀的省法 | 【实测·单点】臂 D 在 3,072 unit（0.874×）下 `usage=0.894`、`hits>0`、`CPU→GPU=2.83 GB`；**但它同时是 §8.4 那个信号的来源**，结论要等 cold 判完。 |
| `window` 规则的"归零" | 【实测·不复现】见 §5.3：8 卡上是"部分命中"（2.79×）而不是 0。 |
| 1.000× 的余量 | 【实测·反例】A/A'/B 三条**接近 1.000× 的臂全部级联归零** ⇒ **上线必须 ≥1.2×**（比 `013`/`021` 的结论更严）。 |
| `concurrency > 1` | 【未确认】全程 `concurrency=1`。 |
| 淘汰路径在大池下的压力 | 【实测·弱】A'（7,892 次淘汰）与 B（3,038 次）都没崩，但没做"远超容量"的压测。 |
| 非 1024 整数倍的前缀命中长度 | 【推断】命中长度受 full 组（1024）约束；`hits` 数字与之一致，**没**专门扫。 |

---

## 10. 复现

```bash
# 0) 生成挂载用的 4 个文件（本机，不占卡；内含 D2/021 的 md5 指纹断言）
python3 a2/agents/L3_8card/patch/build_patch.py

# 1) A3：把 per-group 挂载点接进影子包（幂等；只改 shadow-pkg/scripts/serve_a2.sh，已备份）
bash ~/projects/dsv41-upstream-pr/agents/L3_8card/scripts/patch_serve_a2_l3.sh

# 2) 不占卡探针（c0 探针锁 ~10 s）
bash tools/a3_chip.sh c0 --timeout 300 --name l3-probe -- \
  env L3_PATCH_DIR=/work/agents/L3_8card/patched python3 /work/agents/L3_8card/scripts/probe_l3.py

# 3) 一条臂（~10–13 min：起服 ~5–6 min + 压测）
TAG=l3-b2-16x128k-12x PROMPTS=16 PROMPT_TOKENS=131072 REPLAY_PROMPT_TOKENS=65536 \
  MAX_LEN=133120 KV_MEM_BYTES=4294967296 OFFLOAD_BYTES=60666413056 ENGRAM=0 SWA_TRIM=off \
  BPC_JSON='{"default":8,"swa":1}' bash agents/L3_8card/scripts/run_arm_l3b.sh

# 4) 压行 / 算账 / 反解
python3 agents/L3_8card/scripts/summarize_l3.py --dir agents/L3_8card/out/<tag> <tag>
python3 agents/L3_8card/scripts/model_units.py
# 反解每请求 unit：把 lookup-summary 的 tally 相邻两行做差（§3.1）
```

**开关**：`BPC_JSON`（标量 8 = 旧口径；`{"default":8,"swa":1}` = per-group）、`SWA_TRIM=off|window`、
`REPLAY_PROMPT_TOKENS`、`SHARED_PREFIX=1`、`DIAG=1`（token 级诊断）、
`OFFLOAD_BYTES`（记账字节；unit 数 = `OFFLOAD_BYTES / 1,048,576`）。

---

## 11. 产物清单

| 文件 | 作用 |
|---|---|
| `a2/agents/L3_8card/patch/build_patch.py` | 生成 4 个挂载文件（含指纹断言 + 语法/锚点自检） |
| `a2/agents/L3_8card/patched/{scheduler.py,scheduler_window.py,pgp_manager.py,offloading_config.py,cpu_spec.py}` | 实际挂进容器的文件（`scheduler_window.py` = `SWA_TRIM=window` 对照版） |
| `a2/agents/L3_8card/scripts/patch_serve_a2_l3.sh` | 幂等地把两个挂载点接进 `shadow-pkg/scripts/serve_a2.sh` |
| `a2/agents/L3_8card/scripts/run_arm_l3.sh` / `run_arm_l3b.sh` | 单臂运行器（8 项补丁自检 → 压测 → 宿主池实占求和；b 版多一个 `DIAG` 钩子） |
| `a2/agents/L3_8card/scripts/{probe_l3.py,summarize_l3.py,model_units.py,diag_tokens.py}` | 不占卡探针 / 压行 / 算账 / token 级诊断 |
| `a2/agents/L3_8card/scripts/{chain_cd.sh,chain_a2.sh,chain_cold2.sh,chain_b2.sh}` | 臂串行器 |
| `a2/agents/L3_8card/bench/kv_offload_client.py` | 压测客户端（新增 `--shared-prefix`、`finish_reason`/`usage`/逐 token 文本/`text_repr`） |
| `a2/agents/L3_8card/bench/kv_events_probe.py` | KV 事件探针（与 021 逐字同源） |
| `a2/logs/raw/027-l3-8card/` | **各臂的原始产物**（meta/keylines/metrics/kv_events/client.json/pool_bytes + 驱动日志 + 探针输出） |

---

## 12. 红线遵守

* 只用 **Phy-ID 8–15**（`locks/c0.lock` 全程持锁；探针走 `tools/a3_chip.sh c0`）；**没碰** Phy-ID 0–7、`dsv41-a3`、`mooncake-master`、`jitpgo-*`；
* 起跑前 `npu-smi info` 确认 8–15 无进程；
* **不手设** `ASCEND_RT_VISIBLE_DEVICES`；不用 `/tmp`（本机 `~/tmp/20260922/l3_8card/`、A3 侧同路径）；
* **没写** `upstream-v41/`；**没改镜像内源码**（全部 `docker -v` 只读挂载）；只改影子包 `shadow-pkg/scripts/serve_a2.sh`（备份 `.L3_8card.bak`）；
* 跨机传文件走 **coscli**（key `share/xfer/027-*`）；ssh 只跑命令；
* 宿主 `MemAvailable` 施工期间最低 **1,210 GiB**（没到 150 GiB 红线）；
* 结论全部标 **【实测】/【推断】/【未确认】**；缺的格子标 `—`。
