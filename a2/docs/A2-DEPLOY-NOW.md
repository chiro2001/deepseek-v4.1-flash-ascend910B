# A2 现在的部署选项（2026-09-22 11:2x）

> **一句话**：**档 B（现状）与档 C（int8 省 47 GiB 内存）都是"实测可上线"的**；
> 档 D 与 ②c 还在验。**唯一的阻塞仍然是 A2 本机的那条探测**。
> 标记：**【实测】/【推断】/【未确认】**。

---

## 0. 唯一的阻塞（只能你在 A2 上做）

```bash
# 在 A2 宿主上（脚本自己 docker exec 进服务容器）——**服务在跑也不用停**
cd <dsv41-release>/a2/scripts
A2_CONTAINER=dsv41-a2 A2PROBE_FLOOR_GIB=300 LIGHT=1 bash a2_one_shot_probe.sh
```
**它回答四件事**：`host_mem_pool` / `pin_memory` 单次 vs 总量 / ★★ **`aclrtHostRegister` 能不能注册** /
★★★ **那块注册内存**能不能**真的走 H2D/D2H**（= β 路线的生死判据）。
**为什么必须做**：A2 的 `host_mem_pool = 0`，且这个模型在 A2 上 **Engram 206 GiB 注册曾失败** ⇒ **A3 全绿不代表 A2 全绿**。
**判读**：脚本末尾自带 `DECISION`。**看的是 `★ 注册内存的设备往返判据 = True`**（H2H 通过不算数）
⇒ 真 ⇒ `NPU_OFFLOAD_HOST_MEM=registered`；假 ⇒ 回落 `pinned` 并重新量池子上限。
**耗时/占用**【实测·A3 同脚本】：`LIGHT=1`（默认）**18 s**、宿主峰值 ≈40 GiB、**显存只用一个 256 MiB 张量**。

> ★★ **"占不占 NPU"（回答"能不能和服务共存"）**：**需要能用上设备**（`aclInit` + `aclrtSetDevice` + 一条 stream）
> —— 这步省不掉；但**不加载模型、不跑算子、不做图捕获、不抢 HBM 的 KV 池**
> ⇒ **可以和正在服务的 A2 共存**（`COPY_GIB=0` 可把显存占用归零）。详见 `logs/052`。
> ⚠️ **旧版脚本已废弃**：它少了 `aclrtSetDevice`（会让 `aclrtMallocHost` 一律报 `107002`，看着像"A2 不能用 pinned"），
> 而且 ctypes + `torch_npu` 同进程**会段错误**且**吞掉全部输出**（`logs/052` §1，A3 实测）。

> ★ 探测**不需要因 int8 改动** —— 池后端（内存 API）与 KV 量化（页几何）是**正交**的两件事。

---

## 1. 档 B —— 现状，已验证

```bash
MODEL=<模型目录> OFFLOAD_GB=56 MAX_LEN=131072 MAX_SEQS=16 \
BLOCKS_PER_CHUNK='{"default":8,"swa":1}' PREFIX_MATCH_UNIT=32 ENGRAM=0 \
NPU_OFFLOAD_HOST_MEM=registered OFFLOAD_SCHED_PATCH=1 OFFLOAD_NPU_WORKER_PATCH=1 \
bash a2/scripts/serve_a2_offload.sh
```

| 判据 | 8 卡真权重实测 |
|---|---|
| replay / fill TTFT | 1,420.6 / 18,202.6 ms = **12.81×**（×1.2 池 ⇒ **14.31×**） |
| `CPU→GPU` | 25.69 GB（208 load job） |
| `hits` | 1,062,400 |
| `BlockRemoved:CPU` | **0** |
| 宿主实占 | **197.21 GiB**（= 24.65 GiB/worker × 8） |
| HBM KV cache | **427,643 token** |

**保留投机解码**（`--speculative-config` dspark，接受长度中位 3.58）、**图模式可用**。

---

## 2. ★★ 档 C —— int8 省 47 GiB 内存（**8 卡实测通过；★ 新 md5 上待一条复跑**）

> ⚠️ **上线前必读（2026-09-22 12:0x）**：下面这些实测是在 **md5 `1cc9e992…`** 上得到的；
> 而当前发布件是 **`94aeebb7…`**，它除了修档 D 的 segfault 之外，**还改了 `_kv8_graph_rows_bound`
> —— 那正是档 C 走的窗口面**。⇒ **档 C 在 `94aeebb7…` 上还需要 `sg-c-c-graph-b` 一条复跑**
> （判据 = `EE1016=0` + `replay1 sha` 与 eager 逐字相同）。复跑通过前，**档 C 按"待复跑"对待**。
> 详见 `patches/kv8-graphsafe/README.md` §3.0「每条实测对应哪个 md5」。

```bash
# 在档 B 之上：
KV8_SWA=1 KV8_RING_FP16=1        # ← 脚本会自动置 APC_ALIGN=3 与 GRAPH_SAFE=1
# ★ 前提：挂上 patches/kv8-graphsafe/dsa_v41.py（md5 94aeebb757d6d5708268754481a05e0a）
bash a2/scripts/serve_a2_offload.sh
```

| 判据 | 8 卡真权重实测（`FULL_DECODE_ONLY`） |
|---|---|
| 起服 | ★ `EE1016 = 0`、`capture failed = 0`、就绪 659 s、`static_kernel` 无降级 |
| **HBM KV cache** | **427,643**（与档 B **逐字相同** ⇒ 容量零退化） |
| **宿主实占** | ★ **150.01 GiB**（= 18.75 GiB/worker × 8）⇒ **×1.3146，省 47.20 GiB** |
| `BlockStored:CPU` | 29,436（逐字相同） |
| `CPU→GPU` | 21,188,968,448 B = **21.19 GB**（> 0 ⇒ 真命中） |
| `hits` / `queries` | **901,120 / 3,145,984** |
| replay vs fill | **1,608.2 vs 19,880.0 ms = 12.36×** |
| `BlockRemoved:CPU` | **0** |
| `fill sha` | `d524172f9f5ae368…`（与档 B **逐字相同**） |
| ★★ `replay1 sha` | `bc2e797ab069f09ced…`（**与档 C-eager 逐字相同** ⇒ 图模式 = eager） |

**代价**：int8 的 **+1.3~1.8%** decode 时延（`logs/028`/`034` 的区间）。
★ **它保留投机解码、图模式可用** ⇒ **是当前"性价比最高"的一档**。

**为什么 A2 上 HBM 容量不涨（×1.0000）**：A2 的池子实际是 **"3 个投机解码窗口页 + 1 个 long-KV 页"**
（`540,928 = 3×131,072 + 147,712`），而 draft 的窗口面**硬卡 BF16**（源码 `DeepseekV41DraftSWASpec.__post_init__`）
⇒ int8 只能压得动第 4 页。**tiny 没有 draft 组（`num_nextn_predict_layers=0`）⇒ 所以 tiny 上是 ×1.4655**。

---

## 3. 档 D —— ×1.1356 HBM 容量（**待验**）

```bash
KV8_SWA=1 KV8_RING_FP16=1 KV8_FULL=1 KV8_PREFILL=1    # 同理自动置 APC_ALIGN/GRAPH_SAFE
```
| 判据 | 状态 |
|---|---|
| HBM KV cache | ★ **485,610**（×1.1356）【实测·图模式】 |
| 图模式四条判据 + J2 | ⏳ 待跑（`sg-a-d-graph` 排队） |
| ⚠️ **"静默读错"风险** | 【实测·机制】`[SG-PPR]` 证明捕获期 `cmp_mcs=6 ⇒ ppr=1`，而 replay 的压缩前缀有几百块；【未确认·后果】越界读在**算子层**最可能静默错（torch 层已实测是**响亮失败**） |

⇒ **档 D 的修法已在同一份 `dsa_v41.py` 里**（`_kv8_cmp_plane` 的 graph_safe 分支），**但需要 `sg-a-d-graph` + 决策臂 `sg-a-d-cmplegacy` 确认。**

---

## 4. ②c —— draft block 128→64（保留投机、无精度风险，**在验**）

| | 值 |
|---|---|
| 收益 | ★ **HBM ×1.8177**（**777,318** token，相对档 B 427,643） |
| 保留投机解码 | ✅ |
| 精度风险 | **无**（不改 dtype，draft 仍 BF16） |
| 改动面 | ★ **2 文件 / 2 处，默认关**（`dspark.py` 给 draft 自己的 `block_size`（env `VLLM_V41_DRAFT_BLOCK`）+ 放宽 `plan_cache_slots` 的**相等**检查为**整除**检查）—— 详见 `logs/051` |
| 已落地的证据 | ★ **`64` 档本来就在算子的块大小表里**（`_DSV4_BLOCK_SIZES[64][0][1] == 64`、`page_size_padded_t2 == 65,536`）；**三臂对称单元自检全绿**（`upstream` raise / `draftaware` ×1.0000 与 8 卡逐字同 / **`patched` 档 C 369,280、档 D 282,880**） |
| 已查清的风险 | 窗口跨 3~4 块（算子/块表/KV manager **都无假设**）；⚠️ 唯一硬编码 `kv8_ori_plane` 的 `pages_per_req=2` **只在 int8 平面上跑 ⇒ ②c 不走它** |
| 副作用 | DRAM 池 `sw_chunks` 1→2 ⇒ 该组每段 unit 2→3 ⇒ 总需求 **+5.0%**（`OFFLOAD_GB=56` 要复算） |
| ⏳ 还差什么 | **一条真权重端到端臂**（判据：容量 595,404（档 C）/ 777,318（档 D）+ 图捕获成功 + **`SpecDecoding` 四项不降** + sha 与冷算参考一致） |

---

## 5. 落地顺序建议

> ★★ **硬约束（用户决策 2026-09-22）：保留投机解码。** A2 是单流场景，DSpark 的收益不可替代
> ⇒ **⑤a（关投机换容量）已否决**；下表每一档、每一条待验路线都在 `--speculative-config dspark` 下成立。

```
1. ★ 先跑 A2 的探测（a2_one_shot_probe.sh）⇒ 决定池后端
2. ★ 上档 B（已验证）⇒ 拿到"16×128K + 12.8~14.3× 加速"
3. ★★ 再上档 C（已实测）⇒ 池从 197 GiB 降到 150 GiB，代价 +1.3~1.8% 时延
4. ⏳ 等 ②c / 档 D 的结论 ⇒ 若成立，HBM 容量再 ×1.82 / ×1.14
```

**回滚**：所有能力都在 `PYTHONPATH` / `docker -v` 挂载层 ⇒ **去掉挂载即回现状**；
逐条回滚 = 对应 env 置 0（`KV8_*=0` / `APC_ALIGN=0` / `GRAPH_SAFE=0` / `P2_POOL_PATCH=0`）。
**没有一处写镜像。**
