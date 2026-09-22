# A2 现在的部署选项（2026-09-22 15:3x）

> **一句话**：**档 B / 档 C / 档 D 的"容量 + 功能三判据"都已在 8 卡真权重上实测通过**；
> ★★ **但有两条挂在台面上的保留意见**（2026-09-22 14:2x 新增第 2 条，**正在定性**）；
> **唯一的阻塞（要用户做的那件事）仍然是 A2 本机的池后端探测**（§0）。
> 标记：**【实测】/【推断】/【未确认】**。
>
> ### ⚠️⚠️ 两条保留意见（**不藏，直接放最前面**）
> 1. **档 D 的接受率样本不足**（`max_tokens=1`、`Drafted` 分母还不同）⇒ 见 §3 的说明与已排的三臂判决实验；
> 2. ★★★ **已定性（2026-09-22 14:5x）：那 2/16 的差异不是 int8 缺陷，是「判据判别力不足」**
>    —— **决定性证据：档 B（BF16 无损池，四个 int8 开关全 0）的热臂，给出的退化 token 与档 C（int8）逐字相同**
>    ⇒ 差异来自**热/取回路径本身**；加上 `max_tokens=1` 只生成 **1 枚 token**、候选是**近平局**（全是空白类 token）
>    ⇒ ★ **「逐字相同」这条加强判据在该口径下不成立是正常的**（详见 `logs/062`）。
>    ★ **目标三判据不受影响**（档 C **12.50×** / 档 D **12.87×**）。
>    ⚠️ **但仍不许外推成「int8 已证明保真」** —— **正面判据（KV 级逐字节）仍未跑**；
>    且 **档 D 在 prompt 9 上留一格【未确认】**（`' '` vs 档 B/C 的 `'_'`）。

---

## ★★ 三条命令走完（在 A2 上照抄即可）

```bash
# ① 池后端探测（不占卡、不加载模型；服务在跑也不用停）—— **唯一的阻塞**
A2_CONTAINER=dsv41-a2 A2PROBE_FLOOR_GIB=300 LIGHT=1 bash a2/scripts/a2_one_shot_probe.sh
#    ★ 看 `★ 注册内存的设备往返判据 = True/False`（H2H 通过不算数）

# ② 造 shadow-pkg（在 A2 本机；不依赖任何开发机）
PKG=<dsv41-release 路径> DST=$HOME/shadow-pkg bash a2/scripts/make_shadow_pkg.sh

# ③ 干跑（**会打印真实挂载清单**）→ 起服
DRY=1 SHADOW_PKG=$HOME/shadow-pkg MODEL=<模型目录> KV8_SWA=1 KV8_RING_FP16=1 \
    bash a2/scripts/serve_a2_offload.sh
#    ★ 看 `[a2-dry] MOUNTS(NN):` 里有没有那 **7 个 kv8-int8-pkg 件**（档 C/D 的必需件）

SHADOW_PKG=$HOME/shadow-pkg MODEL=<模型目录> OFFLOAD_GB=56 MAX_LEN=131072 MAX_SEQS=16 \
  NPU_OFFLOAD_HOST_MEM=registered KV8_SWA=1 KV8_RING_FP16=1 \
    bash a2/scripts/serve_a2_offload.sh
```
★ **这三条已在发布包布局下从 GitHub 全新 clone 验过**（档 B / 档 C 两条路径都走通、**dry-run 能打出真实挂载**、发布仓零污染）—— 见 `logs/055` §5.0 / §5bis。

> ### ★★★ 2026-09-22 15:3x：**档 C/D 的挂载件从 1 个变成 7 个**（一个已修的交付缺口）
> 此前文档只写"挂 `kv8-graphsafe/dsa_v41.py` 就能起档 C" —— **那是错的**：
> 档 C/D 实跑时挂的是 **7 个整文件**，而发布包里原本**只有 1 个**。
> ⇒ 已新增 `patches/kv8-int8-pkg/`（6 个整文件 + README，md5 与 `arm.out` 台账**逐字相同**），
> 并由 `make_shadow_pkg.sh` 在检测到 `A2_KV8` / `A2_KV8_SWA` / `A2_RING_FP16` 时**自动挂上 7 个**，
> **缺一个就 die**（不静默降级成档 B）。详见 [`logs/055`](../logs/055-a2-launch-path.md) §5bis。

---

## ⚠️ 一条贯穿全部结论的**外推**（必须先说清）

★ **本文件、`DELIVERY.md`、`logs/*` 里的所有「实测」数字，全部来自 A3**
（`A3-node1` 的 **Phy-ID 8–15**，**8 × 910C**）—— 而目标是 **A2（8 × 910B3）**。

| 维度 | 为什么可以外推 | 为什么**不能**想当然 |
|---|---|---|
| **容量算术**（`GPU KV cache size`） | 它只由**页几何**（`Σslot_pages` / BPR / `avail`）决定，**与芯片无关** | ⚠️ `avail` 取决于**每卡可用显存**，而 B3 与 C 的 HBM 容量/驱动占用**不一定相同** ⇒ **`427,643` 这类绝对数要重测** |
| **卸载功能的四条判据** | 是**卸载层**的行为，与芯片无关 | ⚠️ 池后端不同（A2 `host_mem_pool=0`）⇒ **`P1_pinned` 那条路径必须重跑**（这正是 §0 探测要回答的） |
| **int8 的页几何**（`page_bytes=66,560` 等） | 纯算术 + 已由**单元自检**验证（`051` 三臂） | ⚠️ **算子是否支持**是该芯片的 kernel 属性 —— `015` 已实测「TND KV 在 arch22 没 kernel」，同类假设在 B3 上要重验 |
| **图模式兼容性**（`EE1016=0`） | 判据是**软件路径**（host 标量 / D2H），与芯片无关 | ⚠️ capture 的具体行为是**驱动/CANN** 的事 ⇒ A2 上第一次起图模式**仍要盯 `EE1016`** |
| **性能数字**（`12.50×` / `ms/step`） | 趋势可借 | ⛔ **绝对数不可借** —— A2 是 910B3（设备更慢、HCCS 带宽更低、TP8 通信更贵）⇒ **必须在本机重测** |

⇒ ★★ **正确的读法**：**「能力」（能不能跑通、页几何对不对、功能判据成不成立）可以借；
「数字」（多少 token、多少 ms、多少倍）必须在本机重测。**
⇒ ★ 这也是为什么 §0 的池后端探测被列为**唯一的阻塞** ——
它是「能力」层里**唯一一个我们不能从 A3 借的**。

## ★★ 判据账（2026-09-22 16:0x）—— **上线后照着这张表核**

### A. 目标要求的「三判据」（**DRAM 卸载到底有没有生效**）

| # | 判据 | 档 B | 档 C | 档 D | 怎么核 |
|---|---|---|---|---|---|
| 1 | `BlockStored(medium=CPU) > 0`（**能存**） | 29,436 | 29,436 | 29,436 | `curl :PORT/metrics | grep kv_offload_store` |
| 2 | `CPU→GPU` 搬了字节（**能取**） | 21.52 GB | **21.19 GB** | 12.11 GB | `grep kv_offload_load_bytes_total` |
| 3 | `external_prefix_cache_hits > 0`（**真命中**） | 901,120 | 901,120 | 901,120 | `grep external_prefix_cache_hits_total` |
| ★ | **replay ÷ fill**（**取回比重算快**） | 12.87× | **12.50×** | **12.87×** | 两次 TTFT 之比 |
| ★ | `BlockRemoved(medium=CPU) == 0`（**没被踢**） | 0 | 0 | 0 | `grep kv_offload_block_removed` |

★ **全部为 8 卡真权重实测**（`logs/042` / `048` / `050`）；**上线后必须自己再核一遍**，
因为 A2 的池后端（`registered` vs `pinned`）与 A3 不同。

### B. 容量判据（**先记下期望值，再对比**）

| 档 | HBM `GPU KV cache size` 期望 | 宿主实占期望 | 依据 |
|---|---:|---:|---|
| 档 B | **427,643** | **197.21 GiB** | `logs/042` |
| 档 C | **427,643**（★ 与档 B 相同 —— **容量不涨是正常的**，收益在宿主） | **150.01 GiB** | `logs/048` |
| 档 D | **485,610** | **144.63 GiB** | `logs/050` / `R_8card_int8` |
| ②c（预测） | **777,318** | 待测 | `logs/051`（**8 卡端到端在跑**） |

> ⚠️ ★ **最容易误判的一格**：**档 C 的 HBM 容量与档 B 逐字相同**（427,643）——
> 因为 slots 0–2 的 binding 是 **draft 组（BF16）**，int8 只压得动第 4 个 slot。
> ⇒ **别拿「容量没变」当「int8 没生效」**（`053` 专门记过这个陷阱）。
> ★ int8 的收益要**看宿主内存**（197.21 → 150.01 GiB）或 `[R8-SLOTS]` 的 slot 数值。

### C. 服务健康判据（**起服后先看这三条**）

```bash
grep -c 'P1_pinned.*ret=0'            <serve.log>   # 期望 8   ← 池后端生效
grep -c 'D2_offload'                  <serve.log>   # 期望 >0  ← 卸载层装载
grep -c 'alignment_chunk_count.*8'    <serve.log>   # 期望 >0  ← per-group bpc 生效
grep -c 'P2_poolsizing'               <serve.log>   # 期望 >0  ← L1 生效
```
★ 任一为 0 ⇒ **停，别压测**（脚本末尾也会打印这几条）。

### D. ★ 提交前的最后一道（**2026-09-22 新增**）
```bash
DRY=1 SHADOW_PKG=$HOME/shadow-pkg MODEL=<模型目录> KV8_SWA=1 KV8_RING_FP16=1 \
    bash a2/scripts/serve_a2_offload.sh | grep -E 'a2-dry.*MOUNTS'
#   ★ 档 C/D 必须含那 7 个 kv8-int8-pkg 件（MOUNTS 条数约 24）
```
⇒ 这一道能挡住「文件没进包」那一类缺口（本轮就抓到过三个）。

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

## 0b. ★★ 第二步：造 shadow-pkg（**此前这一步会卡住**）

`serve_a2_offload.sh` 依赖 **shadow-pkg**，而它原来**只存在于开发机**（`~/projects/dsv41-upstream-pr/shadow-pkg`，
手工改出来的、**从没进过发布包**）⇒ 探测即使全绿，**第二条命令也会立刻打印「⚠ 找不到 shadow-pkg」并退出**。

```bash
# 在 A2 本机从本仓库自己造（不依赖任何开发机）
PKG=<dsv41-release 路径> DST=$HOME/shadow-pkg bash a2/scripts/make_shadow_pkg.sh
# 干跑确认参数（不起服务）：
DRY=1 SHADOW_PKG=$HOME/shadow-pkg MODEL=<模型目录> bash a2/scripts/serve_a2_offload.sh
```

生成器做 **5 处精确锚点插入**（锚点必须恰好命中一次，否则 **fail-closed 且不落盘**）+ 4 条 grep 自检；
**不写 dsv41-release 一个字节**（已实测）。详见 `logs/055-a2-launch-path.md`。
★ **注意**：这份 shadow **与开发机上那份不等价**（开发机还含别的任务的注入块）
⇒ **不要拿开发机的 arm 结论直接套 A2 的 shadow**（见 `patches/ARTIFACT-IDENTITY.md`）。

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

## 2. ★★ 档 C —— int8 省 47 GiB 内存（★ **8 卡实测通过，发布件已复跑**）

> ✅ **2026-09-22 13:0x 解除警示**：此前那条"新 md5 上待复跑"**已完成** ——
> `sg-c-c-graph-b`（档 C 图模式，8 卡，md5 **`94aeebb7…`**）**全绿**，且与 `22cbf20c` 那轮**逐字节相同**：
> ```
> 捕获 9/9 [00:54] · EE1016=0 · 容量 427,643（= 档 B）
> fill  sha = d524172f9f5ae368…   ← ★ 与 22cbf20c 那轮逐字相同
> replay1 sha = bc2e797ab069f09ced… ← ★ 与 22cbf20c 那轮逐字相同
> hits 901,120 / load_bytes 21,188,968,448 B / replay 1,594.8 ms vs fill 19,936.0 ms = 12.50×
> ```
> ⇒ **档 C 在发布件上已成立**。详见 `patches/kv8-graphsafe/README.md` §3.0 与 `patches/ARTIFACT-IDENTITY.md` §1.1。

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

## 3. 档 D —— ×1.1356 HBM 容量（★ **已全绿**，2026-09-22 12:3x）

```bash
KV8_SWA=1 KV8_RING_FP16=1 KV8_FULL=1 KV8_PREFILL=1    # 同理自动置 APC_ALIGN/GRAPH_SAFE
```
| 判据 | 状态（`sg-c-d-graph`，8 卡真权重，md5 `94aeebb7…`） |
|---|---|
| 起服 + 捕获 | ✅ 捕获 **9/9 [06:11]**、`/health=200`、`static_kernel` 无降级 |
| ★ 判据 0（致命项） | ✅ `EE1016=0 / Segfault=0 / Engine core init=0 / Worker died=0` |
| HBM KV cache | ★ **485,610**（×1.1356）—— 与 R 的档 D 臂 **逐字相同** |
| 四条判据 | ✅ `CPU→GPU` 12.11 GB>0 / `hits` 901,120>0 / `BlockRemoved:CPU`=0 / replay **12.87×** |
| ★★ 判据④（最强） | ✅ `replay1 sha` 与**同几何 eager 臂逐字节相同**（`8600507eb6b43bfa…`） |
| ★ 越界读是否被消灭 | ✅ `[SG-PPR]` 证明捕获期 cmp 面页数**由 shape 决定**（768 / 1536 页），不再是 `ppr=1` |
| ⚠️ 越界读的后果（更正） | **不是单一形态**：决策臂 `sg-c-d-cmplegacy` 实测到的是**崩引擎**（`507057 SUSPECT REMOTE ERROR`，第一个真实请求即死）；**同几何 A/B**（唯一差别=补丁开关）证明因果。⇒ 正确表述 = **"可能崩、也可能静默算错"** ⇒ **必须验 sha**，不许用"反正会崩"自我安慰 |
| ⚠️ 接受率读数 | 本几何（`max_tokens=1`）档 D `1.00 / Drafted 15`、档 C `1.50 / 10` —— ★ **样本太小，不构成"档 D 降低接受率"的证据**（档 D graph 与 eager 读数**完全相同** ⇒ 与图/补丁无关）。**要判档 D 是否保投机，需要一条专门的接受率 A/B**【未确认】 |

⇒ **档 D 的修法就在同一份 `dsa_v41.py` 里**（`_kv8_cmp_plane` 的 graph_safe 分支），
**已在 8 卡真权重上验完** ⇒ **档 D 可以上**（唯一保留意见是上面那条"接受率需另测"）。

---

## 4. ★★ 两条「解开 draft 天花板」的路 —— **②a 已实测失败，交付推荐是 ②c**

> ### ★★★ 2026-09-22 14:5x：【实测】②a（draft INT8，block **保持 128**）
> ```
> ddi-d-i8-graph（tier D）: capture_finished=1 ee1016=0 not_supported=0 capture_failed=0   ← ★ 图捕获成功
> probe: block=128 dtype=torch.int8 scale_dim=4 page_bytes=66560                        ← ★ draft 页 66,560
> GPU KV cache size = 39,846   ← ★ 与零参数模型的预测【逐字相同】
> ```
> ★★ **算术上 ②a 严格优于 ②c**（两者 Σ 相同 = 282,880，但 ②a 的 BPR 更小 = 2471 vs 2600，
> 因为 **②a 保持 block=128、没有「窗口跨块」的副作用**）：
> ```
> 8 卡预测：②c = 777,318（×1.8177）   ②a = 817,898（★ ×1.9126）
> tiny 实测：②c = 36,825               ②a = 39,846（★ 逐字命中模型）
> ```
> ⇒ **算术上 ②a 是唯一能碰到原始 ×1.84 目标的那条路**（预测 ×1.9126）——
> ★★★ **但 Q3 实测失败，②a 不可用**（`logs/056`）：
> ```
> RuntimeError: The previous device metadata submission has not been released
>   @ worker/device_metadata.py:74（触发形状 num_scheduled_tokens=6 + 5 个 spec token）
> ②a 臂：16/16 请求失败、0 条 SpecDecoding 读数
> 对照臂（同包同参数，唯一变量 DRAFT_INT8=0）：ok=16/16、sha 0ebccb55b30c…（与 054 四臂逐字相同）
> ```
> ⇒ ★★ **②a 特有，不是 harness**。
>
> ★★ **诊断臂进一步定位到「泄漏的那一次提交」**（`056` §4.4b）：
> ```
> submit#1 in_flight=False tasks=7 → release#1 ✅
> submit#2 in_flight=False tasks=1 frontiers=[(2, …)]      → ★★ 无 release#2   ← 泄漏点
> submit#3 in_flight=True  tasks=7（与 #1 逐字相同的 7 个）  → ⛔ 抛 RuntimeError
> ```
> ★ `submit#2` 的形状与其余每次**都不同**（**只 1 个任务**、`group_id` 在 target 的 7 任务提交里**一次都没出现过**）
> ⇒ 来自**另一个 builder（draft 侧）** ⇒ 「**release 缺口在 draft 侧的 execute 路径上**」这条
> **从推断升到有实测支撑**。
>
> ⚠️⚠️ **但「病灶是 `dsa_v1.py` 缺量化存取」这条必须降级**（`D_draftINT8` **主动**提出，主代理采纳）：
> 它的探针钩在 `dsa_v1.py::AscendDSAImpl` 上，**横幅打出来了但从未被调用** ——
> 真身是 `models/layer/attention/layer.py::DSAAttention`（`ops/dsa.py:35`）。
> 而且**首个异常在日志里彻底看不见**（`tuple` / `AttributeError` / `npu_scatter` / `ori_kv` 命中**全 0**）。
> ⇒ **正确的说法**：「要移植量化存取」是**【推断】，不是已证实的病因**；
> 下一个探针 target 应该是 **`DSAAttention`**，**不是** `dsa_v1.py`。
> ⇒ ★★ **别把「病因已证实」写进文档** —— 本条目只到「**②a 在单 die tiny 上不可用**」
> （**足以否决交付选项**），**不到「病因已定位」**。

### ②c 的细节（**仍是 ②a 失败时的回落**，8 卡端到端在 c0 排队）

| | 值 |
|---|---|
| 收益 | ★ **HBM ×1.8177**（**777,318** token，相对档 B 427,643） |
| 保留投机解码 | ✅ |
| 精度风险 | **无**（不改 dtype，draft 仍 BF16） |
| 改动面 | ★ **2 文件 / 2 处，默认关**（`dspark.py` 给 draft 自己的 `block_size`（env `VLLM_V41_DRAFT_BLOCK`）+ 放宽 `plan_cache_slots` 的**相等**检查为**整除**检查）—— 详见 `logs/051` |
| 已落地的证据 | ★ **`64` 档本来就在算子的块大小表里**（`_DSV4_BLOCK_SIZES[64][0][1] == 64`、`page_size_padded_t2 == 65,536`）；**三臂对称单元自检全绿**（`upstream` raise / `draftaware` ×1.0000 与 8 卡逐字同 / **`patched` 档 C 369,280、档 D 282,880**） |
| ★★★ **机制已在 slot 层实测**（单 die，`054`） | 四臂实测 `capacity = max(kv+index, aliases_max, draft)`：<br>• **档 B**：`aliases_max=131072` ⇒ draft 131072→65536 **capacity 纹丝不动（131072）** ⇒ 只拿到"draft 页数 130→259"的副作用 ⇒ **容量降 ×0.9242**<br>• **档 D**：`aliases_max=66560` ⇒ draft 131072→65536 **把 draft 从 binding 位置拉下来** ⇒ **capacity 131072→66560（减半）** ⇒ **容量涨 ×1.5570**<br>★ 且 `d128` 那行自带 `[draft-aware]`（`capacity=131072 legacy=66560`）⇒ **"draft 是 slots 0–2 的 binding 项"在 slot 层直接实测**，不再只是算术推断 |
| ★ 单 die 数值判据（`054`） | **输出**：B/D 各 7 轮 sha 逐字节相同、逐 prompt 16/16、跨臂 16/16；**投机**：两臂 `MeanAccLen 1.685 / AvgDraftAcc 13.69%` 完全一致，**提案序列 4367/4367 逐条相同**；**图模式**：b64 捕获 23 s / b128 9 s，`EE1016=0` |
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
