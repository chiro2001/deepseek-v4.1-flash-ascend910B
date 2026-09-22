# A2 现在的部署选项（2026-09-22 13:0x）

> **一句话**：**档 B / 档 C / 档 D 的"容量 + 功能三判据"都已在 8 卡真权重上实测通过**；
> ★★ **但有两条挂在台面上的保留意见**（2026-09-22 14:2x 新增第 2 条，**正在定性**）；
> **唯一的阻塞（要用户做的那件事）仍然是 A2 本机的池后端探测**（§0）。
> 标记：**【实测】/【推断】/【未确认】**。
>
> ### ⚠️⚠️ 两条保留意见（**不藏，直接放最前面**）
> 1. **档 D 的接受率样本不足**（`max_tokens=1`、`Drafted` 分母还不同）⇒ 见 §3 的说明与已排的三臂判决实验；
> 2. ★★ **"取回后输出与全量重算逐字相同"这条加强判据，在 8 卡真权重上出现了 2/16 prompt 的差异**
>    （热臂**自身可复现** 16/16；冷参考 `CPU→GPU=0/hits=0`，自证成立；fill 全网一致）
>    ⇒ **正在用"档 B（BF16 无损池）的 hot vs cold"做判决**（详见 `docs/INT8-CASE.md` §4.1）。
>    ★ **目标要求的三判据（`BlockStored>0`/`CPU→GPU>0`/`hits>0` + replay≪fill）不受影响**；
>    但**对外口径必须降级**：在定性之前，**不要再写"取回与重算逐字一致"**。

---

## ★★ 三条命令走完（在 A2 上照抄即可）

```bash
# ① 池后端探测（不占卡、不加载模型；服务在跑也不用停）—— **唯一的阻塞**
A2_CONTAINER=dsv41-a2 A2PROBE_FLOOR_GIB=300 LIGHT=1 bash a2/scripts/a2_one_shot_probe.sh
#    ★ 看 `★ 注册内存的设备往返判据 = True/False`（H2H 通过不算数）

# ② 造 shadow-pkg（在 A2 本机；不依赖任何开发机）
PKG=<dsv41-release 路径> DST=$HOME/shadow-pkg bash a2/scripts/make_shadow_pkg.sh

# ③ 干跑 → 起服
DRY=1 SHADOW_PKG=$HOME/shadow-pkg MODEL=<模型目录> bash a2/scripts/serve_a2_offload.sh
SHADOW_PKG=$HOME/shadow-pkg MODEL=<模型目录> OFFLOAD_GB=56 MAX_LEN=131072 MAX_SEQS=16 \
  NPU_OFFLOAD_HOST_MEM=registered bash a2/scripts/serve_a2_offload.sh
```
★ **这三条已在发布包布局下从 GitHub 全新 clone 验过一遍**（档 B 与档 C 两条路径都走通、发布仓零污染）—— 见 `logs/055` §5.0。

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

## 4. ②c —— draft block 128→64（保留投机、无精度风险，**在验**）

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
