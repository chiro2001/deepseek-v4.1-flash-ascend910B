# A2（8×910B3）使能：DRAM KV 卸载 + KV8

> **分支**：`feat/kv8-dram-offload-pending`
> **日期**：2026-09-22
> **状态**：★ **档 B（现状）与档 C（int8 省 47 GiB）都已 8 卡真权重实测可上线；档 D 的图模式判据在验**。
> ★★ **唯一阻塞 = A2 本机的一条池后端探测**（见 §0）。

---

## 0. ★★ 先读这一页

**本文正文（§1–§5）已经是当前口径**（2026-09-22 重写）。
下面这张表是给**读过早期版本的人**对的账（早期版本曾写"KV8 不能上线"、"128K×16 要 333 GiB"）：

| 早期版本写的 | ★ 现在正确的 |
|---|---|
| 「128K × 16 并发 ⇒ 宿主 **333 GiB**，在 442 GiB 之内」（§1.5/§2.3/§5） | **档 B（+L1）= 197.21 GiB**【8 卡实测】；**档 C（+int8）= 150.01 GiB** ⇒ 余量从 60% 降到 **34%** |
| 「A2 的长上下文不依赖 KV8」（§2.3） | 仍然成立，但 KV8 **已经不需要"押注"**：档 C 已实测（图模式过、输出 sha 与 eager 逐字相同） |
| 「KV8 当前形态不能上线（时延负收益 +21%）」（§2.2/§2.3） | ⛔ **作废**：+21% 那条来自**未融合**的 rebuild；`logs/026` 融成 1 个 kernel、`logs/049` 修掉图捕获炸点之后，**档 C 在 8 卡真权重上已起服并跑通全部判据**；代价只剩 **int8 的 +1.3~1.8%** decode 时延 |
| 「容量 ×1.135」（§2.2） | **按几何分档写**：tiny 几何 ×1.4655（档 C）/ ×1.9133（档 D）；**A2 真权重**档 C ×1.0000、档 D **×1.1356**（多一个 draft 组顶住，见 `logs/050`） |
| 「`state` 组缩到 BF16 的精度影响未测」（§2.3） | `logs/034`/`045` 已闭合：ring16 本身不是 NaN 源；数值缺陷的**真根因是 APC 命中长度没按压缩比对齐**（`logs/047`，已修） |
| 「并发 × 上下文边界」按 ×6.945 估算（§5） | L1（`P2_POOL_PATCH`）把池子的**结构性浪费从 1.96× 拿回来**（`logs/030`）⇒ 现在的表见 `DELIVERY.md` §4 |
| 「A2 实机验证是最大风险」（§4-1） | **仍然是**，而且已经被收敛成**一条命令**（见下） |

### 0.1 唯一阻塞：A2 的池后端探测（只能在这台机器上做）

```bash
# 在 A2 宿主上（脚本自己 docker exec 进服务容器）——服务在跑也不用停
cd <本仓库>/a2/scripts
A2_CONTAINER=dsv41-a2 A2PROBE_FLOOR_GIB=300 LIGHT=1 bash a2_one_shot_probe.sh
```

### 0.1b ★★ 然后是第二步：造 shadow-pkg（**此前这一步会卡住**）

`serve_a2_offload.sh` 依赖 **shadow-pkg**，而 shadow-pkg 原来**只存在于开发机上、从没进过发布包**
⇒ 池后端探测全绿之后，第二条命令仍会立刻打印「⚠ 找不到 shadow-pkg」。
**现已补上生成器**（在 A2 本机从本仓库自己造，不依赖任何开发机）：

```bash
PKG=<本仓库路径> DST=$HOME/shadow-pkg bash a2/scripts/make_shadow_pkg.sh
# 然后干跑（不起服务）确认参数：
DRY=1 SHADOW_PKG=$HOME/shadow-pkg MODEL=<模型目录> bash a2/scripts/serve_a2_offload.sh
```

生成器做 **5 处精确锚点插入**（锚点必须恰好命中一次，否则 **fail-closed 且不落盘**）+ 4 条 grep 自检；
并且 **不会写 dsv41-release 一个字节**（已实测）。详见 [`logs/055-a2-launch-path.md`](logs/055-a2-launch-path.md)。
* **为什么必须做**：A2 的 `host_mem_pool = 0`，且这台机器上 **Engram 206 GiB 注册曾撞 `207001`**
  ⇒ **A3 全绿不代表 A2 全绿**；它决定池子走 `registered` 还是回落 `pinned`。
* **看哪一行**：`★ 注册内存的设备往返判据 = True/False`（**H2H 通过不算数**）。
* **占不占 NPU**：需要能用上设备（`aclInit`+`aclrtSetDevice`，省不掉），但**不加载模型 / 不跑算子 /
  不建图 / 不抢 HBM KV 池**（显存峰值 = 一个 256 MiB 张量，`COPY_GIB=0` 可归零）
  ⇒ ★ **可以和正在服务的 A2 共存**；耗时【实测·A3】`LIGHT=1` **18 s**。
* 详细说明：[`logs/052-a2-probe-v2.md`](logs/052-a2-probe-v2.md)。

### 0.2 落地顺序

```
1. 跑 §0.1 的探测 ⇒ 定 NPU_OFFLOAD_HOST_MEM
2. 上【档 B】⇒ 16×128K + 重算→取回 12.8~14.3×
3. 再上【档 C】⇒ 宿主 197.21 → 150.01 GiB（图模式可用，输出 sha 与 eager 逐字相同）
4. ⏳ 等 ②c（draft block 64）与档 D 的结论 ⇒ HBM 容量再 ×1.82 / ×1.14
```
★★ **硬约束（用户 2026-09-22）**：**保留投机解码**（A2 是单流场景，DSpark 收益不可替代）；
**"关投机换容量"已否决**，只作诊断/归因臂。

---

## 1. DRAM KV 卸载 —— ✅ **已验证**（8 卡真权重 · 16 请求）

### 1.1 解决的问题

DeepSeek-V4.1 上曾出现 **"能存不能取"**：
`BlockStored(medium="CPU") = 12,288`、`GPU_to_CPU` 搬了 **393.6 GB**，
但 **`CPU_to_GPU = 0`**、**`external_prefix_cache_hits = 0`**、
replay TTFT 4167 ms ≈ fill 4192 ms ⇒ **对服务质量零帮助**。

### 1.2 根因（三条独立机制，缺一条都修不好）

| # | 机制 | 说明 |
|---|---|---|
| **①** | **`state` 组一票否决** | compressor 的 state ring（`prefix_cacheable = False`，每请求只有 1 页）在**取回路径**上仍被当成 full-attention 组参与命中判定，而它在**存储路径**上永远存不出 chunk（`len(block_ids)//blocks_per_chunk = 1//8 = 0`）⇒ 它的 key 必然 MISS ⇒ 上游 `_lookup()` 里 `if num_hit_chunks == 0: return 0` 把**整轮请求**（含 full 组与 40 个 SWA 资源）的命中**全部判死** |
| **②** | **SWA 池条目太粗** | SWA 窗口只有 128 token，而池条目是 1024 token；每请求每 SWA 组存 32 条，其中**30 条永不取回** ⇒ 池子需求被放大 **4.89×**（`logs/021`） |
| **③** | **pinned 池在真实 8 卡上要不到** | `aclrtMallocHost` 单次要 8 GiB 就报 `207001`（宿主 `MemAvailable` 仍有几百 GB）⇒ 池子上限被卡死 |
| **④**（后补） | **`bpc>1` 的组每 chunk 只搬 1 个块** | `scheduler.py::_build_store_jobs` 里 `blocks_per_chunk` **局部变量泄漏**（收集 loop 复用同名变量，spec loop 不再赋值）⇒ 该搬 492 个只搬了 64 个（`logs/043b`，已修 + 端到端验过） |

### 1.3 效果（`logs/016` / `logs/022`，均已复现）

| 判据 | 修复前 | **修复后** |
|---|---|---|
| `BlockStored(CPU)` | 12,288 | **6,144**（= 16 × 384，一轮） |
| **`CPU_to_GPU`** | **0** | **12.44 GB** |
| **`external_prefix_cache_hits`** | **0** | **507,904 / 1,048,832**（**replay 轮 96.9%**） |
| **replay / fill TTFT** | 4167 / 4192 ms（+0.6%） | **253.4 / 4429.4 ms（17.5×）** |

**拐点已验证**（三条臂夹死）：池子 **0.667× 工作集 ⇒ 归零**、**1.000× ⇒ 17.9×**、1.209× ⇒ 17.5×。
★ 但 **1.000× 那格是"贴着临界"**（`logs/040`/`041`）⇒ **`units_ratio` 要留 ≥1.05 的余量**，别按 1.000× 配。

### 1.4 补丁（`patches/`）

| 文件 | 作用 |
|---|---|
| `0001-offload-scheduler.patch.py` | `scheduler.py` 替换版。**含 `state` 组参与位修复 + per-group `blocks_per_chunk` + `bpc` 泄漏修复**（是三者超集，**只需挂这一份**） |
| `0001b-offload-per-group-bpc-manager.patch.py` | `PerGroupBPCManager`（池的格子 = 1 个 GPU block）+ 三种静默失败的 fail-fast 加固 |
| `0001c-offload-per-group-bpc-hooks.patch.py` | 配置解析钩子（`blocks_per_chunk` 支持 `{"default":8,"swa":1}`） |
| `0002-offload-cpu-pool-host-registered.patch.py` | `cpu_npu.py` 替换版：池子改走 `aclrtHostRegister`（绕开 `aclrtMallocHost` 的 `207001`），**注册失败自动回落 `pinned`** |
| `kv8-graphsafe/` | int8（档 C/D）的必需件 —— 见 `patches/kv8-graphsafe/README.md` |

详见 [`patches/README.md`](patches/README.md)（含 md5、挂载方式、**补丁生效自检**、参数定值）。

### 1.5 参数定值（★ 已按 L1 重算）

| 场景 | `cpu_bytes_to_use` | 宿主实占（档 B / 档 C） | 依据 |
|---|---:|---:|---|
| 32K × 16 并发 | 10 GiB | ≈35 / ≈27 GiB | `logs/030`（L1 实测 1.96×） |
| **128K × 16 并发** | **56 GiB** | ★ **197.21 / 150.01 GiB**【8 卡实测】 | `logs/042` / `logs/048` |

★ **两个必须注意的标定**：
* 一个池条目只记 8 MiB 的账（`worker_kv_bytes_per_block = 131,072 B`）；
* **宿主实占 = 记账值 × 常数**（16 张 canonical 张量逐个分配）—— 档 B 是 **3.52×**、档 C（20 张量）是 **2.68×**；
  **L1 那 1.9895× 是把"另一份结构性浪费"拿回来**的（`logs/030`）⇒ 容量规划**必须按宿主算，不能按记账值**。

**其它必须的参数**：`--prefix-match-unit 32`（否则撞 `tokens_per_block=32 % tokens_per_hash=128`）、
`--enable-prefix-caching`、`blocks_per_chunk` 用 per-group 字典、
`--kv-cache-memory-bytes ≥ max_model_len × kv_per_token`
（★ 开了 int8 还必须 `VLLM_V41_APC_ALIGN=3` + `VLLM_V41_KV8_GRAPH_SAFE=1`，`serve_a2_offload.sh` 会自动补并告警）。

---

## 2. KV8（long-KV INT8）—— ★ **档 C 已实测可上线；档 D 在验**

### 2.1 已完成的（技术面全部打通）

| 环节 | 结果 |
|---|---|
| **原方案 `layout_kv="TND"`** | ⛔ **在 A2/A3 上不存在**（arch22 只编译 `TND Q × PA_BBND KV`）；arch22 里那些 `TND` 代码是**死代码**（选择表没实例化） |
| **替代设计** | ✅ **PA_BBND scratch + identity block table + 索引重编号**，**逐比特精确** |
| 引擎集成 | ✅ 接进真实调用点，开关 `VLLM_V41_KV8_SWA` / `VLLM_V41_KV8` |
| 数值 | ✅ 真量化 `rel_L2 = 5.43e-3`、`cos = 0.9999857`、无损臂 `max_abs = 0` |
| 图兼容 | ✅ 全链可 capture、可 replay；★ **档 C 的 `replay1 sha` 与 eager 逐字相同** |

### 2.2 ★ 时延与容量（**都已重测**）

| 指标 | 早期（未融合） | **现在** |
|---|---|---|
| **时延** | **+21%**（+6.41 ms/step） | ★ **+1.3~1.8%**（`logs/026` 融成 1 个 kernel；`logs/028`/`034` 复核） |
| **容量** | ×1.135 | ★ **按几何分档**：tiny 档 C **×1.4655** / 档 D **×1.9133**；**A2 真权重**档 C **×1.0000**、档 D **×1.1356** |
| **宿主内存** | — | ★ **档 C 197.21 → 150.01 GiB（×1.3146）** —— int8 在 A2 上真正的收益在这里 |

**两条根因都已被修掉**：
* 时延：不是带宽，是**"算子个数 × 每核延迟"**（每设备算子 ≈4–6 µs、rebuild ~30 个算子）⇒ **融成 1 个 kernel**（`logs/026`）；
* 容量：**FP32 compressor state ring**（`131072 B`）顶住 3 个 ratio-2 槽的页 ⇒ `ring16` 缩到 65536；
  ★ 但 A2 真权重上**真正的天花板是 draft 组**（`logs/050`）⇒ 见 §2.4。

### 2.3 「待测」的含义（★ 已收窄）

**档 C 已在发布件上验完**（`logs/048`/`049` + 13:0x 的 `sg-c-c-graph-b`）：图模式捕获期炸点已修，
宿主内存省 47.20 GiB，HBM 容量与档 B 逐字相同，输出 sha 与 eager 逐字相同。
> ✅ **md5 对号已完成**：`sg-c-c-graph-b`（档 C，8 卡，md5 **`94aeebb7…`**）全绿，且
> `fill`/`replay1` 两枚 sha 与 `22cbf20c` 那轮**逐字节相同** ⇒ **"改动能平移"这个【推断】已被实测坐实**。
> 档 D 也在同一个 md5 上全绿（`sg-c-d-graph`，含 `replay1 sha == 同几何 eager`）。
> 见 `patches/kv8-graphsafe/README.md` §3.0 与 `patches/ARTIFACT-IDENTITY.md` §1.1。

**还差判据的只有档 D**：
1. **档 D 的图模式四条判据 + 输出 sha**（`sg-c-d-graph`，`logs/049`）；
2. **②c（draft block 128→64）的真权重端到端臂** ⇒ 预测 HBM **×1.8177**（`logs/051`）。

---

## 3. 目录

| 路径 | 内容 |
|---|---|
| `patches/` | 可交付补丁（卸载层 4 份 + `kv8-graphsafe/` int8 补丁集）+ 挂载说明 |
| `scripts/` | ★ **`a2_one_shot_probe.sh`（A2 第一条命令）** + **一键起服**（`serve_a2_offload.sh`） |
| `docs/` | **`A2-DEPLOY-NOW.md`（上线清单，先读这个）** + `A2-GO-LIVE.md`（长版）+ `INT8-CASE.md` |
| `logs/` | 全部实验日志（**带序号索引在 `logs/README.md`**） |
| `DELIVERY.md` | ★ **交付单一入口**（配置、证据分层、边界、回滚） |
| `CHANGELOG.md` | 本分支相对 `main` 的逐项变更 |

### 3.1 一键起服

```bash
# 档 B（推荐起点）
MODEL=<模型目录> OFFLOAD_GB=56 MAX_LEN=131072 MAX_SEQS=16 \
NPU_OFFLOAD_HOST_MEM=registered SHADOW_PKG=<shadow-pkg 路径> \
  bash a2/scripts/serve_a2_offload.sh

# ★ 档 C（宿主再省 47 GiB；会自动补 APC_ALIGN=3 与 GRAPH_SAFE=1 并告警）
KV8_SWA=1 KV8_RING_FP16=1 ... 同上 ...

# 先干跑看参数（不启动）：
DRY=1 MODEL=<模型目录> NPU_OFFLOAD_HOST_MEM=registered bash a2/scripts/serve_a2_offload.sh
```

★ **为什么需要 shadow-pkg**：`scripts/serve_a2.sh` 是生产脚本，**不改它**。
shadow-pkg 是它的副本，多了两个注入点（认 `KV_ARGS_EXTRA` 与 `OFFLOAD_*_PATCH`）。
本脚本负责把 `a2/patches/` 的补丁复制进去，再调它的 `serve_a2.sh`。

**起服后必须先跑自检**（脚本末尾会打印命令；任一为 0 就停，别压测）：
```bash
grep -c 'P1_pinned.*ret=0'          <serve.log>   # 期望 8（池后端生效）
grep -c 'D2_offload'                <serve.log>   # 期望 >0
grep -c 'alignment_chunk_count.*8'  <serve.log>   # 期望 >0（per-group 生效）
grep -c 'P2_poolsizing'             <serve.log>   # 期望 >0（L1 生效）
grep -a 'P2_WORKER_HOST_BYTES'      <serve.log>   # ★ 宿主实占
```

---

## 4. 并发 × 上下文的容量边界（★ 用 L1 之后的新数）

16 张 canonical 张量 + L1 之后，池子的宿主实占（**8 卡实测**）：

| 场景 | 记账 `cpu_bytes_to_use` | **宿主实占（档 B / 档 C）** | A2（余量 442 GiB） |
|---|---:|---:|---|
| 16 × 32K | 10 GiB | ≈35 / ≈27 GiB | ✅ 宽裕 |
| **16 × 128K** | **56 GiB** | ★ **197 / 150 GiB** | ✅ **可以**（档 C 只占 34%） |
| 32 × 128K | 112 GiB | ≈394 / ≈300 GiB | ⚠️ 档 B 紧、档 C 勉强 |
| 64 × 128K | — | — | ⛔ 不可能 |

★ **但这是最坏口径**（每条请求前缀互不相同）。真实 agent 流量是
**共享 system prompt + 变长历史**，此时池子只需覆盖**唯一前缀总量**：

| | 要覆盖的 | 1.000× 池 |
|---|---|---|
| 最坏 | Σ 各请求长度 | 16 × 128K |
| **典型（共享前缀）** | 唯一前缀总量 | ≈1 个前缀 |

**⇒ 差一个数量级**，所以"能扛多少并发"要用**真实 trace 的前缀唯一率**算，不能按请求数乘积估。
（A2 现场实测 prefix cache 命中率 **96%** ⇒ 属于"共享前缀"那一档。）

---

## 5. 尚未验证的（★ 已更新）

| # | 事项 | 影响 / 状态 |
|---|---|---|
| 1 | ★★ **A2 实机**池后端（`aclrtHostRegister` 在 `host_mem_pool=0` 的机器上能不能用） | **唯一的阻塞**；已收敛成一条命令（§0.1） |
| 2 | 档 D 的图模式判据（含输出 sha） | ⏳ `sg-c-d-graph` 在跑 |
| 3 | ②c（draft block 64）真权重端到端 | ⏳ 排队；改动清单与三臂单元自检已定稿（`logs/051`） |
| 4 | 8 卡真权重 **KV 级逐字节**比对 | ⚠️ 仍未做（现有最强判据是"跨臂同名轮次 sha 逐字相同"） |
| 5 | ★ **纯 BF16（档 B）也踩在同一条 ring 别名暴露面上** | `logs/045`：**潜伏缺陷、非现行故障**（`nan=0`，但 `31/32` 行是"别家平面"的字节）—— 需长期观察 |
| 6 | 服务在 `temperature=0` 下**同臂内也会抖** | `logs/037`：32K 2/4、512 短 prompt 3/4 首 token 不同 ⇒ 与容量无关的**独立**质量问题 |
