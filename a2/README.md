# A2（8×910B3）使能：DRAM KV 卸载 + KV8

> **分支**：`feat/kv8-dram-offload-pending`
> **状态**：**DRAM 卸载已在 8 卡真权重上验证通过；KV8 待测**（见 §2）。
> **日期**：2026-09-22

---

## 1. DRAM KV 卸载 —— ✅ **已验证**（8 卡真权重 · 16 请求）

### 1.1 解决的问题

DeepSeek-V4.1 上曾出现 **"能存不能取"**：
`BlockStored(medium="CPU") = 12,288`、`GPU_to_CPU` 搬了 **393.6 GB**，
但 **`CPU_to_GPU = 0`**、**`external_prefix_cache_hits = 0`**、
replay TTFT 4167 ms ≈ fill 4192 ms ⇒ **对服务质量零帮助**。

### 1.2 根因（两条独立机制，缺一条都修不好）

| # | 机制 | 说明 |
|---|---|---|
| **①** | **`state` 组一票否决** | compressor 的 state ring（`prefix_cacheable = False`，每请求只有 1 页）在**取回路径**上仍被当成 full-attention 组参与命中判定，而它在**存储路径**上永远存不出 chunk（`len(block_ids)//blocks_per_chunk = 1//8 = 0`）⇒ 它的 key 必然 MISS ⇒ 上游 `_lookup()` 里 `if num_hit_chunks == 0: return 0` 把**整轮请求**（含 full 组与 40 个 SWA 资源）的命中**全部判死** |
| **②** | **SWA 池条目太粗** | SWA 窗口只有 128 token，而池条目是 1024 token；每请求每 SWA 组存 32 条，其中**30 条永不取回** ⇒ 池子需求被放大 **4.89×**（`logs/021`） |
| **③** | **pinned 池在真实 8 卡上要不到** | `aclrtMallocHost` 单次要 8 GiB 就报 `207001`（宿主 `MemAvailable` 仍有几百 GB）⇒ 池子上限被卡死 |

### 1.3 效果（`logs/016` / `logs/022`）

| 判据 | 修复前 | **修复后** |
|---|---|---|
| `BlockStored(CPU)` | 12,288 | **6,144**（= 16 × 384，一轮） |
| **`CPU_to_GPU`** | **0** | **12.44 GB** |
| **`external_prefix_cache_hits`** | **0** | **507,904 / 1,048,832**（**replay 轮 96.9%**） |
| **replay / fill TTFT** | 4167 / 4192 ms（+0.6%） | **253.4 / 4429.4 ms（17.5×）** |

**拐点已验证**（三条臂夹死）：池子 **0.667× 工作集 ⇒ 归零**、**1.000× ⇒ 17.9×**、1.209× ⇒ 17.5×。

### 1.4 补丁（`patches/`）

| 文件 | 作用 |
|---|---|
| `0001-offload-scheduler.patch.py` | `scheduler.py` 替换版。**含 `state` 组参与位修复 + per-group `blocks_per_chunk`**（是两者的超集，**只需挂这一份**） |
| `0001b-offload-per-group-bpc-manager.patch.py` | `PerGroupBPCManager`（池的格子 = 1 个 GPU block） |
| `0001c-offload-per-group-bpc-hooks.patch.py` | 配置解析钩子（`blocks_per_chunk` 支持 `{"default":8,"swa":1}`） |
| `0002-offload-cpu-pool-host-registered.patch.py` | `cpu_npu.py` 替换版：池子改走 `aclrtHostRegister`（绕开 `aclrtMallocHost` 的 `207001`），**注册失败自动回落 `pinned`** |

详见 [`patches/README.md`](patches/README.md)（含 md5、挂载方式、**补丁生效自检**、参数定值）。

### 1.5 参数定值

| 场景 | `cpu_bytes_to_use` | 宿主实占 | 依据 |
|---|---:|---:|---|
| 32K × 16 并发（per-group bpc） | **16 GiB** | ≈111 GiB | `logs/021` §6.1 |
| **128K × 16 并发（per-group bpc）** | **48 GiB** | **≈333 GiB** | 同上 |
| 32K，**不**用 per-group bpc | 48 GiB | 333 GiB | `logs/016` 实测 |
| 128K，**不**用 per-group bpc | — | **1,333 GiB** | 不可行 |

★ **两个必须注意的标定**（都推翻了早期估算）：
* **一个池条目只记 8 MiB 的账**（`worker_kv_bytes_per_block = 131,072 B`）；
* **但宿主实占 = 记账值 × 6.945**（16 个 canonical 张量逐个分配）⇒ 容量规划要按宿主算。

**其它必须的参数**：`--prefix-match-unit 32`（否则撞 `tokens_per_block=32 % tokens_per_hash=128`）、
`--enable-prefix-caching`、`blocks_per_chunk` 用 per-group 字典、
`--kv-cache-memory-bytes ≥ max_model_len × kv_per_token`。

---

## 2. KV8（long-KV INT8）—— ⏳ **待测**

### 2.1 已完成的（技术面全部打通）

| 环节 | 结果 |
|---|---|
| 目标容量 | 4421 → 2405 B/token（×1.84） |
| **原方案 `layout_kv="TND"`** | ⛔ **在 A2/A3 上不存在**（arch22 只编译 `TND Q × PA_BBND KV`） |
| **替代设计** | ✅ **PA_BBND scratch + identity block table + 索引重编号**，**逐比特精确** |
| 引擎集成 | ✅ 接进真实调用点，开关 `VLLM_V41_KV8=1` |
| 数值 | ✅ 真量化 `rel_L2 = 5.43e-3`、`cos = 0.9999857`、无损臂 `max_abs = 0` |
| 图兼容 | ✅ 全链可 capture、可 replay、改输入跟着变 |

### 2.2 ⛔ 但两条硬指标不达标（**这就是"待测"的原因**）

| 指标 | 目标 | 实测 |
|---|---|---|
| **时延** | ≤ +0.2% | **+21%**（+6.41 ms/step） |
| **容量** | ×1.84 | **×1.135** |

**两条根因都已定位**：
* 时延：**不是带宽，是"算子个数 × 每核延迟"** —— 生产图里每个设备算子 ≈4–6 µs、
  rebuild 有 ~30 个算子 ⇒ 只能**融成 1 个 kernel**（正在评估）；
* 容量：**FP32 compressor state ring**（`32×1024×4 = 131072 B`）顶住 3 个 ratio-2 槽的页
  ⇒ 只量化 SWA 拿不到收益；缩 state ring 后是 ×1.91。

### 2.3 「待测」的含义

**KV8 当前形态不能上线**（时延负收益）。待测的是：
1. **读侧 rebuild 能否融成 1 个 kernel**（KV8 成立的唯一门槛）；
2. **state ring 缩到 BF16 的精度影响**（它是累积状态）。

★ 好消息：**A2 的长上下文不依赖 KV8** —— §1 的 per-group bpc 已经把
「128K × 16 并发」的宿主占用从 1,333 GiB 降到 **333 GiB**，在 A2 的 442 GiB 余量之内。

---

## 3. 目录

| 路径 | 内容 |
|---|---|
| `patches/` | 四个可交付补丁 + 挂载说明 |
| `scripts/` | **一键起服**（`serve_a2_offload.sh`，依赖 shadow-pkg） |
| `logs/` | 关键实验日志（8 卡终验、per-group bpc、KV8 裁决） |
| `CHANGELOG.md` | 本分支相对 `main` 的逐项变更 |

### 3.1 一键起服

```bash
MODEL=<模型目录> SHADOW_PKG=<shadow-pkg 路径> bash a2/scripts/serve_a2_offload.sh

# 32K 场景（默认）：宿主实占 ≈111 GiB
# 128K 场景：
MODEL=<模型目录> OFFLOAD_GB=48 MAX_LEN=131072 bash a2/scripts/serve_a2_offload.sh

# 先干跑看参数（不启动）：
DRY=1 MODEL=<模型目录> SHADOW_PKG=<路径> bash a2/scripts/serve_a2_offload.sh
```

★ **为什么需要 shadow-pkg**：`dsv41-release/scripts/serve_a2.sh` 是生产脚本，**不改它**。
shadow-pkg 是它的副本，多了两个注入点（认 `KV_ARGS_EXTRA` 与 `OFFLOAD_*_PATCH`）。
本脚本负责把 `a2/patches/` 的四个文件复制进去，再调它的 `serve_a2.sh`。

---

## 4. 尚未验证的

| # | 事项 | 影响 |
|---|---|---|
| 1 | **A2 实机验证**（`aclrtHostRegister` 在 `host_mem_pool=0` 的机器上是否可用） | **最大风险**，决定 §1 能否上线 |
| 2 | per-group bpc 的 8 卡验证 | 进行中 |
| 3 | `state` 组跳过的数值正确性 | 只做了语义推断（与 GPU 前缀缓存路径一致），**未做精度对比** |
| 4 | `×6.945` 乘数在 A2 上 | 需重测 |
