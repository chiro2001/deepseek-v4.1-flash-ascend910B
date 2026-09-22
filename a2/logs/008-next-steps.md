# 下一步怎么做（2026-09-21 23:5x 决策）

> 依据：B4（`noengram`）**跑通** + `dram32` 已就绪 + 三份精度数据（`logs/002/003/004`）+ 576 溯源（`logs/007`）。

---

## 一、先说这条**改变了优先级**的实测：DRAM 卸载在 DSV4.1 上通了

| 臂 | 配置 | 结果 |
|---|---|---|
| ctl | 无卸载 | HBM KV 46,387 token；replay TTFT 4335.7 ms ≈ fill ⇒ **全量重算**；`CPU` 事件为 0 |
| dram16 | +16 GiB，**带 Engram** | ✘ `tokens_per_block=32 % tokens_per_hash=128` 断言 |
| b3 | +PMU32，**带 Engram** | ✘ `aclrtMallocHost` **207001**（1 GiB 要不到） |
| **B4 `noengram`** | +PMU32，**去掉 Engram** | ✅ **起来（595 s）**，`GPU_to_CPU` **1280 次 / 393.6 GB / 40.8 s** |
| **dram32** | +PMU32 + **32 GiB**，去掉 Engram | ✅ **364 s 就绪**（正在跑） |

### ★ 三条结论

1. **卸载层本身在 A3 上是通的** —— 393.6 GB 搬出去了；
2. **`207001` 是"卸载池 + Engram 表"叠加导致的**，不是卸载池本身要不到内存
   ⇒ 之前那条"卸载层拿不到 pinned 池是拦路虎"**不成立**（那是**两者共存**时的问题）；
3. ⛔ ★ **但它是"只写不读"的退化解 —— 已判定（见 §一.0）**。

### ★ 一.0 判定结果：**卸载没起作用，只是当了回收站**

【实测，23:5x 读 `off8-noengram` 的现有产物】

| 量 | 值 | 说明 |
|---|---|---|
| `kv_offload_size_count{CPU_to_GPU}` | **0.0** | **一次都没读回来** |
| `kv_offload_size_count{GPU_to_CPU}` | 1280 次 / 393.6 GB / 40.8 s | 写得很勤 |
| `kv_offload_cpu_cache_usage_perc` | **0.0** | 池子始终是空的（写完就被丢） |
| `fill` TTFT p50 | **4304.3 ms** | 基线 |
| **`replay1` TTFT p50** | **4208.0 ms** | ⇒ **与 fill 持平 = 全量重算** |
| 对比 ctl（无卸载） | fill 4356.4 / replay 4335.7 | 同样持平 |

⇒ **三条判据里第 3 条（replay ≪ fill）不成立**，而 `CPU_to_GPU = 0` 说明**读回路径根本没走**。
**和单卡 Qwen3-1.7B 那次（`logs/045`：replay 38.4 ms vs fill 90.4 ms、命中 100%）形成鲜明对比。**

**为什么**（最可能，待验）：`logs/045` §5.2 那条坑 —— **`blocks_per_chunk` 与请求长度不匹配**
（当时 `=64` ⇒ `BlockStored(CPU)=0`）。这里写是写了，但**读的时候哈希对不上**
⇒ 要么 `--prefix-match-unit 32` 与卸载层的 chunk 粒度不一致，
要么 **DSV4.1 的 13 个 KV group 里只有部分被卸载**（compressor state 组 `prefix_cacheable=False`），
导致"存进去的块永远匹配不上读请求"。

⇒ **这变成 P0 的第一件事，而且它决定了 DSV4.1 的卸载是否可行。**

> ⇒ **对 A2 的含义**：A2 的 `host_mem_pool=0` 会影响**Engram**（我们已经默认关了），
> **但不一定影响卸载池** —— 需要 A2 上的 `a2_probe.sh` 来定。

---

## 二、下一步：**两条线并行**，但优先级不同

### ★ 线 1（P0，先做）：**先修 DSV4.1 的"只写不读"，再上 A2**

**为什么最高**：这是**已验证机制 + 无精度损失 + 直接解决用户痛点**（KV 被踢出就重 prefill）。
而且不需要写任何新算子。
**但 §一.0 已判定当前配置下它是个"回收站"** ⇒ 得先修好，否则搬到 A2 上也是白搬。

| 步骤 | 动作 | 成本 | 判据 |
|---|---|---|---|
| ~~1.1~~ | ~~判定是不是退化解~~ | ✅ **已完成** | ⛔ **是退化解**（`CPU_to_GPU=0`、replay=fill） |
| **1.1b** | ★ **定位为什么读不回来**：① 查 `--prefix-match-unit 32` 与卸载 chunk 的粒度是否一致；② 查 13 个 KV group 里**哪些真的被卸载**（compressor state 组 `prefix_cacheable=False`）；③ 对比单卡 Qwen3 那次的成功配置（`logs/045`） | 1–2 h | 找到让 `CPU_to_GPU > 0` 的配置 |
| **1.1c** | 用修好的配置复跑 replay 臂 | 40 min | **replay TTFT ≪ fill** |
| 1.2 | A2 上跑 `a2_probe.sh`（**用户粘贴**） | 5 min | `host_mem_pool` / `aclrtMallocHost` / `pin_memory` 三个读数 |
| 1.3 | 按探测结果定 A2 的卸载参数（`--prefix-match-unit 32`、`blocks_per_chunk=8`、`cpu_bytes_to_use≈260 GB`） | 30 min | 能起服 |
| 1.4 | A2 上验三判据：`BlockStored(CPU)>0`、`external_prefix_cache_hits` 增长、**replay TTFT ≪ fill TTFT** | 1 h | 三条都成立 |

### 线 2（P1，并行起步）：INT8 KV 的"量化存储 + 读时反量化"

**为什么第二**：收益大（**×1.84 容量**，与卸载**相乘**），但**要改代码 + 必须实测净收益**，
且**有精度代价**（虽然 `logs/002` 显示很干净）。

| 步骤 | 动作 | 成本 | 判据 |
|---|---|---|---|
| 2.1 | **算清反量化量**：V4.1 每步 `ori_kv`（SWA 窗口）+ `cmp_kv`（压缩块）到底多少字节 | 30 min，纯算 | 总量 ≪ 全上下文 |
| 2.2 | 单卡**离线**做 A/B：同一批 Q，BF16 KV vs "INT8 存 → 反量化 → BF16"，测 attention 输出差 | 2 h | `rel_L2 ≤ 2.3%`（`logs/003` 的 INT8 g128 实测） |
| 2.3 | 单卡**在线**：改 `dsa_v41.py` 的 `scatter_cache_sk` 与 SMLA 调用点，测 step 时间 | 半天 | **净收益为正** |
| 2.4 | 若 2.3 通过 ⇒ 端到端（GSM8K / Vision） | 半天 | 不退化 |

### 线 3（P2，只读评估，不投入）：FP4 / 自写 attention 算子

| 事项 | 结论 |
|---|---|
| FP4 KV | `logs/003`：dtype 符号有、能分配，但 **bf16→fp4 转换 `161002`**、CANN 无 fp4 算子 ⇒ **只能"存"不能"算"** |
| 自写 attention 算子 | `logs/007`：**不需要** —— `npu_sparse_flash_mla` 已吃 448+64；要写的只是量化/反量化 |
| `mixed_quant_sparse_flash_mla` | A5 专属（A2/A3 ×）⇒ **排除** |
| Level 2（改 CANN kernel 泛化到 448） | 成本高、要维护 CANN 分支 ⇒ **不做主路线** |

### 线 4（背景）：等 D_off8 收尾 + 把 B4 的发现补进文档

---

## 三、★ 下一步的**第一件事**

**定位"写进去为什么读不回来"**（§一.1b）。这是整条卸载路线的**唯一拦路虎**，
而且它同时决定 A2 该不该上。三个具体排查方向：

1. **粒度对齐**：`--prefix-match-unit 32` 是给 `patch_kv_cache_utils` 用的哈希粒度，
   而**卸载层的 chunk 粒度由 `blocks_per_chunk × block_size` 决定**（现为 `8 × 128 = 1024` token）。
   两者是否必须一致？`logs/045` 的单卡成功配置用的是 **默认 prefix-match-unit**，
   我们这里为了绕过 compressor state 组（block_size=32）才设了 32 ⇒ **怀疑是这里引入的**。
2. **哪些 group 真的被卸载**：DSV4.1 有 **13 个 KV group**，其中 compressor state 组
   `prefix_cacheable=False`。如果卸载层只处理"可前缀缓存"的组，**可能整体被静默跳过**。
3. **对照**：把 `logs/045` 单卡 Qwen3 成功那次的 `blocks_per_chunk` / `prefix-match-unit` /
   `max_model_len` 三个值列出来，与本次逐项对比。

---

## 四、给两个方向各配一个子代理（建议）

| 代理 | 槽位 | 任务 | 预算 |
|---|---|---|---|
| **N1_offload** | 不占卡（读数据 + 写脚本）；A2 侧等用户 | ① 判定 B4 是否退化解；② 把 `logs/045` 的四条坑 + B4 的新发现合成"**A2 卸载上线清单**"（含参数定值、判据、回滚） | 90 min |
| **N2_kv8** | c0（die 3） | 线 2 的 **2.1 + 2.2**：算清反量化量 + 单卡离线 A/B | 90 min |

**为什么不马上做 2.3（在线改代码）**：2.1/2.2 是它的前提 —— 如果反量化量是"全长上下文"，
这条路直接死；如果 2.2 的精度差就超过 `logs/003` 的 2.21%，也不用改代码了。
