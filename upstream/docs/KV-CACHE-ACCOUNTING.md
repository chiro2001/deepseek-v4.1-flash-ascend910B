# DeepSeek-V4.1-Flash 的 KV cache 账 —— 4421 B/token 是怎么来的，为什么官方是 890 B

> 2026-09-21 起。触发：**论文说 890 B/token，我们的实测是 4421 B/token，差 4.97×。**
> 这份文档把账拆到"每一字节来自哪个数据结构的哪一维"，并回答两个问题：
> ① 这 5 倍差在哪；② 是我们实现错了，还是这一代硬件做不到。
>
> 结论先行：**不是实现错误。是"精度"（3.76×）+ 少量口径差（1.3×），而精度那一刀卡在 A5 独占的硬件能力上。**

---

## 1. 实测：两点独立吻合

| 机器 | `Available KV cache memory` | `GPU KV cache size` | **B/token/rank** |
|---|---:|---:|---:|
| A3（910C，`a1_20260921_172649`） | 15.82 GiB | 3,842,534 | **4420.7** |
| A2（910B3，用户 09-20 那次，util=0.90） | 14.40 GiB | 3,498,354 | **4419.8** |

两次相差 **0.02%** ⇒ 这个数字是稳的，不是某次配置的偶然。

**换算**：

```
1M 上下文 = 4421 B/token × 1,048,576 = 4.32 GiB / rank
         × 8 rank              = 34.5 GiB        （TP 内是复制，见 §4）
```

---

## 2. 拆账：4421 = 4096 + 325（差 0.3 B）

### 2.1 结构：40 层里只有 4 层保留全上下文 KV

config（`text_config`）里的三个关键字段：

```python
num_hidden_layers      = 40
compress_ratios        = [0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,...,1,0,0,0]
kv_source_layer_ids    = [2, 8, 14, 20]          # ← 只有这 4 层拥有全上下文 KV
index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
candidate_source_layer_id = 20
index_topk             = 512
sliding_window         = 128
head_dim               = 512
num_key_value_heads    = 1
```

这就是论文说的 **CSA2 跨层复用**（4 种 static mode：**Full / Reindex / Reuse**）：
**只有 4 个 "Full" 层生成 global KV**，其余 36 层通过**窗口 128 的 SWA + 层间共享**复用它们。

### 2.2 逐项

`vllm_ascend/models/deepseek_v41/model.py` 里那四个平面是这么建的：

```python
self.long_kv_cache = DeepseekV41CacheLayer(
    vllm_config, f"{prefix}.long_kv_cache",
    DeepseekV41FullSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=width,            # = head_dim = 512
        dtype=torch.bfloat16,       # ← 硬编码
        compress_ratio=role.compress_ratio,
    ),
)
```

| 项 | 算法 | B/token |
|---|---|---:|
| **4 个共享 long-KV 平面** | 4 平面 × 512 维 × **2 B(BF16)** × 1 KV head | **4096** |
| 4 个 indexer 平面 | 3 个 ratio-2 平面 `(128+2)/2` + 1 个 ratio-1 平面 `(128+2)`；INT8 + FP16 scale | **325** |
| SWA 平面 | `window=128` 有界 ⇒ **不随 token 增长**（每请求 40×128×512×2 = **5.24 MB** 固定） | **~0** |
| compressor 状态 | `DeepseekV41CompressorStateSpec`，每请求一个 FP32 环 | **~0** |
| **合计** | | **4421** |
| **实测** | | **4421**（差 0.3 B） |

⇒ **账是闭合的**：4421 B/token 全部来自"**4 个 512 维 BF16 平面 + indexer**"，
没有浪费、没有重复计数。

**两个非显然的点**：
1. **`compress_ratio` 只减少 `block_size` 的存储量，不减少 `head_size`**：
   `storage_block_size = block_size // compress_ratio`，而 plane 大小 = `storage_block_size × head_size × 2B`。
   所以 ratio-2 平面恰好是 ratio-1 的一半 —— 与上表一致。
2. **SWA 与 compressor 状态虽然"不随 token 增长"，但它们是 per-request 的**：
   每请求 SWA 约 **5.24 MB**（40 层合计，见上表），`max_num_seqs=32` 时合计约
   **168 MB** —— 在 1M 上下文的场景下可以忽略（SWA 占 global KV 的 0.113%），
   但在"很多短会话"的场景下它会变成与 global KV 同量级的开销。

---

## 3. 论文的 890 B/token 是怎么来的

出处：`DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression`
（[arXiv 2609.19969](https://arxiv.org/html/2609.19969v1) / [HF model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)）。原文三句关键：

> *"…reduce its **global KV cache footprint (always in HBM)** to **890 bytes per token**…"*
> *"…combined with **FP4 main KV caching (E2M1 format, one E4M3 scale per 16 channels)**…"*
> *"We retain **FP8 for the SWA KV cache** due to its sensitivity to quantization."*

**注意论文分了两个口径**（这点容易混）：

| 论文口径 | 值 | 机制 |
|---|---|---|
| **global KV（恒在 HBM）** | **890 B/token** | FP4 主 KV + CSA2 跨层复用 |
| **persistent KV（SSD/host）** | V4-Flash 的 **1/8** | **SWA Bounded Replay** ⇒ 不持久化 SWA |

### 差距的来源

| 因素 | 倍数 | 说明 |
|---|---:|---|
| **★ 主 KV 精度：BF16 vs FP4** | **3.76×** | BF16 = 2 B/元素；FP4 E2M1 + 每 16 通道 1 个 E4M3 scale ≈ 0.531 B/元素 ⇒ 2/0.531 = 3.76 |
| 其余（indexer 精度、平面数、口径） | ~1.3× | 我们 indexer 是 INT8+FP16 scale；论文未逐项给出 |
| **合计** | **4.97×** | 4421 / 890 |

### 换成别的精度会怎样

| 配置 | B/token | 1M/rank | 8 rank 合计 | 同 15.82 GiB 能装 |
|---|---:|---:|---:|---:|
| **BF16（现况）** | 4421 | 4.32 GiB | **34.5 GiB** | 3.84M token |
| FP8 主 KV | 2373 | 2.32 GiB | 18.5 GiB | **7.2M（×1.9）** |
| FP4 主 KV | 1413 | 1.38 GiB | 11.0 GiB | **12.0M（×3.1）** |

---

## 4. ★ 为什么用不了 8-bit —— 三道门，全关着

### 门 ①　硬件能力：**`DSV4_COMPRESSED_CACHE` 只在 A5**

```python
# vllm_ascend/attention/dsa_attn_kv_plan.py
def _supports_dsv4_compressed_cache() -> bool:
    return get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)

def get_dsv4_attn_kv_dtype(vllm_config) -> torch.dtype:
    return (
        torch.bfloat16
        if not _supports_dsv4_compressed_cache() or is_a5_bf16_kv_enabled(vllm_config)
        else torch.float8_e4m3fn          # ← 只有 A5 能到这里
    )
```

而 `DSV4_COMPRESSED_CACHE` / `FP8_ATTENTION` **只出现在 `AscendDeviceType.A5` 的 capability 集合里**
（`vllm_ascend/device/hardware_profile.py`，A5 块从第 211 行开始）。
A2/A3 用的是 `_STANDARD_CAPABILITIES` / `_A3_CAPABILITIES`，**都没有它**。

设备类型映射（`vllm_ascend/device/hardware.py`）：

```python
"910b"      → AscendDeviceType.A2
"910c"      → AscendDeviceType.A3
"ascend950" → AscendDeviceType.A5      ← 昇腾 950，下一代
```

⇒ **这是硬件代差，不是配置选项。**

### 门 ②　模型路径：`compress_ratios` 把 C8 判定排除掉了

唯一的 8-bit 开关 `enable_sparse_sfa_c8` 的准入：

```python
# vllm_ascend/utils.py:130
def model_uses_sfa_sparse(model_config) -> bool:
    return (
        hasattr(hf_text_config, "index_topk")
        and not hasattr(hf_text_config, "compress_ratios")     # ← 关键
        and not hasattr(hf_config, "compress_ratios")
    )

# vllm_ascend/ascend_config.py:643
self.enable_sparse_sfa_c8 = self.enable_sparse_sfa_c8 and use_sparse    # 与门
```

**DSV4.1 的 config 恰恰有 `compress_ratios`** ⇒ `model_uses_sfa_sparse()` = **False**
⇒ **C8 被强制关闭**，连 INT8 回退都用不上。这条**与硬件无关**，是模型路径的门。

### 门 ③　V4.1 的 attention 实现里没有量化路径

```bash
$ grep -i "c8\|float8\|int8" vllm_ascend/attention/dsa_v41.py
（空）
```

V4.1 走独立的 `DeepseekV41EagerAttentionImpl`，四个主 KV 平面的 dtype **在构造时就写死 BF16**
（见 §2.2），而且**不受 `--kv-cache-dtype` 影响** —— 这就是为什么启动命令里写
`--kv-cache-dtype bfloat16` 却"没得选"。

### 关于命名：`C4` / `C128` **不是位宽**

```python
# vllm-ascend/attention/dsa_v1.py
class AscendDSAC4Backend:      # "Ascend's physical 32/64/128-token C4 pages
                              #  represent 128/256/512 raw scheduler tokens"
class AscendDSAC128Backend:    # "physical 32/64/128-token C128 pages
                              #  represent 4096/8192/16384 raw scheduler tokens"
```

⇒ **`C4` = 压缩比 4，`C128` = 压缩比 128**（那是 **V4** 的架构；V4.1 的 `compress_ratios` 只用 0/1/2）。
**真正的 4-bit 在代码里叫 FP4 / `torch_npu.float4_e2m1fn_x2`。**

---

## 5. 4-bit（真 FP4）的可能性

**dtype 有，而且我们在用**（作 MoE 权重，不是 KV）：

```
torch_npu.float4_e2m1fn_x2 = 296     ← 正是论文的 E2M1
torch_npu.float8_e4m3fn    = 292
torch_npu.float8_e8m0fnu   = 293
```

`vllm_ascend/ops/fused_moe/moe_quant.py:85`、`prepare_finalize.py:415` 已经在用它做 W4A8 的 W4。

**而且另一条产品线做过 FP4 cache**：`glm5next/sparse_attn_indexer_kpool.py` 有 `use_fp4_cache`
—— 但那是 **indexer 的 FP4 cache，不是主 KV**。

**DSV4.1 的主 KV 没有任何 FP4 路径**，要分两层看：

| 层面 | 状态 |
|---|---|
| dtype / 量化算子 | ✅ 有（torch_npu 提供，我们在用） |
| **attention kernel 能读 FP4 KV 并按 E4M3-per-16-ch 解量化** | ❌ **不存在** |

---

## 6. 结论：能动的与不能动的

| 目标 | B/token | 挡在哪 | 性质 | 可行性 |
|---|---:|---|---|---|
| **主 KV → FP8** | 2373 | `DSV4_COMPRESSED_CACHE` + `FP8_ATTENTION` 只在 A5 | **硬件代差** | ❌ |
| C8（packed 8-bit） | — | `model_uses_sfa_sparse` 排除 V4.1；且实现里无 c8 代码 | 软件/架构 | ❌ 等于从零实现 |
| FP4 主 KV | 1413 | 无任何实现、kernel 不存在 | 软件 | ❌ 工作量大（要写 kernel） |
| **DRAM 卸载** | 4421（不变） | 无 | —— | ✅ **最现实**，机制已验证 |

### ★ 一个必须说清的判断：**我们用 BF16 是"保守"，不是"错误"**

模型是按 **FP4 KV 做 QAT 训练的**：

> *"we use **FP4 global KV caches during training** with only marginal performance degradation"*
> *"To enable FP4 main KV cache storage … we introduce **QAT during post-training**"*

我们用 BF16 存 ⇒ **精度只会更好，不会更差**；代价纯粹是**内存**（4.32 GiB/rank vs 1.38 GiB）。

⇒ **不要把 4421 B/token 当成缺陷去修**；它是"这一代硬件 + 保守精度"的合理选择。
要缓解内存压力，能动的只有**精度之外**的手段 —— 也就是 **DRAM 卸载**。

---

## 7. 对 DRAM 卸载方案的含义（可直接用）

### 7.1 需要搬的就是 global KV —— 正好是 4421 B/token

因为 SWA 与 compressor 状态**不随上下文增长**（§2.2），
**DRAM 层的容量需求 ≈ 4421 B/token × 序列长度**。

### 7.2 ★ MLA 在 TP 内是**复制**的，不是切分

证据：我们推出的 4421 B/token **正好等于四个平面的完整大小**，而不是 1/8。
原因是 `num_kv_heads = 1`（MLA 的 latent 被所有头共享），
每个 rank 都要有完整的 latent 才能算自己那部分 query。

⇒ **1M 上下文的 DRAM 需求是 34.5 GiB（8 份副本），不是 4.32 GiB。**

### 7.2.1 ★★ 为什么是 8 份 —— 算子侧的机制，以及一个 8× 的可优化点

`vllm/v1/kv_offload/cpu/spec.py` 里 CPU 层的容量是这么算的：

```python
num_copies = 1 if self.replicated_layout else world_size      # ← 决定存 1 份还是 8 份
kv_bytes_per_block  = config.worker_kv_bytes_per_block * num_copies
kv_bytes_per_chunk  = kv_bytes_per_block * self.blocks_per_chunk
self.num_chunks     = int(cpu_bytes_to_use) // aligned_kv_bytes_per_chunk
```

⇒ **当 `replicated_layout=False` 时，`cpu_bytes_to_use` 要覆盖 `world_size` 份副本**，
每个 token 的 DRAM 成本 = `4421 B × 8 = 35,368 B`。

**而 `replicated_layout` 在昇腾上必然为 False**，有**三个独立**的原因：

| # | 条件 | 我们这边的实际情况 |
|---|---|---|
| 1 | `self.replicated_layout = config.replicated_layout and self._uses_shared_region()`；而 `_uses_shared_region()` 定义为 `current_platform.is_cuda_alike() and not is_rocm()`；`is_cuda_alike()` = `PlatformEnum in (CUDA, ROCM)` | 昇腾是 `PlatformEnum.NPU` ⇒ **False** |
| 2 | 还要求 **`type(single_group_spec) is MLAAttentionSpec`**（**精确类型**，注释写明 "fail closed on wrappers and sliding-window variants"） | DSV4.1 用的是 `DeepseekV41FullSpec` / `DeepseekV41IndexerSpec` / `DeepseekV41SWASpec`（**子类**）⇒ 不匹配 |
| 3 | 同一段还要求 **`len(kv_cache_groups) == 1`** | DSV4.1 **有多组**（long-KV / indexer / SWA / compressor 状态）⇒ 不匹配 |

⇒ **在 CUDA 类平台上，纯 MLA + 单组 + TP-only 的模型（正是 DeepSeek 这一类）
可以走"单份共享 host 层"，DRAM 只要 1/8；昇腾这条路径拿不到。**

**【推断，未验证】** 这是一个独立于本文档、值得报给上游的可优化点：
把 single-copy host layout 推广到 (a) 昇腾平台、(b) 多组 MLA 模型。
若成立，同样 300 GB 能缓存的上下文从 8.5M 变成 **68M token（8×）**。
**验证方式**：改造后跑同一 workload，看 `num_chunks` 是否还随 `world_size` 变化。
**⚠️ 措辞**：这不是 bug —— upstream 注释里写的是 "Safe MVP boundary"，是**有意划的边界**，
所以应该说"扩展适用面"，不要说"修复错误"。

### 7.3 A2 的容量规划（768 GB DRAM，Engram 占 206 GiB，余 442 GB）

> 换算口径：`cpu_bytes_to_use` 是**服务级总量**、且要覆盖 **8 份副本**（见 §7.2.1），
> 所以 `tokens = cpu_bytes_to_use / (4421 × 8)`。

| `cpu_bytes_to_use` | DRAM 实际占用（×1.7 记账） | 可缓存的上下文 | 相对 HBM（3.84M） | 可同时容纳的 1M 会话 |
|---:|---:|---:|---:|---:|
| 100 GB | 170 GB | **~2.8M token** | 0.74× | 2 |
| 200 GB | 340 GB | **~5.7M token** | 1.5× | 5 |
| **300 GB**（用户计划值） | 510 GB | **~8.5M token** | **2.2×** | **8** |
| 442 GB（全给） | 751 GB | ~12.5M token | 3.3× | 12 |

> ⚠️ 300 GB 的 ×1.7 = 510 GB **超过 442 GB 余量**。要么把 `cpu_bytes_to_use` 设成
> 实际能吃下的值（≈260 GB ⇒ ×1.7 ≈ 442 GB），要么先测准 ×1.7 这个系数在本机的实际值
> （M_offload 是在 A3 上测的，A2 未必相同）。
>
> ⚠️ ×1.7 是在 **`cpu_bytes_to_use=8 GiB`** 上测的单点，**外推到 300 GB 是否仍线性未经验证**
> （可能含一部分不随容量增长的开销）。

### 7.4 收益（相对"被踢出就重算"）

| | 从 DRAM 取回 1M | 重新 prefill 1M |
|---|---:|---:|
| **时间** | **0.2–0.7 s** | **180–480 s** |
| 比值 | 1× | **约 300–1000×** |
| 资源 | 拷贝引擎 + 主机链路（**可与计算重叠**） | 整个 engine 的算力 |
| 对其他请求 | 基本无影响 | **全排队** |

（DRAM 取回按 M_offload 实测 28.5 GB/s；A2 走 PCIe 会更慢，**在 A2 上未实测**。
prefill 按 A2 实测 4,258–5,602 tok/s 外推到 1M，**未在 1M 上实测**。）

---

## 8. 还缺什么（诚实清单）

| # | 项 | 状态 |
|---|---|---|
| 1 | **A2 上的 DRAM 取回带宽** | 【未测】M_offload 的 28.5 GB/s 是 A3/HCCS；A2 是 PCIe，会明显更低 |
| 2 | **A2 上的 ×1.7 DRAM 记账系数** | 【未测】（A3 实测值；A2 未必相同） |
| 3 | **1M 上下文的重算时间** | 【外推】192 s 是最乐观假设（吞吐持平 5,600）；长上下文会退化 |
| 4 | **论文 890 B 的逐项分解** | 【口径差】论文说"不存 SWA 几乎让持久层减半"，而我们算出 SWA 只占 0.113% —— 两者对不上，**未解决** |
| 5 | **`--swa-bounded-replay` 在我们这套里是否生效** | 【未查】vLLM 侧有该开关（`cache.py:230`），vllm-ascend 里 grep 不到。**但它只影响"持久化 SWA 的存储"，不影响 global KV 的驱逐重算** —— 对本题不是主要矛盾 |
