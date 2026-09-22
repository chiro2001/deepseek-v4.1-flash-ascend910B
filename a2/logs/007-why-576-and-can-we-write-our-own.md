# 为什么那个 INT8 算子要 576？我们自己写量化算子可行吗

> 2026-09-21 23:4x。回答两个问题：① `kv_quant_sparse_flash_attention` 为什么硬要
> `q_head_dim=576`；② 如果自己写量化算子，能不能实现。
>
> **结论先说**：576 是**那个算子的实现限制**（源码注释原文 `// 576:当前不泛化`），
> 不是硬件限制 —— **证据是 A5 上有一个几何完全对得上 V4.1 的量化算子**。
> 而"自己写"的**现实形态不是写 attention 算子**，是**写量化/反量化 + 复用现成的 BF16 算子**。

---

## 一、576 到底是什么：算子的几何是"经典 MLA"，不是 V4.1 的

### 1.1 576 与 656 的构成（源码原文）

`csrc/attention/kv_quant_sparse_flash_attention/README.md`：

> 「KV_D 值仅支持 656，即 **nope + rope\*2 + dequant_scale\*4 = 512 + 64\*2 + 4\*4**」

`op_host/..._tiling.cpp:1282`：

```
OP_CHECK_IF(qHeadDim_ != 576, // 576:当前不泛化        ← ★ 注释原文
OP_CHECK_IF(kHeadDim_ != 656, // 656:当前不泛化
```

⇒ **q = 576 = nope(512) + rope(64)**；K = 656 = nope(512) + rope×2(128，K/V 共享) + scale×4(16)。
这是**经典 MLA**（V3/V4 系）：`kv_lora_rank = 512` 与独立的 `k_pe = 64` 分开。

### 1.2 ★ 更深一层：kernel 把 nope **硬编码成 128 或 512**

`op_kernel/..._common.h`：

```
enum class ATTENTION_MODE {
    GQA_MHA    = 0,   // QKV headDim 相等
    MLA_NATIVE = 1,   // Dn=128, Dr=64
    MLA_ABSORB = 2,   // Dn=512, Dr=64
};
```

⇒ **Dn ∈ {128, 512}，没有 448**。所以就算改 tiling 放行，**kernel 也没有对应的特化**。

### 1.3 V4.1 的几何是第三套

【实测/代码，K8_int8 + `models/deepseek_v41/model.py:361-373`】

```python
self.head_dim      = config.head_dim                # 512（**总宽**，不是 nope）
self.rope_head_dim = config.qk_rope_head_dim        # 64
self.nope_head_dim = head_dim - rope_head_dim       # **448**
self.scale         = head_dim ** -0.5               # 1/sqrt(512)
inplace_partial_rotary_mul(..., partial_slice=[448, 512])   # rope 在同一条 512 向量尾部**原地**旋转
```

⇒ V4.1 是 **`448 + 64` 内嵌在同一条 512 宽向量里**，既不是 `512 + 64`（算子要的），
也不是 kernel 支持的 `128` 或 `512`。
而且 **tile_size=128 不整除 448**（448/128 = 3.5）⇒ 连量化粒度都对不上。

---

## 二、★ 但这不是"做不到"——A5 上就有一个几何完全对的量化算子

【实测，CANN 源码树】`mixed_quant_sparse_flash_mla`：

```
产品支持：  950PR/950DT √   |   Atlas A3 ×   |   Atlas A2 ×

q_d 仅支持 512；
KV 由 nope、rope、scale、padding 拼接；quant_mode=2 时：
    nope(448, FLOAT8_E4M3FN) + rope(64, bfloat16) + scale(7) + pad(1B)
rope_head_dim 仅支持 64。
```

**⇒ 它的 nope 就是 448、q_d 就是 512 ⇒ 和 V4.1 一模一样。**

三条推论：

1. **576 不是硬件或算法上的必然** —— CANN 自己就能做 448+64；
2. **A5（950）那条线已经为 V4.1 的几何写好了量化算子**；**A2/A3 那条线还没做**；
3. ⇒ 这**不是"写不出来"的问题，是"他们只给 A5 写了"**。

> ⚠️ **更正 K8_int8 的一条建议**：它在日志 §4.3/§7 里推荐
> "把 `dsa_v41.py:495` 换成 `mixed_quant_sparse_flash_mla`" ——
> **该算子 A2/A3 不支持**（README 明写 ×），所以这条对**我们**走不通。
> 它的价值在于**证明了几何可做**，不在于可以直接用。

---

## 三、那"自己写"到底要写什么？三个层次

### ★ Level 1（推荐）：不写 attention 算子，只写"量化 / 反量化"，复用现成的 BF16 算子

**关键洞察**：V4.1 现在调的 `npu_sparse_flash_mla` **原生就吃 448+64、而且是 BF16**
（`dsa_v41.py:495` 的实参就是 `ori_kv=swa_cache_layer.kv_cache[0]`、`cmp_kv=source_cache`）。
⇒ **我们根本不需要那个 576 的 INT8 算子。**

要走的路：

```
存:  scatter 进 long-KV 时，存 INT8 + scale（而不是 BF16）
读:  调用 SMLA 之前，把本次要用到的 KV 反量化成一块 BF16 scratch
算:  用**原封不动的** npu_sparse_flash_mla，指向 scratch
```

要写的东西：一个**逐行量化**（存） + 一个**逐行反量化**（读）。**都不是 attention 算子。**

| 环节 | A2 上现成能用吗 |
|---|---|
| 量化（存） | ✅ `torch_npu.npu_dynamic_quant`（内置，W8A8 路径在用） |
| 反量化（读） | ✅ 一次逐元素 `mul`（scale 广播） |
| BF16 attention | ✅ `npu_sparse_flash_mla`（**不量化、不改**） |

**代价与收益**：

* 收益：容量 **4421 → 2405 B/token ⇒ ×1.84**（`logs/002` §5 的字节账）；
* 代价：**多一遍读写**（写 INT8、读 INT8、写 BF16 scratch、读 BF16）。**必须实测**；
  但注意 V4.1 每步并不读整个上下文：`ori_kv` 只有 **SWA 窗口**那部分，
  `cmp_kv` 是**压缩过**的（`cmp_ratio` 2/4/128）⇒ **反量化量 = 窗口 + 压缩块，不是全长上下文**【推断，待实测】。

**为什么这是唯一现实的路**：不改 attention 算子 ⇒ 不怕 CANN 升级；
不需要 AscendC kernel 开发 ⇒ 我们自己能做完。

### Level 2：把 A2/A3 的 INT8 算子泛化到 448

要改两处（都在 **CANN 侧代码**，不在我们仓里）：

1. `AttentionMode` 加一个 448 的特化（现在只有 128/512）；
2. `tile_size` 从 128 放宽到 64（448/128 ∤，448/64 = 7 ⇒ 整除）。

| 项 | 评估 |
|---|---|
| 工作量 | 中偏大（AscendC kernel 特化 + 搬运/流水要重排） |
| 归属 | CANN 的代码，**改了我们自己维护**，升级可能被覆盖 |
| 收益 | 比 Level 1 少一遍读写（直接吃量化 KV） |
| 结论 | **不做主路线**，除非 Level 1 的开销被证明不可接受 |

### Level 3：把 A5 的算子移植到 A2/A3

源码在机器上（`/home/wzj/qli-optimize/solutions/S5_combined/ops-transformer/attention/mixed_quant_sparse_flash_mla`），
但那是 **arch35（A5）kernel**，搬到 A2/A3 等于重写。**不现实**
（而且那是**别人的工作目录**，不该动）。

---

## 四、回答"能不能自己写"：能，而且比想象的浅

| 问题 | 答案 |
|---|---|
| 576 是硬件的吗？ | ❌ **不是**。注释 `// 576:当前不泛化`；且 A5 上有 448 的版本 |
| 我们要写 attention 算子吗？ | ❌ **不需要**。V4.1 的 BF16 算子已经吃 448+64 |
| 我们要写什么？ | ✅ **量化（存）+ 反量化（读）两个小 kernel**，其余复用 |
| 主要风险 | ⚠️ **多一遍读写的开销**（要实测）+ **KV 布局与 cache spec 的改动**（页几何、scale 存哪） |
| 主要收益 | **容量 ×1.84**（3.50M → 6.4M token），与 DRAM 卸载**相乘** |

**⇒ 这条路把"改模型几何 / 重训"整个绕开了。** K8 说的"要重训"是针对
**用那个 576 的算子**；**用 Level 1 就不需要**。

---

## 五、下一步（按性价比）

| 优先 | 实验 | 成本 | 回答什么 |
|---|---|---|---|
| **P0** | 算 **SWA 窗口 + cmp_kv 的实际字节量**（从 `dsa_v41.py` 入参 + config 的 `compress_ratios` 推） | 30 min，纯算 | Level 1 的**反量化量**多大 |
| **P0** | 单卡实测 "INT8 存 → 反量化到 BF16 scratch → 喂 SMLA" 的**端到端 step 时间** | 半天 | Level 1 的净收益是正是负 |
| **P1** | 核实 `mixed_quant_sparse_flash_mla` 在本机 CANN 里**是否真的只有 A5 kernel** | 20 min | 彻底排除 Level 3 |
| **P2** | 评估 Level 2 的 kernel 改动面（只读评估） | 半天 | 是否值得 |

---

## 六、取证

```bash
cd vllm-ascend-upstream
# ① 576/656 的硬检查 + "当前不泛化"注释
git show origin/main:csrc/attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.cpp | sed -n '1282,1290p'
# ② kernel 支持的 Dn 只有 128/512
git show origin/main:csrc/attention/kv_quant_sparse_flash_attention/op_kernel/kv_quant_sparse_flash_attention_common.h | sed -n '35,45p'
# ③ 656 = 512 + 64*2 + 4*4（README 原文）
git show origin/main:csrc/attention/kv_quant_sparse_flash_attention/README.md | grep -n 656
# ④ V4.1 真实几何 + 实际调的算子
git show origin/main:vllm_ascend/models/deepseek_v41/model.py | sed -n '361,375p'
git show origin/main:vllm_ascend/attention/dsa_v41.py | sed -n '495,520p'
# ⑤ A5 那个算子（在机器的 CANN 源码树里，**别人的目录，只读**）
sed -n '1,20p' /home/wzj/qli-optimize/solutions/S5_combined/ops-transformer/attention/mixed_quant_sparse_flash_mla/README.md
```

---

## 七、★ 三方对账：模型设计说 FP4，Ascend 实现用的是 **FP8**

K4_int4 交付后，把三份材料放一起看，出现一个**必须点明**的差异：

| 来源 | KV 的格式 | 证据 |
|---|---|---|
| **模型设计**（官方 tech report / 教程） | **FP4 main KV cache** | `docs/source/tutorials/models/DeepSeek-V4.1-Flash.md` §1：*"FP4 main KV cache … 890 bytes per token"* |
| **Ascend A5 的量化实现** | **FP8（E4M3）** | `mixed_quant_sparse_flash_mla` README：`nope(448, **FLOAT8_E4M3FN**) + rope(64, bf16) + scale(7, bf16)`，scale 7 个/448 ⇒ **group 64** |
| **专家权重的格式**（我实测解码） | **MXFP4（E2M1 + E8M0 block 32）** | `w1.weight I8[N, K/2] + scale F8_E8M0[N, K/32]` |

**⇒ Ascend 在 KV 上选了 FP8，不是 FP4。** 这可能是因为：A5 没有 FP4 的 KV 通路、
或他们评估后认为 FP4 精度不够、或是另一条设计线。**我们不需要猜**——但它有直接含义：

**★ 我们做 A2 低精 KV 时，"跟 Ascend 自己走"的安全选择是 FP8/INT8，不是 FP4。**

而 K4 的实测正好把这条量出来了：

| 档 | rel_L2 | **cos_min** | **frac(cos<0.99)** | B/token | 省 |
|---|---:|---:|---:|---:|---:|
| **INT8 g128**（≈FP8 精度档） | **2.21%** | **0.990** | **0** | 2405 | 45.6% |
| **FP4 E2M1 g16** | 7.54% | 0.823 | 0.97% | 1477 | 66.6% |
| INT4 g32 | 13.0% | 0.203 | **30.5%** | 1477 | 66.6% |
| INT4 g128 | 23.6% | 0.013 | **99.6%** | 1381 | 68.8% |

**读法**：
* **INT8/FP8 那一档没有尾部**（`frac(cos<0.99) = 0`，最差向量仍 0.990）⇒ 与 Ascend A5 的选择一致；
* **FP4 省得多（66.6% vs 45.6%）**，但有 **0.97% 的 token** cosine 掉到 0.99 以下
  —— 值不值，取决于这 0.97% 会不会伤到长链推理；
* **均匀 INT4 不要碰**（g32 就有 30.5% 的 token 尾部）。

⇒ Level 1 这条路的**默认档建议定为 INT8**（与 A5 的 FP8 同精度量级、无尾部、
且 `npu_dynamic_quant` 现成可用）；**FP4 作为"如果容量还不够"的第二档**，但要先把那 0.97% 的尾巴评估清楚。
