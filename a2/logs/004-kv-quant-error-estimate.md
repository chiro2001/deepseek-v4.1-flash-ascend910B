# KV 低精：算子能力 + 量化误差**估计**（仿真）

> 2026-09-21 23:0x，前台分析。**不碰 C8 / 结构改动**，只谈两件事：
> ① A2 上有哪些算子可用；② 量化误差有多大。
>
> ⚠️ **口径**：本节数字全部来自 **CPU 数值仿真**（`scripts/kv_quant_error_estimate.py`），
> **不是真机测量**。真实 KV 分布未知 ⇒ 扫了 4 种分布看**敏感性**。
> 原始数据：[`raw/004-quant-error.json`](raw/004-quant-error.json)。
> 真机验证由 K8_int8 / K4_int4 两个子代理在做（`logs/002` / `logs/003`）。

---

## 一、算子：A2 到底有什么

我把仓里**所有** `csrc/**/README.md` 的产品支持表机械解析了一遍（脚本在 `/tmp/parse_support.py`，
逐行取 `Atlas A2/A3/950` 那一格）：

| 算子 | A2 | A3 | A5 | 与低精 KV 的关系 |
|---|:--:|:--:|:--:|---|
| `sparse_attn_sharedkv` | **√** | √ | × | **V4.1 现在用的**（非量化） |
| `kv_quant_sparse_flash_attention` | **√** | √ | √ | **INT8**，但语义是 **per-layer SFA**（V4 的路线） |
| `kv_quant_sparse_attn_sharedkv` | **×** | **×** | √ | 量化版 shared-KV（FP8 E4M3、tile 64）—— **只有 A5** |
| `quant_lightning_indexer_v2` | √ | √ | √ | indexer 的量化 |
| `sparse_flash_mla` | √ | √ | √ | A5 BF16 路径用的 |
| `fused_sparse_attention_overlap` | √ | √ | √ | 融合 |
| **`compressor`** | **×** | **√** | √ | ← **A2 缺、A3 有**（但见下方说明） |
| `grouped_matmul_swiglu_quant` | √ | √ | – | MoE 侧 |

### 1.1 两条**通用**量化算子（不依赖上面任何一个专用 kernel）

【代码事实】`attention/dsa_v1.py:1824` 和 `_310p/quantization/methods/w8a8_dynamic.py:217` 都在用：

```python
hs_int8, hs_pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)   # BF16/FP16 → INT8 + per-token scale
torch_npu.npu_quant_matmul(...)                                           # 量化矩阵乘
```

⇒ **这是关键**：`npu_dynamic_quant` 就是"**per-token INT8 量化 + 出 scale**"，
是 torch_npu 的内置算子（不是 csrc 里的专用 kernel），**A2/A3 都在用**。
而**反量化只是 `mul`（逐元素乘 scale）**，任何平台都有。

⇒ **结论一：做"INT8 存储 + 读时反量化"（L1）不需要新算子、不需要 CANN 改动。**
需要的是 `npu_dynamic_quant`（存的时候）+ 一次 `mul`（读的时候）。

### 1.2 ★ `compressor` A2 不支持 —— 但 V4.1 在 A2 上**没用它**

【代码事实】`git grep "npu_compressor" vllm_ascend` → **零命中**；
V4.1 的 compressor 走的是 **`vllm_ascend/ops/triton/compressor/compressor_triton.py`**（Triton 实现）。
⇒ csrc 那个 `compressor` 是给别的路径用的，**A2 跑 V4.1 不受影响**。
（这条排除了一个可能的误解，值得记下来。）

---

## 二、量化误差：仿真结果

### 2.1 设置

| 项 | 值 |
|---|---|
| 形状 | `D = 512`（`config.head_dim`，MLA latent）、N = 2048 token、H = 32 head |
| 分布 | ① Gaussian ② Student-t(3)（重尾）③ **1% 通道 ×10**（LLM 激活的经典 outlier）④ Logistic |
| 量化 | 对称均匀；**per-token-group**；scale 用 fp16 |
| 指标 | attention 输出的**相对误差**（`‖Δo‖/‖o‖`）、cosine、**top-k 重合度**、B/token |
| 空测 | **BF16 roundtrip** 作为噪声地板 |

### 2.2 主表（稠密 attention，多分布中位数）

| 方案 | 输出 rel_err | cosine | K rel_err | **B/token** | **容量增益** |
|---|---:|---:|---:|---:|---:|
| **BF16**（地板） | 0.0027 | 0.9999967 | 0.0017 | 4421 | 1.00× |
| FP8 E4M3 (tile1) | 0.0188 | 0.99982 | 0.0125 | 2381 | 1.86× |
| **INT8 tile128**（真实算子口径） | **0.0134** | **0.99991** | 0.0108 | 2405 | 1.84× |
| INT8 tile64 | 0.0110 | 0.99994 | 0.0089 | 2437 | 1.81× |
| INT8 group32 | 0.0093 | 0.99996 | 0.0072 | 2501 | 1.77× |
| **INT4 group128** | **0.2209** | **0.9763** | 0.176 | 1381 | **3.20×** |
| INT4 group64 | 0.1855 | 0.9833 | 0.148 | 1413 | 3.13× |
| **INT4 group32** | **0.1560** | **0.9880** | 0.123 | 1477 | 2.99× |
| INT4 per-token | 0.2772 | 0.9625 | 0.242 | 1357 | 3.26× |

### 2.3 ★ 稀疏 attention（top-512 / 2048，模拟 DSA 的块选择）

这一栏更接近生产（DSA 只算选中的块），但误差**更大** —— 因为注意力更集中，
平均效应减弱：

| 方案 | 输出 rel_err | cosine | **top-k 重合度** |
|---|---:|---:|---:|
| BF16 | 0.0107 | 0.99993 | 0.9991 |
| FP8 E4M3 | 0.0403 | 0.99916 | 0.9933 |
| **INT8 tile128** | **0.0370** | **0.99932** | **0.9939** |
| INT8 group32 | 0.0289 | 0.99958 | 0.9962 |
| **INT4 group128** | **0.452** | **0.889** | **0.896** |
| **INT4 group32** | **0.244** | **0.970** | **0.939** |

> ⚠️ 这一栏的 `top-k 重合度` 是**保守**口径：真实 DSA 的块选择用的是
> **indexer 自己那份 K**（`deepseek_v41/indexer.py:76`，**已经是 INT8**），
> 而这里为了把最坏情况算进去，用**同一个 K** 选。
> ⇒ 真实系统的 top-k 重合度应该**好于**这个表。**这一条要用真机确认**（K8/K4 的任务）。

### 2.4 ★ outlier 敏感性 —— 比"位数"更要命的是"分组"

| 方案 | gauss | t3 | **1% outlier** | logistic |
|---|---:|---:|---:|---:|
| BF16 | 0.0024 | 0.0050 | 0.0042 | 0.0024 |
| **INT8 tile128** | 0.0092 | 0.0174 | **0.0504** | 0.0112 |
| INT8 group32 | 0.0076 | 0.0102 | **0.0208** | 0.0086 |
| INT4 group128 | 0.168 | 0.245 | **0.803** | 0.202 |
| INT4 group32 | 0.139 | 0.157 | **0.369** | 0.155 |

⇒ **同一个 8-bit，tile128 → group32 让 outlier 场景的误差降 2.4×**。
而 per-token（整 512 维一个 scale）在 outlier 下是 **0.0698**，比 tile128 还差。

**结论二：低精 KV 的第一风险不是"用了几 bit"，而是"量化粒度扛不扛得住通道 outlier"。**

### 2.5 对照锚点：模型已经在容忍多大的量化误差？

【实测-仿真】同样是 4-bit 对称量化，作用在**权重**上（典型 std=0.02 的 4096×4096 矩阵）：

| | rel_err |
|---|---:|
| 权重 INT4 group128 | **11.72%** |
| 权重 INT4 group32 | 9.70% |
| 权重 INT8 group128 | 0.65% |

⇒ 这台机器跑的就是 **W4A8**，所以**权重侧已经在吃 ~11.7% 的量化误差**。
把它当尺子：

| KV 方案 | 稠密 rel_err | 是权重锚点的 | 稀疏 rel_err | 是权重锚点的 |
|---|---:|---|---:|---|
| INT8 tile128 | 1.34% | **0.11×**（小一个数量级） | 3.70% | **0.32×** |
| INT4 group32 | 15.6% | **1.33×**（比权重还大） | 24.4% | **2.08×** |
| INT4 group128 | 22.1% | **1.88×** | 45.2% | **3.86×** |

**读法**：
* **INT8 ⇒ 误差是"模型已经接受量级"的 1/9 ~ 1/3**，几乎肯定安全；
* **INT4 ⇒ 误差比权重侧还大 1.3–3.9 倍**，而且 KV 是**逐 token、每步都要读**的，
  误差会**在 40 层里逐层累积**（本仿真只算了单层 attention）⇒ **不能直接接受**。

> ⚠️ **本仿真最大的局限**：只算了**单层**的 attention 输出误差。
> 真实模型 40 层，KV 误差会**逐层传播/放大**（尤其 decoder 层之间有残差）。
> ⇒ **表里的数字是下界**。要判"能不能接受"必须有真机端到端（GSM8K / 困惑度）。

---

## 三、结论与建议

### 3.1 算子侧

| 需求 | A2 上有没有 | 结论 |
|---|---|---|
| per-token INT8 量化（存） | **有**：`torch_npu.npu_dynamic_quant`（内置，A2/A3 都在用） | ✅ |
| 反量化（读） | **有**：逐元素 `mul` | ✅ |
| 融合的"量化 shared-KV attention" | **没有**（A5 专属） | ❌ 但 L1 不需要它 |
| INT4 量化/反量化 | **没有专用算子**（`float4` 在 Ascend 上零命中） | ❌ 需要自己做 pack/unpack |

⇒ **L1（INT8 存储 + 读时反量化）在算子层面是"零新增依赖"的**：
`npu_dynamic_quant` 存、`mul` 反量化、`npu_sparse_attn_sharedkv` 照常算。

### 3.2 精度侧

| 判定 | 依据 |
|---|---|
| **INT8 值得做** | 误差是权重锚点的 **0.11–0.32×**；cosine ≥ 0.9993；top-k 重合度 99.4% |
| **INT4 暂不建议** | 误差是权重锚点的 **1.3–3.9×**，且只算了单层；cosine 掉到 0.89–0.97 |
| **分组粒度比位数更关键** | outlier 场景下 tile128→group32 让 INT8 误差降 2.4× |
| **必须真机验证** | 本仿真是单层 + 合成分布 ⇒ 只能给"量级"，不能给"够不够" |

### 3.3 与容量账的关系（**INT8 的收益被 scale 吃掉一部分**）

```
BF16  : 4 × (512 × 2 B)           = 4096 B   → 总 4421，A2 3.50M token
INT8  : 4 × (512 × 1 B + 8 B scale) = 2080 B   → 总 2405，**1.84×  ⇒ 6.4M token**
INT4  : 4 × (512 × 0.5 + 8 B)     = 1056 B   → 总 1381，**3.20×  ⇒ 11.2M token**
```

（scale 按 fp16、tile128 算。若改用 tile64，INT8 的总量升到 2437 B，增益降到 1.81×
—— **用 0.03× 的容量换 1.2× 的精度改善，很划算**，见 §2.3。）

### 3.4 建议顺序

| 优先 | 动作 | 为什么 |
|---|---|---|
| **P0** | 等 **K8_int8** 的真机结果（真算子 + 真分布） | 把本仿真的"量级估计"换成"实测" |
| **P0** | 若真机也支持 INT8 ⇒ 按 **tile64 或更细** 设计 L1 | §2.4：粒度是抗 outlier 的关键 |
| **P1** | 让 **K4_int4** 在**同一协议**下跑，与本表对齐 | 才能放进同一张表 |
| **P2** | 端到端（GSM8K / 困惑度）—— **本仿真给不了** | 40 层累积效应只有真机能看 |
| **✗ 暂缓** | INT4 | 误差比权重锚点还大，且无算子 |

---

## 四、复现

```bash
cd ~/projects/dsv41/a2
python3 scripts/kv_quant_error_estimate.py --tokens 2048 --heads 32 --trials 3 \
        --out logs/raw/004-quant-error.json
# 约 8 秒（纯 numpy，不需要 NPU）
```

脚本里每个数字都由 `quant_uniform()` / `attention()` / `sparse_attention()` 现场算，
**没有任何手打常数**；JSON 里同时存了逐 trial 的 raw 记录（`raw` / `sparse` 两个数组）。
