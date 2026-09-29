# DP/TP 切法对性能与 KV 容量的影响（A3 单机实测）

**日期**：2026-09-29　**机器**：A3-21，8 chip（chips 0–7，两个配置**共用同一批卡**）
**标记**：【实测】/【推断】/【未确认】

---

## 0. 结论速览

| | A = **TP4/DP2** | B = **TP2/DP4** | 谁赢 |
|---|---|---|---|
| `EP = DP × TP` | **8** | **8** | 平（对照前提） |
| **单 rank KV 容量**（5 GiB 预算） | **1,096,072 tok** | **1,096,072 tok** | **逐位相同** |
| **总 KV 容量**（同预算 5 GiB） | 2.19 M tok | **4.38 M tok** | B 是 A 的 2.00× |
| **总 KV 容量**（各自吃满空闲 HBM，§4.1） | 9.24 M tok | **15.63 M tok** | **B 是 A 的 1.69×** |
| 单流 decode（并发 1） | **41.9 tok/s** | 39.1 tok/s | **A +7.2%** |
| 单流 decode（并发 2） | **41.4 tok/s** | 35.0 tok/s | **A +18.3%** |
| 单流 decode（并发 8） | 29.4 tok/s | **31.0 tok/s** | **B +5.4%** |
| 总吞吐（并发 8） | 221.7 tok/s | **232.8 tok/s** | **B +5.0%** |

### 0.1 两条独立的结论

**① KV 容量：TP 对每 token 字节数完全无效，DP 才是容量的来源**【实测，逐位相同】
两个配置在**同样的 5 GiB 预算**下单 rank 容量**精确相等**（1,096,072 token，来自
服务端自报的 `GPU KV cache size`）⇒ 每 token 字节数相同（4898 B/tok）
⇒ **TP 不改变 KV 的单位成本**。

但"总容量 B = A 的 2 倍"**只在预算相同时成立**。若各自吃满空闲 HBM，
因为降 TP 会让非专家权重多复制 3.24 GiB/rank，真实比值是 **1.69×**（不是 2×），
相对 TP8 是 **3.14×**（不是 4×）。推导见 §4.1，权重账见 §5.1。

**② 性能：交叉点在并发 4–8 之间**【实测，单次采样见 §7】
- **低并发（1–2）A 明显更快**：并发 2 时 A 领先 18.3%
- **高并发（8）B 略胜**：单流 +5.4%、总吞吐 +5.0%
- 即：**加 DP 会牺牲低并发单流速度，换取高并发吞吐和 KV 容量。**

这对"该不该上 DP4/TP2"是个**权衡**，不是单纯的赢或输 —— 取决于你的负载偏交互
（低并发、要快）还是偏吞吐（高并发、要容量）。

### 0.2 最硬的一条证据

【实测】**单 rank 覆盖 1M 上下文所需的 KV，在 TP4/DP2 与 TP2/DP4 上完全相同（都是 4.78 GiB）**
—— 这是两个配置**各自独立**起服时被 vLLM 拒绝所附的数字（详见 §4）。

机制（vLLM 通用逻辑，镜像里 200+ 个模型文件都是这一句）：

```python
num_kv_heads = max(1, total_num_kv_heads // tp_size)
```

本模型 `num_key_value_heads = 1`（MLA 把 KV 压成一个 latent）⇒ TP>1 时
`max(1, 1//TP) = max(1, 0) = 1` ⇒ **每个 rank 仍存完整 KV**。

⇒ **推论**：TP 只增加 KV 的**副本数**（KV HBM 效率 = 1/TP，**TP8 时 87.5% 是浪费**），
不增加容量。**想让容量增长只有两条路：加 DP、或用 DCP 沿序列维切**（§5.4）。

---

## 1. 实验设计（为什么这样设计）

### 1.1 唯一变量 = DP/TP 切法

```
A: TP4/DP2   →  EP = 2 × 4 = 8
B: TP2/DP4   →  EP = 4 × 2 = 8
```

**两者 EP 都是 8**，而专家权重按 EP 切分、且 MoE 占模型总参数的 98%
（543 B / 552 B，从 `config.json` 的 `n_routed_experts=384`、`moe_intermediate_size=2304`、
`hidden_size=5120`、40 层逐位算出）
⇒ **专家权重的切分方式完全不变**，唯一变化的是那 3% 非专家权重按 TP 切分的方式，
以及 TP/DP 的通信拓扑。

### 1.2 两个配置共用同一批卡

A 与 B 都在 **chips 0–7** 上先后运行（不是一半一半）。
理由：两个半区的 PCIe/NUMA 未必对称，同卡先后跑能把这个变量消掉。
代价是起服时间翻倍。

### 1.3 一致的参数（两个配置逐字相同）

| 项 | 值 |
|---|---|
| `SPEC` | **0**（关推测解码 —— 去掉 DSpark 的干扰，A 恒 1.0） |
| `MAX_SEQS` | 16 |
| `MAX_LEN` | 1048576 |
| `BAT_TOKENS` | 8192 |
| `GPU_UTIL` | 0.85 |
| `KV_CACHE_MEMORY_BYTES` | 5 GiB |
| `ENGRAM_DEVICE_INDEX` | 0 |
| `DROPCACHE` | 0 |
| `PATCH_MODE` | mount |
| 负载 | 1024 prompt / 128 output，`ignore_eos`，并发 1/2/4/8 |

---

## 2. A 臂（TP4/DP2）实测

【实测】`results/dptp_seq_0929_103149/A_bench.log`

| 并发 | 单流 tok/s | 总吞吐 tok/s | 加速比 | 单流效率 | TTFT |
|---:|---:|---:|---:|---:|---:|
| 1 | **41.9** | 41.7 | 1.00× | 100.0% | 0.22 s |
| 2 | **41.4** | 83.5 | 2.00× | 99.0% | 0.21 s |
| 4 | 34.8 | 136.3 | 3.27× | 83.0% | 0.30 s |
| 8 | 29.4 | **221.7** | 5.32× | 70.2% | 0.46 s |

（`ok=8/8` 全部成功，无错误）

---

## 3. B 臂（TP2/DP4）实测

【实测】`results/dptp_seq_0929_103149/B_bench.log`

| 并发 | 单流 tok/s | 总吞吐 tok/s | 加速比 | 单流效率 | TTFT |
|---:|---:|---:|---:|---:|---:|
| 1 | 39.1 | 39.4 | 1.00× | 100.0% | 0.27 s |
| 2 | 35.0 | 69.6 | 1.77× | 89.5% | 0.28 s |
| 4 | 34.6 | 131.5 | 3.34× | 88.5% | 0.34 s |
| 8 | **31.0** | **232.8** | 5.92× | 79.4% | 0.57 s |

### 3.1 两臂并排

| 并发 | A 单流 | B 单流 | Δ(B−A) | A 总吞吐 | B 总吞吐 | Δ(B−A) | A 单流效率 | B 单流效率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **41.9** | 39.1 | **−6.7%** | 41.7 | 39.4 | −5.5% | 100% | 100% |
| 2 | **41.4** | 35.0 | **−15.5%** | 83.5 | 69.6 | −16.6% | 99.0% | 89.5% |
| 4 | 34.8 | 34.6 | −0.6% | 136.3 | 131.5 | −3.5% | 83.0% | 88.5% |
| 8 | 29.4 | **31.0** | **+5.4%** | 221.7 | **232.8** | **+5.0%** | 70.2% | 79.4% |

**读法**：
- A 的曲线在并发 2 处**几乎完美线性**（99.0% 效率），到并发 8 掉到 70.2%；
- B 的曲线从并发 2 就开始掉（89.5%），但**到并发 8 只掉到 79.4%** —— 曲线更"平"；
- ⇒ **A 在低并发更省，B 在高并发更抗压。** 交叉点在 4–8 之间。

### 3.2 完整三元组（ms/step、A、tok/s）

本轮 **`SPEC=0`**，即**没有推测解码** ⇒ 每步恰好产出 1 个 token
⇒ **`A ≡ 1.0`（无草稿，谈不上接受长度）**，且 `ms/step ≡ ms/token = 1000 / 单流 tok/s`。

| 并发 | A ms/step | A tok/s | B ms/step | B tok/s | A | Δms/step(B−A) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **23.87** | 41.9 | 25.58 | 39.1 | 1.0 | **+7.2%** |
| 2 | **24.15** | 41.4 | 28.57 | 35.0 | 1.0 | **+18.3%** |
| 4 | 28.74 | 34.8 | 28.90 | 34.6 | 1.0 | +0.6% |
| 8 | 34.01 | 29.4 | **32.26** | 31.0 | 1.0 | **−5.1%** |

（`ms/step` 由单流 tok/s 反算；两者在本轮口径下等价。**注意**：与历史上带 DSpark
的数字不可比 —— 那里 `A≈2.8`，`ms/step` 与 `tok/s` 的关系是
`tok/s = A × 1000 / ms_per_step`，不是 `1000 / ms_per_step`。）

**总吞吐侧（聚合，含并发摊薄）**：

| 并发 | A 总吞吐 tok/s | B 总吞吐 tok/s | Δ(B−A) |
|---:|---:|---:|---:|
| 1 | 41.7 | 39.4 | −5.5% |
| 2 | 83.5 | 69.6 | −16.6% |
| 4 | 136.3 | 131.5 | −3.5% |
| 8 | 221.7 | **232.8** | **+5.0%** |

---

## 4. KV 容量的实测证据

【实测】两个配置在第一次起服时**都被 vLLM 拒绝**，报错逐字相同：

```
ValueError: To serve at least one request with the model's max seq len (1048576),
(4.78 GiB KV cache is needed, which is larger than the available KV cache memory (4.0 GiB).
Based on the available memory, the estimated maximum model length is 849536.
```

两次报错（A 与 B）**数值完全相同**：都需要 **4.78 GiB**、都只能覆盖 **849536 token**。

由此可算每 token 字节数：
```
4.0 GiB / 849536 tok = 5055 B/tok
```
与另一次独立测量交叉验证（TP8/DP1，KV=14.65 GiB → 3,168,174 token）：
```
15728022528 B / 3168174 tok = 4964 B/tok
```
两者差 1.8%（padding/对齐差异），量级一致 ⇒ **BF16 KV ≈ 5 KB/token/rank**。

★ 注意：官方文档说 "890 bytes per token"，那是 **FP4 主 KV** 下的数字（§1 明确写 "FP4 main KV cache"），
比我们的 BF16 小 5.6 倍。**两者不可混用。**

### 4.0 ★★ 最强的一条证据：两个配置的单 rank 容量**逐位相同**

【实测】A 与 B 的 serve.log 里服务端自报的 `GPU KV cache size`：

```
A (TP4/DP2, 5 GiB 预算): GPU KV cache size: 1,096,072 tokens
B (TP2/DP4, 5 GiB 预算): GPU KV cache size: 1,096,072 tokens
```

**两者精确相等（不是"接近"，是同一个数）。** 这就是"MLA 的 KV cache 完全不随 TP
切分"的直接实测证据 —— 如果 KV 像 GQA 那样按 head 切，TP4 的单 rank 容量应该是
TP2 的一半。它没有。

由此算每 token 字节数：`5368709120 B / 1096072 tok = 4898 B/tok`，
与 §4 的另两次独立测量（4964 / 5055 B/tok）落在同一档（差异 ≤3%，来自 padding 与对齐）。

### 4.1 总容量的正确算法（★ 修正一处早期的简化）

§0 那条"总容量 = DP × 单 rank"**只有在每 rank 可用 HBM 相同时才成立**。实际上
**TP 越小、非专家权重被复制得越多、留给 KV 的 HBM 越少**，所以 DP 的收益会被部分抵消。

正确的算法要三步（每 die HBM = **61.27 GiB**，日志实测）：

```
单 rank 可用 KV = 61.27 GiB − 权重/rank − 固定开销
总 KV 容量     = DP × (单 rank 可用 KV ÷ 每 token 字节数)
```

固定开销 = **4.56 GiB**（实测：激活 3.15 + 非 torch 0.63 + NPU 图 0.78）；
每 token 字节数 = **4898 B**（本次实测 `5 GiB → 1,096,072 token` 反算）。
权重/rank 见 §5.1 的逐张量账。

| 配置 | 权重/rank | 可用 KV/rank | 单 rank 容量 | DP | **总 KV 容量** | 相对 A |
|---|---:|---:|---:|---:|---:|---:|
| **A = TP4/DP2** | 35.64 GiB | 21.07 GiB | 4.62 M tok | 2 | **9.24 M tok** | 1.00× |
| **B = TP2/DP4** | 38.88 GiB | 17.83 GiB | 3.91 M tok | 4 | **15.63 M tok** | **1.69×** |
| 参照 TP8/DP1 | 34.02 GiB | 22.69 GiB | 4.97 M tok | 1 | 4.97 M tok | 0.54× |
| 参照 TP1/DP8 | 45.36 GiB | 11.35 GiB | 2.49 M tok | 8 | 19.90 M tok | 2.16× |

### 4.2 ★ 直接回答"B 是不是 A 的 2 倍、TP8 的 4 倍"

**都不是。**

| 问题 | 答案 | 为什么 |
|---|---|---|
| B 是不是 A 的 **2 倍**？ | **1.69 倍** | DP 翻倍（2→4）但 TP 减半，非专家权重从 3.24 涨到 6.48 GiB/rank ⇒ 每 rank 少 3.24 GiB 可给 KV |
| B 是不是 TP8 的 **4 倍**？ | **3.14 倍** | TP8→TP2 让非专家从 1.62 涨到 6.48 GiB/rank（+4.86），再加上 DP 4 vs 1 |

**净效应可以写成**：

```
总 KV ∝ DP × ( 61.27 − 32.40 − 12.96/TP − 4.56 )
                HBM    专家     非专家    固定开销
              （不变） （不变）  （TP↓则↑）（不变）
```
其中「非专家」那一项**随 TP 变小而变大**，是抵消 DP 收益的唯一来源。

⇒ **加 DP 的同时降 TP，KV 收益会被"非专家复制"吃掉一部分**（本例吃掉约 16%）。


---

## 5. KV cache 到底是怎么放的（逐张量权重账 + 13 个 group）

### 5.1 权重账（从 checkpoint 逐张量算出，不是估算）

【实测】解析 `quant_model_weights-*.safetensors` 全部 80 个分片的 header
（按 dtype × shape 累加字节，不需要读 273 GB 的权重本体）：

| 类别 | 大小 | 占比 | 切分方式 |
|---|---:|---:|---|
| **routed experts**（`layers.N.ffn.experts.M.w{1,2,3}.*`） | **259.19 GiB** | **95.2%** | 按 **EP** |
| attention 等（MLA / indexer / compressor / norm） | 10.19 GiB | 3.7% | 按 **TP** |
| shared experts | 1.32 GiB | 0.5% | 按 TP |
| embed + lm_head | 1.23 GiB | 0.5% | 按 TP |
| 其它 | 0.22 GiB | 0.1% | — |
| **合计** | **272.15 GiB** | 100% | |

⇒ **非专家权重只有 12.96 GiB（4.8%）。**

因为 `EP = DP × TP`，**A 与 B 的 EP 都是 8 ⇒ 那 259 GiB 专家权重完全不变**，
变的只有这 4.8%：

| 配置 | 专家/rank | 非专家/rank | **权重/rank** |
|---|---:|---:|---:|
| TP8/DP1 | 32.40 | 1.62 | 34.02 GiB |
| **A = TP4/DP2** | 32.40 | 3.24 | **35.64 GiB** |
| **B = TP2/DP4** | 32.40 | 6.48 | **38.88 GiB** |
| TP1/DP8 | 32.40 | 12.96 | 45.36 GiB |

（`routed_experts / 8`、`non_expert / TP`）

### 5.2 13 个 KV group 的切分行为（逐组不同）

【实测·结构】来源 `a2/logs/001-dsv41-dram-offload-8card.md` §3（第一手清单）：

| # | group | 内容 | block_size | 按 token 增长 | TP 维度 |
|---|---|---|---|---|---|
| 0 | `full` | 4× `layers.{2,8,14,20}.long_kv_cache` + 4× 同名 `.indexer.k_cache` | 128 | ✅ 是 | **复制** |
| 1 | `state` | 3× `layers.{2,8,14}.compressor.state_cache`（FP32 **32 行环**） | 32 | ❌ **每请求固定 1 页** | **复制** |
| 2–11 | `swa0`…`swa9` | 40 个 SWA 资源（window=**128**）按 slot 轮转分 10 组、每组 4 层 | 128 | ⚠️ 每组每请求 ≥1 页 | **复制** |
| 12 | `dspark` | 3× `mtp.{0,1,2}` draft SWA（aliasing 到 target slot） | 128 | ⚠️ 同上 | **复制** |

★ **全部 13 组在 TP 维度上都是复制的** —— 这不是从代码读出来的，是**本次 A/B 直接测出来的**：
TP4/DP2 与 TP2/DP4 的单 rank 容量**逐位相同**（都是 `1,096,072 tokens`）。
如果**任何一组**按 TP 切，这两个数就该差 2 倍。

### 5.3 ★ 是否存在"必须复制"的情况：**有，三种**

#### 必须复制 ①：MLA 在 TP 下【必然，我们正踩】

vLLM 的通用逻辑（镜像里 200+ 个模型文件都是这一句）：

```python
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
```

我们模型 `num_key_value_heads = 1`（MLA 把 KV 压成**一个** latent）：

| TP | `max(1, 1//TP)` | 结果 |
|---:|---:|---|
| 1 | max(1, 1) = 1 | — |
| 2 | max(1, **0**) = 1 | **完整复制** |
| 4 | max(1, **0**) = 1 | **完整复制** |
| 8 | max(1, **0**) = 1 | **完整复制** |

**1 个 head 除不动 TP ⇒ 只能复制。** 对比 GQA 模型（8 个 KV head）：TP8 时每 rank 1 head，
完美切分。**MLA 拿不到这个好处** —— 这是 MLA "省了 KV 带宽/体积" 付出的对价。

**代价可以写成一个干净的公式**：

```
KV 的 HBM 效率 = 1 / TP        （只有 DP 维度带来有效容量增长）
```

| TP | KV HBM 的浪费比例 |
|---:|---:|
| 8 | **87.5%** |
| 4 | 75% |
| 2 | 50% |
| 1 | 0% |

★ **我们现在 CED-PD 是 TP8** ⇒ 8 张卡**各存一份完整 KV**，87.5% 的 KV 显存是纯浪费。

#### 必须复制 ②：DP 维度【必然，但这是特性不是浪费】

每个 DP rank 是**独立引擎**（`group_ranks` 里 DP 是最外层），服务**不同的请求**，
KV 天然不共享。⇒ 总容量 = `DP × 单 rank`，这是**线性收益**，不是冗余。

#### 必须复制 ③：PD 分离的两个引擎各持一份

P 算完的 KV 经 `kv_producer` → `kv_consumer` 传给 D。CED 形态下 P 只产出 layer 0–19 的 KV、
D 收下后做 128-token bounded replay（见 `experimental/ced/mooncake_hybrid_connector.py`
与 `core_scheduler_replay.patch`）。

#### 附带两个"每请求固定"的隐性成本（不是复制，但同样吃容量）

| 项 | 成本 | 影响 |
|---|---|---|
| `state` 组 | **每请求固定 1 页**（32 行 FP32 环），且 `prefix_cacheable = False` | 低并发长上下文可忽略；**高并发短请求时会主导** |
| SWA 组（含 dspark） | window=128 ⇒ 每组每请求至少 1 页 | 同上 |

### 5.4 能不能不复制？两条路

#### 路 1：TP 按 head 切 —— ❌ 对我们**不可用**

需要 `num_kv_heads ≥ tp_size`。MLA 只有 1 个 head，**物理上不可切**。

#### 路 2：DCP（Decode Context Parallel）—— ✅ **MLA 唯一的出路**

沿**序列维**切，而不是 head 维。代码依据（`vllm/v1/kv_cache_interface.py:228`）：

```python
def max_num_blocks_per_req(self, vllm_config, max_len):
    kv_shard_count = parallel_config.decode_context_parallel_size
    return cdiv(max_len, self.block_size * kv_shard_count)   # ← 除以 DCP
```

官方支持矩阵（`docs/source/user_guide/feature_guide/context_parallel.md`，我拉的 `main@b64b4d7`）：

| Device | Attention Backend | Chunked Prefill | Prefix Caching | Graph | **P/D 分离** |
|---|---|---|---|---|---|
| **A2/A3** | **MLA/GQA** | ✅ Full | ✅ Full | ✅ Full | **✅ Full** |

CLI（官方 Usage 段逐字）：

```bash
--decode-context-parallel-size <N>
```

**收益是双重的**：

1. **省 HBM**：DCP=N 时每 rank 只存 1/N 的序列 ⇒ **直接修掉 §5.3 那个 `1/TP` 的浪费**
2. **能服务更长上下文**：单 rank 不再需要装下整个 1M 序列

#### 顺带发现（对我们**不可用**）

官方另有 `sparse_kv_offload`（KV 卸载到 host，仅 D 节点、需 PD 分离），
但它的前置检查会**拒绝我们的模型**：

```python
if hasattr(vllm_config.model_config.hf_text_config, "compress_ratios"):
    raise ValueError("Sparse KV offload don't support compress now.")
```

我们模型 `compress_ratios = [0,0,2,2,...,1,...,0,0,0]` 存在 ⇒ **直接被拒**。


### 5.5 小结：三种复制、两条路、一个首选

| 复制/浪费 | 性质 | 能否消除 |
|---|---|---|
| **① MLA 在 TP 下** | **成本**（`max(1, 1//TP)` 除不动） | ✅ **DCP 可以**；按 head 切不行 |
| **② DP 维度** | **特性**（不同引擎服务不同请求） | ❌ 也不该消除（它就是容量来源） |
| **③ P/D 两引擎各持一份** | **架构代价**（KV 要从 P 搬到 D） | ❌ 消除不了，只能靠 CED 只搬必要层来压小 |

**可行动项**（按投入产出比排序）：

1. ★★★ **给 D 侧加 DCP**（`--decode-context-parallel-size N`）—— 官方对
   **A3 + MLA + P/D 分离**标的是 **Full compatibility**。它**同时**解决两件事：
   修掉 §5.3 那个 `1/TP` 的 KV 浪费（TP8 时 87.5%），以及让单 rank 不必装下整个 1M 序列。
   这是目前看到的**性价比最高的一条**。
   ⚠️ 唯一需要先验的是：**CED 的 P→D 交接在 DCP 下是否仍然正确**
   （CED 特殊在 P 只产出 layer 0–19、D 做 128-token bounded replay，
    与 DCP 的"序列切分 + allgather 拼上下文"要同时成立）。官方矩阵里那格是
    通用 MLA 的结论，**没有覆盖我们的 CED 扩展** ⇒ 属【未确认】。
2. ★★ **优先给 D 加 DP**（而不是 P）—— P 侧权重只有全模型的一半左右、显存极宽裕，
   是**算力/带宽受限**；D 侧才是显存受限（全 40 层权重 + 全部 KV）。
   §4.1 的账也支持：DP 的收益全在 D。
3. ★ **别用 TP 去换容量** —— 实测已证 TP 对每 token 字节数无效，只增加副本。
   若必须用大 TP（为了单请求算力），就接受 `1/TP` 的 KV 效率损失。
4. ❌ **`sparse_kv_offload` 不用考虑**（`compress_ratios` 存在 ⇒ 官方直接 raise）。

---

## 6. 待验证的【推断】（本次实验能回答的部分）

| # | 推断 | 本实验如何回答 |
|---|---|---|
| 1 | TP8→TP2 让 allReduce 变便宜（参与者 8→2，ring 长度 1.75→1.0） | B 的单流 tok/s 应与 A 相当或更好 |
| 2 | DP 让单流从"高并发区间"回到"低并发区间" | 已由 A 的曲线支持（见 §2）；B 的曲线会更平 |
| 3 | 空闲 DP rank 的 dummy batch（`_dummy_run(uniform_decode_query_len)`）有代价 | B 的并发 1 单流若明显低于 A，则该代价可见 |
| 4 | MoE 的 all2all 跨 DP 全局，是逐 step 锁步的 | 两者并发 1 的差异体现 |

---

## 7. 实验进度：两个配置均已测完【实测】

| 项 | 状态 |
|---|---|
| A = TP4/DP2 (EP=8) | ✅ 已测（并发 1/2/4/8） |
| B = TP2/DP4 (EP=8) | ✅ 已测（并发 1/2/4/8） |
| KV 容量 | ✅ 两配置单 rank 逐位相同 + 总容量 2× |
| 起服日志 | `results/dptp_0929_103149/`（A）、`results/dptp_seq_0929_103149/`（B） |
| 原始 JSON | 各自目录下的 `A_bench.json` / `B_bench.json` |

### 7.1 踩过的坑（4 次起服失败，全部有明确根因）

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| 1 | A 起服被拒：`Free memory on device (54.42/61.27 GiB) less than desired` | **device 2（Phy-ID 2）有 6.85 GB 驱动级残留**（`npu-smi` 显示 "No process in device" 但 HBM 9744 MB vs 其它 ~2890 MB）；不是我们的容器 | `GPU_UTIL` 0.92→0.85 |
| 2 | A/B 都拒：`4.78 GiB KV needed > available 4.0 GiB` | 我给 KV 设了 4 GiB，而 1M 上下文单 rank 需 4.78 | 改 5 GiB（**这个数字本身就是 §4 的实测证据**） |
| 3 | A/B 都超时：`TimeoutError: Timed out waiting for engine core processes to start` | 两个实例并行起 = 6 个 EngineCore 抢同一台机的磁盘（读 490 GB 权重）+ CPU | (a) 加 `VLLM_ENGINE_READY_TIMEOUT_S=7200`；(b) **改成串行起** |
| 4 | 起服被拒：`选中的卡里有正在被占用的：7` | a3-21 是**共用机**，别人的临时进程（`python`，139 MB）间歇占用 device 7 | 加"等卡空闲 + 重试"（最多 60 min），**不抢别人的卡** |

### 7.2 当前的起服代价（给后续实验参考）

- 单实例冷启动 ≈ **15–20 min**（静态内核 + 图捕获）
- 串行跑两个配置 ≈ **40–50 min**（含测量）
- 两个实例并行起 **不可行**（坑 #3）

---

## 8. 尚未确认的（诚实边界）

### 8.1 性能数据的边界

1. **每次并发只采一次（`--repeats=1`）**，没有做重复取中位数。因此 §3.1 里
   ±5% 量级的差异（如并发 8 的 +5.4%）**可能在噪声范围内**；而并发 1–2 的
   6.7% / 15.5% 差距较大，更可信。要确认 ±5% 那几格需要重跑
   （每个配置 ~20 min 起服 + ~5 min 测量）。
2. 本轮 **SPEC=0**（关推测解码），所以结果里没有 DSpark；
   与历史带 DSpark 的数字（如 90.3 tok/s）**不可直接比较**。
   ★ 而且 DSpark 会改变空闲 rank 的 dummy batch 大小
   （`_dummy_run(uniform_decode_query_len)`，K=7 时是 8 个 token），
   对 **B 这种 DP 更大的配置影响可能不同** ⇒ 带 SPEC 的 A/B 需要重测。
3. **A 与 B 共用 chips 0–7，其中 device 2 带 6.85 GB 驱动级残留**（非我们的进程，
   容器全清后仍在）—— 两个配置受同样影响，所以对照公平；但绝对性能低于干净机器。

### 8.2 容量与权重账的边界

4. **§4.1 的"总 KV 容量"是估算，不是本轮实测**：本轮把 KV 固定成 5 GiB 后
   vLLM **跳过了显存 profiling**（日志逐字：`reserved 5.00 GiB for KV Cache as
   specified by kv_cache_memory_bytes, skipping memory profiling`），
   所以没有本轮的 weights / activation 实测行。
   - 权重：**从 checkpoint header 逐张量算的**（可信度高）；
   - 固定开销 4.56 GiB：**借用另一次 TP2/EP2 配置**的 profiling 值 ——
     EP8 下通信 buffer 更大，**真值可能更高 ⇒ 我算的容量可能偏高**。
5. **每 token 4898 B 含 padding 与对齐**，且其中 `state`（每请求固定 1 页）与
   SWA（window=128，每组每请求 ≥1 页）两个"每请求固定"成本的占比**未单独量化** ——
   它们对"低并发长上下文"影响小、对"高并发短请求"影响大。
6. **权重 272.15 GiB 只含 `quant_model_weights`**，未含 MTP 的 9.9 GB
   （`mtpq-*.safetensors`）与 host 上的 Engram 206 GB。
7. **每 token 5 KB 是两次独立测量的交叉验证**（5055 / 4964 / 4898 B），
   最大差 3.2%，未定位是 padding、对齐还是别的。

### 8.3 DCP 那条路的关键未知

8. ★ **官方支持矩阵里"A3 + MLA + P/D 分离 = Full compatibility"是通用 MLA 的结论，
   没有覆盖我们的 CED 扩展。** CED 特殊在 P 只产出 layer 0–19 的 KV、
   D 收下后做 128-token bounded replay，而 DCP 要做**序列维切分 + 跨 rank 拼上下文**；
   两者能否同时成立**属【未确认】**，必须先做小规模验证再上线。
9. DCP 与 CED 的 `ced_prefix_tokens` / `ced_missing_swa_groups` 这套交接标记
   在序列切分下如何对应，**未查**。
