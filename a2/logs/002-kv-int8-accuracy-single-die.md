# 002 — 单 die 验证 8-bit KV cache 精度（K8_int8）

> 2026-09-21。机器：**A3（A3-node1）槽位 c0 = die 3，910C**。
> **⚠️ 这是 A3 单 die，不是 A2 实机** —— A2（内网 8×910B3）本机 ssh 不可达。
> 选 A3 单 die 做代理的理由：算子支持矩阵里 `kv_quant_sparse_flash_attention` 在
> **A2/A3 都是 √**，单 die 与 A2 的算子可用性一致（**A2 侧仍未实机确认，见 §9**）。
> 本次没有动任何容器、没有碰 Phy-ID 8–15、没有写 `/tmp`、没有手设 `ASCEND_RT_VISIBLE_DEVICES`。

---

## 0. 结论摘要（先看这 6 行）

1. **【实测】算子存在且可跑**：nightly 单卡测试在 c0 上一次通过（`1 passed in 4.29s`）；
   INT8 KV 前向在本机 **逐比特可复现**（跑两遍 `max_abs = 0.0`）。
2. **【实测】INT8 的精度代价很小**：BF16 KV vs INT8 KV（都是真算子、同一批 Q）
   `cos_p50 = 0.999980`、`cos_p99 = 0.999982`、`cos_min = 0.999976`，
   相对 L2 `p50 = 6.3e-3 / p99 = 6.9e-3`，`max_abs = 4.9e-4`（输出 RMS 0.016）。
3. **【实测】噪声地板**：算子自身跑两遍 **差值恒等于 0**；把「无损 int8（能精确还原 BF16 值）」
   喂进同一个算子，算子 vs 精确 golden 的差是 `rel_p99 = 2.7e-3`、`cos = 0.999997`
   —— **量化带来的误差（6.4e-3）是算子自身数值噪声（2.7e-3）的 ~2.4 倍**，两者都远小于任何有意义的模型精度阈值。
4. **★【实测】V4.1 用不了这个算子 —— 卡在形状，不是精度。**
   `npu_kv_quant_sparse_flash_attention` **只接受 `q_head_dim = 576`（nope 512 + rope 64）**；
   V4.1 的 `config.head_dim = 512`、`qk_rope_head_dim = 64`、**`nope_head_dim = 448`**
   （rope 是在**同一个 512 宽的 latent 内部**旋转最后 64 维，见 `model.py:361-373`）。
   ⇒ 这不是「换算子」能解决的，需要**改模型几何 + 改算子**（§4）。
5. **【实测】两个算子都彻底忽略 `value` 入参**：把 `value` 置成全 0，输出**逐比特不变**。
   V 实际取的是 K 的 int8 段（MLA 语义）。⇒ V4.1 的 V 是整个 512 宽 latent（含 rope 那段），
   而该算子的 V = int8 nope 段（V4.1 只有 448 维）⇒ **V 表达不出来**。
6. **【实测】容量收益取决于几何**，见 §5：按真实 V4.1 几何 **39.1%**（4421 → 2693 B/token）；
   按算子要求的 576 几何算是 33.3%。

---

## 1. 必答 ① —— 算子是否存在且可调用

| 项 | 结果 | 证据 |
|---|---|---|
| `torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention` 存在 | 【实测】✅ | nightly 测试通过 |
| 在**单 die（910C）**能跑起来 | 【实测】✅ | 见下 |
| CANN 侧 kernel 支持本硬件 | 【实测】✅ | **运行期证据**：c0 上 31 个 case 的量化算子前向全部返回合理结果（不是报错），直接证明 910C 有可用 kernel。旁证（**只查到一次，第二次 `find` 没复现，谨慎采信**）：`.../kernel/config/ascend910_93/ops_transformer/kv_quant_sparse_flash_attention.json` |
| A2（910B3）侧同样支持 | **【未确认】** | 支持矩阵写 A2 √，但本机只能看到 A3 的 CANN；A2 的 CANN 安装不在我可达范围 |

```
[a3_chip] slot=c0 container=prbench-c0 task=k8-int8-gate start=2026-09-21 22:44:56
.                                                                        [100%]
1 passed, 14 warnings in 4.29s
```

原始日志：`a2/logs/raw/002-gate_nightly_test.log`（同目录 `.gate_out`）。

## 2. 必答 ② —— INT8 量化的精度损失

### 2.1 协议（严格按要求的四条腿）

| 要求的臂 | 本次怎么实现 | 说明 |
|---|---|---|
| **参考臂 BF16 KV，不量化** | ① 精确 fp32 golden（BF16 KV）；② `npu_sparse_flash_attention`（**非量化**、A2/A3 都支持的真算子，BF16 KV） | ②是「同父系」的非量化算子，用来给端到端一个有物理意义的对手 |
| **空测 BF16 vs BF16** | **N1**：同一份输入过同一个算子两遍；**N2**：`lossless` 数据（int8 值 × 2⁻⁷，BF16 可精确表示 ⇒ 解码后与 BF16 逐比特相同）过算子 vs 精确 golden | **字面意义的「BF16 过同一个算子」做不到**：`npu_kv_quant_sparse_flash_attention` 按定义只收 packed int8 KV（喂 BF16 直接报错）。N2 是更强的替代 —— 数据无损，差的只有 kernel 数值 |
| **测试臂 INT8** | 对称量化、per-token-head-tile、带 scale，真的喂给那个算子 | tile **只能 128**（见 2.4），64 被算子拒绝 |
| 量化误差本身 | golden(INT8 反量化 KV) vs golden(BF16 KV) | 纯量化误差，不含算子实现差 |

形状取值：`block_size ∈ {128,256}`、`sparse_block_size=1`、`sparse_mode=3`(causal)、
`attention_mode=2`、`quant_scale_repo_mode=1`、`key/value_quant_mode=2`，
query heads 64、kv heads 1，topk 见各表。**参数取自 nightly 测试与 `dsa_v41.py`，没有自己编。**

### 2.2 主表：算子真能跑的形状（nope 512 + rope 64，即 V4/V3 的 MLA 形状）

数据：合成高斯（q~N(0,1)、latent~N(0,0.5)），tile=128、block=256、topk=2048、batch=1。
对照列 `Q` = 纯量化误差；`地板 N2` = 空测；`端到端` = INT8 算子 vs BF16 算子（**同一批 Q**）。

| 场景 | **Q** rel_p99 / cos_p50 | **地板 N2** rel_p99 / cos_p50 | **端到端** rel_p99 / cos_p50 / cos_min | max_abs |
|---|---|---|---|---|
| decode 4K  b1 | 6.36e-3 / 0.9999828 | 2.70e-3 / 0.9999970 | **6.86e-3 / 0.9999799 / 0.999976** | 4.9e-4 |
| decode 32K b1 | 6.29e-3 / 0.9999828 | 2.72e-3 / 0.9999970 | **6.76e-3 / 0.9999802 / 0.999977** | 4.9e-4 |
| decode 128K b1 | 6.24e-3 / 0.9999827 | — | 6.53e-3 / 0.9999816（golden 口径，见注①） | 3.9e-4 |
| decode 4K  b8 | 6.50e-3 / 0.9999825 | — | 6.76e-3 / 0.9999811（golden 口径，见注②） | 4.7e-4 |
| prefill 4K  b1 | 8.22e-3 / 0.9999807 | 3.98e-2* / 0.9999969 | **8.47e-3 / 0.9999779 / 0.999953** | 5.9e-3 |
| prefill 32K b1 | 7.84e-3 / 0.9999813 | — | **8.09e-3 / 0.9999799** | 2.4e-3 |
| prefill 128K b1 | 7.97e-3 / 0.9999812 | — | **8.12e-3 / 0.9999797** | 2.0e-3 |
| prefill 4K b8 | 8.09e-3 / 0.9999814 | — | 1.04e-1 / 0.9999800（golden 口径，见注②） | — |
| decode 4K b1 block=128 | 6.36e-3（与 256 相同） | — | 6.7e-3（与 256 相同） | 4.2e-4 |
| **更尖的 softmax**（K 幅度 ×4） | 1.76e-2 / 0.9999542 | — | **1.79e-2 / 0.9999518 / 0.999883** | 3.1e-2 |
| nightly 分布（uniform(-5,10)） | 1.93e-2 / 0.9999739 | — | 2.0e-2 / 0.9999728 | 1.9e-1 |

**注①** decode 128K 那行的"端到端"用的是「量化算子 vs BF16 **golden**」（6.53e-3 / 0.9999816），因为 BF16 算子在 128K 上的独立臂没单独跑。
**注②** batch=8：`npu_sparse_flash_attention` 在这次调用形态下**自身就没跑对**（TND 多请求 layout 没配对，见 §9-③），
所以 b8 两行给的是 **golden 口径**（量化算子 vs BF16 golden）：decode = 6.76e-3 / 0.9999811；
prefill = 1.04e-1 / 0.9999800（p99 被少数「序列开头、有效 token 极少」的行拉高，cos 仍 0.99998）。
**注③** prefill 的 rel_p99 比 p50 高（4.2e-2 vs 2.5e-3）是同一个原因：causal + 下标落在序列开头几位时有效稀疏 token 极少，分母小。cos 指标不受影响。
**注④** prefill 32K / 128K 两行的"端到端"也是「量化算子 vs BF16 **golden**」（8.09e-3 / 8.12e-3），因为那两例没跑独立的 BF16 算子臂；同例 `Q` 列分别是 7.84e-3 / 7.97e-3，可见算子只额外贡献 ~0.2e-3。
**`*`** 那个 3.98e-2 是无损数据 prefill 的 p99，同样被退化行拉高；该例 cos_p50 仍是 0.9999969。

**逐行原始数据：`a2/logs/raw/002-grid5.json`、`002-grid6.json`。**

### 2.3 空测（噪声地板）明细

| 空测 | 结果 | 判读 |
|---|---|---|
| **N1** INT8 算子跑两遍 | `max_abs = 0.000e+00`（全部 31 个 case） | **【实测】逐比特确定**，算子无随机性 ⇒ 「跑两遍」这条路的地板是 0 |
| **N2** `lossless` 数据下 golden(INT8) vs golden(BF16) | `rel_p99 = 0.000e+00`、`cos = 1.0000000` | **【实测】**无损构造成立（int8×2⁻⁷ 在 BF16 里精确），量化臂与 BF16 臂的数据**逐比特相同** |
| **N2'** 同一份无损数据，**算子** vs 精确 golden | decode `rel_p99 = 2.69e-3`、`cos_p50 = 0.9999970`、`cos_min = 0.999996` | **【实测】算子自身的 kernel 数值噪声地板** |

⇒ **判定规则**：任何小于 **`cos` 偏差 3e-6 / `rel_L2` 2.7e-3（decode 单 token）** 的差异不可判定。
本次量化带来的差异（`cos` 偏差 1.7e-5、`rel_p99` 6.4e-3）**高于地板约 2.4 倍 ⇒ 可判定**，且绝对值极小。

### 2.4 协议偏差（必须写清楚）

1. **`tile_size` 只能是 128**：传 64 时算子直接报
   `Parameter tile_size of KvQuantSparseFlashAttention has incorrect value 64. Reason: tile_size should be 128.`（【实测】，见 `002-grid5.json` 的 `A_v4_tile64_rejected_*`）
   ⇒ 「tile 64 与 128 两种」这个要求**测不了 64**。64 的精度我改用 golden 在 **V4.1 真实几何**上算（§2.5），两者差别在噪声量级。
2. **BF16 KV 过同一个量化算子**做不到（算子拒收 BF16 packed）。已用 N1+N2 替代，理由见 2.3。
3. **golden 只覆盖抽样的 query 行**（48 行：末尾 24 行 + 均匀 24 行），因为 128K 的全量 golden 在 CPU 上不可行。
   **算子输出是全序列的**，只是对齐比较时取这 48 行。每例用的行与被丢弃的退化行都在 JSON 的 `rows_used` / `rows_dropped_lt8_valid` 里。
4. **sparse 下标是合成的**（每行互不相同的均匀随机 token），不是真 indexer 的输出。真 indexer 会挑「更相关」的 token ⇒ softmax 更尖 ⇒ 误差更大。
   我用把 K 幅度放大 4 倍的方式做了**加严测试**（表里「更尖的 softmax」行）：`rel_p99` 从 6.4e-3 涨到 1.8e-2，`cos_p50` 仍有 0.99995。

### 2.5 V4.1 真实几何上的量化误差（golden，精确数学）

V4.1：nope 448 + rope 64（rope 在同一个 512 宽 latent 里），topk = `index_topk` = 512，block 256，
tile 64（能整除 448）与 tile 128（448/128 = 3.5，用 4 个 scale，最后一格 64 宽）。
**这些行算子跑不了（§3），所以只有 golden 这一条腿 —— 纯量化误差。**

| 场景 | tile 64 rel_p99 / cos_p50 / cos_min | tile 128 rel_p99 / cos_p50 |
|---|---|---|
| decode 4K b1 | 6.97e-3 / 0.9999784 / 0.999975 | 7.51e-3 / 0.9999757 |
| decode 32K b1 | 7.72e-3 / 0.9999751 | — |
| decode 128K b1 | 6.97e-3 / 0.9999794 | — |
| decode 4K b8 | 7.29e-3 / 0.9999785 | — |
| prefill 4K b1 | 7.66e-3 / 0.9999776 / 0.999967 | 8.26e-3 / 0.9999741 |
| prefill 32K b1 | 7.74e-3 / 0.9999778 | — |
| prefill 128K b1 | 7.66e-3 / 0.9999779 | — |
| prefill 4K b8 | 7.69e-3 / 0.9999781 | — |
| **V = 整个 512 宽 latent（V4.1 真值）** | **8.10e-3 / 0.9999719** | — |
| 更尖的 softmax（decode / prefill） | 1.76e-2 / 0.9999467；2.01e-2 / 0.9999514 | — |
| topk=128 / topk=2048 | 7.80e-3 / 0.9999758；5.66e-3 / 0.9999861 | — |

⇒ **【实测】在真实 V4.1 几何上，量化误差与算子可跑形状同阶（rel_p99 ≈ 7e-3、cos_p50 ≈ 0.99998），
且 tile 64/128、block 128/256、4K/32K/128K、batch 1/8 之间没有实质差别。**
误差随「softmax 尖锐度」变化最大（≈3×），随序列长度几乎不变。

---

## 3. ★ 本次最重要的发现：算子与 V4.1 的形状对不上

三条**独立**的实测证据：

**【实测 E1】** 量化算子拒绝 V4.1 的 query 宽度：
```
Parameter qHeadDim_ of KvQuantSparseFlashAttention has incorrect value 512.
Reason: q_head_dim only support 576.
```
**【实测 E2】** 非量化同胞算子也拒绝：
```
qk_head_dim only support 512, but got 448[FUNC:CheckFeatureMlaNoQuantShape]
      [FILE:sparse_flash_attention_tiling.cpp]
```
⇒ 这两个算子的 KV cache 布局被**硬编码成 MLA 的 512(nope)+64(rope)**：query 576、value 512。

**【实测 E3】** 两个算子都**彻底忽略 `value` 入参**（`v_probe.py`，`a2/logs/raw/002-v_probe.json`）：
```
quant: {"max_abs_diff_base_vs_zeroV": 0.0, "value_argument_has_effect": false}
bf16 : {"max_abs_diff_base_vs_zeroV": 0.0, "value_argument_has_effect": false}
```
（`value` 张量 RMS = 0.4999，输出 RMS = 0.0161，不是「V 本来就小」的假象。）
配套交叉验证：故意让 V≠K 时，算子输出与「V := K 的 int8 段」的 golden 吻合到 `rel_p99 = 2.7e-3`
（= 噪声地板），而与自己传入的 V 差 `rel_p99 = 1.31`。⇒ **V 取的是 K 的 int8 段。**

**V4.1 的实际几何**（`models/deepseek_v41/model.py:361-373`、`attention/dsa_v41.py:425-437`）：
```python
self.head_dim = config.head_dim               # 512
self.rope_head_dim = config.qk_rope_head_dim  # 64
self.nope_head_dim = config.head_dim - config.qk_rope_head_dim   # 448
self.scale = self.head_dim ** -0.5            # 1/sqrt(512)，不是 1/sqrt(576)
latent = latent.view(-1, 1, attn.head_dim)    # 512 宽
inplace_partial_rotary_mul(..., partial_slice=[attn.nope_head_dim, attn.head_dim])
```
⇒ **Q 512 宽、KV 平面 512 宽，rope 是同一条 512 向量里的最后 64 维（原地旋转）；V 也是这条 512 向量。**
与算子的「512 nope 平铺 + 额外 64 rope」**不是同一个东西**。

---

## 4. 必答 ③ —— V4.1 走哪条路、能不能吃到 INT8、要多少代码改动

### 4.1 先纠正一处交接文档的过时结论（【实测-代码】）

**`dsa_v41.py` 现在调的不是 `sparse_attn_sharedkv`。** 在 A3 容器里（commit `e43cf1e9f`）实测：

```
vllm_ascend/attention/dsa_v41.py:495   torch.ops._C_ascend.npu_sparse_flash_mla(...)      <- V4.1 走的
vllm_ascend/attention/dsa_v1.py        get_dsa_attn_kv_plan() -> npu_sparse_attn_sharedkv / npu_kv_quant_sparse_attn_sharedkv  <- V4 的 DSA 路径
```
`grep -rn npu_sparse_attn_sharedkv vllm_ascend/` 只命中 `dsa_attn_kv_plan.py`（被 `dsa_v1.py` 与
`context_parallel/dsa_cp.py` 消费）。⇒ 交接文档里「V4.1 现在走的就是 `sparse_attn_sharedkv`」
在当前这棵树上是**过时的**（可能描述的是 A2 已部署的那个 revision）。
**这不影响结论**：两条路都吃不到 INT8，只是理由不同。

### 4.2 用 `npu_kv_quant_sparse_flash_attention` 给 V4.1 省 KV —— 不行，三道门全关

| 门 | 内容 | 结论 |
|---|---|---|
| ① 形状 | 算子要求 `q_head_dim = 576`（nope 512 + rope 64） | V4.1 是 512（448 + 64 内嵌）【实测 E1/E2】 |
| ② V 语义 | 算子 V = K 的 int8 段，`value` 入参被忽略 | V4.1 的 V 是整个 512 宽 latent（含 rope 段）⇒ 少 64 维【实测 E3】 |
| ③ tile | `tile_size` 必须 = 128 | V4.1 nope = 448，448/128 = 3.5 ⇒ 最后一格只有 64 维，跨格 scale 语义未定义【实测】 |

### 4.3 要多少代码改动 —— **不是换算子，是改几何**

* **换算子（改一行）**：**不可行**。V4.1 的路径上没有第二个「支持 448+64」的量化算子。
* **加 dequant 前置**：**不可行**。KV 里存的就是 8-bit；K 的语义（哪几维是 rope）是**常量布局**，
  跟存成什么精度无关 ⇒ 前置 dequant 不改变形状不匹配。（而且每次 attention 前解回 BF16，省内存的目的当场失效。）
* **真正要做的事**：把 KV 平面几何从「512 宽、rope 内嵌」改成「**nope 512 宽 + 独立 rope 64 宽**」
  （= 576 B/plane/token）。这意味着 compressor 的输出从 512 变成 512 + 一个独立的 64 维 rope 输入 ——
  **是模型参数化的改变，要重新训练 / 至少重新导出权重**，不是推理侧补丁。
  （MLA 里 V3/V4 本来就是这么做的：`kv_lora_rank=512` + 独立的 `k_pe=64`。）
* **另一条更短的路（【推断】，值得先验证）**：`cann_ops_transformer` 里有一个
  **`mixed_quant_sparse_flash_mla`** —— 正是 V4.1 在用的 `npu_sparse_flash_mla` 的**量化版**，
  接口一模一样（`ori_kv/cmp_kv/ori_sparse_indices/...`），额外多出
  `quant_mode` / `key_dtype` / `value_dtype`：
  ```
  /usr/local/Ascend/cann-9.1.0/python/site-packages/cann_ops_transformer/ops/mixed_quant_sparse_flash_mla.py
  ```
  ⇒ **V4.1 要低精 KV，最可能的正解是把 `dsa_v41.py:495` 的 `npu_sparse_flash_mla` 换成
  `mixed_quant_sparse_flash_mla` + 把 KV 平面按它的布局存 INT8**，而不是用 `npu_kv_quant_sparse_flash_attention`。
  **【未确认】**：① 该算子在 A2/A3（`ascend910b` / `910_93`）是否真有 kernel；
  ② 它接受的 KV 几何是不是 V4.1 的 448+64（从签名看**不要求 576**，但要实测）。
  **这是下一步最该做的 30 分钟实验。**

---

## 5. 必答 ④ —— 容量收益

### 5.1 现在（实测口径）
```
4421 B/token/rank = 4 个共享 long-KV 平面 x 512 dim x BF16(2B) = 4096
                  + 4 个 indexer 平面 (INT8 + FP16 scale)      =  325
```
### 5.2 INT8 打包后每 token 每平面的字节
packed 布局 = `int8(nope) | bf16(rope) | fp32(scale) x ceil(nope/tile)`：

| 几何 | nope | rope | tile | scale 数 | **每平面 B** | 4 平面 B | +indexer | **B/token** | **相对 4421** |
|---|---|---|---|---|---|---|---|---|---|
| **V4.1 真实** | 448 | 64 | 128* | 4 | **592**【实测】 | 2368 | 325 | **2693** | **−39.1%** |
| V4.1 真实 | 448 | 64 | 64 | 7 | 604【实测】 | 2416 | 325 | 2741 | −38.0% |
| 算子要求的 576 几何 | 512 | 64 | 128 | 4 | **656**【实测】 | 2624 | 325 | **2949** | −33.3% |
| 理想上界（512 全 int8，无 rope/scale） | 512 | — | — | — | 512 | 2048 | 325 | 2373 | −46.3% |

`*` tile 128 时 448/128 = 3.5，最后一格只有 64 维 —— **这个布局在算子上不存在**，仅作上界参考。
「每平面 B」全部取自跑出来的 `kv_cache.shape[-1]`（JSON 字段 `k_pack_bytes_per_token`），不是纸面推算。

### 5.3 换算到 A2

| 配置 | B/token/rank | 15.82 GiB 能装 | A2 现在 3,498,354 token 会变成 |
|---|---|---|---|
| 现在（BF16） | 4421 | 3.84M | 3.50M |
| **INT8（V4.1 几何，−39.1%）** | **2693** | **6.31M（×1.64）** | **5.74M** |
| INT8（576 几何，−33.3%） | 2949 | 5.76M（×1.50） | 5.24M |
| 理想（−46.3%） | 2373 | 7.16M（×1.86） | 6.51M |

⇒ **【实测量级】INT8 long-KV 把 KV 容量放大 1.64×（V4.1 几何）。**

---

## 6. ★ 关键假设验证：量化 long-KV 会不会影响 sparse index 选择

**结论：【实测-代码】不会 —— 但前提是「只在写入 long-KV 平面那一步量化」。**

证据（都来自 A3 容器里的真代码）：

1. **indexer 有自己独立的 KV 平面，而且本来就是 INT8**（`models/deepseek_v41/indexer.py:65-84`）：
   ```python
   self.k_cache = DeepseekV41CacheLayer(..., AscendMLAAttentionSpec(
       head_size=self.width,   # index_head_dim = 128
       dtype=torch.int8,       # 本来就量化
       scale_dim=1, tokens_per_state=compress_ratio, ...))
   ```
   long-KV 是另一个 cache group：`{prefix}.long_kv_cache`，`head_size=512, dtype=torch.bfloat16`。
2. **两条读路径指向不同张量**：
   * 选下标：`dsa_v41.py:447-455` -> `context[self.index_k_source_prefix].kv_cache[0]`（indexer 的 `k_cache`）
   * 算 attention：`dsa_v41.py:470` -> `context[self.long_kv_source_prefix].kv_cache[0]`（`long_kv_cache`）
3. **写入是从同一个 `latent` 分两路**（`dsa_v41.py:~424-437`）：
   ```python
   attn.indexer.update_keys(latent, index_slots, ...)                    # 投影到 128 维 + int8 -> 自己的平面
   scatter_cache_sk(attn.long_kv_cache.kv_cache[0], long_slots, latent)  # -> 512 宽 BF16 平面
   ```
   indexer 读的是「latent 经 `idx.wk`(512->128) 投影后的 int8 结果」，
   **没有任何一个字节来自 long-KV 平面**。

⇒ **【实测-代码】把 long-KV 平面改成 INT8，不会改变 indexer 看到的任何数据，因此不会改变 top-k 选择。**

**反过来的红线**：如果为了省事在 **compressor 输出 `latent` 上**就量化（而不是在 scatter 进 long-KV 那一步），
indexer 也会吃到量化后的 latent，**选择就会变**。实现时必须把量化点卡在 store 那一侧。

**没做到的（【未确认】）**：没有在设备上端到端跑一次真 indexer（需要真权重 + 完整模型实例），
所以这是「代码级实测 + 逻辑推断」，不是「跑出来的下标对比」。

---

## 7. 方法、脚本、原始数据

| 东西 | 路径 |
|---|---|
| 探针脚本（本地） | `a2/agents/K8_int8/int8_kv_accuracy.py`（主网格）、`a2/agents/K8_int8/v_probe.py`（value 入参对照实验） |
| 脚本（A3 上） | `/work/agents/K8_int8/`（= A3 `~/projects/dsv41-upstream-pr/agents/K8_int8/`） |
| 原始 JSON | `a2/logs/raw/002-grid5.json`（31 例全网格）、`002-grid6.json`（修正 BF16 臂 + V≠K 对照）、`002-v_probe.json` |
| nightly 门禁 | `a2/logs/raw/002-gate_nightly_test.log` |
| golden 语义 | 照抄 nightly 的 `_reference_attention`（含 `probs.to(bf16)` 这一步），只改成只在抽样行上算 |

复现命令（A3 上）：
```bash
bash ~/projects/dsv41-upstream-pr/tools/a3_chip.sh c0 --timeout 900 --name k8-grid -- \
  bash -c 'cd /work/agents/K8_int8 && export TMPDIR=/work/agents/K8_int8/tmp && \
           python3 int8_kv_accuracy.py --out /work/agents/K8_int8/raw/grid.json --rows 48'
```
（`exit 75` = 没抢到锁，重试即可。全程用 c0，没碰 Phy-ID 8–15。）

---

## 8. 这对 A2 意味着什么

1. **8-bit KV 的精度不是障碍**【实测】：单层 attention 输出 `cos ≈ 0.99998`，
   而算子自身的数值噪声就有 `cos ≈ 0.999997` —— 量化的代价只是噪声地板的 ~2.4 倍，
   且在 4K/32K/128K、batch 1/8、block 128/256、tile 64/128 上都没有实质变化。
   **要担心的顺序是「几何/算子能不能用」 -> 「精度」，不是反过来。**
2. **`npu_kv_quant_sparse_flash_attention` 对 V4.1 是死路**【实测】：
   它要 576（512+64），V4.1 是 512（448 内嵌 64）；而且它的 V 只能是 K 的 int8 段。
   **不要在这条路上继续投入** —— 除非接受「改模型几何 + 重新训练」。
3. **最短的下一步是 `mixed_quant_sparse_flash_mla`**【推断，未确认】：
   它是 V4.1 现在用的 `npu_sparse_flash_mla` 的量化版，接口一致。
   **验证它在本机能不能编、能不能跑，是 30 分钟的活，收益是 1.64× KV 容量。**
4. **容量收益 1.64×**【实测字节数换算】：3.50M -> 5.74M token（单 rank，15.82 GiB 口径）。
   与 DRAM 卸载（~260 GB ⇒ ~5.7M token 可缓存上下文）叠加，A2 的有效上下文能上千万级。
5. **索引选择不受影响**【实测-代码】，但**量化点必须卡在 long-KV store 那一步**。
6. **可移植性**：全部结论来自 **A3 单 die（910C）**。算子矩阵说 A2 也是 √，
   但**我没有 A2 实机数据**；A2 的 CANN 版本、`tile_size=128` 限制、`q_head_dim=576` 限制是否一致，见 §9。

---

## 9. 未确认清单（不许用相邻数字顶替）

| # | 未确认的事 | 为什么没确认 | 怎么补 |
|---|---|---|---|
| ① | **A2（910B3）上这个算子能不能跑** | A2 本机 ssh 不可达 | 把 nightly 那条 pytest 让用户粘到 A2 跑一次（30 秒） |
| ② | **`mixed_quant_sparse_flash_mla` 在 A2/A3 能否编译/运行、接受什么几何** | 本次时间用尽 | import `cann_ops_transformer.ops` 调一次最小 case |
| ③ | **batch>1 的非量化算子臂没跑对** | 我的 TND 多请求 layout 没配好（`bf16_op` 在 b8 自差 1.5） | 修 `run_bf16_op` 的 cu_seqlens / indices 维度后重跑 |
| ④ | **真权重单层**（加分项） | 未做：90 分钟预算用尽 | 用 `~/models/DeepSeek-V4.1-Flash` 抽一层 |
| ⑤ | **合成分布 != 真实 latent 分布** | 只有合成数据 | ④ 做完即可给真实分布下的数字 |
| ⑥ | **端到端模型质量影响（perplexity / KL）** | 只测了单层 attention 输出 | 需要起服务跑评测集 |
| ⑦ | **`value` 入参被忽略是不是「设计如此」** | 只测到现象，没找到 CANN 文档/注释 | 查 CANN 算子文档或问 CANN 侧 |
| ⑧ | **A2 侧 opp kernel 是否也提供该算子** | 我看到的只是 A3 的 CANN 安装目录 | A2 上 `ls .../kernel/config/ascend910b/ops_transformer/` |
