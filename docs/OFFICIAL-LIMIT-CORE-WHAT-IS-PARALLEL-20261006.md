# 官方的「控核」到底把什么和什么并行（源码级）

> 起因：用户追问 —— 「官方的控核，具体是能把什么和什么进行并行？」
> 依据：`~/opensrc/cann-recipes-infer`（a3-21，master `2225cae`）逐行读源码，
> 不依赖文档概括。全部标注行号，可复核。

## 0. 一句话

`limit_core_num` 在官方代码里只有 **5 个使用点**，全部服务于 **2 组"同一层 attention 内部、
互不依赖的两条支路"** 的并行：

1. **MLA 的 KV 支路 ‖ Q 支路**（`wkv` ‖ `wq_b`）—— 核预算 **12 + 8**
2. **Compressor ‖ Indexer 的 Q 支路** —— 核预算 **16 + 8**

而且有一句注释把因果关系写死了：

```python
if self.compress_ratio == 4:  # c4a supports compressor parallel only if it supports limit core num
    self.enable_compressor_parallel = self.enable_limit_core
```

⇒ **控核不是"锦上添花的优化"，而是 c4a 层 compressor 并行的前置条件**。

---

## 1. 全部 5 个使用点

| # | 位置 | 参数 | 限的是谁 |
|---|---|---|---|
| 1 | `models/deepseek_v4/models/modeling_deepseek.py:1419` | `kv_aic_num = 24//2 = 12`，AIV=24 | `kv_norm` + kv 的 partial RoPE（跑在 **mla_stream**） |
| 2 | `.../modeling_deepseek.py:1440` | `qb_aic_num = 24 − cmpr_aic_num = 8`，AIV=16 | `q_b_norm` + q 的 RoPE + quant（跑在 **主流**） |
| 3 | `.../modeling_deepseek.py:1614` | `cmpr_aic_num = 16`，AIV=32 | attention 的 `compressor`（跑在 **compressor_stream**） |
| 4 | `.../models/modules/indexer.py:161` | `cmpr_aic_num = 16`，AIV=32 | Indexer 自己的 `weights_proj` + `compressor`（跑在**主流**） |
| 5 | `.../modules/indexer.py:184` | `rope_aic_num = 24 − 16 = 8`，AIV=16 | Indexer 的 `wq_b` + RoPE + Hadamard + quant（跑在 **indexer_stream**） |

A3 每 die = **24 cube / 48 vector**，`aiv_to_aic_ratio = 2`。所以 **12+8=20**、**16+8=24** ——
都在 24 以内（第 1 组留了 4 个 cube 余量）。

---

## 2. 第 1 组：MLA 内部 **KV 支路 ‖ Q 支路**

```python
# modeling_deepseek.py:1398-1445（删节，只留结构）
kv_aic_num = self.total_aic_num // 2   # half of total corenums to support parallelism with q_b qbmm
qb_aic_num = self.total_aic_num - self.cmpr_aic_num          # 24 - 16 = 8
enable_cmpr_stream = self.enable_compressor_parallel and not is_prefill

if self.platform_version == PlatformVersion.A3:   # ← A3 专有
    qa = self.wq_a(x)
record_event(enable_multi_streams, self.mla_events, 0)
if self.platform_version != PlatformVersion.A3:
    qa = self.wq_a(x)
qr, qr_scale = self.apply_norm_dynamic_quant(qa)

with npu_stream_switch(enable_multi_streams, attn_metadata.get('mla_stream', None)):
    wait_event(enable_multi_streams, self.mla_events, 0)
    kv = self.wkv(x)                       # 无核限制（大 matmul）
    record_event(enable_multi_streams, self.mla_events, 1)
    with limit_core_num(enable_limit_core, kv_aic_num, kv_aic_num*2, ...):  # ← #1 12 AIC
        kv = self.kv_norm(kv)
        inplace_partial_rotary_mul(kv...)  # kv 的 RoPE
        record_event(enable_multi_streams, self.mla_events, 2)

wait_event(enable_multi_streams, self.mla_events, 1)   # 等 wkv 完成
q = self.q_b_proj_tp(qr, qr_scale)                     # 主流：wq_b，无核限制
...
with limit_core_num(enable_limit_core, qb_aic_num, qb_aic_num*2, ...):  # ← #2 8 AIC
    q = self.q_b_norm(q) + q 的 RoPE + quant
```

**并行的双方**：

| | 流 | 算子 | 核预算 |
|---|---|---|---|
| 甲 | `mla_stream` | `kv_norm` + **kv 的** partial RoPE | **12 AIC / 24 AIV** |
| 乙 | 主流 | `q_b_proj_tp`（wq_b）+ `q_b_norm` + **q 的** RoPE + quant | **8 AIC / 16 AIV** |

**为什么能并行**：`wkv(x)` 走 KV 支路、`wq_b(qr)` 走 Q 支路 ——
**两条支路都只依赖本层输入，互不依赖**（KV 支路写 KV cache，Q 支路出注意力 query）。

**A3 上官方反而更保守**：`if platform_version == A3: qa = wq_a(x)` 被提到 event 之前，
配合注释 `# ensure wkv matmul does not overlap with wq_a or wq_b`
⇒ **A3 上 wkv 必须等 wq_a 完成**（950 上不必）。这是个 A3 专属的额外串行约束。

---

## 3. 第 2 组：**Compressor ‖ Indexer 的 Q 支路**

```python
# modeling_deepseek.py:1610-1620
if self.compress_ratio > 1:
    with npu_stream_switch(enable_cmpr_stream, attn_metadata.get('compressor_stream', None)):
        wait_event(enable_cmpr_stream, self.cmpr_events, 0)
        with limit_core_num(enable_limit_core, self.cmpr_aic_num, self.cmpr_aic_num*2, ...):  # ← #3 16 AIC
            self.compressor(x, attn_metadata, is_prefill)
        record_event(enable_cmpr_stream, self.cmpr_events, 1)

if self.compress_ratio == 4:
    # wait self.cmpr_events[1] in self.indexer
    topk_idxs = self.indexer(x, qr, qr_scale, attn_metadata, enable_cmpr_stream, self.cmpr_events, 1, is_prefill)

if self.compress_ratio > 1:
    wait_event(enable_cmpr_stream, self.cmpr_events, 1)   # finish compressor before sfa
```

```python
# modules/indexer.py:159-227
with limit_core_num(enable_limit_core, self.cmpr_aic_num, self.cmpr_aiv_num, ...):  # ← #4 16 AIC
    weights = self.weights_proj(x_for_weights_proj) * (self.softmax_scale * self.n_heads ** -0.5)
    self.compressor(x, attn_metadata, is_prefill)          # Indexer 自己的 compressor

with npu_stream_switch(enable_multi_streams, attn_metadata.get('indexer_stream', None)):
    wait_event(enable_multi_streams, self.indexer_events, 0)
    with limit_core_num(enable_limit_core, self.rope_aic_num, self.rope_aiv_num, ...):  # ← #5 8 AIC
        q = self.wq_b(qr, dynamic_scale=qr_scale)
        ... partial_rotary_mul + Hadamard + dynamic_quant ...
    record_event(enable_multi_streams, self.indexer_events, 1)

wait_event(enable_multi_streams, self.indexer_events, 1)   # ← 汇合
topk_idxs = self.forward_li_quant(q, q_scale, li_cmp_kv, li_key_dequant_scale, weights, attn_metadata)  # LI 融合算子
```

**并行的双方（c4a 层，一个 step 内三条流同时活跃）**：

| | 流 | 算子 | 核预算 |
|---|---|---|---|
| 甲 | `compressor_stream` | attention 的 compressor（写 `c4a_cmp_kv`，供 SFA 用） | **16 AIC / 32 AIV** |
| 乙 | `indexer_stream` | Indexer 的 Q 支路：`wq_b` + RoPE + Hadamard + quant | **8 AIC / 16 AIV** |
| 丙 | 主流 | Indexer 自己的 compressor（写 `li_cmp_kv`，供 LI 用）+ `weights_proj` | 16 AIC / 32 AIV |

**汇合点**：**LI 融合算子**（`forward_li_quant`，即 lightning indexer 选 top-k）——
它同时需要 `indexer_stream` 产出的 `q/q_scale` 和 compressor 产出的 `li_cmp_kv`
⇒ 所以有两道 `wait_event`。

**★ 因果写在注释里**：

```python
# modeling_deepseek.py:810-820
if self.enable_multi_streams and self.platform_version == PlatformVersion.A3:
    self.enable_compressor_parallel = self.compress_ratio == 128
    if self.compress_ratio == 4:   # c4a supports compressor parallel only if it supports limit core num
        self.enable_compressor_parallel = self.enable_limit_core
else:
    self.enable_compressor_parallel = False
self.total_aic_num = 24   # enable_limit_core only suppots A3 (24 cube and 48 vector cores)
self.cmpr_aic_num = 0
if self.enable_compressor_parallel:
    self.cmpr_aic_num = 16
```

读法：
* **`enable_compressor_parallel` 只对 A3 生效**（`platform_version == A3`）；
* **c128a 层**：只要 multi_streams 开就并行；
* **c4a 层**：**必须同时开 `enable_limit_core`**，否则不并行；
* 全部只在 `not is_prefill`（decode）生效。

---

## 4. 另外三组并行（**不限核**，只靠多流）

把这三组也列出来，是为了说明"控核只管前面两组"：

| # | 并行双方 | 流 | 汇合点 | 限核 |
|---|---|---|---|---|
| 3 | **共享专家** ‖ **gating + routed expert dispatch/combine** | `shared_expert_stream` ‖ 主流（`modeling_deepseek.py:481`） | MoE 相加 | ❌ |
| 4 | **整个 step 的 kernel metadata**（c1a/c4a/c128a/LI） ‖ **全部 40 层主计算** | `metadata_stream`（`modeling_deepseek.py:2853-2890`） | `wait metadata_event[1]` | ❌ |
| 5 | **Engram 的 Hash/Embedding/WKV 预计算** ‖ 主层计算 | `engram_precompute`（`modeling_deepseek.py:1733`） | 第 1/14 层 Gate | ❌ |

> **第 4 组特别值得注意**：`generate_kernel_metadata()` 在**模型 forward 的最开头、每步调一次**
> （`modeling_deepseek.py:2914`），整步的 SFA/LI metadata 全部丢到 `metadata_stream`，
> **与 40 层主计算完全重叠**。
> 而我们的 profile 里，等价物（`SparseFlashMlaMetadata` 258 µs × 3）**落在步的最后 10%**
> （见 `docs/DECODE-PARALLELISM-WHAT-IS-HIDDEN-20261006.md` §3）——
> 这是**同一件事的两种排法**，官方是"提前铺开"，我们是"堆在末尾"。

---

## 5. ★★ 我们和官方的根本差异：**用"顺序"代替"控核"**

我们的实现（`patches/files/dsa_v1.py:1884-1893`）里，注释写得非常直白：

```python
# kv_matmul and q_b_matmul are both Cube ops. Ensure kv_matmul (launched on
# aux_stream) completes before q_b_matmul starts so they do not contend for
# the Cube units. kv_norm (Vector) follows kv_matmul on aux_stream and is
# unaffected as it overlaps with q_b_matmul.
main_stream.wait_event(e_kv_matmul_done)
```

| 维度 | **官方** | **我们** |
|---|---|---|
| 避免两条支路抢 cube 的手段 | **`limit_core_num` 分配核预算**（12 + 8） | **`wait_event` 强制先后** |
| kv_matmul 与 q_b_matmul 的关系 | **真并行**（各拿一部分 cube） | **严格串行**（kv 跑完 q_b 才开始） |
| 靠什么重叠 | 两个 matmul 同时在跑 | 只有 `kv_norm`(Vector) 搭在 q_b_matmul 上 |
| 代价 | 各自变慢（12→1.92×、8→2.81×），但时间重叠 | 各自满核很快，但不重叠 ⇒ **AIC 空转** |

**这正好解释了我们实测到的三个现象**（见 `docs/DECODE-AIC-AIV-PIPELINE-20261006.md`）：

1. **AIC ∩ AIV 只有 3.50 ms**（AIV 的 14.75 里 76% 在等 AIC）；
2. **主流上 AIC 连续块的中位长度 = 1 个算子**（271.9 段/步）—— 每个 AIC 块跑 37 µs 就得让位；
3. **AIC 利用率只有 55%**（22.05 / 40.02 ms），而 AIV 28%、通信 0% 重叠。

> 我们**有** compressor 的侧流（`dsa_v1.py:1971/2010` 有 `compressor_done` 事件），
> **有** `multistream_dsv4_dsa_overlap` 开关，
> **但整个 `vllm_ascend` 里 `limit_core_num` 零使用** —— 所以按官方注释，
> **我们的 c4a compressor 并行在"是否成立"这一点上就存疑**（官方需要一个前置的控核）。

---

## 6. 结论：可直接动手的两处

| # | 动作 | 期望 | 风险 |
|---|---|---|---|
| **1** | 在 `dsa_v1.py` 的 `aux_stream` 路径上，把 `wait_event(e_kv_matmul_done)` **换成 `limit_core_num`**（kv 侧 12 / 主侧 8） | 让 kv_matmul 与 q_b_matmul 真并行 | 需实测：12+8 的分法在我们模型的 shape 下是否真的更快（各自的 1.92×/2.81× 惩罚 vs 重叠收益） |
| **2** | 把整个 step 的 SFA/LI metadata 从"步尾"**提到步首的侧流**（对标官方 `generate_kernel_metadata` 每步一次的做法） | 回收 0.8~1.2 ms/步（现落在最后 10%） | 需确认我们的 metadata 生成是否依赖本步中间结果（官方说只依赖 block table） |

> 注意：**第 1 项不能照搬 12/8**。官方的 12/8 是针对它的 shape 调出来的；
> 我们的 `kv_matmul`/`q_b_matmul` 规模不同（TP8、W4A8），
> 应该用 `tools/tiny_limit_*.py` 先量出"在我们 shape 下、给多少 cube 才能让两边耗时持平"，
> 再做 A/B。判据用 `[bneck] hp`（ms/step）。

---

## 7. 复现

```bash
R=~/opensrc/cann-recipes-infer
# 5 个使用点
grep -rn "limit_core_num(" $R/models/deepseek_v4/ | grep -v "import\|def "
# 因果注释（c4a 需要 limit core）
sed -n '810,822p' $R/models/deepseek_v4/models/modeling_deepseek.py
# 第 1 组（MLA KV ‖ Q）
sed -n '1395,1448p' $R/models/deepseek_v4/models/modeling_deepseek.py
# 第 2 组（Compressor ‖ Indexer）
sed -n '1608,1640p' $R/models/deepseek_v4/models/modeling_deepseek.py
sed -n '155,230p'  $R/models/deepseek_v4/models/modules/indexer.py
# 第 4 组（metadata 每步一次、整步重叠）
sed -n '2853,2890p' $R/models/deepseek_v4/models/modeling_deepseek.py
sed -n '2910,2918p' $R/models/deepseek_v4/models/modeling_deepseek.py
# 我们的对照实现
sed -n '1880,1900p' ~/cedpd-repo/patches/files/dsa_v1.py
```
