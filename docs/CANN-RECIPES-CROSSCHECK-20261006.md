# ★ 对 CANN 官方 recipes 的交叉核对：我们的想法官方实现了多少？

> 起因：用户要求「找本地有没有 `cann-recipes-infer` 的 clone，更新一下，然后找有没有我们这个想法的实现」。
> 结论：**找到了、更新了，而且我们的三个想法官方都有实现** —— 其中一个（控核）**明确标注只支持 A3**。
> 同时**推翻了我上一轮的一条建议**（见 §4）。

## 0. 仓库位置与更新

```bash
# a3-21 上（本地 ~/projects 与 a3-22 都没有）
/home/l00886679/opensrc/cann-recipes-infer
git remote -v   # https://gitcode.com/cann/cann-recipes-infer.git

git fetch --all && git checkout master && git pull --ff-only
# 76b9cb8 (2026-09-15) → 2225cae (2026-09-30)，共落后 37 个提交
```

本次更新带来的关键内容（对我们直接相关）：

| 提交 | 内容 |
|---|---|
| `7098115` | **`feat: feat deepseek-v4.1-flash high performance inference`**（完整 V4.1-Flash 实现） |
| `2225cae` | `feat(dsv4.1): 打开 type-2 grouplist 开关 + swiglu 归一自定义算子` |
| `0322b30` / `1f73d95` | dsv41 单卡部署脚本 / 手册 |
| `54017b4` | **dsv4 AFD 文档**（Attention/FFN 解耦） |

新增的关键路径：
```
models/deepseek_v4_1/                 ← 完整 V4.1-Flash 实现（含 engram / dspark / compressor / indexer）
models/deepseek_v4/models/modules/    ← V4 版（含 multi_stream + limit_core 的完整实现）
ops/ascendc/                          ← 21 个 AscendC 自定义算子 + 文档（逐个列了产品支持）
docs/models/deepseek_v4_1/
  ├── deepseek_v4.1_low_latency_tp_guide.md      ★ 低时延优化报告（多流/控核/Superkernel/预取）
  ├── deepseek_v4.1_flash_cann_tech_report.md    ★ 技术报告（Engram 多流）
  └── deepseek_v4.1_asc_comm_tech_report.md
```

---

## 1. ★ 我们的三个想法，官方逐条实现（附原文）

低时延报告的 "Highlights" 原文：

> **"框架特性优化：Npugraph EX、多流、Superkernel、预取。"**

### 1.1 想法 A（多流 + 控核）—— 完整实现，且**只支持 A3**

`models/deepseek_v4/models/modules/indexer.py:99-190` 的原文：

```python
aic_total = 24          # enable_limit_core only suppots A3
aiv_to_aic_ratio = 2    # aiv_num is 2 * aic_num
self.cmpr_aic_num  = 16                    # Compressor 拿 16 个 cube
self.cmpr_aiv_num  = 16 * 2                # 32 个 vector
self.rope_aic_num  = aic_total - 16 = 8    # Q投影+RoPE 拿 8 个 cube
self.rope_aiv_num  = 8 * 2                 # 16 个 vector

enable_multi_streams = self.enable_multi_streams and not is_prefill   # 只在 decode
enable_limit_core    = self.enable_limit_core    and not is_prefill

# 主流：Compressor + weights_proj，**限到 16 AIC / 32 AIV**
with limit_core_num(enable_limit_core, self.cmpr_aic_num, self.cmpr_aiv_num, exe_mode=...):
    weights = self.weights_proj(x) * ...
    self.compressor(x, attn_metadata, is_prefill)

# 侧流：wq_b + RoPE，**限到 8 AIC / 16 AIV**
with npu_stream_switch(enable_multi_streams, attn_metadata.get('indexer_stream', None)):
    wait_event(enable_multi_streams, self.indexer_events, 0)
    with limit_core_num(enable_limit_core, self.rope_aic_num, self.rope_aiv_num, exe_mode=...):
        q = self.wq_b(qr, dynamic_scale=qr_scale)
        q = partial_rotary_mul_quant(...)
```

**这就是"把 Cube 核在两条并发流之间分预算"的完整范式**：
16 + 8 = 24（A3 的 cube 数）、32 + 16 = 48（A3 的 vector 数），**恰好用满**。

官方文档对"为什么必须控核"的说明（低时延指南原文）：

> "多流通过事件依赖将无直接数据依赖的 MLA、Indexer、Compressor 和共享专家阶段调度到独立 Stream……
> **多流收益取决于计算和核资源是否形成有效重叠；如果并发分支同时占满 Cube/Vector 核，
> 执行时间线仍可能串行或产生拖尾**。"

**这句话正是我们上一轮实测到的现象**：`MIX_AIC ∩ MIX_AIV = 0.000`（混合核内部相位串行）、
AIC ∩ AIV 只有 3.50 ms / 14.75 ms。

### 1.2 想法 B（Engram 多流）—— 实现，且我们已经吃了一部分

技术报告 §344-389 原文：

> "其中 **Hash、Embedding 和 WKV Projection 不读取当前 Decoder Layer 的 `hidden_states`，
> 具备提前计算条件**；Gate 必须在对应层主流中执行，因此是多流预计算的汇合点。"
>
> "当 `enable_engram_multi_stream=True` 时，模型创建 `engram_precompute` 副流……
> Layer0 返回后，副流继续执行 WKV1 并记录 `ready1`……这部分计算**与主流 Layer2 至 Layer13 重叠**。"

代码里还有一条**防 Cube 争抢**的关键注释
（`models/deepseek_v4_1/models/modeling_deepseek.py`）：

```python
# After Layer-0 completes: submit wkv (Stage 2) to the side stream.
# wkv waits on ev[5] (post-GroupedMM) so it only starts after
# GroupedMM (Cube) finishes, avoiding Cube contention.  The main
# stream continues to Layer-1 immediately.
```

**这解释了我们实测里的那个"例外"**：profile 里 `通信 ∩ AIV = 4.114 ms`，
而 4.114 恰好等于 `AivKernel` 的 4.114 —— 即 **engram 的 vector 核已经和 HCCL 通信完全并行**。
**这条机制在我们栈里已经跑通了**，与官方设计一致。

### 1.3 想法 C（预取 / Superkernel）—— 官方有，我们缺依赖

| 项 | 官方 | 我们的栈 |
|---|---|---|
| **预取** | 列在框架优化 Highlights；"面向十万亿参数级模型……通过计算与数据搬运重叠"、"数据预取" | 【未确认】需进一步定位具体 API（低时延报告只列名，细节在 950 路径） |
| **Superkernel** | `enable_superkernel`（`executor/utils/common_utils.py:59` → `tng.scope.super_kernel`） | ❌ **没有 `torchair`**（我们栈用 `torch_npu.dynamo`），该实现依赖 GE-graph 栈 |
| **Npugraph EX** | `exe_mode: "npugraph_ex"` | ✅ 我们已用（`NPUGRAPH_EX=1`） |

---

## 2. ★★ 最重要的可执行发现：`limit_core_num` 我们完全没用

### 2.1 API 在我们的栈里**可用**

```python
torch.npu.npugraph_ex.scope.limit_core_num(aic_num, aiv_num)   # ✅ 存在
```

实测（容器内）：
```
torch_npu: 2.10.0.post4
has npugraph_ex: True
  scope: True
  成员: ['limit_core_num']
  has limit_core_num: True
```

### 2.2 但我们在整个 vllm-ascend 里零使用

```bash
$ grep -rn "limit_core_num\|npugraph_ex.scope" /vllm-workspace/vllm-ascend/vllm_ascend/
(空)
```

⇒ **我们有这个 API、有 A3 硬件、有官方的完整参考实现，但没接线。**

### 2.3 现状：我们有多流，但没有控核

| 机制 | 我们 | 官方 |
|---|---|---|
| 多流编排 | ✅ `multistream_dsv4_dsa_overlap`（`aux_stream=dsv4_dsa_overlap_stream()`） | ✅ `enable_multi_streams` |
| 事件同步 | ✅ `torch.npu.Event` / `ExternalEvent` | ✅ 同 |
| **控核（核预算）** | ❌ **零使用** | ✅ `enable_limit_core` + `limit_core_num` |
| Engram 侧流 | ✅（实测已与通信并行 4.114 ms） | ✅ `enable_engram_multi_stream` |

**这解释了为什么我们的多流收益有限**：两条流同时跑时会**互相抢同一个 24-cube 池**，
于是时间线退化回串行 —— 正是官方文档警告的"执行时间线仍可能串行或产生拖尾"。

---

## 3. 官方 AscendC 算子：我们用了 12 个，6 个 A3 可用但未用

`ops/ascendc/docs/` 共 21 个算子，**15 个支持 A3**。逐个核对我们的使用情况：

### 3.1 已经在用（12）

| 官方算子 | 对应我们 profile 里的 | 我们代码位置 |
|---|---|---|
| `npu_hc_pre_v2` | **`HcPre`** | `models/deepseek_v41/model.py:815` |
| `npu_hc_post` / `mhc_post` | **`HcPost`** | `models/deepseek_v41/model.py:828` |
| `inplace_partial_rotary_mul` | **`InplacePartialRotaryMul`** | `attention/dsa_v1.py:1975,2033` |
| `npu_rms_norm_dynamic_quant` | （融合版 RmsNorm+Quant） | `attention/dsa_v1.py:2015,2143` |
| `npu_moe_gating_top_k` | `MoeGatingTopKHash` | `device/device_op.py:915` |
| `quant_lightning_indexer` | **`QuantLightningIndexerV2`** | `attention/dsa_v41.py:911` |
| `sparse_flash_mla` | **`SparseFlashMla`** | `attention/dsa_v41.py:544` |
| `npu_sparse_attn_sharedkv` | `SparseAttnSharedkv` | `attention/dsa_attn_kv_plan.py:147` |
| `mega_moe` | MoE 主路径 | `ascend_config.py:264` |
| `lightning_indexer` / `flash_attn` | indexer / attention | `sfa_v1.py` / `platform.py` |

### 3.2 **A3 可用但我们没用**（可下手清单）

| 官方算子 | 对应我们 profile 里的开销 | 备注 |
|---|---:|---|
| `npu_scatter_nd_update_asc` | **`ScatterNdUpdateSk` 1.180 ms/步** | 我们用的是 aclnn 版 |
| `npu_compressor` | Compressor 路径 | 我们用 triton 版 |
| `npu_swiglu_clip_quant` | `DequantSwigluQuant` 0.561 ms/步 | |
| `npu_gather_selection_kv_cache` | KV gather | |
| `mixed_quant_sparse_flash_mla` | 混合量化 SFA | |
| `npu_rms_norm_dynamic_quant` | profile 里 `RmsNorm`(2.012) 与 `DynamicQuantV2`(0.747) **是分开的** | 融合版只在 w8a8 分支被调用（`_is_w8a8_dynamic` 门控） |

### 3.3 950-only（我们没有对应实现）

`npu_kv_compress_epilog` / `_v2`、`npu_indexer_compress_epilog`、
`npu_moe_init_routing_group_quant`、`npu_swiglu_group_quant`。

> 注意 `npu_moe_init_routing_group_quant` 是 950-only，而我们的 profile 里有
> `MoeInitRoutingV3`（0.633 ms/步）—— 这一项**没有 A3 版官方融合算子**。

---

## 4. ⚠️ 更正上一轮我的一个建议

上一轮我写："**小张量 elementwise 群（AivKernel+HcPre+HcPost+RmsNorm+RoPE）值得融合**"，
并建议"HcPre/HcPost 是一对，可以融合"。

**核对后这条不成立**：`HcPre` 与 `HcPost` **已经是官方融合算子**的实测成本：

```python
# vllm_ascend/models/deepseek_v41/model.py:815
return torch.ops._C_ascend.npu_hc_pre_v2(...)      # 一次调用产出 y + post + comb + pre
# :828
return torch.ops._C_ascend.npu_hc_post(...)
```

同理 `InplacePartialRotaryMul`、`SparseFlashMla`、`QuantLightningIndexerV2` 都是融合算子。
**所以"这 6.6 ms 可以靠融合拿回来"是错的** —— 它们已经是融合后的成本。
真正还没融合的只有 profile 里那对 **`RmsNorm` + `DynamicQuantV2`**（2.759 ms/步，
且受 w8a8 门控）。

**修正后的靶点排序**：

| # | 靶点 | 真实 ms/步 | 依据 |
|---|---|---:|---|
| 1 | **接线 `limit_core_num`（控核）让现有多流真并发** | 上限 ~11.3 | §1.1 + §2 |
| 2 | `RmsNorm + DynamicQuant` 融合（非 w8a8 路径） | ≤2.8 | §3.2 |
| 3 | `ScatterNdUpdateSk` → `npu_scatter_nd_update_asc` | ≤1.2 | §3.2 |
| 4 | `SwiGLU` 融合 | ≤0.6 | §3.2 |

---

## 5. 复现

```bash
# 1) 更新仓库
ssh a3-21 'cd ~/opensrc/cann-recipes-infer && git fetch --all && git checkout master && git pull --ff-only'

# 2) 读官方多流+控核的参考实现
ssh a3-21 'cd ~/opensrc/cann-recipes-infer && sed -n "90,200p" models/deepseek_v4/models/modules/indexer.py'

# 3) 查 limit_core_num 是否可用
ssh a3-21 'docker exec dsv41-tp8k5 python3 -c "
import torch, torch_npu
print(hasattr(torch.npu.npugraph_ex.scope, \"limit_core_num\"))"'

# 4) 核对算子使用情况
ssh a3-21 'docker cp ~/tmp/op_usage.py dsv41-tp8k5:/tmp/ && docker exec dsv41-tp8k5 python3 /tmp/op_usage.py'
```
