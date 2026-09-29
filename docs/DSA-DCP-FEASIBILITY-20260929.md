# V4.1 实现 DCP（Decode Context Parallel）的可行性与实施路径

**日期**：2026-09-29　**目标**：在 8 chip 上实现 `TP8 + DCP8`，把 KV 容量从 3.50 M token 提到 ~28 M
**标记**：【实测】/【推断】/【未确认】　**状态**：调研完成，待实施

---

## 0. 一句话结论

**cannbot-skills 内没有可直接用的 DCP 实施参考；但 vllm-ascend 自己**（`sfa_cp.py`）
**有一份针对"同类稀疏注意力"的完整 DCP 实现 —— 这是最好的蓝本。**

缺的不是"从零实现 DCP"，而是"**把 V4.1 特有的 `compress_ratios`（CSA2）纳入已经存在的 DCP 框架**"。

---

## 1. ★★ 关键发现：V3.2 与 V4.1 的 DSA 差异是**代码级分界线**，且它决定谁能用 DCP

用户提出的"V3.2 和 V4.1 的 DSA 可能有差异"被证实，而且 vllm-ascend 里有一条**明确的判定函数**：

```python
# vllm_ascend/utils.py:172
def model_uses_sfa_sparse(model_config) -> bool:
    hf_text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if hf_text_config is None:
        return False
    if model_uses_kpool_indexer(model_config):
        return False
    return (
        hasattr(hf_text_config, "index_topk")
        and not hasattr(hf_text_config, "compress_ratios")     # ★ 唯一分界
        and not hasattr(hf_config, "compress_ratios")
    )
```

代进两个模型：

| | `index_topk` | `compress_ratios` | 判定 | DCP |
|---|---|---|---|---|
| **DeepSeek-V3.2-Exp** | ✅ | ❌ **无** | **SFA** | ✅ **有** |
| **DeepSeek-V4.1-Flash（我们）** | ✅ 512 | ✅ **有** `[0,0,2,2,...]` | **DSA** | ❌ **无** |

⇒ **分界线就是 `compress_ratios`（即 CSA2 压缩注意力）。**
这正是官方 DCP 支持表只列 `MLA/GQA` 与 `SFA`、**不列 DSA** 的原因。

【实测】我们模型 `config.json` 的 `text_config.compress_ratios = [0,0,2,2,2,...]`
（位置 20 开始是 1，40 之后是 0）⇒ 被判为 DSA。

### 两侧的结构差异（也用数据证实）

| 结构 | V3.2-Exp | V4.1 |
|---|---|---|
| `compressor` | ❌ 0 命中 | ✅ `modules/compressor.py` |
| `mHC` / `hc_mult` | ❌ 0 命中 | ✅ `op_impls/mhc.py`（`hc_mult=4`） |
| `sliding_window` / SWA | ❌ 0 命中 | ✅ window=128，10 个 group |
| `compress_ratio` | ❌ 0 命中 | ✅ 18 处（`dsa_v1`）/ 7 处（`dsa_v41`） |
| `engram` | ❌ | ✅ `engram_gate/hash/hbm.py` |

（来源：`cann-recipes-infer/models/deepseek_v3_2_exp/` 全文检索，均为 0 命中）

⇒ **V3.2 的 DSA 是"带 index_topk 的稀疏注意力"；V4.1 的 DSA 是"带 index_topk + CSA2 压缩 + SWA + mHC"的复合结构。**
⇒ 所以 **V3.2 的 CP 参考不能照搬**，但 **SFA 的 DCP 代码可以**（见 §3）。

---

## 2. cannbot-skills 调研结论（用户要求的调查）

### 2.1 仓库位置

`/home/chiro/projects/vllm/model-comparing/cannbot-skills/`（CANNBot Skills，CANN 官方）

### 2.2 相关 skill 与结论

| skill | 与本任务的关系 |
|---|---|
| `model/model-infer-parallel-analysis` | ★ **最相关**。决策树里有"长序列附加 CP"这一层，但没有 DCP 的实施细节 |
| `model/model-infer-parallel-impl` | ❌ **明确不覆盖** |
| `model/model-infer-kvcache` | ❌ 只有 MLA absorb / Paged / Legacy，无 DCP |

`model-infer-parallel-analysis` 的关键原文（逐字）：

> **CP / KVP 实施支持**：本 skill 输出可包含 `cp_size` / `kvp_size` 候选，但
> `model-infer-parallel-impl` skill **当前不直接支持这两个维度的代码实施**，
> 需参照仓内已有模型手动改造（CP 参考 `cann-recipes-infer/models/deepseek-v3.2-exp/`，
> KVP 参考 `cann-recipes-infer/models/longcat-flash/`）。

### 2.3 它指向的两个参考实现，**都不适合我们**

| 参考 | 机制 | 为什么不适合 |
|---|---|---|
| `cann-recipes-infer/models/deepseek-v3.2-exp/`（CP） | **prefill-only**：配置里逐字写着 `cp_size: 32  # only active at prefill stage`；代码 `cp_size = self.cp_size if is_prefill else 1`；KV 结构体叫 `PrefillCPMetaData` | ① 是 **prefill CP，不是 decode DCP**；② V3.2 无 compressor/SWA/mHC |
| `cann-recipes-infer/models/longcat-flash/`（KVP） | KV 按 **head 维**切 | MLA/DSA 的 `num_key_value_heads=1`，**head 维切不动** |

### 2.4 `cann-recipes-infer/models/deepseek_v4_1/`（顺带发现）

**存在**，但：
- 配置 `world_size: 8`、所有 `*_tp_size: 1`、**无 `cp_size`**
- 且 `platform_version: "950"`（不是 A3）
- 代码里 `cp_size` 只在 **prefill** 用（`cp_size = self.cp_size if is_prefill else 1`，同 V3.2）

⇒ **官方 recipes 仓在 V4.1 上也没有 decode CP。**

### 2.5 结论

**cannbot-skills + cann-recipes-infer 都提供不了我们要的 DSA-DCP 实现或直接参考。**
唯一有价值的是它**确认了"长序列加 CP"是标准方向**，以及给出了"CP 只在 prefill 生效"这个反面参照。

---

## 3. ★★★ 真正可用的蓝本：vllm-ascend 自己的 SFA-DCP

这是本次调研**最有价值的发现**。vllm-ascend 里已经有一份**针对同类稀疏注意力的完整 DCP**：

| 文件 | 行数 | 内容 |
|---|---:|---|
| `vllm_ascend/attention/context_parallel/sfa_cp.py` | **1315** | SFA 的 DCP + 与 DSA-CP 组合的实现 |
| `vllm_ascend/attention/context_parallel/sfa_dcp_utils.py` | 129 | 复制的 indexer-cache 布局、block table 构造 |
| `vllm_ascend/attention/context_parallel/common_cp.py` | 259 | ★ **通用**的 `DCPImplMixin` / `DCPMetadataBuilderMixin` |
| `vllm_ascend/ops/triton/dcp/dcp_a2a.py` | — | ★ DCP 的 a2a 算子 |
| `vllm_ascend/ops/triton/dcp/dcp_a2a_batched.py` | — | ★ DCP 的 batched a2a 算子 |

### 3.1 为什么它适用

**SFA 和 DSA 共享同一个核心难题**：稀疏注意力的 **top-k 候选选择**。
DCP 下每个 rank 只有 1/N 的序列，但 top-k 必须在**全局**选 —— SFA 的 DCP 已经解决了这个问题：

```python
# sfa_dcp_utils.py（逐字）
"""SFA-specific CP helpers shared by the attention and indexer builders.
These helpers encode the replicated SFA indexer-cache layout..."""
```

### 3.2 更强的证据：SFA 的 DCP 能**与 DSA-CP 组合**

```python
# sfa_cp.py:296（逐字注释）
class AscendSFADSADCPMetadata(AscendSFADCPMetadata):
    """SFA metadata for the combined DSA-CP and DCP execution path."""
    dsa_cp_context: DSACPContext | None = None
```

⇒ **存在"prefill 用 DSA-CP + decode 用 DCP"的组合路径**，说明两个机制在框架里是**兼容**的。

### 3.3 通用原语（这两条最省工作量）

```python
# common_cp.py:55 / 108
class DCPMetadataBuilderMixin:
    def _get_dcp_context_lens(...)
    def _get_dcp_rank_context_lens(...)

class DCPImplMixin:
    def _dcp_all_gather(self, tensor, dim)
    def _dcp_all_gather_fragments(...)
    def _merge_dcp_attention_output(...)     # ← 含 LSE 合并（softmax 分母）
```

⇒ 跨 rank 的收集与合并**不用自己写**。

---

## 4. 各层现状盘点

| 层 | 状态 | 证据 |
|---|---|---|
| **① Rank 分配** | ✅ **就绪** | `vllm/config/parallel.py:526`：`# DCP reuses the TP ranks when PCP is disabled`，`tp % dcp == 0` 即可 ⇒ TP8 支持 DCP8 |
| **② KV 容量缩减** | ✅ **就绪** | `core/kv_cache_interface.py:228`（`cdiv(max_len, block_size × dcp)`）与 `:265`（`max_memory_usage_bytes`）；`vllm_ascend/core/kv_cache_interface.py:103` 同样逻辑，注释写着 *"each dcp rank only need save max_model_len//dcp_world_size tokens locally"* |
| **③ 我们 5 个 spec 的继承** | ✅ **就绪**【未实测】 | 5 个 spec 都继承到 `AttentionSpec`/`FullAttentionSpec` 的 dcp 逻辑（逐个查了继承链） |
| **④ 多 group + dcp≠1 的 block size 对齐** | ✅ **就绪** | `patch/platform/patch_kv_cache_utils.py` 有专门的 `if dcp != 1:` 分支处理 LCM 对齐，**正是为多 group 场景写的** |
| **⑤ 通用跨 rank 原语** | ✅ **就绪** | `DCPImplMixin` / `DCPMetadataBuilderMixin` / triton `dcp_a2a` |
| **⑥ 同族参考实现** | ✅ **就绪** | `sfa_cp.py`（1315 行） |
| **⑦ DSA 的 attention backend 选 CP 实现** | ❌ **缺** | `dsa_v41.py:1060` `get_impl_cls()` 硬返回 `AscendDSAV41Impl`，**不看 `decode_context_parallel_size`** |
| **⑧ 把 compress_ratios/CSA2 纳入序列切分** | ❌ **缺（主要工作量）** | 无 |
| **⑨ SWA(10 组)/state/dspark 的切分语义** | ❌ **缺** | 无（SWA window=128 跨 rank 需要 halo） |
| **⑩ 安全网** | ❌ **缺** | `dsa_v41.py:1076` `supports_dcp = False` —— 但**全树 `.supports_dcp` 零命中**，是死代码 ⇒ **没有任何框架级拦截** |

### ★ 4.1 一个必须知道的危险组合

如果现在直接设 `--decode-context-parallel-size 8`（不设 `enable_dsa_cp`）：

- **② 会生效**：KV spec 按 `cdiv(max_len, dcp)` 缩小 **8 倍**
- **⑦ 不会生效**：attention 仍是 `AscendDSAV41Impl`（非 CP），它**假设本地有完整序列**
- ⇒ **KV 只存 1/8，注意力却要读全序列** ⇒ 必然算错
- ⇒ 而且 **⑩ 无安全网**，不会有人拦你

**这是"静默算错"类事故，与本仓踩过的"draft 图静默无 attention（A 恒 1.0）」同族。**

---

## 5. V4.1 的 CP 现状（最新 main vs 我们的镜像）

【实测】最新 main（`b64b4d7`）比我们用的镜像（`e43cf1e9f`）**多了一个 V4.1 专用 CP 文件**：

```python
# vllm_ascend/attention/context_parallel/dsa_v41_cp.py（240 行，最新 main 独有）
"""V4.1 replicated-cache TP-token DSA CP adapter."""
```

以及 `dsa_v41.py` 已接线：

```python
@staticmethod
def get_builder_cls():
    from vllm_ascend.attention.context_parallel.dsa_v41_cp import get_v41_cp_classes
    return get_v41_cp_classes()[0]

@classmethod
def supports_pcp(cls) -> bool:
    return False
```

**但注意 docstring 里的 `replicated-cache`** —— 它的 KV 是**复制**的，只在 TP 组内切 **token** 做 prefill 并行，**不缩减 KV 容量**。

| 机制 | 开关 | V4.1 有？ | KV 缩减？ |
|---|---|---|---|
| **DSA-CP** | `additional_config.enable_dsa_cp` | ✅ 最新 main 有（我们镜像无） | ❌ **replicated-cache，不减** |
| **DCP** | `--decode-context-parallel-size` | ❌ **无实现** | ✅ 会减（spec 层） |

⇒ **升级到最新 main 能白拿 DSA-CP（prefill 加速），但拿不到 DCP。**

---

## 6. 实施路径（建议）

### 6.1 先做 Step 0（20 分钟，零改码）

```bash
# 在 8 chip 上起 TP8/DP1 + --decode-context-parallel-size 8（DSA backend）
# 只看两件事：
#   ① 能不能起来 / 在哪报错
#   ② GPU KV cache size 有没有变成 8 倍（3.50M → 28.0M）
```

**期望结果**：能起来（无校验拦截），KV 显示 8 倍，但**推理结果是错的**
⇒ 这就把 §4.1 那个"危险组合"从【推断】变成【实测】，并确认 ② 确实生效。

### 6.2 然后实现（按 §3 的蓝本）

| Step | 内容 | 依据 |
|---|---|---|
| 1 | 在 `dsa_v41.py` 的 `get_impl_cls` / `get_builder_cls` 加 DCP 分支 | 照 `sfa_v1.py:396` / `mla_v1.py:112` 的写法 |
| 2 | 新建 `dsa_v41_dcp.py`：`AscendDSAV41DCPImpl(DCPImplMixin, AscendDSAV41Impl)` | 照 `sfa_cp.py` 的 DCP 部分 |
| 3 | 把 **CSA2 compressor** 纳入序列切分（主要工作量） | 无现成参考 |
| 4 | 处理 **SWA 的跨 rank halo**（window=128，10 组） | 无现成参考 |
| 5 | 处理 `state`（FP32 32 行环）与 `dspark`（aliasing）的切分语义 | 无现成参考 |
| 6 | 加**离线自检**：判据绑"KV 容量 8 倍"+"结果正确"两个可观测痕迹 | 本仓纪律 |
| 7 | 真机验容量 + 正确性（144K/1M 针）+ 性能三元组 | 本仓纪律 |

### 6.3 明确的取舍

| 换来 | 代价 |
|---|---|
| KV 8 倍（3.50M → 28.0M token） | **放弃动态推测解码**（`platform.py` 明确：*"Dynamic speculative decoding and decode context parallelism is not supported by vLLM Ascend"*） |
| 单 rank 不必装下 1M 序列 | 每步跨 rank gather KV 的时延（**未量化**） |
| — | 与 **Engram HBM sharing 互斥**（`engram_hbm.py`：*"Engram HBM sharing requires EP and PP=PCP=DCP=1"*）⇒ 需确认我们是否走这条路 |

---

## 7. 诚实边界

1. **SFA 的 DCP 代码我只读了结构与 docstring**（`sfa_cp.py` / `sfa_dcp_utils.py` 的函数名与注释），**没有逐行读完 1315 行**，所以"可复用程度"是【推断】。
2. **`DCPImplMixin._merge_dcp_attention_output` 的 LSE 合并是否适用于 DSA 的 sparse attention 未验证** —— SFA 能用不代表 DSA 能用（虽然两者都是 sparse）。
3. **Step 0 未做**：§4.1 那个"危险组合"目前仍是【推断】，没有实测确认"KV 缩 8 倍但 attention 读全序列"这个失败形态。
4. **SWA/state/dspark 的 DCP 语义完全没查** —— 这三类占 13 个 group 里的 12 个，是**主要工作量的来源**。
5. **性能代价未量化**：DSA 的 `index_topk=512` 在 DCP 下是否需要跨 rank 交换候选、量级多大，未知。
6. **未向官方确认**是否正在做 DSA-DCP（官方文档只说 DSA-CP *"will be removed once PCP support is stable"*，没提 DSA-DCP 计划）。

---

## 8. 参考资料（本地路径）

| 内容 | 路径 |
|---|---|
| cannbot-skills（官方技能库） | `~/projects/vllm/model-comparing/cannbot-skills/` |
| 并行分析 skill（含 CP 决策树） | `.../model/model-infer-parallel-analysis/SKILL.md` |
| 参考配置索引（含 cp_size 表） | `.../model/model-infer-parallel-analysis/references/config-index.md` |
| cann-recipes-infer（CP 参考，prefill-only） | `~/tmp/cri/models/deepseek_v3_2_exp/` |
| cann-recipes-infer 的 V4.1（无 decode CP） | `~/tmp/cri/models/deepseek_v4_1/` |
| **vllm-ascend 最新 main（含 `dsa_v41_cp.py`）** | `~/tmp/va-latest/`（`b64b4d7`） |
| **SFA 的 DCP 实现（最佳蓝本）** | `~/tmp/va-latest/vllm_ascend/attention/context_parallel/sfa_cp.py` |
| 通用 DCP 原语 | `~/tmp/va-latest/vllm_ascend/attention/context_parallel/common_cp.py` |
| DCP triton 算子 | `~/tmp/va-latest/vllm_ascend/ops/triton/dcp/` |
