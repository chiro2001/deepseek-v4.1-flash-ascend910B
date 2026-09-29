# SFA-DCP 照抄手册：给 DeepSeek-V4.1 的 DSA 实现 DCP

**日期**：2026-09-29
**配套**：`DCP-OPERATOR-INVENTORY-20260929.md`（算子级清单，已说清的结论本文不再复述）
**本文定位**：把 `vllm-ascend` 里**已经跑起来的 SFA-DCP** 逐行读透，产出「照着改」的移植细节 + 与我们 V4.1 树的精确差距。
**约定**：【参考】= 蓝本树 `/home/chiro/tmp/va-latest/vllm_ascend`（vllm-ascend main，`b64b4d7`）；【镜像】= 实机镜像解包树 `/home/chiro/tmp/v41img2/vllm_ascend`（**真正要改的树**）。行号只对这两份快照有效。
**只读声明**：本次未改动任何被读代码，未起服务，未 commit。

---

## 0. 先对齐版本：两棵树不是同一份代码

这一点决定「哪些能整段抄、哪些要重写」。

| 项 | 【参考】va-latest | 【镜像】v41img2 | 影响 |
|---|---|---|---|
| `attention/context_parallel/sfa_cp.py` | 1699 行，DCP 核心在 657-1573 | **1315 行，DCP 核心在 579-1264** | 镜像自带的 SFA-DCP 更早，但**机制相同**；建议以镜像为准抄，参考树只看后续修复 |
| `common_cp.py` | 259 行；`get_dcp_local_seq_lens` 从 vLLM 导入 | 233 行；**自带** `get_dcp_local_seq_lens`（`common_cp.py:11-31`） | 镜像这个本地版返回 `[num_reqs, dcp]`，取本 rank 要 `[:, dcp_rank]`；参考树/新版 vLLM 是 `(seq_lens, dcp_size, dcp_rank, interleave)` 直接返回本 rank。**抄调用点时必须按镜像的语义写** |
| `sfa_dcp_utils.py` | 有（129 行） | **没有** | 复制态地址公式在镜像里是 `sfa_cp.py` 的类方法（`_build_block_table_replicated_view` 685-720、`_build_slot_mapping_replicated_view` 721-763） |
| DCP 合并算子的注册名 | `torch.ops.vllm.dcp_a2a_fused`（`ops/triton/dcp/dcp_a2a.py:616-622`） | **`torch.ops.vllm.sfa_dcp_a2a_fused`**，由 `vllm_ascend.ops.triton.sfa_cp` 注册（导入点 `sfa_cp.py:11`） | 移植时用镜像侧的名字/模块；`ops/` 目录本次未解包，**落地前先确认镜像里存在 `ops/triton/sfa_cp.py`、`sparse_index_remap.py`、`query_gather_prep.py`** |
| DCP 尺寸校验 | `dcp == pcp` 或 `tp*pcp`（`platform.py:1579-1599`） | **`dcp == tp`**（`platform.py:1519-1532`） | V4.1 若走 SFA 同款实现，DCP8 ⇒ TP8 |
| `sparse_index_remap` Triton 版 | `ops/triton/sparse_index_remap.py:26-79`（int32 整数除法 + 两段式压缩） | 镜像里只有 torch/fp32 回退（`sfa_cp.py:998-1046`）；Triton 版未解包 | 先抄 fp32 回退版验证正确性，再换 Triton |

【镜像】的 `_merge_dcp_outputs` 调用签名（照抄锚点，`sfa_cp.py:1047-1086`，return 在 1080-1085）：

```python
return torch.ops.vllm.sfa_dcp_a2a_fused(
    sfa_output,          # [T, H_local, D_model]
    softmax_lse,         # [T, H_local, 1] float32
    self.dcp_size,       # int
    scatter_dim,         # 1 = 按 head 切；0 = 按 token 切
    self.dcp_group.unique_name,
)
```

---

## 1. Q1：KV 到底怎么切

### 1.1 三条权威公式（互相对得上）

**(a) 上游 slot-mapping kernel**（`vllm/v1/worker/block_table.py:445-470`）——DCP 的唯一事实来源：

```python
virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
vb_idx  = pos // virtual_block_size
vb_off  = pos - vb_idx * virtual_block_size
is_local = (vb_off // CP_KV_CACHE_INTERLEAVE_SIZE) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
local_off = (vb_off // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)) * CP_KV_CACHE_INTERLEAVE_SIZE \
            + vb_off % CP_KV_CACHE_INTERLEAVE_SIZE
slot = block_table[vb_idx * BLOCKS_PER_KV_BLOCK + local_off // block_size] * block_size + local_off % block_size
slot = is_local ? slot : PAD_ID(-1)
```

**(b) 等价的全局 token 归属式**（把 (a) 化简，前提 `I | KV_CACHE_BLOCK_SIZE`，SFA-DCP 与 V4.1 都满足）：

```
owner(pos) = (pos // I) % dcp        # 全局第 k 个 I-token 块归 rank k%dcp
local_index(pos) = (pos // (I*dcp)) * I + (pos % I)
I = cp_kv_cache_interleave_size
```

**(c) 每 rank 的本地 KV 长度**（【镜像】`common_cp.py:11-31`，语义同上游 `vllm/v1/attention/backends/utils.py:1091-1128`）：

```
base      = (L // I // dcp) * I
local_len = base + clip(L - base*dcp - rank*I, 0, I)
```

### 1.2 所以是「块轮转」，不是连续区间

- `I = 1`：token 级轮转，`pos % dcp` 归属。
- `I = block_size`：**块级轮转/块循环（block-cyclic）**——全局每 `I*dcp` 个 token 一个超块，超块内按 `[rank0 I个][rank1 I个]…[rank(dcp-1) I个]` 划走，rank 内部再把自己的片段**压缩成连续地址**（`local_off`）。
- 都不是「连续区间切分」；也**不是**「整块轮流分配物理块」。

**`cp_kv_cache_interleave_size` 的作用**：决定「一个 rank 一次拿多少连续 token」，也就是决定压缩组（ratio=2）会不会跨 rank。
【参考】`vllm/config/parallel.py:382-393` 的注释即此意：`=1` 时 token i 落 rank `i%dcp`；`=block_size` 时先填满靠前 rank 的同一块。

**SFA-DCP 强制 `I == cache_config.block_size`**：
【镜像】`platform.py:1320-1327`、【参考】`platform.py:1384-1391` —— 判据是 `model_uses_sfa_sparse(model_config)`：

```python
if use_sparse and cp_size > 1 and parallel_config.cp_kv_cache_interleave_size != cache_config.block_size:
    logger.warning_once("... Override cp_kv_cache_interleave_size to {block_size}")
    vllm_config.parallel_config.cp_kv_cache_interleave_size = cache_config.block_size
```

⚠️ **V4.1 陷阱**：`model_uses_sfa_sparse()` 显式排除带 `compress_ratios` 的模型（【参考】`utils.py:172-183`），所以**这个强制覆盖对 V4.1 不生效**；V4.1 开 DCP 必须自己加校验/覆盖，否则 `I` 会停在默认 1，压缩槽边界与 rank 边界错位。

### 1.3 每个 rank 到底存什么

| 平面 | 归属 | 每 rank 存储 | 依据 |
|---|---|---|---|
| 主 KV（SFA 的 nope/rope；V4.1 的 `long_kv` 压缩态） | **DCP 分片** | 全序列的 `1/dcp` | 【镜像】`core/kv_cache_interface.py:103-113`（`max_model_len // dcp`）+ 1.1(a) 的 `-1` 掩码 |
| SFA 的 LightningIndexer K/scale | **每 rank 全量复制** | 全序列 1 份/rank | 【镜像】`sfa_cp.py:565-576` 设计注释 + `core/kv_cache_interface.py:126-140` 页放大 |
| V4.1 `indexer.k_cache` | 目标：全量复制 | 全序列 1 份/rank | 现状是 `DeepseekV41IndexerSpec ⊂ AscendMLAAttentionSpec`（【镜像】`core/deepseek_v41.py:30-41`）⇒ 会被当 full-attention 分片，**必须改成复制态**（改造点见 §8.2-⑪） |
| V4.1 `swa` | 目标：每 rank 全量 | window 内 token（本来就是 window-sized） | 见 §4：spec 是 SlidingWindow 系（复制态），但 slot_mapping 仍被 rank 过滤，**必须修写侧** |
| V4.1 `compressor.state_cache` | 每 rank 全量（32 行 FP32 环） | 1 页/请求 | `AscendCircularBufferSpec ⊂ AttentionSpec` 但**不是** `FullAttentionSpec`，`dcp_world_size_for_kv_cache_spec` 返回 1 ⇒ 复制；环由 metadata 寻址，不用 slot_mapping |

### 1.4 主 MLA cache 是否真的按 1/dcp 分片？——是

两层证据：
1. **显存层**：`AscendMLAAttentionSpec.max_memory_usage_bytes`（【镜像】`core/kv_cache_interface.py:103-113`）
   ```python
   if dcp_world_size > 1:
       max_model_len = cdiv(max_model_len, dcp_world_size)   # 每 rank 只留 1/dcp
   return cdiv(max_model_len, self.block_size) * self.page_size_bytes
   ```
2. **写入层**：slot-mapping kernel 给非本 rank 的 token 写 `PAD_ID`，写入算子（`reshape_and_cache` / `scatter_cache_sk`）直接跳过 `-1`。

### 1.5 indexer cache 为什么必须复制、复制多少倍、代价多大

**为什么**：DCP 下每个 rank 只拿到一部分 KV，但**稀疏 top-k 必须逐字一致**（否则各 rank 的 softmax 归一化域不同，LSE 合并出来的结果不等于全局 softmax）。SFA 的解法是「**让每个 rank 都能看到全量 indexer cache，各自独立地算出同一份全局 top-k**」，再各自把不属于自己的 index 置 `-1`（§2）。这是一种**用 indexer cache 显存换通信**的设计：不引入任何 top-k 相关通信。

**复制多少倍**：每 rank 一份全量 ⇒ 相对「同样按 DCP 分片」的布局是 **dcp 倍**；存储上通过把 spec 的页按 `dcp` 放大实现：

```python
# 【镜像】core/kv_cache_interface.py:126-140
class AscendSFAIndexerCacheSpec(MLAAttentionSpec):
    sfa_dcp_replicated_indexer_size: int = 1        # runner 里在 DCP>1 时置为 dcp
    @property
    def real_page_size_bytes(self) -> int:
        num_heads_per_page = self.block_size * self.num_kv_heads
        return self.sfa_dcp_replicated_indexer_size * num_heads_per_page * (
            self.head_size * get_dtype_size(self.dtype)
            + self.scale_dim * get_dtype_size(self.scale_dtype))
```
配合继承来的 `max_memory_usage_bytes`（除以 dcp）⇒ `cdiv(L/dcp, B) * (B*dcp*130B) ≈ L*130B`，即**每 rank 一整份全量 indexer cache**。
赋值点【参考】：`worker/model_runner_v1.py:507-509`、`worker/v2/attn_utils.py:116-120`（`enable_sfa_dcp_replicated_indexer()` 为真时 `= self.dcp_size`）。

**代价（V4.1 实参数）**：`index_head_dim=128`（int8）+ `scale_dim=1`（fp16）= **130 B/压缩 token**。
V4.1 只有 4 个 `compressor/index` 物理 cache（`kv_source_layers=[2,8,14,20]`，`compress_ratios[2,8,14]=2`、`[20]=1`；`index_source_layers` 里 24/28/32/36 复用 20 的 cache，见 `models/deepseek_v41/model.py:710-713`）：

```
每 rank ≈ L * 130B * (3 层 /2 + 1 层 /1) = L * 325 B
L=1M  ⇒ 325 MB/rank，8 rank 合计 2.6 GB
（对照：若按 DCP 分片只需 40 MB/rank）⇒ 额外 ~285 MB/rank
```

> 结论：**indexer 复制是这套 DCP 方案最贵的一项显存开销**，也是它唯一「不能省」的代价。V4.1 若要把 indexer 也分片，就必须改成「分布式选 top-k + 通信」，那是另一条自研路线。

### 1.6 复制态地址公式（可直接照抄）

【镜像】`sfa_cp.py:685-720`（【参考】抽成了 `sfa_dcp_utils.py:57-88`）：

```python
def build_block_table_replicated_view(dcp_block_table, replicated_col_idx,
                                      dcp_size, blocks_per_phys_block,
                                      seq_lens, num_reqs):
    # replicated_col_idx = arange(0, local_cols * dcp)
    local_col_idx = (replicated_col_idx // (dcp_size * blocks_per_phys_block)
                     * blocks_per_phys_block
                     + replicated_col_idx % blocks_per_phys_block)
    rank_in_view = (replicated_col_idx // blocks_per_phys_block) % dcp_size
    local_logical_blocks = torch.index_select(dcp_block_table, 1, local_col_idx)
    if blocks_per_phys_block == 1:
        replicated = local_logical_blocks * dcp_size + rank_in_view
    else:
        sub = local_logical_blocks % blocks_per_phys_block
        phys = local_logical_blocks // blocks_per_phys_block
        replicated = (phys * dcp_size + rank_in_view) * blocks_per_phys_block + sub
    return replicated * (seq_lens[:num_reqs] > 0).view(-1, 1)   # 空请求整行清零
```

复制态 slot（【镜像】`sfa_cp.py:721-763`）：**用全局 position 直接算，不做 rank 过滤**

```python
logical_block_idx = positions // replicated_view_block_size
block_offsets     = positions %  replicated_view_block_size
block_numbers = replicated_block_table.flatten()[req_indices * width + logical_block_idx]
slot_mapping[:num_actual_tokens] = block_numbers * replicated_view_block_size + block_offsets
# 其余位置预先 fill_(-1)
```

语义：`dcp_block_table` 的第 `c` 列 = 「全局第 c 个 I-token 块」在本 rank 的物理块；复制视图把它展开成 `dcp` 列/块，`local_phys*dcp + r` 这种「块号 ×dcp + 副本序」的排布**正是 `sfa_dcp_replicated_indexer_size=dcp` 所需的物理布局**（每个 rank 用同一套地址写满自己那一整份）。

---

## 2. Q2：top-k 怎么全局化

### 2.1 端到端链路（SFA 实测）

```
indexer 写 cache ──> QLI/lightning indexer 选 top-k（复制态坐标 = 全局坐标）
                 ──> _remap_sparse_indices（全局 -> 本 rank 本地，非本 rank 置 -1 并压到行首）
                 ──> sparse attention（本地 block table + 本地 seq_lens）
```

关键点：**没有分布式 top-k，也没有 top-k 的 all-gather**（只有 DSA-CP 叠加时才 all-gather token 分片，见 §2.5）。

### 2.2 张量形状与语义

| 阶段 | 形状 | 语义 | 位置 |
|---|---|---|---|
| indexer 选出的 top-k | `[T, K]` 或 `[T, 1, K]` int32（remap kernel 先 flatten 成 2D 再还原）；V4.1 先 `[T, K]`，再由 `pad_sparse_indices` 补成 `[T, 1, K]` | **全局 token 坐标**（V4.1 是**全局压缩 token 坐标**） | 【镜像】`models/deepseek_v41/indexer.py:227-243`；V4.1 `dsa_v41.py:229-238` |
| remap 后 | 同形状，尾部填 `-1` | 本 rank 本地 KV 行号，有效项稳定压到前部 | 【镜像】`sfa_cp.py:998-1046` |
| 送算子 | `[T, K]` 或 `[T, 1, K]`（按后端；V4.1 的 `npu_sparse_flash_mla` 要 `[T, 1, K]`） | `-1` = 跳过 | `device_op.py:458-526`；【镜像】`dsa_v41.py:229-238` |

### 2.3 remap 实现（两版都可照抄）

**torch/fp32 回退版**（【镜像】`sfa_cp.py:998-1046`，语义与 Triton 版逐位一致）：

```python
if self.dcp_size <= 1: return topk_indices
idx = topk_indices.to(torch.float32)
I = self._dcp_interleave_size
blk   = torch.floor(idx / I)                       # 全局 I-块号
owner = blk - torch.floor(blk / self.dcp_size) * self.dcp_size
valid = (idx >= 0) & (owner == self.dcp_rank)
if I == 1:
    remapped = torch.floor(idx / self.dcp_size)
else:
    remapped = torch.floor(idx / (self.dcp_size * I)) * I + (idx - blk * I)
out = torch.where(valid, remapped, self._remap_invalid_index).to(topk_indices.dtype)  # -1
# 稳定压缩：有效项按原顺序压到行首，尾部保持 -1
pack_keys = self._remap_order[:K].expand_as(topk_indices) + (~valid).float() * K
_, order = torch.sort(pack_keys, dim=-1)
return torch.gather(out, -1, order.to(torch.int32))
```

**Triton 版**（【参考】`ops/triton/sparse_index_remap.py:26-79` + `82-120`）与上面同式，但：
* owner 判定用整数除法而不是 fp32；
* 压缩必须「先分 chunk 本地压缩，再按 chunk 顺序 gather」，**不要用 `tl.cumsum` 前缀和**（该文件 128-140 行注释：a2 上不确定、a3 上会挂住 vector core）；输出尾部 `-1` 要在 fused kernel 里用 `in_bounds` 掩码预填，不要在 gather kernel 里补。

### 2.4 被置无效的 index 用什么值、后续算子怎么跳过

- 值：**`-1`（int32/fp32 均可，最终转回 index 的 dtype）**；V4.1 的 `pad_sparse_indices` 补齐也用 `-1`（【镜像】`dsa_v41.py:229-238`）。
- 跳过机制（**在算子内部，不需要额外 mask**）：vendored CSA kernel
  `sparse_flash_mla_csa_block_vector.h:525-535` `GetRealS2Idx()` → `realS2Idx = -1`；
  `:538-542` `GetKeyGmOffset()` → `if (realS2Idx < 0 || realS2Idx >= s2IdLimit) return -1`；
  `:576-580` `CopyInSingleKv()` → `if (keyBNBOffset < 0) return;`
  ⇒ 越界/负值既不取 page，也不参与 softmax。
- **`s2IdLimit` 是 `seqused_cmp_kv`**（即 `cmp_seq_lens`）。DCP 下这个限额必须是**本 rank 的本地长度**，否则 remap 后越界（见 §5.3）。

### 2.5 V4.1 专用：压缩域 remap（**本条是自研，但推导已闭合**）

V4.1 的 top-k 在**压缩 token 坐标**：QLI 的 `seqused_k = cache_seq_lens`、`cmp_ratio = ratio`（【镜像】`models/deepseek_v41/indexer.py:196-215`），返回的 selected 又被 `prepare_indexer_indices` 过滤成「可见、按位置升序」的**压缩 token 下标**（`ops/triton/prepare_indexer_indices.py:36-49`：`visible = (pos+1)//ratio`）。

压缩 token `g` 由原始 token `[g*ratio, g*ratio+ratio)` 合成。当 **`ratio | block_size` 且 `I = block_size`** 时，这一整组必落在同一个 I-块内，于是：

```
owner(g) = (g // (I // ratio)) % dcp
local(g) = (g // (dcp * I//ratio)) * (I//ratio) + (g % (I//ratio))
本 rank 的压缩长度 = get_dcp_local_seq_lens(L, dcp, rank, I) // ratio
cmp_residual（全局）= L % ratio（所有 rank 相同，不需要分片）
```

**结论：压缩平面的 remap 就是同一个 `_remap_sparse_indices`，把 `interleave_size` 换成 `block_size // compress_ratio`。** 这正是 `I = block_size` 必须强制的原因：它让「压缩组」与「rank 边界」天然对齐（V4.1：`ratio∈{1,2}`，`block_size≥64`，整除成立）。

**candidate 两段式筛选（`candidate_topk_blocks=2048`, `candidate_block_size=8`）在 DCP 下无需改动**：候选是 **index-K cache 的物理块 ID**（【镜像】`models/deepseek_v41/indexer.py:225-243`，`candidate_mode=1/2/3`），而 index-K cache 是**每 rank 全量复制**的，物理块 ID 空间在 8 个 rank 上完全一致 ⇒ 第一段（layer 20）产出的候选块、第二段（layer 24+）消费的候选块天然一致。**唯一要改的是最终 position top-k 的 remap**（上面那条）。

---

## 3. Q3：partial attention + LSE 合并的确切调用链

### 3.1 decode 路径（SFA-DCP 实测，【镜像】行号）

```
forward()
 └─ exec_kv()                                   # 写本 rank 的 KV（slot 已按 rank 过滤）
 └─ _prepare_kv_for_parallel()                  # 纯 decode 时=空操作
 └─ ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
 └─ _record_query_gather_context(ql_nope, q_pe) # sfa_cp.py:1113-1126 ↔ va-latest 1416-1429
        └─ _start_dcp_query_gather                 # 1088-1111 ↔ 1373-1414
              torch.cat([ql_nope, q_pe], -1) → all_gather_async(dim=1) → DCPGatherContext(handle)
 └─ indexer(...)                                 # 本地选 top-k（复制态 indexer cache，无通信）
 └─ _execute_sparse_flash_attention_process(...)  # 1171-1264 ↔ 1475-1573
        1) topk = self._remap_sparse_indices(topk)          # 全局→本地，尾部 -1
        2) ql_nope, q_pe = self._finish_dcp_gather(handle)   # 973-981：等 Q gather，得全 H
        3) out, sm_max, sm_sum = DeviceOperator.execute_sparse_flash_attention_process(
               ..., block_table=dcp_context.block_table,   # 本地 block table
               actual_seq_lengths_key=dcp_context.seq_lens, # 本地 KV 长度
               sparse_mode=0, return_lse=True)
        4) softmax_lse = sm_max + torch.log(sm_sum)                       # 1250：自然对数
        5) softmax_lse = softmax_lse.permute(1,0,2).reshape(T, -1, 1)     # → [T,H,1] float32
        6) output = self._merge_dcp_outputs(out, softmax_lse)              # head 维合并
```

### 3.2 合并算子的入参/形状/返回

【参考】`ops/triton/dcp/dcp_a2a.py`（【镜像】对应 `ops/triton/sfa_cp.py`，名字 `sfa_dcp_a2a_fused`）：

| 层 | 函数 | 入参 | 返回 |
|---|---|---|---|
| 入口 | `dcp_a2a_fused(partial_output, softmax_lse, dcp_size, scatter_dim, group_name, ...)`（516-568） | `out[T,H,D]`、`lse[T,H,1] fp32`（**必须 fp32**，254-295 校验） | `[T, H/dcp, D]` |
| 打包 | `pack_dcp_output_lse`（298-363） | 同上 | `send[dcp, H/dcp, T, D+p]`，`p=_lse_pack_dim(dtype)` |
| 通信 | `dist.all_to_all_single(recv, send, group)`（490-497） | — | `recv` 同形状 |
| 合并 | `fused_dcp_lse_combine`（366-472） | `recv[dcp, s, r, D+p]`，可选 `local_output/local_lse` | `[T, H/dcp, D(+1)]` |

`_lse_pack_dim`（246-251）：**bf16/fp16 → 4，fp32 → 1**。
bf16 时 LSE 用「符号+指数码+3 位 base-256 尾数」的整数编码（`_pack_dcp_output_lse_kernel:71-110`）——因为 bf16 只有 8 位尾数，直接存 fp32 LSE 会丢精度。

### 3.3 LSE 合并的数学式

参考实现（【参考】`common_cp.py:248-259`，注释即公式）：

```
LSE_final = logsumexp_i(LSE_i)
O_final   = Σ_i exp(LSE_i - LSE_final) · O_i
```

工程实现（`_fused_dcp_lse_combine_kernel:113-243`）是其数值稳定版：

```
m = max_i LSE_i ;  w_i = exp(LSE_i - m) ;  O = (Σ_i O_i·w_i) / (Σ_i w_i) ;  LSE = m + log(Σ_i w_i)
```

（【镜像】`common_cp.py:121-133` 的 `_merge_dcp_attention_output` 走的是另一条路：`_process_attn_out_lse` 把 `[T,H,D+1]` `permute(1,2,0)` 后 `all_to_all_single`，再 `torch_npu.npu_attention_update(lse_list, out_list, 0)`；MLA/GQA 用它，SFA 用融合算子。两条数学等价。）

### 3.4 LSE 是自然对数

三处一致，**是 ln，不是 log2**：
1. 【参考】`sfa_cp.py:1565`：`softmax_lse = softmax_max + torch.log(softmax_sum)`（`torch.log` = ln）。
2. 合并算子用 `tl.exp / tl.log`（`dcp_a2a.py:196-243`）。
3. V4.1 的融合 MLA 算子：`sparse_flash_mla_csa_block_vector.h:400-410` `Log(outLSE, sum)` 后 `Add(outLSE, outLSE, max)`（AscendC `Log` = 自然对数）。
   （上游用 `lse_base_on_e` 区分两者，默认 `True`：`vllm/v1/attention/backend.py:793-816`。）

### 3.5 空 partial（某 rank 一条 top-k 都没命中）

正确性由三层兜底，**不需要额外分支**：
1. 该 rank 的 `softmax_sum = 0`、`softmax_max = -inf` ⇒ `softmax_lse = -inf`。
2. 打包时 `finite_lse = (lse==lse) & (lse != ±inf)` 为假 ⇒ `exponent_code = 0`（`dcp_a2a.py:88-104`）⇒ 接收侧 `packed_valid = exponent_code != 0` 为假（`:183-190`）。
3. 合并时 `weight = where(valid, exp(lse-m), 0)`，且**先选后乘**：`partial_output = tl.where(valid_lse, partial_output, 0.0)` 再乘权重（`:227-232`，注释明说这是为了防止无效 rank 的 NaN 污染结果）。`denominator = weight_sum if >0 else 1`（`:237-239`）。
4. 全部 rank 都空（理论上只有 0 token 请求）：输出 0，`merged_lse = -inf`。

> 注意：每个被选中的 index 只有唯一 owner，所以 DCP8 下 `index_topk=512` 平均每 rank ~64 条，长序列下**确实会出现某 rank 0 条**，这是正常路径。

---

## 4. Q4：滑窗 / 局部窗口

### 4.1 SFA：「没有滑窗，因此无先例」

* `AscendSFAImpl.__init__` 收 `sliding_window` 参数（【镜像】`sfa_v1.py:425`、`sfa_cp.py:872/885`；【参考】`sfa_v1.py:791`），但**全 SFA 代码再无任何使用**；`rg sliding_window` 在 SFA 的 impl/builder/CP 实现里只命中这几处「接收即转发」的参数。
* SFA 没有 SWA cache plane（KV cache 只有 main nope/rope + indexer K/scale）。
* SFA 的「局部性」来自两处、与 DCP 无关：
  * indexer 的因果可见性（复制态视图天然包含因果规则，注释见 【镜像】`sfa_cp.py:1243-1246`）；
  * sparse 算子 `sparse_mode=3`（prefill，右下降因果裁剪）↔ **decode remap 后必须 `sparse_mode=0`**（【镜像】`sfa_cp.py:1247`，注释：本地 KV 长度与全局 query 长度不再同坐标系）。

### 4.2 MLA-DCP 也没处理局部窗口

`rg "sliding|window|swa"` 在【参考】`mla_cp.py` / `attention_cp.py` / `common_cp.py` **零命中**。
上游明确把滑窗类 cache 排除在 DCP 分片之外：`dcp_world_size_for_kv_cache_spec` 只对 `FullAttentionSpec` 返回 dcp（`vllm/v1/core/kv_cache_utils.py:678-697`），SlidingWindow 保持 `dcp_world_size=1`；`SlidingWindowManager` 里还有 `assert dcp_world_size == 1, "DCP not support sliding window attn now."`（`vllm/v1/core/single_type_kv_cache_manager.py:926`）。
⇒ **「DCP + 滑窗」在 vLLM/vllm-ascend 里都没有先例，包括 SFA 和 MLA 两条线。**

### 4.3 V4.1 的滑窗必须自研（但有明确最小改法）

现状：每个 layer 都有一张 `swa` plane（`window_size=128`，【镜像】`core/deepseek_v41.py:43-51`、`models/deepseek_v41/model.py:588-600`），写入走 `scatter_cache_sk(attn.dsa_attn.swa_cache_layer.kv_cache[0], swa_metadata.slot_mapping, kv)`（【镜像】`dsa_v41.py:305-313 / 368-381`），读取走 `ori_kv=attn.dsa_attn.swa_cache_layer.kv_cache[0]`（`dsa_v41.py:546-553`）。

**风险**：SWA 组在 DCP 下仍是「复制态 spec」，但 `BlockTable` 的 slot-mapping kernel 用的是**进程 DCP 度**（`vllm/v1/worker/block_table.py:147` 的 `total_cp_world_size=self.dcp_world_size`），于是每个 rank 只会写自己那 `1/dcp` 的 token ⇒ **每 rank 的 SWA cache 有空洞，滑窗 attention 直接错**。这不是 SFA 的 bug（SFA 没有 SWA plane），是 V4.1 新增面。

三个候选方案：

| 方案 | 做法 | 评价 |
|---|---|---|
| **A（推荐）复制写入** | 在 SWA 的 metadata builder 里用**不做 rank 过滤**的 slot mapping（公式同 §1.6 的复制态 slot，但 block table 用 SWA 自己的全宽表，`rank_in_view` 恒为 0）。每 rank 写全部 token，读本地窗口 | 无通信、显存不变（窗口本来就复制）、代价只有重复写带宽；与 SFA「复制 indexer」同构 |
| B 分片 + 跨 rank 取窗口 | SWA 也按 DCP 分片，attention 前用 halo 交换邻居窗口 | 需要 halo/通信，且 window=128 < block_size=128×8，跨 rank 边界必须通信，收益极小 |
| C 把窗口搬进 long_kv | 让压缩平面承担局部注意力 | 改变语义，风险最大 |

⇒ 结论写进 §8：**【无先例需自研】**，但方案 A 的实现量约 30-60 行（一个 slot mapping 覆盖 + 一处 spec 判据）。

---

## 5. Q5：metadata 侧要塞什么

### 5.1 SFA-DCP 依赖的字段（已存在，可复用）

| 字段 | 位置（定义） | 生产者 | SFA-DCP 消费点 |
|---|---|---|---|
| `AscendCommonAttentionMetadata.dcp_local_seq_lens` / `_cpu` | 【参考】`attention/utils.py:286-287` | runner：`worker/dcp_utils.py:170-190`；上游 `gpu_model_runner.py:2490-2505` | 【参考】`sfa_cp.py:890-905`（拷进持久 buffer，成为 `dcp_context.seq_lens`） |
| `context_parallel_metadata: AscendDCPMetadata{num_computed_tokens_of_dcp, query_lens_cpu, max_query_len, dcp_mtp_attn_mask}` | 【参考】`attention/utils.py:209-216`、`264` | `worker/dcp_utils.py:588-601`（legacy）、`618-664`（spec/MLA） | SFA **不直接用**（用 `dcp_local_seq_lens`）；GQA/MLA 用 |
| `block_table_tensor`（本 group 的物理表） | vLLM 公共字段 | runner `BlockTable.block_table.gpu` | 【参考】`sfa_cp.py:773-778`（`_get_dcp_local_block_table`）截本地视图 → `dcp_context.block_table` |
| `slot_mapping`（rank 过滤，非本 rank = `-1`） | vLLM 公共字段 | kernel `block_table.py:445-470` | main KV 写 `_get_sfa_kv_slot_mapping`（【参考】1431-1441）；indexer 写则用**复制态**临时替换 |
| `dcp_size` / `dcp_rank` | `common_cp.DCPMetadataBuilderMixin.__init__`（55-71） | `get_dcp_group()` | 全部 |

### 5.2 镜像树里最值得抄的一段：`_build_with_metadata_view`

【镜像】`sfa_cp.py:779-838`（【参考】`869-936` 更完整）——**用 try/finally 临时换视图，再调用原 builder**：

```python
dcp_slot_mapping     = common_attn_metadata.slot_mapping      # 原始（rank 过滤）
full_dcp_block_table = common_attn_metadata.block_table_tensor
dcp_block_table      = self._get_dcp_local_block_table(full_dcp_block_table, num_reqs)
replicated_bt   = self._build_block_table_replicated_view(dcp_block_table, common_attn_metadata.seq_lens)
replicated_slot = self._build_slot_mapping_replicated_view(common_attn_metadata, replicated_bt)
common_attn_metadata.slot_mapping      = replicated_slot
common_attn_metadata.block_table_tensor = replicated_bt
try:
    metadata = build_metadata()          # 原 builder 原封不动地跑
finally:
    common_attn_metadata.slot_mapping      = dcp_slot_mapping
    common_attn_metadata.block_table_tensor = full_dcp_block_table
# 本地视图单独挂在 metadata 上，给「写主 KV / 读本地 KV」用
metadata.dcp_context = DCPContext(slot_mapping=dcp_slot_mapping, block_table=dcp_block_table,
                                  seq_lens=local_seq_lens_buf, kv_gather_block_ids=..., kv_gather_block_table=...)
```

这不污染共享表（`finally` 恢复），是 SFA-DCP 能「零侵入」复用原 builder 的关键。

### 5.3 V4.1 需要的映射（本手册核心结论）

V4.1 的 4 个平面各自由 `DeepseekV41MetadataBuilder` 的一个实例负责（同一个类，靠 `self.kv_cache_spec` 分流：`dsa_v41.py:739-760`，`cache_kind ∈ {swa, long_kv, index_k, compressor_state}`），而且**所有 cache 写入/读取都是 metadata 驱动的**：

* SWA 写：`swa_metadata.slot_mapping`（`dsa_v41.py:305-313, 368-381`）
* long_kv 写：`compressor_metadata.cache.slot_mapping`（`dsa_v41.py:390-450`）
* index_k 写：`indexer_metadata.cache.slot_mapping`（`dsa_v41.py:390-450` → `indexer.update_keys`，`models/deepseek_v41/indexer.py:93-116`）
* 读：`metadata.swa.{block_table,seq_lens,query_start_loc}` 与 `metadata.attention.{block_table,cache_seq_lens}`（`dsa_v41.py:508-538`）

⇒ **只需给每个平面的 builder 注入正确的视图**（一个 mixin 即可，见 §8 的 `V41DCPMetadataMixin` 签名）：

| 平面 | block_table 视图 | slot_mapping 视图 | seq_lens |
|---|---|---|---|
| `swa` | 原样（全宽，复制态） | **复制态（不过滤 rank）** | 全局 `seq_lens`（`seqused_ori_kv`） |
| `long_kv` | 本地（原样，rank 过滤） | 原样（rank 过滤，天然只写自己那段） | **本地压缩长度** `local_seq//ratio` |
| `index_k` | **复制态** | **复制态 + `//ratio`** | 全局压缩长度（QLI 在复制态上选） |
| `compressor_state` | 原样 | 不用（环由 `c2_ring_metadata` 寻址） | 原样 |

**好消息**：`compressed_slot_mapping`（`dsa_v41.py:175-185`）作用在 `common.slot_mapping` 上，所以只要**在调用 builder 之前把 `common.slot_mapping` 换成复制态**，它就会自动算出「复制态的压缩 slot」（`slot//ratio` 与 `(slot+1)%ratio==0` 在 `I=block_size` 下与全局 position 对齐，推导见 §2.5）——**这一行都不用改**。

**block table 宽度**：`max_local_block_table_cols = cdiv(max_model_len, block_size * dcp) * blocks_per_phys_block`（【镜像】`sfa_cp.py:620-624`），复制视图宽 `local_cols * dcp`。
落地时按 V4.1 的**每个平面各自的 `storage_block_size`** 重算（long_kv/index 的 `storage_block_size = block_size//ratio`，`dsa_v41.py:1020-1035`），别直接抄 SFA 的单一 plane 常量。

---

## 6. Q6：KV 写入侧

### 6.1 decode：新 token 写哪个 rank

**只写 owner rank**，公式 §1.1(b)。`slot_mapping` 由 runner 的 kernel 产出，非本 rank 的位置已经是 `PAD_ID(-1)`，写入算子跳过 `-1`，**因此不需要任何通信**：

* SFA：`_get_sfa_kv_slot_mapping()` 返回 `dcp_context.slot_mapping[:num_input_tokens]`（【镜像】`sfa_cp.py:1128-1134`）。
* V4.1：`long_kv` 写用 `compressor_metadata.cache.slot_mapping`，压缩后 `slot//ratio`，天然只写本 rank 的完整压缩组（`valid` 掩码 `(slot+1)%ratio==0` 在 `I=block_size` 下等价于「全局 position 是组内最后一个 token」，`dsa_v41.py:800-838`）。

### 6.2 indexer/K 的写入是「全员都写」

* SFA：indexer cache 每个 rank 都要有全量 ⇒ 写侧用**复制态 slot mapping**。
* V4.1：`index_k` 同样——把复制态 slot mapping 注入 `index_k` builder 即可，`indexer.update_keys` 不用改。
* 注意：SFA 在 DSA-CP（token 切分）叠加时还要在写前 all-gather `k_li`（【参考】`attention/indexer.py:352-395`）；**纯 DCP 不需要**（DCP 不切 token，每个 rank 都有全部 token 的 hidden_states）。

### 6.3 prefill / mixed batch：KV 全 gather（模式 A）

SFA-DCP 对「有 prefill 的 batch」不用 LSE 合并，而是把所有被引用的 KV 块 all-gather 到每个 rank 直接算：

* 构造紧凑视图：`dcp_block_table.flatten().unique(return_inverse=True)` → `(valid_block_ids, compact_block_table)`，再按 **HCCL 段序**（`dcp_collective_rank_order`，因为 `dist.new_group` 会按全局 rank 排序，逻辑 DCP 序 ≠ 段序）重排成 `compact + rank_order*num_blocks`（【参考】`sfa_cp.py:851-867`；【镜像】`765-777` 是简版，无 PCP 的全局视图）。
* 发起：`torch.index_select(kv_cache[0], 0, valid_block_ids)`（+ rope/cat）→ `all_gather_async(..., dim=0)`（【参考】`1200-1238`）。
* 消费：`_finish_dcp_gather` → 用 `kv_gather_block_table` + `sparse_mode=3`、`return_lse=False`（【参考】`1489-1521`）。
* **发起时机**：紧接本层 KV 写完之后、indexer 选 top-k 之前（【参考】`1443-1473`），用来和 top-k 计算重叠。

V4.1 移植注意：V4.1 的 long_kv 是**共享 cache**（只有 kv_source 层写，consumer 层只读，`models/deepseek_v41/model.py:710-713`），gather 的发起者与消费者不是同一层 ⇒ 需要按层角色决定谁发起（无 SFA 先例）。

---

## 7. Q7：图捕获（ACL Graph / FULL_DECODE_ONLY）

**能跑，但有 4 条硬约束**：

1. **只支持 DecodeOnly/SpecDecoding 的 dummy 构建**：`build_for_graph_capture` 对其它 attn_state 直接 `NotImplementedError`（【镜像】`sfa_cp.py:840-855`，【参考】`938-956`）。V4.1 的 `build_for_cudagraph_capture` 目前无此限制（`dsa_v41.py:663-673`），改造时要照 SFA 加上。
2. **所有跨 rank 视图必须是持久 buffer**（地址稳定，图重放时按地址读）：
   `dcp_local_seq_lens_buf`、`block_table_replicated_view_buf`、`slot_mapping_replicated_view_buf`、`arange_buffer` 都在 `__init__` 里预分配（【镜像】`sfa_cp.py:583-645`），每步用 `copy_()` 原地刷新（`779-838`, `721-763`）。**不要在 forward 里新建张量再传给图**。
3. **collective 在图内是普通算子**：`all_gather_async` 就是 `dist.all_gather_into_tensor(..., async_op=True)` 返回 `Work`（`vllm_ascend/distributed/utils.py:17-26`），SFA-DCP 在 `_finish_dcp_gather` 里 `handle.wait()`；SFA 全文件**没有任何 `_EXTRA_CTX.capturing`/`graph_task_group` 特判**（对照 MLA-DCP 要显式 `torch.npu.graph_task_group_begin/end` 包 FIA，见【参考】`mla_cp.py` 与 `attention_cp.py:340-390`）。也就是说 **SFA-DCP 依赖 HCCL collective 本身可被 ACL Graph 捕获**。
4. **shape 由捕获 bucket 决定**：`AscendSFAMetadataBuilder.get_cudagraph_support` 返回 `AttentionCGSupport.UNIFORM_BATCH`（【镜像】`sfa_v1.py:270-277`），因此 all-to-all 的 `[dcp, H/dcp, T, D+p]` 在捕获时形状固定，T 为该 bucket 的 token 数。

**V4.1 额外约束**：`validate_cache_runtime` 只允许 `NONE` 或 `FULL_DECODE_ONLY`（【镜像】`core/deepseek_v41.py:318-330`），且 `dsa_v41_forward` 是 `@eager_break_during_capture` 的 custom op（`dsa_v41.py:51-75`）——这意味着 V4.1 的注意力体在捕获期是「eager 边界」，DCP 的 collective 也会被捕获到同一张图里。**验证点**：`all_to_all_single` + `handle.wait()` 在 DCP 下的图捕获行为需要实机跑一次 `FULL_DECODE_ONLY` + DCP8 的 dummy run。

---

## 8. Q8：与 V4.1 的差距清单

### 8.1 SFA-DCP 有、V4.1 没有（要搬过来的）

| # | 机制 | SFA 位置 | 判定 | 改造点（函数签名级） |
|---|---|---|---|---|
| ① | DCP 双视图 metadata builder（复制态 + 本地） | 【镜像】`sfa_cp.py:579-838` | **可直接复用** | 新增 `class V41DCPMetadataMixin(DCPMetadataBuilderMixin)`，重写 `_build_with_metadata_view(common, build_metadata) -> metadata`；按 `self.kv_cache_spec` 分流四种视图（§5.3） |
| ② | 复制态 block table / slot mapping | 【镜像】`sfa_cp.py:685-763` | **需改造** | `build_replicated_block_table(dcp_block_table, replicated_col_idx, dcp_size, blocks_per_phys_block)`；V4.1 需按平面传 `storage_block_size`（long_kv/index 为 `block//ratio`） |
| ③ | top-k 本地化 remap | 【镜像】`sfa_cp.py:998-1046` | **需改造** | `remap_sparse_indices(topk, dcp_size, dcp_rank, interleave)`：**压缩平面传 `interleave = block_size // compress_ratio`**（推导见 §2.5） |
| ④ | Q all-gather + 输出/LSE all-to-all 合并 | 【镜像】`1088-1111`, `1171-1264`, `1047-1086` | **可直接复用** | `_start_dcp_query_gather(ql_nope, q_pe) -> DCPGatherContext`；`_merge_dcp_outputs(out, lse, scatter_dim=1)`；V4.1 的 q 张量是 `[T, H_local, head_dim]`，与 SFA 形状一致 |
| ⑤ | LSE 通道（`return_lse=True` + `sm_max+log(sm_sum)`） | 【镜像】`1231-1257` | **需改造** | `_native_attention(..., return_softmax_lse: bool)`：V4.1 现在是 `dsa_v41.py:566` 写死 `return_softmax_lse=False`、返回 `(output, _)`；要改成返回 `(output, softmax_lse)`，LSE buffer 用 `[T, H, 1] fp32` |
| ⑥ | `can_return_lse_for_decode = True` 能力声明 | 【镜像】`sfa_cp.py:862` | **可直接复用** | V4.1 的 `DeepseekV41EagerAttentionImpl` 加类属性 `can_return_lse_for_decode = True`、`need_to_return_lse_for_decode`（上游 `cp_utils.py:46` 会校验 `layer.impl`，V4.1 目前 `DeepseekV41CacheLayer` 没有 `.impl`，**这条校验是静默跳过的**，要么补 `.impl` 要么在平台侧 fail-fast） |
| ⑦ | prefill KV 全 gather 紧凑视图 | 【参考】`1200-1238`、`851-867` | **需改造** | V4.1 long_kv 是共享 cache（仅 kv_source 写），gather 发起者需按 `role.is_kv_source` 决定 |
| ⑧ | 解析范式（按开关选 impl/builder） | 【镜像】`1290-1314` | **可直接复用** | `resolve_v41_impl(vllm_config) -> type`：`enable_v41_dcp()` 为真时返回 `DeepseekV41DCPAttentionImpl` |

### 8.2 V4.1 有、SFA 没有（要自研的）

| # | 机制 | V4.1 现状位置 | 判定 | 方案/改造点 |
|---|---|---|---|---|
| ⑨ | **swa plane（window=128，40 层）** | 【镜像】`core/deepseek_v41.py:43-51`、`dsa_v41.py:305-313/368-381/546-553` | **无先例需自研** | 方案 A（§4.3）：SWA builder 注入「不过滤 rank」的 slot mapping；不改窗口语义、不加通信 |
| ⑩ | **compressor state ring（32 行 FP32/请求）** | 【镜像】`core/deepseek_v41.py:68-80`、`compressor.py:60-129`、`dsa_v41.py:954-1013` | **无先例需自研（可解）** | 环本身复制（`AscendCircularBufferSpec` 不是 FullAttention，`dcp_world_size=1`）；DCP 不切 token ⇒ 每 rank 独立推进得到**完全相同的 pooled latent**；只有 owner rank 把它写进 long_kv，所有 rank 写 index_k。需要保证 ring metadata 全部由全局 position 计算（现状即是） |
| ⑪ | **compress_ratio=2 的 CSA2 压缩槽** | 【镜像】`dsa_v41.py:175-185`（`compressed_slot_mapping`）、`_write_compressed_source:383-450` | **需改造（推导已闭合）** | 保持 `I = block_size`（强制覆盖，§1.2），则 `ratio` 整除 `I` ⇒ 压缩组不跨 rank；remap 用 `I//ratio`，本地长度 `local//ratio`。**必改 3 处**：<br/>• `core/deepseek_v41.py:_cache_plane_sizes(spec)`（122-128）要计入 index 复制倍数，否则 `plan_cache_slots`(137-198) 的共享槽容量算小<br/>• `DeepseekV41IndexerSpec` 加 `dcp_replicated_size`（参照 `AscendSFAIndexerCacheSpec`，`core/kv_cache_interface.py:114-140`）<br/>• `DeepseekV41MetadataBuilder.build` 里 long_kv 的 `cache_seq_lens` 与 `max_cache_seq_len` 改本 rank 值（`dsa_v41.py:837-846`, `1001-1006`） |
| ⑫ | **dspark draft layers** | 【镜像】`models/deepseek_v41/dspark.py`、`platform.py:1542-1550` | **无先例需自研** | 平台已直接拒绝「动态投机 + DCP」；draft SWA 走 `AscendSlidingWindowMLASpec`（同 ⑨）。建议 DCP8 首版**不支持 dspark**，先 fail-fast |
| ⑬ | **indexer `candidate_topk_blocks` 两段式筛选** | 【镜像】`models/deepseek_v41/indexer.py:227-243`、`models/deepseek_v41/model.py:459-461,572` | **可直接复用（无需改）** | 候选是**物理块 ID**，index cache 全量复制 ⇒ 8 个 rank 上候选集合天然一致；只有最终 position top-k 要 remap |
| ⑭ | 显式 DCP 拒绝 | 【镜像】`core/deepseek_v41.py:337-348`：`PP=DCP=PCP=1` 否则 `NotImplementedError` | **需放开** | 这就是今天 V4.1 开 DCP 的**唯一失败点**（fail-fast，不会静默算错）；先放开这一条，再按 §1.2 加 `cp_kv_cache_interleave_size = block_size` 的强制 |

---

## 9. Q9：DCP8 的性能代价估算

### 9.1 公式（每层、每 decode step）

记号：`T`=本 step 的 token 数（≈batch 大小），`H`=参与合并的**总** head 数（=64，TP8×DCP8 时每 rank 本地 8 个），`D`=输出维（512），`b`=元素字节（bf16=2），`p`=LSE 打包维（bf16 → 4，fp32 → 1）。

| 项 | 每 rank 发送 | 每 rank 接收 | 全局合计 |
|---|---|---|---|
| Q all-gather（decode） | `T·(H/dcp)·(D_q)·b` | `T·H·D_q·b` | `dcp·T·H·D_q·b`（D_q=512=448+64） |
| 输出+LSE all-to-all | `T·H·(D+p)·b` | `T·H·(D+p)·b` | `dcp·T·H·(D+p)·b` |
| prefill KV gather（仅 prefill/mixed） | `n_blocks · blk · (kv_lora + rope) · b` | 同左 | `dcp ×` 同左 |

### 9.2 V4.1 / DCP8 实参

`H=64, D=512, b=2, p=4`（bf16 打包），TP8×DCP8 ⇒ 本地 8 head：

```
Q gather     : 每 rank 发 8 ∶ 收 64 个 head
               = 1·8·512·2 = 8 KB 发 / 64 KB 收
输出+LSE a2a : 发 = 64·1·516·2 = 66.0 KB/rank（每个对端 8.25 KB）
               收 = 66.0 KB
合计 ≈ 发 74 KB / 收 130 KB（每 rank·层·token-step）
```

**同步次数**：每层每个 step **2 次 collective**（1 次 all_gather + 1 次 all_to_all），与序列长度无关。
对照：主 KV 的片内 1/dcp 使 decode 期的 KV 读取量降为 1/dcp，这正是 DCP 的收益来源。

**规模感**：`T=128` 时每层每 rank 发 ≈ 9.5 MB / 收 ≈ 16.6 MB；`T=128`、40 层 ≈ 378 MB 发 / 665 MB 收 per step——**通信量与序列长度无关，只随 token 数线性增长**。所以 DCP8 的收益必须来自「稀疏 KV 读取量降到 1/dcp」：`index_topk=512`、`D=512` bf16 时，未开 DCP 每 token 每层读 ~512 KB 的 cmp KV，开 DCP8 后本 rank 只需读 ~64 行 ≈ 64 KB（省 ~8×），换 66 KB 的 a2a 发送 + 66 KB 接收。是否划算取决于「HBM 读」与「HCCL 传输」的相对成本，必须实机量化（建议：DCP1 vs DCP8，固定 batch，比较单层 attention 时间与 HCCL 时间占比）。

### 9.3 能否 overlap

可以，且 SFA-DCP 是**故意分开「发起」与「等待」**设计的：

* Q gather 在 KV 写完之后立刻 `all_gather_async(async_op=True)` 发起（【镜像】`_record_query_gather_context:1113-1126`），中间隔着 **indexer 选 top-k**（纯本地计算），到 `_finish_dcp_gather`（`973-981`）才 `handle.wait()` ⇒ Q 通信与 top-k 计算重叠。
* prefill 的 KV gather 同理：在 `_store_parallel_kv` 里发起（`1136-1167`），消费在注意力前（`1171-1200`）。
* 输出侧 all-to-all 是**同步阻塞**的（`dist.all_to_all_single` 无 async_op），只在 `_merge_dcp_outputs` 里出现一次，没有 overlap 空间（除非另加通信流）。
* 没有使用多流：SFA-DCP 全文件无 `npu_stream_switch`/`graph_task_group`；V4.1 已有成熟的多流骨架（`dsa_v41.py:315-381`，`dsv4_dsa_overlap_stream()`），若要进一步压延迟，可把 `pack_dcp_output_lse` 的打包放到辅助流上与 `v_up_proj/o_proj` 重叠。

---

## 10. 落地顺序（照抄执行清单）

| 步 | 动作 | 文件/函数 | 验收 |
|---|---|---|---|
| 1 | 放开 DCP 拒绝，强制 `cp_kv_cache_interleave_size = block_size` | 【镜像】`core/deepseek_v41.py:337-348`、`platform.py:1302-1330`（把 `use_sparse` 判据扩到 V4.1） | 起 DCP8 不再 `NotImplementedError`，且日志出现 interleave 覆盖 |
| 2 | 加 `can_return_lse` 能力 + 打开 LSE 输出 | 【镜像】`dsa_v41.py:495-569`（`_native_attention` 返回 `(out, lse)`）、`dsa_v41.py:566` | 单层单请求：`return_softmax_lse=True` 能拿到 `[T,H,1] fp32`，且 == 手算 `logsumexp` |
| 3 | index_k 复制态 spec + 页放大 | 【镜像】`core/deepseek_v41.py:30-41`、`_cache_plane_sizes:122-128`、`plan_cache_slots:137-198` | `pool_bytes_per_block` 随 dcp 增长 dcp 倍；无 OOM；index 缓存内容全量 |
| 4 | builder 双视图 mixin（swa/index_k 复制，long_kv 本地） | 新增 `V41DCPMetadataMixin`（§8.1-①） | dump metadata：三种 slot mapping/block table 形状与 `-1` 分布符合 §5.3 |
| 5 | top-k remap（压缩域 `I//ratio`） | 新增 `remap_sparse_indices(...)`（§2.5） | 单请求 1M 序列：remap 前后 top-k 集合一致（仅坐标变），`-1` 数 = `K - 本rank命中数` |
| 6 | decode 链路（Q gather + LSE a2a） | `_start_dcp_query_gather` / `_merge_dcp_outputs`（§3.1） | DCP8 vs DCP1 输出误差 < 1e-2（bf16 打包），与 §3.3 公式一致 |
| 7 | prefill 链路（KV gather 或逐 rank partial + LSE） | 参考 §6.3 | prefill 首 token 与 DCP1 对齐 |
| 8 | 图捕获 | `build_for_graph_capture`（§7） | FULL_DECODE_ONLY 下 DCP8 dummy run 通过，重放结果稳定 |

### 仍待实机确认（我无法只读验证）

1. `torch.ops._C_ascend.npu_sparse_flash_mla` 的 **torch 层签名**是否暴露 `return_softmax_lse` / `softmax_lse` 输出（TBE 层已确认有：`dynamic/sparse_flash_mla.py:71` `output_names=['attn_out','softmax_lse']`、`:219-223` attr）。建议实机打印 `torch.ops._C_ascend.npu_sparse_flash_mla.default._schema`。
2. LSE 的 **layout**：kernel 按 `lseOffset = (tBase+s1)*gSize + n2*qSeqSize*gSize` 写（`csa_block_vector.h:382-395`），单 kv-head 时等价 `[T, H]` 连续；V4.1 需要 `[T,H,1]`，需确认分配的 buffer 形状能被 kernel 接受。
3. **SWA 平面今天的实际行为**：DCP>1 时 SWA 组是否真的被 rank 过滤（本文按 kernel 代码推断）。验证方法：DCP2、单请求，dump `swa_metadata.slot_mapping`，看是否有一半位置是 `-1`。
4. 镜像树的 vLLM 版本（决定 `get_dcp_local_seq_lens` 签名、`block_table.py` 行号、`dcp_a2a` op 名）。镜像解包树里的 `common_cp.py:11-31` 是自带实现，可规避签名问题。
5. `plan_cache_slots` 里 index 页放大后，`capacity = max(kv_bytes + index_bytes, aliases...)` 是否仍能让 SWA/state 别名落在同一槽内（容量变了会影响 `alias` 组的 `page_size_padded`）。
