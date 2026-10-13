# DSA-DCP 算子级清单（DeepSeek-V4.1 / vllm-ascend / DCP8）

**日期**：2026-09-29
**目标**：为「在 vllm-ascend 上给 DeepSeek-V4.1 的 DSA attention 实现 DCP（decode context parallel，DCP8）」找到可直接复用的算子与实现蓝本。
**标记**：【实测】（我读到的代码事实）/【推断】（由代码推导）/【未确认】（我没验证）

---

## 0. 摘要：4 个必须先知道的结论

1. **main 上已经有 DSA 的 CP 实现，但没有 DSA 的 DCP。**
   `context_parallel/` 里 `dsa_cp.py`（2665 行）与 `dsa_v41_cp.py`（240 行）实现的是
   **DSA-CP（TP 内 token 切分，KV 全量复制）** 与 **PCP（预填充上下文并行）**；
   全文件 `grep -n "decode_context_parallel"` 在 `dsa_cp.py` / `dsa_v41_cp.py` **零命中**。
   DCP 只对 SFA（`enable_sfa_dcp_replicated_indexer`）和 MLA（`mla_cp.py`、`attention_cp.py`）开启。

2. **SFA 的稀疏 top-k 在 DCP 下不是分布式 top-k，也不是运行时 gather 再选**，而是：
   **indexer 的 K cache 在每个 DCP rank 上物理全量复制**，于是每个 rank 用同一份全局 indexer cache
   独立算出**完全相同的全局 top-k**；选完之后再用 `_remap_sparse_indices` 把不属于本 rank 的
   index 置 `-1` 并压到行首。证据：`sfa_cp.py:657-671`（设计注释）、`sfa_cp.py:1283-1330`（remap）、
   `sfa_cp.py:869-896`（builder 把 replicated block table/slot mapping 换进 metadata）。

3. **`_merge_dcp_attention_output` 的数学形式对 sparse 完全成立**（前提见 §4）。
   SFA-DCP 的 decode 路径就是这么做的：本地 top-k 过滤后各算 partial attention，
   再 `dcp_a2a_fused` 做 all-to-all + LSE 合并。证据：`sfa_cp.py:1523-1573`、
   `common_cp.py:147-221`、`ops/triton/dcp/dcp_a2a.py:113-243, 475-568`。

4. **V4.1 的 fused 稀疏注意力算子本身支持返回 LSE**，这是 DCP 能被复用的关键前提：
   `aclnn_sparse_flash_mla.h:44,79-81`（`returnSoftmaxLse` + `softmaxLseOutOptional`），
   CSA（压缩稀疏）kernel 里 LSE 分支真实存在：`arch22/sparse_flash_mla_csa_kernel.h:227,301-303`，
   LSE 计算在 `arch22/sparse_flash_mla_csa_block_vector.h:378-416`（`Log(sum) + max`，自然对数）。
   当前 V4.1 调用点把开关写死为 `False`：`dsa_v41.py:524`。
   **最大工作量不在「合并」，而在「把 compress_ratios/CSA2 的压缩 token 与 indexer cache 纳入 DCP 分片」。**

---

## 1. 快照核对（先说清楚我读的是哪份）

| 内容 | 路径 | 版本 | 状态 |
|---|---|---|---|
| vllm-ascend main | `/home/chiro/tmp/va-latest` | `b64b4d714484feaa6ca71edc99b98318fe6d2f0d`（`git status` 干净） | 【实测】本次所有 vllm-ascend 行号均出自这里 |
| cann-recipes-infer | `/home/chiro/tmp/cri` | `92d9e1f9e57696e4c4762fa8509c015a9e8f9591`（干净） | 【实测】 |
| vllm 本体 | `/home/chiro/miniforge3/envs/pypto-x-w8j/lib/python3.12/site-packages/vllm` | dist-info `vllm_cpu-0.29.0` | 【实测】上游 vLLM 的 DCP 引用出自这里 |
| a3-21 上的 `~/tmp/va-latest`、`~/tmp/cri` | — | — | 【未确认】以我 ssh 落到的账号（`l00886679`）在 a3-21 上**找不到**这两个目录（`find / -maxdepth 4 -name va-latest` 无命中）。本地两份的 commit 与你给的完全一致，故按本地为准；若 a3-21 另有副本，请复核差异 |

### 1.1 与既有结论不一致的地方（重要，别沿用旧认知）

| 旧认知 | 现状（b64b4d7） |
|---|---|
| `sfa_cp.py` 1315 行 | **1699 行**（`wc -l`），DCP 核心在 1109-1573 |
| `context_parallel/` 只有 sfa/mla/attention 三个实现 | 新增 `dsa_cp.py`(2665)、`dsa_v41_cp.py`(240)：**V4.1 的 DSA-CP 适配层** |
| `dsa_v41.py` 完全不看 CP 开关 | `get_impl_cls()` 仍硬返回 `AscendDSAV41Impl`（`dsa_v41.py:1060-1061`），但 `get_builder_cls()` 已经分支：`get_v41_cp_classes()`（`dsa_v41.py:1064-1067` → `context_parallel/dsa_v41_cp.py:22-25`）。impl 的分支发生在模型构造处：`models/deepseek_v41/model.py:738-742`、`models/deepseek_v41/dspark.py:111` |
| `supports_dcp = False` 是死代码 | **仍是死代码**：vllm 侧 `supports_dcp` 只有定义（`vllm/model_executor/layers/attention_layer_base.py:24`、`vllm/v1/attention/backend.py:809`）和 4 处赋值，**全树没有任何读取点**；被消费的是 `supports_dcp_with_varlen`（`vllm/v1/attention/backend.py:623-653`） |
| V4.1 因 `compress_ratios` 被判 DSA 而拿不到 DCP | 成立，但根因更具体：`enable_sfa_dcp_replicated_indexer()` = `model_uses_sfa_sparse() and dcp>1`（`vllm_ascend/utils.py:186-193`），而 `model_uses_sfa_sparse()` 显式排除 `compress_ratios`（`vllm_ascend/utils.py:172-183`） |

---

## 2. 交付 1：`context_parallel/` 算子级清单

### 2.0 文件清单与行数

| 文件 | 行数 | 角色 |
|---|---|---|
| `context_parallel/__init__.py` | 0 | 空 |
| `context_parallel/common_cp.py` | 259 | **DCP 公共底座**：mixins + LSE 合并数学 |
| `context_parallel/attention_cp.py` | 621 | GQA/Dense 的 DCP（head 分片 + chunked prefill） |
| `context_parallel/mla_cp.py` | 906 | MLA 的 DCP（decode LSE 合并 + prefill KV gather+reorg） |
| `context_parallel/sfa_cp.py` | 1699 | **SFA 的 DSA-CP / PCP / DCP 全部实现（本任务主蓝本）** |
| `context_parallel/sfa_dcp_utils.py` | 129 | SFA 复制态 indexer 的 block table / slot mapping 地址构造 |
| `context_parallel/dsa_cp.py` | 2665 | V4 的 DSA-CP（TP token 切分）+ DSA-PCP；**无 DCP** |
| `context_parallel/dsa_v41_cp.py` | 240 | **V4.1 的 DSA-CP 适配层**（复用 dsa_cp 的 builder，改 V4.1 的 Q/KV 切法）；**无 DCP** |
| `ops/triton/dcp/dcp_a2a.py` | 622 | DCP 输出/LSE 打包、all-to-all、融合合并 custom op |
| `ops/triton/dcp/dcp_a2a_batched.py` | 231 | 上述的 A5/BF16 batched kernel |
| `ops/triton/sparse_index_remap.py` | 见文件 | top-k 全局 index → DCP 本地 index 的 Triton remap |
| `ops/triton/query_gather_prep.py` | 见文件 | DCP query head-major 打包（避免两次 all-gather） |

### 2.1 `common_cp.py`（259 行）

| 类 / 函数 | 行号 | 用途 | 入参形状语义 | DSA 可用性 | 底层算子 |
|---|---|---|---|---|---|
| `get_cp_local_query_key_lens(qsl, cum_ql, seq_lens, local_start, local_end)` | 11-31 | 给 **token 切分**（DSA-CP）算每请求的本地 query 累计长度与因果 KV 长度 | 返回 `(local_query_lens[T_req], local_key_lens[T_req])`，`int32` | 【可直接复用】DSA-CP 已在用（`sfa_cp.py:393-401`） | torch 逐元素（`clamp/cumsum/where`） |
| `build_pcp_ordered_slot_mapping(global_slot_mapping, pcp_context, buf)` | 34-52 | PCP 汇聚顺序的 slot 重排 | `global_slot_mapping[T] → buf[T_gathered]`，写不上的置 `-1` | 【需改造】（仅 PCP+DCP 场景） | `torch.index_select` + `masked_fill_` |
| `DCPMetadataBuilderMixin` | 55-105 | 提供 `dcp_size/dcp_rank`，从 `common_attn_metadata.context_parallel_metadata.num_computed_tokens_of_dcp` 取「每个 rank 已算 token 数」 | `_get_dcp_rank_context_lens() → [num_reqs]` | 【可直接复用】 | 纯 Python/torch |
| `DCPImplMixin`（`supports_dcp = True`、`_dcp_all_gather`、`_dcp_all_gather_fragments`） | 108-145 | DCP 组生命周期 + 「把多段张量拼一起再一次 all-gather 再 split」的公共工具 | `_dcp_all_gather(x, dim) → [T*dcp, ...]` | 【可直接复用】 | `GroupCoordinator.all_gather`（HCCL） |
| **`_merge_dcp_attention_output(attn_output, softmax_lse, head_size)`** | 147-162 | **DCP 输出合并入口** | `attn_output[T,H,D]`、`softmax_lse[T,H,1]`（float32） → `[T,H,D]` | 【可直接复用】（数学与 sparse 无关，见 §4） | `_process_attn_out_lse` + `_npu_attention_update` |
| `_process_attn_out_lse(out, lse, dcp_size, dcp_device_group)` | 165-191 | 把 out 与 lse 拼成 `[T,H,D+1]` → `permute(1,2,0)` → **all_to_all_single** | 输入 out/lse 转 float32；输出 `[T, DCP*H, D+1]` | 【可直接复用】 | `dist.all_to_all_single`（HCCL） |
| `_npu_attention_update(head_size, attn_out_lse, dcp_size)` | 194-221 | 把 `[S, DCP*H, D+1]` 拆回 out/lse 列表，调 NPU 融合算子做跨 rank LSE 合并 | `S`=token 数，`H`=本地头数，`D`=head_size | 【可直接复用】 | **`torch_npu.npu_attention_update(lse_list, out_list, 0)`** |
| `_npu_attn_out_lse_update` / `_out_lse_reshape` | 224-245 | 两路 mask/no-mask partial attention 的 LSE 合并（prefill chunk 场景） | 见函数 | 【可直接复用】 | `torch_npu.npu_attention_update(lse, out, 0)` |
| `_update_out_and_lse(out_list, lse_list)` | 248-259 | **LSE 合并数学的参考实现（注释即公式）** | `out[N,B,H,D]`、`lse[N,B,H,1]`；`N`=rank 数 | 【可直接复用】 | torch `logsumexp` / `exp` / `sum` |

> 注意：`_merge_dcp_attention_output` 与 `dcp_a2a_fused` 是**两条并列实现**。SFA 走后者（Triton 融合），MLA/GQA 走前者（`npu_attention_update`）。

### 2.2 `sfa_cp.py`（1699 行）——本任务主蓝本

#### A. 类与 DSA-CP（TP token 切分，**不是 DCP**）

| 类 / 函数 | 行号 | 用途 | 形状语义 | DSA 可用性 | 底层算子 |
|---|---|---|---|---|---|
| `AscendSFAPCPImpl` | 63-245 | PCP（prefill CP）的 SFA impl，O-proj 权重按 PCP 切 | — | 【需改造】 | `all_gather_async`、`DeviceOperator.reshape_and_cache` |
| `_sfa_preprocess_prolog_v3` | 172-225 | PCP 下把本地 KV 写入、pack、跨 PCP all-gather、再 scatter 到全局 slot | `packed[local_tokens, D]` → `gathered[dcp/pcp*tokens, D]` | 【需改造】 | `copy_pcp_kv_cache` + `group.all_gather` + `reshape_and_cache` |
| `DSACPContext` | 248-258 | **DSA-CP（token 切分）的元数据契约**：`num_tokens/num_tokens_pad/local_start/local_end/slot_mapping_cp/actual_seq_lengths_query/key` | 全是 token 区间与每请求长度 | 【需改造】（DCP 复用其 padding 约定） | 纯数据类 |
| `AscendSFADSACPMetadata` / `AscendSFADCPMetadata` / **`AscendSFADSADCPMetadata`** | 260-299 | 三个 metadata 叠加：DSA-CP 字段 + DCP 字段 + 二者组合（docstring「combined DSA-CP and DCP execution path」在 **296-299**） | `dcp_context: DCPContext \| None` | 【可直接复用】作为 DSA-DCP metadata 的模板 | 纯数据类 |
| `DCPGatherContext` / `DCPContext` | 267-285 | 异步 gather 的句柄 + DCP 视图（slot_mapping/block_table/seq_lens/kv_gather_*） | `DCPContext.block_table`=**本地物理** block table | 【可直接复用】 | 纯数据类 |
| `AscendSFADSACPMetadataBuilder` | 302-434 | 在 TP 组内把 token 切给各 rank，算 `local_start/local_end_with_pad`、pad cos/sin 与 slot_mapping、每请求本地 q/k 长度 | `get_cp_local_query_key_lens` 复用点：393-401 | 【需改造】（V4.1 需要按 cache plane 复制） | torch pad / copy |
| `AscendSFADSACPImpl` | 437-654 | DSA-CP 的执行体：本地 token 切片、KV 全量 all-gather、O-proj all2all | `_prepare_native_hidden_states` 470-478 切 token；`_prepare_kv_for_parallel` 556 一次 all-gather 打包 k_pe/k_nope/scale | 【需改造】 | `all_gather_async`、`torch.distributed.all_to_all_single`（645） |
| `_finalize_o_proj` | 603-654 | token 切分下 O-proj 的 all-gather / all2all 回填 | `output[T,...]` 用 `local_start/local_end` 定位 | 【需改造】 | `tp_group.all_gather` / `all_to_all_single` |

#### B. DCP（KV 切分）——**核心**

| 类 / 函数 | 行号 | 用途 | 形状语义 | DSA 可用性 | 底层算子 |
|---|---|---|---|---|---|
| **设计注释：replicated-indexer 布局** | **657-671** | 4 条铁律：① indexer cache 每个 DCP rank 全量复制，② SFA KV cache 保持 DCP 本地（省显存），③ top-k 从复制视图产出后 remap 成本地 KV index，④ replicated view 的 block size = kernel block size | — | 【可直接复用】作为 DSA-DCP 的设计基线 | — |
| `AscendSFADCPMetadataBuilder.__init__` | 676-763 | 预分配 replicated block table/slot mapping 缓冲；`dcp_collective_rank_order`（**HCCL 段序 ≠ 逻辑 DCP 序**，744-758） | `block_table_replicated_view_buf[max_reqs, max_local_cols*dcp]`、`slot_mapping_replicated_view_buf[max_tokens]` | 【需改造】（V4.1 有 4 类 cache plane，要各自建视图） | torch 显存 + `torch.arange` |
| `_get_dcp_local_seq_lens` | 765-771 | **每个 rank 的本地 KV 长度**（由 interleave 决定） | `seq_lens[req] → local_seq_lens[req]` | 【可直接复用】 | `vllm.v1.attention.backends.utils.get_dcp_local_seq_lens`（`utils.py:1091-1130`） |
| `_get_dcp_local_block_table` | 773-778 | 取 DCP 本地物理 block table 视图 | `block_table[reqs, cols]` 截断到 `max_local_block_table_cols` | 【可直接复用】 | `sfa_dcp_utils.py:47-54` |
| `_build_block_table_replicated_view` | 808-828 | 把本地 block table 展开成**复制态 indexer 寻址** | 见 `sfa_dcp_utils.py:57-88` | 【需改造】（DSA 的 indexer 是压缩 token 域） | torch index_select / 整数运算 |
| `_build_slot_mapping_replicated_view` | 830-849 | 生成本次写入的 **复制态 slot**（让每个 rank 都写到全量地址上） | `slot_mapping[T]`；未写位置 `-1` | 【需改造】 | `sfa_dcp_utils.py:91-129` |
| `_build_compact_kv_gather_metadata` | 851-867 | **prefill/mixed 用**：把本地 block table 压紧成「有效块列表 + 跨 DCP 重映射后的 block table」 | 返回 `(valid_block_ids, remapped_block_table)`；`remapped = compact + rank_order*num_blocks` | 【可直接复用】 | torch `unique(return_inverse)` |
| `_build_with_metadata_view` | 869-936 | **关键胶水**：临时把 `common.slot_mapping/block_table_tensor` 换成复制态，调用原 builder，再换回；最后挂 `metadata.dcp_context` | finally 恢复（894-896）保证不污染共享表 | 【可直接复用】 | 上下文式替换 |
| `build_for_graph_capture` | 938-956 | 只允许 DecodeOnly/SpecDecoding 的 dummy 构建 | — | 【需改造】（V4.1 有 FULL graph 与 4 类 plane） | — |
| `AscendSFAPCPDCPMetadataBuilder` | 959-1106 | PCP+DCP：用全局请求视图保证各 rank 打包相同 block ID 顺序；PCP 顺序 indexer slot | `_build_compact_kv_gather_metadata` 987-1015；`_build_pcp_ordered_indexer_slot_mapping` 1017-1046 | 【暂不需要】（我们只上 DCP8，不开 PCP） | `torch.searchsorted`、`build_pcp_ordered_slot_mapping` |
| **`AscendSFADCPImpl`** | 1109-1573 | DCP 执行体 | — | 见下 | — |
| `__init__` | 1113-1160 | 关掉 MLAPO（1143，注释说明 MLAPO 内部写 cache，与 DCP slot 冲突）；读取 `index_topk`（1147-1157，**没有就报错**）；建 remap 常量 | `self._dcp_index_topk`、`_remap_order`、`_remap_invalid_index` | 【需改造】`index_topk` 读取逻辑对 DSA 同样适用 | — |
| `_register_remap_buffers` | 1167-1192 | 把 remap 常量注册成 layer 的 non-persistent buffer（level-2 sleep 后不失效） | — | 【可直接复用】 | `register_buffer` |
| `_has_prefill` | 1194-1198 | `num_prefills>0 or pcp_has_global_prefill` | — | 【可直接复用】 | — |
| **`_record_dcp_kv_gather_context`** | 1200-1238 | **prefill/mixed 的 KV 复制路径**：`index_select` 出有效块 → 把 nope/rope（或 C8 打包态）在 dim=-1 concat → 异步 all-gather | `kv_cache[0][valid_block_ids]` 形状 `[n_blocks, blk, 1, D]` | 【需改造】V4.1 要 gather 的是 **long_kv（压缩态）**，且 SWA cache 是 replicated spec（见 §5.3） | `torch.index_select` + `all_gather_async` |
| `_start_dcp_gather` / `_finish_dcp_gather` / `_all_gather_dim_async` | 1240-1281 | 异步 all-gather 的封装（dim≠0 时 permute→gather→permute 回来） | `gathered[d_x*dcp, ...]` | 【可直接复用】 | `all_gather_async`（`vllm_ascend.distributed.utils`） |
| **`_remap_sparse_indices(topk_indices)`** | **1283-1330** | **全局 top-k → 本 rank 本地 KV index**；非本 rank 的置 `-1`，并把有效项压到行首（不改变 top-k 相对顺序） | `[T, K] int32` → `[T, K] int32`（尾部 `-1`） | 【需改造】V4.1 的 index 是**压缩 token 坐标**，且要处理 `compress_ratio` 与 `cmp_residual`；算法本体（owner 判定 + 除法重映射 + 稳定压缩）可直接照搬 | **Triton `remap_sparse_indices_triton`**（`ops/triton/sparse_index_remap.py:26-79`）；无 Triton 时的 fp32 回退：1305-1330（**舍入误差与排序压缩在 1326-1330**） |
| **`_merge_dcp_outputs`** | **1332-1371** | 输出合并调度：DSA-CP 时 `scatter_dim=0`（token），纯 DCP 时 `scatter_dim=1`（head） | 调 `torch.ops.vllm.dcp_a2a_fused(out, lse, dcp_size, scatter_dim, group)` | 【可直接复用】 | **`vllm_ascend::dcp_a2a_fused`**（`dcp_a2a.py:616-622` 注册） |
| `_start_dcp_query_gather` | 1373-1414 | **decode 用**：把 ql_nope/q_pe 拼起来在 dim=1 做 all-gather（每个 rank 拿到全部 head）；A5 上走 `prep_query_head_major` 快路径 | `ql_nope[T, H_local, D]` → `[T, H_local*dcp, D]` | 【可直接复用】 | `prep_query_head_major`（`ops/triton/query_gather_prep.py`）+ `all_gather_async` |
| `_record_query_gather_context` | 1416-1429 | 决定「prefill 走 KV gather、decode 走 Q gather」的分叉（**这是性能分水岭**） | — | 【可直接复用】 | — |
| `_get_sfa_kv_slot_mapping` | 1431-1441 | 返回 DCP 本地 slot mapping | — | 【需改造】4 个 plane 各自需要 | — |
| `_store_parallel_kv` | 1443-1473 | 写完本层 KV 后**立即启动** prefill 的 KV all-gather，与 indexer 选 top-k 重叠 | — | 【需改造】 | 同上 |
| **`_execute_sparse_flash_attention_process`** | **1475-1573** | **DCP 的两条执行路径总入口** | 见下 | 见下 | `DeviceOperator.execute_sparse_flash_attention_process` |
| ├ prefill/mixed 分支 | 1489-1521 | 等 gather 完成 → 用**紧凑 block table + 全局 top-k** 直接算，**不需要 LSE 合并、不需要 Q gather、不需要 remap**（1504-1507 注释明说） | `block_table=dcp_context.kv_gather_block_table`，`sparse_mode=3`，`return_lse=False` | 【需改造】 | `npu_sparse_flash_attention`（`device_op.py:458-518`） |
| ├ decode 分支：DSA-CP 叠加 | 1527-1535 | token 切分时先把 top-k **跨 DCP all-gather** 回全 token 顺序（1535） | — | 【需改造】 | `dcp_group.all_gather` |
| ├ decode 分支：remap | 1536 | `_remap_sparse_indices` 本地化 | — | 【需改造】 | Triton kernel |
| ├ decode 分支：sparse_mode=0 | 1558-1562 | **关键细节**：remap 后 top-k 已按本地 KV 坐标，不能再用右侧因果裁剪，故 `sparse_mode=0` | — | 【可直接复用】 | op 参数 |
| └ decode 分支：LSE 计算与合并 | 1563-1573 | `softmax_lse = softmax_max + log(softmax_sum)`（1565）→ `permute(1,0,2)` 成 `[T,H,1]`（1566）→ `_merge_dcp_outputs` | `[T,H,1] float32` | 【可直接复用】 | `dcp_a2a_fused` |
| `AscendSFAPCPDCPImpl` / `AscendSFADSADCPMetadataBuilder` / `AscendSFADSADCPImpl` | 1576-1663 | PCP×DCP 组合与 DSA-CP×DCP 组合（后者只是多继承把两组 collective 拼起来，**说明框架是可组合的**） | — | 【可直接复用】作为组合范式 | — |
| **`resolve_sfa_metadata_builder` / `resolve_sfa_impl`** | **1666-1699** | 按三个开关（`enable_dsa_cp` / `enable_sfa_dcp_replicated_indexer` / `pcp>1`）返回实现类 | — | 【可直接复用】**DSA-V4.1 应该仿照这里加 `resolve_dsa_v41_*`** | — |

### 2.3 `sfa_dcp_utils.py`（129 行）

| 函数 | 行号 | 用途 | 形状语义 | DSA 可用性 |
|---|---|---|---|---|
| `get_sfa_pcp_global_metadata` | 13-34 | 用 PCP 的全局 batch 视图替换 metadata 字段（只为构造地址） | — | 【暂不需要】 |
| `get_sfa_dcp_max_local_block_table_cols` | 37-44 | 本地 block table 列数 = `cdiv(max_model_len, blk*dcp) * blocks_per_phys_block` | 标量 | 【可直接复用】 |
| `get_sfa_dcp_local_block_table` | 47-54 | 截取本地视图 | `[reqs, cols]` | 【可直接复用】 |
| `build_sfa_dcp_replicated_block_table` | 57-88 | **复制态寻址核心**：`replicated_block = local_phys_block*dcp + rank_in_view`（`blocks_per_phys_block==1` 时），否则先拆子块再做 | `dcp_block_table[reqs, cols] → [reqs, cols*dcp]` | 【需改造】公式需带 `compress_ratio`（压缩 token → 物理块） |
| `build_sfa_dcp_replicated_slot_mapping` | 91-129 | 由 `positions` 与复制态 block table 算出写 slot | `[T] int32`，非法置 `-1` | 【需改造】V4.1 需要按 plane（SWA/long_kv/index_k）分别算 |

### 2.4 `mla_cp.py`（906 行）与 `attention_cp.py`（621 行）

| 类 / 函数 | 行号 | 用途 | 与 DSA 的关系 |
|---|---|---|---|
| `AscendMLADCPDecodeMetadata` | `mla_cp.py:79-108` | DCP decode 的 local seq lens / q 重排 | 【可直接复用】契约参考 |
| `AscendMlaDCPMetadataBuilder` | `mla_cp.py:111-222` | 生成 `dcp_local_seq_lens`、chunked context | 【需改造】 |
| `AscendMlaDCPImpl.reorg_decode_q` / `_forward_decode_split_attention` | `mla_cp.py:359-364, 485-634` | decode 时把 Q 按 head 重排 + 调 `npu_fused_infer_attention_score` 出 out/lse | 【可直接复用】调用模式（`attention_cp.py:380-404` 同样） |
| **`_reorg_kvcache`** | **`mla_cp.py:810-906`** | **prefill KV gather 后的重排**：把各 rank 的 padded local chunk 拼成连续 KV | 【需改造】V4.1 的 prefill 路径不同（有压缩/共享层） |
| `_merge_dcp_attention_output` 调用点 | `mla_cp.py:803-807`、`attention_cp.py:400-404` | decode 合并入口 | 【可直接复用】 |
| `AscendAttentionDCPMetadataBuilder._build_backend_metadata` | `attention_cp.py:124-183` | GQA 的 prefill chunk 切分与每 rank 上下文长度 | 【需改造】 |
| `_compute_prefill_context` / `_load_kv_for_chunk` | `attention_cp.py:439-510` | **prefill = 逐 chunk 拿全局 KV 到本地算**（另一条 prefill 路线） | 【可参考】 |
| `_update_chunk_attn_out_lse_with_current_attn_out_lse` | `attention_cp.py:406-434` | 用 `_npu_attn_out_lse_update` 把 chunk 结果与本地结果合并 | 【可直接复用】 |

### 2.5 `dsa_cp.py`（2665 行）与 `dsa_v41_cp.py`（240 行）——**DSA 的 CP，不是 DCP**

| 类 / 函数 | 行号 | 用途 | 与 DCP 的关系 |
|---|---|---|---|
| `DSACPMetadata` | `dsa_cp.py:110-121` | token 切分元数据（`local_start/local_end/tokens_per_rank/num_tokens_pad`） | **token 切分**，无 DCP |
| `AscendDSAReqMetadata` | `dsa_cp.py:124-162` | 统一 per-request 元数据（含 `compressor_metadata`、`dspark_swa_indices`） | 【需改造】DCP 版需要再叠一层「本地 KV 视图」 |
| `AscendDSACPLayerMetadata` | `dsa_cp.py:205-211` | 4 个 cache plane 的元数据打包：`swa / compressor_cache / compressor_state / indexer_cache / indexer_state` | 【需改造】**这是 V4.1 的 plane 清单，DCP 必须逐 plane 处理** |
| `AscendDSACPMetadataBuilder.build` | `dsa_cp.py:353-451` | 生成 token 切分 + 每 rank 的 `_build_local_token_metadata` | 【需改造】 |
| `_build_local_token_metadata` | `dsa_cp.py:1140-1237` | 用 `get_cp_local_query_key_lens` 算本地 q/k 长度 | 【可直接复用】 |
| `AscendDSACPImpl._forward` | `dsa_cp.py:1796-2023` | **V4 的 token 切分执行**：全序列 KV cache 更新 + 本地 token attention | 【需改造】DCP 是「本地 KV + 全量 Q」 |
| `_update_indexer_cache` | `dsa_cp.py:2051-2109` | 调 `torch.ops._C_ascend.compressor` 做 indexer 侧压缩并 scatter | 【可直接复用】compressor op 用法 |
| `_indexer_select_topk` | `dsa_cp.py:2111-2176` | 调 `npu_quant_lightning_indexer_v2`（`cmp_ratio=4`、`mask_mode=3`）选 top-k | 【可直接复用】但 DCP 下输出 index 需要 remap |
| `AscendDSAPCPMetadataBuilder/Impl` | `dsa_cp.py:2184-2665` | V4 的 PCP（prefill CP） | 【暂不需要】 |
| `_ReplicatedCacheMetadataBuilder` | `dsa_v41_cp.py:28-55` | **「复制态 cache + 本地 Q」的 builder 范式**：全局 metadata 与本地 Q metadata 分开建（`_build_global_metadata` 47-55） | 【可直接复用】思路与 SFA-DCP 的 replicated view 同构 |
| `AscendDSAV41CPMetadataBuilder.build` | `dsa_v41_cp.py:67-122` | 切本地 token 区间、裁 qsl/seq_lens、复用 global RoPE slice（111-118）、产出 `cp_token_range` | 【需改造】DCP 版还需要 `dcp_context` |
| `AscendDSAV41CPImpl.multistream_preprocess` | `dsa_v41_cp.py:126-191` | 本地 Q 切片 + 复制态 KV 预处理的流重叠；`_write_compressed_source`（182-190） | 【可直接复用】多流骨架 |
| `_select_sparse_indices` | `dsa_v41_cp.py:218-229` | index_source 才算 top-k，其他层读 `shared.topk_indices` | 【需改造】DCP 下共享的 top-k 需要是「全局坐标」且每 rank 可 remap |
| `_project_output` | `dsa_v41_cp.py:231-240` | pad 到 `per_rank` → `restore_tp_heads` → `_forward_o_proj` | 【可直接复用】 |

### 2.6 `ops/triton/dcp/dcp_a2a.py`（622 行）

| 函数 / kernel | 行号 | 用途 | 形状语义 | 底层算子 |
|---|---|---|---|---|
| `_pack_dcp_output_lse_kernel` | 19-110 | 把 out+lse 打包成 all2all 发送缓冲；`LSE_PACK_DIM=1`（fp32）或 4（bf16/fp16 用指数+3 位尾数整数编码，**注释在 72-110**） | `out[T,H,D]`、`lse[T,H,1]` → `send[dcp, S_local, R, D+p]` | Triton（AIV） |
| `_fused_dcp_lse_combine_kernel` | 113-243 | **融合合并**：先扫一遍求 `lse_max`（159-190），再 `weight=exp(lse-lse_max)`，`merged += out*weight`，`merged/=Σweight`（194-239）；`RETURN_LSE` 时输出 `lse_max+log(Σ)`（240-243） | recv 4D → `out[T,H,D]` | Triton |
| `_lse_pack_dim` | 246-251 | bf16/fp16→4，fp32→1；其它 dtype 报错 | — | — |
| `_validate_dcp_inputs` | 254-295 | 形状/dtype/设备校验；要求 scatter 维能被 dcp 整除 | — | — |
| `pack_dcp_output_lse` | 298-363 | 打包入口（A5 + bf16 + head_scatter + 足够行数时走 batched kernel） | — | Triton |
| `fused_dcp_lse_combine` | 366-472 | 合并入口（可带一个 local 贡献 `HAS_LOCAL`） | — | Triton |
| **`dcp_a2a_fused_combine`** | 475-513 | pack →（可选）`dist.all_to_all_single` →（可选）PCP all-gather → combine | — | HCCL + Triton |
| **`dcp_a2a_fused`** | 516-568 | 注册为 `torch.ops.vllm.dcp_a2a_fused` 的公开入口；`scatter_size=1` 时跳过 all2all；`defer_combine`/`return_lse` 互斥 | 参数 `(partial_output, softmax_lse, dcp_size, scatter_dim, group_name, pcp_group_name=None, return_lse=False, defer_combine=False)` | — |
| `dcp_a2a_fused_fake` / `direct_register_custom_op` | 571-622 | torch.compile 的 fake impl 与注册 | — | — |

### 2.7 `ops/triton/sparse_index_remap.py`

| 函数 | 行号 | 用途 |
|---|---|---|
| `remap_sparse_indices_fused_kernel` | 26-79 | 逐 chunk 做 owner 判定（`block_idx = idx // interleave_size`；`owner = block_idx % dcp_size`），重映射 `remapped = idx//dcp_size`（interleave=1）或带 interleave 的公式（66-67），有效项压到 chunk 前部，chunk 尾部写 `-1` |
| `remap_sparse_indices_triton` | 见文件后段 | Python 入口，由 `sfa_cp.py:1296-1303` 调用 |

### 2.8 上游 vLLM（0.29.0）里的对应实现——**同构、可交叉验证**

| 内容 | 位置 | 说明 |
|---|---|---|
| `triton_filter_and_convert_dcp_index` | `vllm/v1/attention/backends/mla/sparse_utils.py:302-380+` | 与 `_remap_sparse_indices` 同构：owner 过滤 + 重映射 + `compact_valid_to_front` + `return_valid_counts` |
| 调用点（FlashInfer MLA sparse） | `.../mla/flashinfer_mla_sparse.py:385-397` | `topk_indices_physical, seq_lens = triton_filter_and_convert_dcp_index(...)`，**用 valid count 直接改写 seq_lens** |
| 调用点（FlashMLA sparse） | `.../mla/flashmla_sparse.py:788-804` | 同上，但 `compact_valid_to_front=False`（保留中间 `-1`，由 kernel 原生 mask） |
| **上游对压缩 + DCP 的态度** | `.../mla/indexer.py:629-633` | `if self.dcp_world_size > 1 and self.compress_ratio > 1: raise NotImplementedError("DCP is not supported with sparse indexer KV compression (compress_ratio=...)")` ← **上游明确不支持「压缩 indexer + DCP」，与我们要做的事正撞** |
| 上游 interleave 限制 | `.../mla/indexer.py:540-548` | DCP 下 `cp_kv_cache_interleave_size > 1` 直接 `NotImplementedError`（注释说 gsm8k parity 失败） |
| FlashAttention MLA Sparse | `.../mla/flashattn_mla_sparse.py:99-100` | 「FlashAttention MLA Sparse does not support DCP for now」 |
| DCP 配置面 | `vllm/config/parallel.py:349-393, 540-575` | `decode_context_parallel_size`、`dcp_comm_backend∈{ag_rs,a2a}`、`dcp_q_replicate`、`cp_kv_cache_interleave_size` |
| KV cache 分片规则 | `vllm/v1/core/kv_cache_utils.py:651-697` | `resolve_dcp_kv_block_size`：`FullAttentionSpec` 类 → `block_size*dcp`；`dcp_world_size_for_kv_cache_spec`：只有 `FullAttentionSpec` 被分片，**SlidingWindow / ChunkedLocal / Mamba 保持复制态** |
| DCP 合并（通用实现） | `vllm/v1/attention/ops/dcp.py:37-63, 241-340, 704-777, 1100-1300` | `cp_lse_ag_out_rs` / `cp_lse_ag_out_ar` / `dcp_a2a_lse_reduce` / `mask_dcp_empty_shards_` / `DirectDCPKVGatherWorkspace`（对称内存直接 KV gather，非 a2a）+ `MLADCPManager` 选择器 |
| `supports_dcp_with_varlen` 的消费点 | `vllm/v1/attention/backend.py:623-653` | DCP>1 且不支持 varlen 时强制 `reorder_batch_threshold = 1` |

---

## 3. 交付 1（重点）：稀疏 top-k 在 DCP 下如何做到「全局选」

**答案：既不是运行时 gather 再选，也不是分布式 top-k，而是「复制态 indexer cache + 本地独立全局选 + 本地化过滤」。**

### 3.1 证据链（逐段）

1. **设计声明**（`sfa_cp.py:657-671`）：
   > `LightningIndexer cache is replicated on every DCP rank so index selection can run against the full sequence and keep the same sparse topk semantics as non-DCP SFA.`
   > `SFA KV cache remains DCP-local to preserve the KV memory saving. The sparse topk indices produced from the replicated indexer view are remapped to local KV indices before calling sparse flash attention.`

2. **地址层怎么做到复制**：builder 在构造 metadata 时，把 `common_attn_metadata` 的
   `slot_mapping / block_table_tensor` **临时换成复制态视图**（`sfa_cp.py:869-896`，`try/finally` 保证换回），
   于是 indexer 的 `write_cache` 写的是「全量逻辑 token 位置 → 复制态物理块」；
   转换公式在 `sfa_dcp_utils.py:57-129`（`replicated_block = local_phys_block*dcp + rank_in_view`）。
   indexer 侧另有一套**独立**的复制态 builder：`vllm_ascend/attention/indexer.py:520-611`（`use_dcp = enable_sfa_dcp_replicated_indexer(...)`）、
   `_build_dcp_cache_metadata`（751-788）、复用的地址工具（668-711）。

3. **谁来喂满复制态 cache**：
   - **纯 DCP（decode）**：decode token 在每个 DCP rank 上本来就是重复的（KV 才分片），所以每个 rank 在自己的复制态 cache 上都写入**全部** token，天然成为完整副本；`build_sfa_dcp_replicated_slot_mapping`（`sfa_dcp_utils.py:91-129`）用**全局 position** 算地址正是为此。
   - **DSA-CP（token 切分）叠加时**：每个 rank 只有一段 token，于是必须在**写 cache 前把 `k_li` 跨 TP(=DSA-CP 组) all-gather**：
     `indexer.py:352-395`（docstring 358-363 明确写「DSA-CP all-gathers the indexer k across the TP group」），
     实现是 `all_gather_async(k_li, get_tp_group(), async_op=True)`（386）+ scale（388-394）。
   - **PCP 叠加时**：prefill 段按 PCP gather 顺序写（`sfa_cp.py:1017-1046`）。

4. **选完之后的本地化**：`_remap_sparse_indices`（`sfa_cp.py:1283-1330`）：
   ```
   block_idx = idx // interleave_size
   owner     = block_idx - (block_idx // dcp_size) * dcp_size     # 该 token 归哪个 rank
   valid     = (idx >= 0) & (owner == dcp_rank)
   remapped  = idx // dcp_size                                    # interleave==1
   # interleave>1: (idx // (dcp*I))*I + (idx - block_idx*I)
   输出 = valid ? remapped : -1，再把有效项稳定压到行首（1326-1330）
   ```
   Triton 版：`ops/triton/sparse_index_remap.py:26-79`；无 Triton 的回退是 fp32 除法（1305-1330）。

5. **上游 vLLM 是同一套语义**：全局 top-k 由 indexer 在全序列上产生，再 `triton_filter_and_convert_dcp_index`
   过滤成本 rank 的物理 slot（`sparse_utils.py:302-325` 的 docstring 与 `compact_valid_to_front` 说明），
   且 `return_valid_counts=True` 时把每行有效数直接当作该 rank 的 `seq_lens`（`flashinfer_mla_sparse.py:386-397`）。

### 3.2 代价与含义（对 V4.1 的直接推论）

- 省的是 **KV cache 显存**，不省的是 **indexer K cache 显存**：indexer cache 必须每个 rank 一份全量。
  **这不是推断，有显式的规格与预算改动**：
  `vllm_ascend/core/kv_cache_interface.py:184-197` 的 `sfa_dcp_replicated_indexer_size` 直接乘进 `real_page_size_bytes`
  （`page_size_bytes` 也返回它），赋值点是 `worker/model_runner_v1.py:507-509`
  （`enable_sfa_dcp_replicated_indexer()` 为真时 `= self.dcp_size`，否则 1）与 `worker/v2/attn_utils.py:116-120`，
  消费点在 `core/kv_cache_placement.py:81-85`、`worker/model_runner_v1.py:5196-5205, 5525-5535, 6171`。
  ⇒ **每 rank 的 indexer cache 页被放大 dcp 倍，即等量于一份全量副本。**
  【推断】按 V4.1 的 `index_head_dim=128`、int8 存储（`models/deepseek_v41/indexer.py:73-87`）与 `tokens_per_state=compress_ratio`，
  1M token 时每 rank 每 index-source 层的 indexer cache ≈ 1M×(128+scale) B ≈ 130 MB 量级；
  由于 V4.1 是 index 共享（`DeepseekV41Topology.index_source_layer_ids`，`models/deepseek_v41/model.py:456-476`），层数应按 index source 层数计，而非全层数。
  **未逐层求和，请以实机显存预算打印为准（【未确认】）。**
- **top-k 一致性是正确性的硬前提**：所有 rank 必须选出**同一个** top-k 集合。
  一旦引入「本地选 + 不通信」或「按 rank 各自压缩」，LSE 合并出来的结果就不再等于全局 softmax。
- 因为每 rank 只保留 top-k 中属于自己的那部分（DCP8 下平均 ~1/8，即 `index_topk=512` → ~64 个），
  **稀疏注意力本身的算力不会变多**，但**每个 rank 仍要跑一次 LSE 合并的 all2all**。

---

## 4. 交付 2：LSE 合并的数学形式与适用范围

### 4.1 数学形式

参考实现（`common_cp.py:248-259`，注释即公式）：

```
LSE_final = logsumexp_i(LSE_i)
O_final   = Σ_i exp(LSE_i - LSE_final) · O_i
```

其中 `i` 遍历 DCP rank（或 attention 分块），`O_i` 是第 i 个分片的**已归一化**部分输出
（即 `Σ_j exp(qk_j - LSE_i) v_j`），`LSE_i = log Σ_j exp(qk_j)`。

工程实现有三条：

| 实现 | 位置 | 机制 | 数值稳定性 |
|---|---|---|---|
| NPU 融合算子 | `common_cp.py:153-162 → 194-221` | `[S, DCP*H, D+1]` 拆 out/lse → `torch_npu.npu_attention_update(lse_list, out_list, 0)` | 由算子内部处理 |
| Triton 融合 | `ops/triton/dcp/dcp_a2a.py:113-243` | 先求 `lse_max`（159-190），`w_i=exp(lse_i-lse_max)`，`O=Σ O_i w_i / Σ w_i`（194-239），`LSE=lse_max+log(Σw)`（240-243） | 显式减最大值 |
| torch 参考 | `common_cp.py:248-259` | `logsumexp` + `exp(lse-lse_final)` | 由 `logsumexp` 保证 |

### 4.2 是否只适用于 dense？——**不，对 sparse 同样成立**，但有 3 个前提

SFA-DCP 的 decode 路径就是稀疏版（`sfa_cp.py:1523-1573`），所以**「sparse + DCP + LSE 合并」已被上游验证可行**。前提：

1. **所有 rank 的 top-k 集合必须完全相同**（复制态 indexer 保证，见 §3）。
2. **每个 rank 必须把自己不拥有的 index 置无效（`-1`）**，让本地 softmax 只覆盖自己的子集
   （`_remap_sparse_indices` 的 `-1` + 压前；`sparse_mode=0` 关闭因果裁剪，`sfa_cp.py:1558-1562`）。
3. **本地 KV 长度与 block table 必须是 DCP 本地坐标**（`dcp_context.seq_lens` / `dcp_context.block_table`，
   `sfa_cp.py:1556-1557`），否则 kernel 会按全局坐标越界。

补充：每条被选中的 index 只有一个 owner（`owner = (idx // interleave) % dcp`），所以各 rank 分到的条数
天然不均（`index_topk=512`/DCP8 时平均 ~64 条，长序列下可能某 rank 0 条），**这不影响正确性**：
没有分到 index 的 rank 贡献空集，其 LSE 为 `-inf`，合并 kernel 里显式做了有效性判定
（`dcp_a2a.py:164, 174, 189-190, 202-215`：`valid_lse = (lse==lse) & (lse!=±inf)` 且
`merged += where(valid_lse, partial_output, 0.0) * weight`，避免 NaN 污染）。
上游还有 `mask_dcp_empty_shards_`（`vllm/v1/attention/ops/dcp.py:37-63`）专门把空分片行 LSE 置 `-inf`。

### 4.3 LSE 的底数问题（易踩坑）

vLLM 用 `lse_base_on_e` 区分自然对数与 log2（`vllm/v1/attention/backend.py:795-804`）。
V4.1 的 fused op 是**自然对数**：`arch22/sparse_flash_mla_csa_block_vector.h:408-410` 是 `Log(sum)` 后 `Add(max)`，
与 `dcp_a2a` 的 `exp/log` 一致【实测：kernel 源码】。

---

## 5. 交付 3：DCP 下的 KV 存取原语

### 5.1 KV cache 的物理分片是怎么发生的

| 层 | 位置 | 行为 |
|---|---|---|
| 规格层 | `vllm/v1/core/kv_cache_utils.py:651-697` | `resolve_dcp_kv_block_size`：AttentionSpec 类 → `block_size * dcp`（一个物理块跨 dcp 个逻辑块）；`dcp_world_size_for_kv_cache_spec`：**只有 `FullAttentionSpec` 分片**，SlidingWindow/ChunkedLocal/Mamba 保持复制态 |
| 显存层 | `vllm_ascend/core/kv_cache_interface.py:161-168` | `AscendMLAAttentionSpec.max_memory_usage_bytes`：`max_model_len → cdiv(max_model_len, dcp_world_size)`，即每 rank 只留 1/dcp |
| 调度层 | `vllm/v1/core/kv_cache_coordinator.py:145` | 每个 group 按其 spec 拿 `dcp_world_size_for_kv_cache_spec` 的结果 |
| 语义层 | `vllm/config/parallel.py:383-393` | `cp_kv_cache_interleave_size`：token 级（1）→ token i 落 rank `i%dcp`；块级（=block_size）→ 先填满前一个 rank 的同一 block |
| 限制 | `vllm_ascend/platform.py:1384-1391` | SFA DCP 要求 `cp_kv_cache_interleave_size == block_size`（不满足会警告并**强制改写**） |
| 校验 | `vllm_ascend/platform.py:1579-1599` | 复制的 indexer + DCP 只允许 `dcp_size == pcp_size` 或 `tp_size*pcp_size`；A5 上 SFA C8 + DCP 还不支持 |

**注意**：以上校验**全部以 `enable_sfa_dcp_replicated_indexer()` 为条件**（即 `model_uses_sfa_sparse` 为真）。
V4.1（`compress_ratios` 存在）走进这些分支的条件恒为假 ⇒ **开 DCP 不会有任何针对 DSA 的校验或报错**（【推断】，未实机验证失败模式）。

### 5.2 三种 KV 存取模式（SFA 实测）

| 模式 | 何时用 | 代码 | 通信 | 是否需要 LSE 合并 |
|---|---|---|---|---|
| **A. KV 全 gather 到本地** | prefill / mixed batch | `sfa_cp.py:1200-1238`（启动）+ `1489-1521`（消费） | 把 batch 内所有被引用块的 KV 一次性 all-gather 到每 rank | **否**（每 rank 看到全部 KV，直接算） |
| **B. Q gather + 分片算 + LSE 合并** | 纯 decode | `sfa_cp.py:1373-1414`（Q gather）+ `1523-1573`（本地算 + 合并）→ `dcp_a2a.py:475-568` | Q 全 gather（head 维）+ 输出/LSE all2all | **是** |
| **C. 对称内存直取（模式 A / Q gather 的上游优化实现）** | 上游 vLLM 的可选优化：KV gather 用于 prefill chunked context（`mla_attention.py:2944-2945` `dcp_manager.kv_gather(...)`），Q gather 用于 decode（`mla_attention.py:966-967`） | `vllm/v1/attention/ops/dcp.py:1104-1203`（`DirectDCPKVGatherWorkspace`）、`948-1058`（`DirectDCPQGatherWorkspace`）、选择器 `1220-1301` | 用 NVLS/对称内存直接 gather（`torch.ops._C.direct_dcp_kv_gather`），省掉「all2all + LSE 合并」两步；需要对称内存硬件，非 Ascend 路径 | KV gather 侧：否；Q gather 侧：否 |

### 5.3 通信量（bytes/token/step）

记号：`H`=rank 上参与合并的头数，`T`=本地 token 数，`D`=head（输出）维度，`b`=元素字节（bf16=2），`p`=LSE 打包维度（bf16 加密打包 `p=4`，fp32 `p=1`，见 `dcp_a2a.py:246-251`），`dcp`=DCP 度。

| 模式 | 每 rank 发送 | 每 rank 接收 | 全局合计 | 备注 |
|---|---|---|---|---|
| B（a2a + 融合合并） | `H·T·(D+p)·b` | 同左 | `dcp·H·T·(D+p)·b` | SFA decode 实际形态；`send` 形状 `[dcp, H/dcp, T, D+p]`（`dcp_a2a.py:309-313`） |
| B（`ag_rs` 老路径） | — | — | 与 a2a 同量级但**多一次 collective**（上游注释：a2a 把每层 3 次 collective 降到 2 次，`vllm/config/parallel.py:361-366`） |
| A（prefill KV gather） | `n_blocks · blk_size · D_kv · b`（其中 `D_kv = kv_lora_rank + qk_rope_head_dim`） | 同左 | `dcp ·` 同左 | 每 rank 得到**全量** KV |
| 复制态 indexer 写入（DSA-CP 时） | `T·128·b_idx`（k_li all-gather） | 同左 | `tp·T·128·b_idx` | `indexer.py:386-394`，每次前向一次 |

**V4.1 实参数代入（【推断】，dim 取自 `engram_ref/official/config.json` 与 `dsa_v41.py`）**：
`H=64`（`num_attention_heads`）、`D=512`（`head_dim`）、bf16、`p=4`（bf16 打包）、`dcp=8`：

- 模式 B，每 token 每层每 rank 发送 ≈ `64 × 1 × 516 × 2 B = 66 KB`；一个 decode step 的 batch `B` 个 token ⇒ `66 KB × B`；
  全局合计（乘 dcp）≈ `528 KB × B`（每层）。
- 若改成 fp32 打包合并（`LSE_PACK_DIM=1`），LSE 部分从 8 B 降到 4 B/token/head，但输出要 fp32 传输 ⇒ 总体更大；所以 bf16 场景默认用 4 维整数打包（`dcp_a2a.py:71-110` 注释说明是为了让 bf16 精确保存 FP32 LSE）。
- 模式 A，长度 `L` 的 prefill 请求：每层每 rank 约 `L × 576 × 2 B`（bf16，`kv_lora_rank=512` + `qk_rope=64`）≈ `1.15 KB × L`；
  `L=32K` 时约 **37 MB/层/rank** ⇒ 这是 prefill 阶段的主要开销，也是 SFA 只在非纯 decode 时才走 A 的原因【推断】。

**结论**：DCP 在 decode 上的通信是 **O(B·H·D·dcp)**（与序列长度无关），在 prefill 上是 **O(L·D_kv·dcp)**（与长度线性）。
把 DCP 用在 decode 阶段收益最直接。

---

## 6. 交付 4：DSA(V4.1) 与 SFA 在算子层面的真实差异

### 6.1 逐段对照

| 维度 | SFA（V3.2-Exp / GLM 类） | DSA（V4.1） | 行号 |
|---|---|---|---|
| Backend 入口 | `AscendSFABackend`，`get_impl_cls/get_builder_cls` 都走 `resolve_sfa_impl/resolve_sfa_metadata_builder` | `DeepseekV41CacheBackend`：`get_impl_cls` **硬返回**（1060-1061），`get_builder_cls` 走 `get_v41_cp_classes()`（1064-1067） | `sfa_v1.py:368-408`、`sfa_cp.py:1666-1699`、`dsa_v41.py:1052-1076` |
| impl 选择时机 | 后端类方法内解析 | **模型构造时**：`self.v41_impl = get_v41_cp_classes()[1](...)` | `models/deepseek_v41/model.py:738-742`、`dspark.py:111` |
| cache plane 数 | 1 个主 KV（nope+rope，可 C8 打包）+ indexer K/scale | **4 类**：`swa`(SlidingWindow MLA)、`long_kv`(MLA, `tokens_per_state=compress_ratio`)、`indexer.k_cache`(MLA+scale, `tokens_per_state=compress_ratio`)、`compressor.state_cache`(CircularBufferSpec) | `models/deepseek_v41/model.py:550-572, 702-715`、`indexer.py:73-87`、`compressor.py:35-45` |
| plane 元数据契约 | `AscendSFAMetadata` | `AscendDSAV41Metadata`(+`global_metadata`/`cp_token_range`)、`DeepseekV41LayerMetadata{attention,swa,compressor,indexer}` | `dsa_v41.py:88-142, 160-174` |
| top-k 语义 | indexer 在全序列 token 坐标选 top-k（`indexer.py:397-503`） | V4.1 indexer 在**压缩 token 坐标**选（`compress_ratio` 参与），候选还可来自共享 source 层 | `dsa_v41.py:438-470`、`dsa_cp.py:2111-2176`（`cmp_ratio`） |
| 稀疏注意力算子 | `npu_sparse_flash_attention`（`sfa_v1.py:231-248`）/ A5 的 `sparse_flash_mla`（`sfa_v1.py:208-229`） | `npu_sparse_flash_mla`，**同时传 ori_kv + cmp_kv + cmp_ratio + cmp_residual + sinks** | `dsa_v41.py:500-526` |
| LSE 输出 | 支持（`return_lse` 透传到 op，`device_op.py:471,517`） | **算子支持但被写死 False** | `dsa_v41.py:524` vs `aclnn_sparse_flash_mla.h:79-81` |
| O-proj | 支持全权重 gather / PCP 权重切分（`sfa_cp.py:117-163, 600-654`） | `_forward_o_proj`（复用 V1），CP 版在 `dsa_v41_cp.py:231-240` 用 `restore_tp_heads` | `dsa_v41.py:302-308` |

### 6.2 DSA 里**完全没有 DCP 处理**的部分（= 工作量清单）

| # | 缺失项 | 代码位置（V4.1 现状） | 为什么 DCP 下必改 |
|---|---|---|---|
| ① | **impl 选择** | `dsa_v41.py:1060-1061` 硬返回；`supports_dcp=False` 死代码（1079） | 没有 impl 分叉就没有 DCP 执行路径。模板：`dsa_v1.py:217-269`（V4 的 backend 两个方法都按开关分支） |
| ② | **metadata 的 DCP 视图** | `AscendDSAV41Metadata` 只有 `global_metadata`/`cp_token_range`（141-142），没有 `dcp_context` | 需要 SFA 那样的 replicated block table / local seq_lens / kv_gather metadata |
| ③ | **compress_ratios / CSA2 压缩槽** | `compressed_slot_mapping`（177-184）按 `slot//ratio` 计算；压缩组完成判定只在原坐标 | DCP 下「一个压缩组」可能横跨多个 rank；组内 token 不齐就不能产出压缩 token。需要按 `cp_kv_cache_interleave_size`/block 对齐或让压缩在 DCP 前完成 |
| ④ | **compressor state 环** | `_write_compressed_source` 的 ratio==2 分支（400-416）依赖 `state_metadata.c2_*`；ring 元数据在 `_c2_ring_metadata`（603）、`c2_complete_mask`（604）、ring owner 计算（933-1013） | 环是**每请求顺序状态**，DCP 分片后每个 rank 只看到部分 token；必须保证「本地 ring 能独立推进」或把 ring 也复制 |
| ⑤ | **SWA（ori_kv）** | `swa_cache` 用 `AscendSlidingWindowMLASpec`（`model.py:557`），写 cache 在 `scatter_cache_sk`（`dsa_v41.py:287`） | SlidingWindow spec 在 DCP 下是**复制态**（`kv_cache_utils.py:678-697`）→ 每 rank 都要写全量 token；但 Q 若被切分，写就不完整 |
| ⑥ | **ori/cmp 两套 mask 与 topk_length** | `ori_mask_mode/ori_win_left/ori_win_right`（129-131）、`sinks`、`ori_topk_length` | 本地化后必须重算长度（如 SFA 的 `sparse_mode=0` + compact 前缀），否则 kernel 用错因果坐标 |
| ⑦ | **top-k 本地化 remap** | `_select_sparse_indices`（438-470）产出的是全局索引，直接交给 op | 缺一个 DSA 版 `_remap_sparse_indices`：要同时处理 `compress_ratio`、`cmp_residual`、以及「压缩 index → owner rank」 |
| ⑧ | **indexer cache 的复制规格** | `indexer.k_cache` 用 `AscendMLAAttentionSpec`（`indexer.py:76`） | 该 spec 属于 `FullAttentionSpec` 分支 ⇒ 默认被 DCP 分片；要复制必须像 SFA 那样换成独立 spec（参考 `AscendSFAIndexerCacheSpec`，`core/kv_cache_interface.py:172-183`） |
| ⑨ | **dspark** | `build_dspark_swa_indices`（`dsa_v1.py:420-500`）在 `dsa_v41.py:835` 使用；Draft SWA 用 `AscendSlidingWindowMLASpec`（`dspark.py:84-94`） | DSpark 的 SWA index 是**位置坐标**，DCP 下需要同样的本地化；另有 `vllm_ascend/platform.py:1602-1620` 对「动态投机 + DCP」直接报错 |
| ⑩ | **校验** | `platform.py:1366-1399, 1579-1599` 全部以 SFA 判据为条件 | DSA 开 DCP 现在是「无校验」状态，需要补 fail-fast 或显式支持 |

---

## 7. 交付 5：可复用性判定（含改造点）

### 7.1 直接复用（不改签名）

| 资产 | 位置 | 说明 |
|---|---|---|
| LSE 合并融合算子 | `ops/triton/dcp/dcp_a2a.py:516-568`（`torch.ops.vllm.dcp_a2a_fused`） | 输入 `(out[T,H,D], lse[T,H,1] fp32)`，与 attention 是 dense/sparse 无关 |
| 稀疏 index 重映射 kernel | `ops/triton/sparse_index_remap.py:26-79` | owner 判定 + 重映射 + 紧凑化 |
| 本地 seq len 计算 | `vllm/v1/attention/backends/utils.py:1091-1130`；`sfa_cp.py:765-771` | 由 `dcp_size/dcp_rank/interleave` 算本地长度 |
| 复制态地址公式 | `sfa_dcp_utils.py:37-129` | 需要加 `compress_ratio` 参数（见下） |
| builder 视图替换范式 | `sfa_cp.py:869-936` | `try/finally` 替换 `common.slot_mapping/block_table_tensor` 再调原 builder |
| 组合范式 | `sfa_cp.py:1637-1663` | 「DSA-CP × DCP」就是用多继承把两套 collective 拼起来 |
| 解析范式 | `sfa_cp.py:1666-1699` | DSA-V4.1 应加同形态的 `resolve_dsa_v41_impl/builder` |
| NCCL/HCCL 原语 | `all_gather_async`（`vllm_ascend/distributed/utils.py`）、`dist.all_to_all_single`、`GroupCoordinator.all_gather` | — |
| NPU 融合 LSE 更新 | `torch_npu.npu_attention_update` | `common_cp.py:219, 237` |

### 7.2 需改造（列出函数签名改动）

| 资产 | 现状签名 | 需要的改造 | 优先级 |
|---|---|---|---|
| `_remap_sparse_indices` | `(self, topk_indices: Tensor) -> Tensor`（`sfa_cp.py:1283`） | 加 `cmp_ratio: int`、`cmp_residual: int`、`ori_topk_length/cmp_topk_length` 的重算；压缩 index → 物理块的映射要同时考虑 `tokens_per_state` 与 `cp_kv_cache_interleave_size`。建议新函数 `remap_csa_sparse_indices(topk, dcp_size, dcp_rank, interleave, cmp_ratio, residual)` | P0 |
| `build_sfa_dcp_replicated_block_table/slot_mapping` | `(dcp_block_table, seq_lens, out, idx, dcp_size, blocks_per_phys_block)`（`sfa_dcp_utils.py:57-129`） | 增加 `tokens_per_state/compress_ratio` 与「压缩组跨 rank」的对齐策略 | P0 |
| `AscendDSAV41Metadata` | 无 DCP 字段（`dsa_v41.py:88-142`） | 挂 `dcp_context: DCPContext`（照抄 `sfa_cp.py:276-299`），并区分 4 个 plane 的本地 block table | P0 |
| `DeepseekV41CacheBackend.get_impl_cls` | 硬编码（1060-1061） | 改为 `get_v41_dcp_classes()` 分支；`supports_dcp` 从死代码改为真实判据 | P0 |
| `AscendDSAV41CacheLayer.spec`(indexer) | `AscendMLAAttentionSpec`（`indexer.py:76`） | 需要「复制态」spec：抄 `AscendSFAIndexerCacheSpec.sfa_dcp_replicated_indexer_size` + `real_page_size_bytes` 的页放大机制（`core/kv_cache_interface.py:184-197`）与 runner 赋值（`worker/model_runner_v1.py:507-509`），否则 indexer cache 会被 DCP 当成 `FullAttentionSpec` 分片 | P0 |
| `_forward_attention` | `(attn, q, metadata, compressed_indices, *, source_cache=None)`（`dsa_v41.py:472`） | 需要 `return_softmax_lse=True` 分支 + 输出 `softmax_lse`；`ori/cmp` 的 `block_table/seq_lens` 换成 DCP 本地视图 | P0 |
| `compress_ratios` 路径 | `compressed_slot_mapping(slot_mapping, ratio)`（177-184） | 增加 DCP 感知：按 `cp_kv_cache_interleave_size` 对齐压缩组，或把压缩组所有权固定在某个 rank | P1 |
| compressor ring | `pool_projected(kv, scores, metadata)`（`compressor.py:53-65`） | 需要 DCP 版的 ring 元数据（跨 rank 组不齐时的兜底） | P1 |
| dspark SWA index | `build_dspark_swa_indices(...)`（`dsa_v1.py:420-500`） | 本地化重写 | P2 |

### 7.3 无参考、需自研

| 项 | 原因 |
|---|---|
| **「压缩域（CSA2）+ DCP」的 top-k 一致性与物理映射** | 上游 vLLM 直接 `NotImplementedError`（`.../mla/indexer.py:629-633`）；vllm-ascend 的 SFA-DCP 复制态假定的是**非压缩** token 级 indexer cache |
| **C2 ring compressor 在 DCP 下的推进** | 全树无先例；`dsa_cp.py` 只在 token 切分下用过它 |
| **DSA 的 prefill KV gather（模式 A）在共享层（kv_source/消费者）下的正确性** | SFA 的 gather 假设每层 KV 由本层写入；V4.1 的 long_kv 只有 kv_source 层写、其他层只读（`DeepseekV41Topology.kv_consumers`，`model.py:471-475`），gather 的发起者与消费者不同 |

---

## 8. 交付 6：cann-recipes-infer 与 cannbot-skills 侧的算子

### 8.1 cann-recipes-infer（`/home/chiro/tmp/cri` @ 92d9e1f）

| 模型 | 并行实现 | 结论 |
|---|---|---|
| `models/deepseek_v4_1/` | **明确不支持 CP**：`models/modeling_deepseek.py:1679-1680` `if parallel_config.cp_size > 1: raise ValueError(f"{parallel_config.cp_size=} is not supported yet!")`；同段还禁 `attn_tp_size > 1`（1676-1677）。整个文件里 `cp_size` 只用于 **prefill 的输入切分**（1335 `cp_size = self.cp_size if is_prefill else 1`、1479-1497 `get_cp_input_ids/get_cp_hidden_states`、2326-2334 `restore_prefill_cp_outputs`），decode 侧没有 CP | **【不可用】**：V4.1 配方没有可借鉴的 DCP decode 实现；但 `deepseek_v4_1/models/modules/{compressor,indexer}.py` 是理解 CSA2/Indexer 数值语义的好材料 |
| `models/deepseek_v3_2_exp/` | **prefill CP（序列切分）+ indexer KV all-gather + 全局 KV 复制**：`models/indexer.py:346-358`（int8 scale all-gather）、`412-422`（indexer KV all-gather 后再按 `cp_metadata.restore_indices` 还原全局顺序）、`models/modeling_deepseek.py:1320-1332`（prolog KV all-gather + restore）、`1334-1345`（**两套 PA cache**：`full_*_cache` 只给本层 prefill FA 用，`decode_*_cache` 保留给后续 decode/PD/offload） | **【需改造】**：这是「prefill 用全局 KV、decode 用本地 KV」的 recipe 版实现，与 vllm-ascend 的模式 A 同构，可以作为**没有 triton 环境时的参考实现**；但没有 decode 期的 LSE 合并与 top-k 本地化 |
| 文档 | `docs/models/deepseek_v4_1/deepseek_v4.1_flash_cann_tech_report.md` | 可用的算子名：`SparseFlashMla`（含 **CSA 压缩路径**，240-244 提到「SMLA/sparse_flash_mla：topk 固定」）、**`GetKVPhyAddr`**（109-145：把稀疏逻辑索引 `s2Idx` 批量换算成 int64 物理地址，`blkIdx=s2Idx/blockSize`、`phyBlk=block_table[blkIdx]`，VF 向量化 + UB gather）——这正是「稀疏 index → 本地物理地址」的 AscendC 参考；`GetRealS2Addr`、`mHC 折叠-扩展`、`ElasticBuffer`（Engram） |

> 对我们最有用的一条：`GetKVPhyAddr` 的公式与 `remap_sparse_indices` 不同（前者是「逻辑块→物理块」查表，后者是「全局 token index → DCP 本地 index」），
> 但两者都要**批量、向量化、int64 地址**，可作为把 remap 从 Triton 迁到 AscendC 时的参考。

### 8.2 cannbot-skills（`/home/chiro/projects/vllm/model-comparing/cannbot-skills/model/`）

| skill | 文件行号 | 结论 |
|---|---|---|
| `model-infer-parallel-analysis` | `SKILL.md:16-18` | **明确写明**：「这一 skill 输出可以包含 `cp_size`/`kvp_size` 候选，但 `model-infer-parallel-impl` **当前不直接支持这两个维度的代码实施**，需参照仓内已有模型手动改造（**CP 参考 `cann-recipes-infer/models/deepseek-v3.2-exp/`**，KVP 参考 `longcat-flash/`）」 |
| 同上 | `SKILL.md:73-74, 217-218, 434` | 只把 `cp_size` 当作并行度旋钮（「长序列 + MLA 叠加 CP」），给的是**策略建议**，不含算子/代码 |
| `model-infer-parallel-impl` | `SKILL.md` 与 `references/{framework_code_examples,framework_moe_parallel,standalone_parallel}.md` | 【不可用】：没有 DCP/LSE 合并/KV 分片的 kernel 清单；`grep -n "dcp\|DCP"` 零命中 |

**结论**：cannbot-skills 侧**没有可直接用的 DCP 算子**；唯一有效指引是把人指向 `deepseek-v3.2-exp` 的 prefill CP 实现（而该实现我们已在 §8.1 读过，且 vllm-ascend 的 `sfa_cp.py` 覆盖得更好）。

### 8.3 本地 vendored 算子包（意外收获）

`/home/chiro/projects/dsv41/graph_prep/src/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/` 里有 **SparseFlashMla 的完整 AscendC/TBE/aclnn 源码**，可直接查证算子能力：

| 资产 | 路径 | 关键事实 |
|---|---|---|
| aclnn 接口 | `op_api/include/aclnnop/aclnn_sparse_flash_mla.h:15-83` | 18 个可选输入 + `returnSoftmaxLse`(bool, 第 79 行) + `softmaxLseOutOptional`(81) |
| TBE 动态 shape | `op_impl/ai_core/tbe/custom_transformer_impl/dynamic/sparse_flash_mla.py:141,219-224,229` | `return_softmax_lse` 是 attrs；`softmax_lse_out_` 是 optional output |
| CSA kernel | `.../ascendc/sparse_flash_mla/arch22/sparse_flash_mla_csa_kernel.h:227,295-315,481,523,569` | 压缩路径读 `returnSoftmaxLse` 并按 TND 布局写 LSE（偏移公式在 298-299） |
| LSE 数值定义 | `.../arch22/sparse_flash_mla_csa_block_vector.h:378-416` | `Log(sum)` → `Add(max)` ⇒ **自然对数**，与 `dcp_a2a` 的 exp/log 一致 |
| metadata 算子 | `op_api/include/aclnnop/aclnn_sparse_flash_mla_metadata.h`、`dsa_v41.py:867`（`npu_sparse_flash_mla_metadata`） | V4.1 的 tiling/metadata 由独立算子产出（`smla_metadata`），DCP 改造时要注意 plan 与 indices 必须来自同一批张量（`sfa_v1.py:175-207` 有同类注释） |

---

## 9. 交付 7：风险与未知（我没验证的东西）

1. **【未确认】`npu_sparse_flash_mla` 的 torch 层签名是否已暴露 `softmax_lse` 输出。**
   我验证了 aclnn 接口（`aclnn_sparse_flash_mla.h:79-81`）、TBE op（`dynamic/sparse_flash_mla.py:229`）与 CSA kernel（`csa_kernel.h:301-303`）三层都支持；
   但 `dsa_v41.py:500-525` 的调用返回 `(output, _)` 且写死 `return_softmax_lse=False`，
   **没有实机验证** `torch.ops._C_ascend.npu_sparse_flash_mla(..., return_softmax_lse=True, softmax_lse=out)` 能否跑通、LSE 的 shape/stride 是否为 `[T, gSize]`（kernel 按 `[T, gSize]` 写，`dsa` 若需要 `[T,H,1]` 需 reshape）。
   建议（只读、无需起服务）：在 a3-21 上 `torch.ops._C_ascend.npu_sparse_flash_mla.default._schema` 打印签名确认。
2. **【未确认】压缩域 top-k 在 DCP 下的正确性**：压缩 token 与 DCP 分片的边界关系（一个压缩组是否可能跨 rank）我**只从代码推断**，没有实机数据；这决定工作量是「改地址公式」还是「改压缩调度」。
3. **【未确认】V4.1 开 DCP 现在的失败模式**：我推理为「无校验、静默算错」，没有实跑（`platform.py` 的 DCP 校验全部以 SFA 判据为条件）。
4. **【未确认】a3-21 上的实际 vllm 版本与本地 0.29.0 是否一致**：本地 vllm 是 `vllm_cpu-0.29.0`；a3-21 上看到的是若干独立源码树（`vllm-dsv41`、`vllm-0.26.0`），未确认运行时实际 import 的是哪份。
   若 a3-21 的 vllm 更旧，`sparse_utils.py:triton_filter_and_convert_dcp_index` 与 `.../mla/indexer.py:629-633` 的「DCP + 压缩 NotImplementedError」可能**不存在**，需要重新核对。
5. **【未确认】通信量公式里的 `H`**：DCP 与 TP 叠加时每 rank 实际参与合并的头数取决于 Q gather 的范围（`sfa_cp.py:1607-1612` 的 `tp_size` 计算），
   我按 TP1/DCP8 的 `H=64` 估算；TP8×DCP8 的实际形态需要按 `_start_dcp_query_gather` 与 `AscendSFAPCPDCPImpl._merge_dcp_outputs` 重新推。
6. **【未确认】性能收益**：本文只给通信量，没有实测 kernel 时间、HCCL 带宽占用与 overlap 效果。
7. **【未确认】`dsa_v41.py` 的 `compress_ratio` 具体分布**：`role.compress_ratio ∈ {0,1,2}`（`dsa_v41.py:390, 476`），
   我只确认了 `_write_compressed_source` 的 ratio==1 与 ratio==2 两条分支；`compress_ratios` 里是否还有其它取值（例如 4）以及它们走哪条路径，未逐层核对。
8. **【限制】本文所有 vllm-ascend 行号对应 `b64b4d7`**；`sfa_cp.py` 从 1315 行涨到 1699 行说明该文件仍在快速演进，落地前需重新对齐行号。

---

## 10. 行号速查（实施时可直接跳）

| 要做什么 | 去看 |
|---|---|
| DCP 总体设计基线 | `sfa_cp.py:657-671` |
| top-k 本地化 remap（数学） | `sfa_cp.py:1283-1330` |
| top-k 本地化 remap（kernel） | `ops/triton/sparse_index_remap.py:26-79` |
| LSE 合并数学 | `common_cp.py:248-259`、`ops/triton/dcp/dcp_a2a.py:113-243` |
| LSE 合并入口 | `ops/triton/dcp/dcp_a2a.py:516-568` |
| decode：Q gather | `sfa_cp.py:1373-1414` |
| prefill：KV gather | `sfa_cp.py:1200-1238`、`1489-1521` |
| DCP 执行分叉（含 sparse_mode 细节） | `sfa_cp.py:1475-1573` |
| 复制态地址公式 | `sfa_dcp_utils.py:37-129` |
| indexer cache「复制」= 页放大 dcp 倍 | `core/kv_cache_interface.py:184-197`、`worker/model_runner_v1.py:507-509`、`worker/v2/attn_utils.py:116-120` |
| builder 视图替换 | `sfa_cp.py:869-936` |
| 实现/构建器解析范式 | `sfa_cp.py:1666-1699`、`dsa_v1.py:217-269` |
| V4.1 现状（要改的地方） | `dsa_v41.py:1052-1093`、`dsa_v41.py:472-526`、`dsa_v41.py:378-436` |
| V4.1 DSA-CP 适配层（可抄骨架） | `context_parallel/dsa_v41_cp.py:28-240` |
| V4.1 cache plane 规格 | `models/deepseek_v41/model.py:550-572, 702-715`、`indexer.py:73-87`、`compressor.py:35-45` |
| DCP 分片规则（上游） | `vllm/v1/core/kv_cache_utils.py:651-697` |
| 上游「压缩 + DCP 不支持」 | `vllm/v1/attention/backends/mla/indexer.py:629-633` |
| 上游 sparse DCP 过滤 | `vllm/v1/attention/backends/mla/sparse_utils.py:302-380` |
| 融合算子 LSE 能力 | `_cann_ops_custom/.../aclnn_sparse_flash_mla.h:79-81`、`.../csa_kernel.h:301-303`、`.../csa_block_vector.h:378-416` |
| cri 的 V4.1「不支持 CP」 | `cri/models/deepseek_v4_1/models/modeling_deepseek.py:1679-1680` |
| cri 的 V3.2 prefill CP 参考 | `cri/models/deepseek_v3_2_exp/models/indexer.py:346-358, 412-422`；`models/modeling_deepseek.py:1320-1345` |
| cannbot 的 CP 结论 | `cannbot-skills/model/model-infer-parallel-analysis/SKILL.md:18` |
