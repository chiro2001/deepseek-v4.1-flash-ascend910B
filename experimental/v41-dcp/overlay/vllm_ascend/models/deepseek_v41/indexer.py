# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 index projections, quantized QLI and cross-layer candidate selection."""

import torch
import torch_npu
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheLayer,
    scatter_cache_sk,
)
from vllm_ascend.core.deepseek_v41 import DeepseekV41IndexerSpec
from vllm_ascend.patch.platform.patch_v41_dcp import v41_dcp_active as _v41_dcp_active
from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices
from vllm_ascend.ops.triton.quantize_indexer_query import quantize_indexer_query
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    wait_for_device_metadata,
)

from .compressor import DeepseekV41RMSNorm, _read


class DeepseekV41Indexer(nn.Module):
    """Small side attention that selects compressed KV positions.

    All index heads are replicated on each TP rank for the correctness path,
    so every rank produces identical sparse indices without an all-reduce.
    """

    def __init__(
        self,
        config,
        owns_k,
        vllm_config,
        prefix,
        compress_ratio,
        quant_config=None,
    ):
        super().__init__()
        self.owns_k = owns_k
        self.vllm_config = vllm_config
        self.compress_ratio = compress_ratio
        self.n_heads = int(_read(config, "index_n_heads"))
        self.width = int(_read(config, "index_head_dim"))
        self.rope_width = int(_read(config, "qk_rope_head_dim"))
        self.index_topk = int(_read(config, "index_topk"))
        self.softmax_scale = self.width**-0.5
        self.weights_scale = self.softmax_scale * self.n_heads**-0.5
        self.wq_b = ReplicatedLinear(
            _read(config, "q_lora_rank"),
            self.n_heads * self.width,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )
        self.weights_proj = ReplicatedLinear(
            _read(config, "hidden_size"),
            self.n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
            return_bias=False,
        )
        if owns_k:
            self.wk = nn.Linear(
                _read(config, "head_dim"),
                self.width,
                bias=False,
                dtype=torch.bfloat16,
            )
            self.k_norm = DeepseekV41RMSNorm(self.width, _read(config, "rms_norm_eps"))
            self.k_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.k_cache",
                DeepseekV41IndexerSpec(
                    block_size=vllm_config.cache_config.block_size,
                    num_kv_heads=1,
                    head_size=self.width,
                    dtype=torch.int8,
                    compress_ratio=compress_ratio,
                    scale_dim=1,
                    scale_dtype=torch.float16,
                    # [V41-DCP 2026-09-29] 复制态：每个 rank 留一份全量 indexer K，
                    # 这样 8 个 rank 各自独立算出**同一份全局 top-k**（零通信）。
                    # 见 DeepseekV41IndexerSpec.dcp_world_size 的注释。
                    dcp_world_size=(
                        int(getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1) or 1)
                        if _v41_dcp_active()
                        else 1
                    ),
                ),
            )

    @staticmethod
    def _output(linear, value):
        output = linear(value)
        return output[0] if isinstance(output, tuple) else output

    def update_keys(self, latent, slots, cos, sin):
        """Publish source-owned index K before latent is RoPE'd as long KV."""
        if not self.owns_k or latent.shape[0] == 0:
            return
        key = self.k_norm(self.wk(latent)).view(-1, 1, self.width)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            key.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.width - self.rope_width, self.width],
        )
        key = key.squeeze(1)
        quantized, scale = torch_npu.npu_dynamic_quant(
            key, dst_type=torch.int8
        )
        k_cache, scale_cache = self.k_cache.kv_cache[0]
        scatter_cache_sk(k_cache, slots, quantized)
        scatter_cache_sk(
            scale_cache,
            slots,
            scale.unsqueeze(-1).to(torch.float16),
        )

    def select(
        self,
        hidden_states,
        qr,
        positions,
        cos,
        sin,
        source_cache,
        source_metadata,
        *,
        is_candidate_source,
        uses_candidate_filter,
        candidate_topk_blocks,
        candidate_block_size,
        candidates,
    ):
        """Score index K, optionally filter blocks, then return position TopK."""
        query = self._output(self.wq_b, qr).unflatten(-1, (self.n_heads, self.width))
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            query.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.width - self.rope_width, self.width],
        )
        weights = self._output(self.weights_proj, hidden_states)
        weights = weights.float() * self.weights_scale

        return self.select_projected(
            query,
            weights,
            positions,
            source_cache,
            source_metadata,
            is_candidate_source=is_candidate_source,
            uses_candidate_filter=uses_candidate_filter,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            candidates=candidates,
        )

    def select_projected(
        self,
        query,
        weights,
        positions,
        source_cache,
        source_metadata,
        *,
        is_candidate_source,
        uses_candidate_filter,
        candidate_topk_blocks,
        candidate_block_size,
        candidates,
    ):
        """Run QLI V2 on paged INT8 K; candidates are block IDs, not positions.

        Source and consumer share [tokens, 1, candidate_topk_blocks] INT32
        block IDs only within this forward. Query quantization and position
        ordering stay outside the native QLI/candidate operator.
        """
        if is_candidate_source and uses_candidate_filter:
            raise ValueError("A candidate source must use the unfiltered position TopK")
        if uses_candidate_filter and candidates is None:
            raise RuntimeError("V4.1 candidate-filtering indexer ran before its source")
        if self.width != 128 or self.n_heads not in (32, 64):
            raise ValueError("A3 QLI requires index_head_dim=128 and 32 or 64 index heads")
        if not 1 <= self.index_topk <= 2048:
            raise ValueError("A3 QLI requires index_topk in [1, 2048]")
        if self.compress_ratio not in (1, 2):
            raise ValueError("Aurora QLI supports compression ratios 1 and 2")
        if is_candidate_source or uses_candidate_filter:
            if not 0 < candidate_topk_blocks <= 2048 or candidate_topk_blocks % 64:
                raise ValueError("candidate_topk_blocks must be a multiple of 64 in [64, 2048]")
            if candidate_block_size != 8:
                raise ValueError("The current A3 candidate kernel requires candidate_block_size=8")
        candidate_shape = (query.shape[0], 1, candidate_topk_blocks)
        if uses_candidate_filter and (candidates.shape != candidate_shape or candidates.dtype != torch.int32):
            raise ValueError("Candidate consumer requires INT32 block IDs with matching query rows")
        topk = self.index_topk
        if query.shape[0] == 0:
            selected = torch.full(
                (0, topk), -1, dtype=torch.int32, device=query.device
            )
            if is_candidate_source:
                candidates = torch.full(candidate_shape, -1, dtype=torch.int32, device=query.device)
            return selected, candidates

        quantized_query, query_scale = quantize_indexer_query(query)
        weights = weights.to(torch.float16)
        key, key_scale = source_cache
        key_scale = key_scale.squeeze(-1)  # Preserve the Hybrid cache page stride.
        cu_seqlens_q = source_metadata.query_start_loc
        seqused_k = source_metadata.cache_seq_lens
        residual = source_metadata.cmp_residual
        common = dict(
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            cmp_residual_k=residual,
            max_seqlen_q=source_metadata.max_query_len,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=self.compress_ratio,
        )
        op_metadata = source_metadata.qli_metadata
        if op_metadata is None:
            raise RuntimeError("V4.1 QLI metadata was not built")
        wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(op_metadata))
        mode = 1 if is_candidate_source else 2 if uses_candidate_filter else 3
        selected, _, candidate_out = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
            quantized_query,
            key,
            weights,
            query_scale,
            key_scale,
            topk,
            2,
            block_table=source_metadata.block_table,
            metadata=op_metadata,
            candidate_topk_index=candidates if uses_candidate_filter else None,
            candidate_mode=mode,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            **common,
        )
        selected = prepare_indexer_indices(selected.squeeze(1), positions, self.compress_ratio)
        selected = self._fix_visibility_for_sharded_k(selected, positions)
        return selected, candidate_out if is_candidate_source else candidates

    def _fix_visibility_for_sharded_k(self, selected, positions):
        """★★ [V41-DCP 2026-09-29] 把 top-k 的**因果可见性过滤**改到局部坐标系。

        `prepare_indexer_indices` 内部用 `visible = (positions+1)//ratio` 过滤，
        其中 `positions` 是**全局**位置。当 indexer K 缓存是**分片态**时
        `selected` 是**本 rank 局部**压缩索引，两者坐标系不一致 ⇒ rank>0 会把
        大量**未来键**判为可见（离线对拍：1803 次错判；rank=5、p=40 时让 query
        看到位置 160+ 的键）。这正是"L≤59 正常、L≥128 开始乱"的根因。

        这里用 `local_visible_positions` 把全局位置换成等价编码，再重跑一遍过滤
        （`prepare_indexer_indices` 本身是幂等的：它只做过滤+排序）。
        """
        from vllm_ascend.patch.platform.patch_v41_dcp import replicate_indexer, v41_dcp_active

        if not v41_dcp_active() or replicate_indexer():
            return selected
        parallel = self.vllm_config.parallel_config
        dcp_size = int(getattr(parallel, "decode_context_parallel_size", 1) or 1)
        if dcp_size <= 1:
            return selected
        from vllm.distributed import get_dcp_group

        from vllm_ascend.attention.context_parallel.v41_dcp import (
            local_visible_positions,
        )

        shifted = local_visible_positions(
            positions,
            interleave=int(getattr(parallel, "cp_kv_cache_interleave_size", 1) or 1),
            ratio=self.compress_ratio,
            dcp_size=dcp_size,
            dcp_rank=int(get_dcp_group().rank_in_group),
        )
        return prepare_indexer_indices(selected, shifted, self.compress_ratio)
