# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 DSA metadata and fused attention execution.

The model file owns the network topology and projection modules.  This module
owns the attention execution boundary: it gathers every cache plane's metadata
before running the compressor, indexer and sparse-attention operators without
moving cache or scheduler knowledge back into the model.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
)

from vllm_ascend.attention.dsa_v1 import dsv4_dsa_overlap_stream
from vllm_ascend.core.deepseek_v41 import (
    DeepseekV41CompressorStateSpec,
    DeepseekV41FullSpec,
    DeepseekV41IndexerSpec,
    DeepseekV41SWASpec,
)
from vllm_ascend.ops.rope_dsv4 import (
    get_cos_and_sin_dsa,
    get_full_cos_and_sin_dsa_for_layer,
)
from vllm_ascend.utils import npu_stream_switch
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    DeviceMetadataTask,
    wait_for_device_metadata,
)

V41_METADATA_BUFFER_SIZE = 1024


def _v41_dcp_on() -> bool:
    """DCP 是否开启（开发期门控，生产环境一律 False）。"""
    from vllm_ascend.patch.platform.patch_v41_dcp import v41_dcp_active

    return v41_dcp_active()


def _v41_dcp_rank() -> int:
    """本进程在 DCP 组内的 rank（DCP 关闭时返回 0）。"""
    if not _v41_dcp_on():
        return 0
    from vllm.distributed import get_dcp_group

    return int(get_dcp_group().rank_in_group)


def _v41_dcp_group():
    from vllm.distributed import get_dcp_group

    return get_dcp_group()


def _v41_dcp_ori_owner() -> str:
    """ori（滑窗）平面由谁携带。

    合并式 `O = Σ_r w_r·O_r / Σ_r w_r` 要求共享的 ori 窗口只被计一次，
    实现方式有两类，本函数选择用哪一种（可用 env 切换以便真机二分）：

    * `seqused0`（**默认**）：所有 rank 都挂 `ori_kv`/`ori_block_table`（因为算子在
      op 层把 `ori_kv` 当**硬必填**，传 None 直接
      `tensor of oriKv is nullptr[CheckRequiredInOutExistence]`），
      但 rank>0 把 `seqused_ori_kv` 设 0 ⇒ 该 rank 的 ori 贡献为空。
    * `rank0`：只有 rank 0 传 `ori_kv`，其余传 `None`。**实测在 A3 上不可用**
      （见上），保留仅为在别的硬件上复现。
    """
    import os as _os

    return _os.environ.get("V41_DCP_ORI_OWNER", "seqused0")


def _v41_dcp_merge_attention(
    output: torch.Tensor,
    lse: torch.Tensor,
) -> torch.Tensor:
    """把各 rank 的局部 partial attention 用 LSE 加权合并成全局结果。

    数学：全局 `softmax(A ∪ B_0 ∪ … ∪ B_{d-1})`，其中 `A` 只由 rank 0 携带、
    `B_r` 是 rank r 的私有分片。各 rank 上报 `(O_r, L_r)`，
    则 `O = Σ_r e^{L_r}·O_r / Σ_r e^{L_r}` **精确**等于全局结果
    （`Σ_r e^{L_r}` 里 A 恰好出现一次）。分段互不重叠地覆盖全局集时该式恒等，
    已被离线夹具用真算子验证（`relout=0.0`）。

    实现用 `all_gather(LSE)` + 两次 `all_reduce`（输出与权重），**不做 head 维
    all_gather / all_to_all**：因为 TP 已经按 head 切分，每个 rank 的 q 就是自己
    那 8 个 head，各 rank 合并的是**同一批 head 的不同 KV 分片** ⇒ 只需在 rank 维
    做归约，不需要重排 head。这比 SFA 的 all-to-all 方案少一次大通信。
    """
    group = _v41_dcp_group()
    dcp_size = group.world_size
    if dcp_size <= 1:
        return output
    # LSE 需要 float32 且布局一致；算子返回 TND `(N2,T1,G)` ⇒ 转成 `[T, H, 1]`。
    lse = lse.to(torch.float32)
    if lse.ndim == 3 and lse.shape[0] == 1:
        lse = lse.permute(1, 2, 0).contiguous()
    elif lse.ndim == 2:
        lse = lse.unsqueeze(-1).contiguous()
    else:
        lse = lse.reshape(lse.shape[1], -1, 1).contiguous()

    gathered = torch.empty(
        (dcp_size, *lse.shape), dtype=torch.float32, device=lse.device
    )
    torch.distributed.all_gather_into_tensor(gathered, lse, group=group.device_group)
    lse_max = gathered.amax(dim=0)
    finite = torch.isfinite(lse) & torch.isfinite(lse_max)
    # 全 rank 都无有效键（例如全 -inf）时退化为 0，避免 NaN 污染后续层。
    weights = torch.where(finite, torch.exp((lse - lse_max).clamp(min=-80.0)), torch.zeros_like(lse))
    scaled = output.to(torch.float32) * weights
    torch.distributed.all_reduce(scaled, group=group.device_group)
    wsum = weights.clone()
    torch.distributed.all_reduce(wsum, group=group.device_group)
    denom = torch.where(wsum > 0, wsum, torch.ones_like(wsum))
    return (scaled / denom).to(output.dtype)


@eager_break_during_capture
def dsa_v41_forward(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Execute V4.1 attention behind an explicit graph side-effect boundary."""
    forward_context = get_forward_context()
    attn = forward_context.no_compile_layers[layer_name]
    attn.v41_impl.forward(attn, None, hidden_states, output)


def dsa_v41_forward_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="dsa_v41_forward",
    op_func=dsa_v41_forward,
    mutates_args=["output"],
    fake_impl=dsa_v41_forward_fake,
    dispatch_key="PrivateUse1",
)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read one field from either an HF config object or a raw config dict."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass
class DeepseekV41Metadata(AttentionMetadata):
    """Scheduler and cache-plane contract for one V4.1 cache resource.

    ``seq_lens``/``query_start_loc`` always stay in original-token
    coordinates, matching the common vLLM metadata. The ``cache_*`` fields
    describe the rows visible to the concrete cache plane. Keeping both
    coordinate systems here lets future fused kernels replace the eager path
    without rebuilding scheduling metadata in the model.
    """

    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    compress_ratio: int
    storage_block_size: int
    is_compressor_state: bool
    cache_kind: str = "unknown"
    positions: torch.Tensor | None = None
    cos: Any = None
    sin: Any = None
    num_actual_tokens: int = 0
    num_input_tokens: int = 0
    num_reqs: int = 0
    num_actual_reqs: int = 0
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    logical_block_size: int = 0
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    cache_seq_lens: torch.Tensor | None = None
    max_query_len: int = 0
    max_seq_len: int = 0
    max_cache_seq_len: int = 0
    attn_state: Any = None
    is_prefilling: torch.Tensor | None = None
    causal: bool | torch.Tensor = True
    ori_win_left: int = 0
    ori_win_right: int = 0
    smla_metadata: torch.Tensor | None = None
    qli_metadata: torch.Tensor | None = None
    cmp_residual: torch.Tensor | None = None
    c2_ring_metadata: torch.Tensor | None = None
    c2_complete_mask: torch.Tensor | None = None
    c2_source_positions: torch.Tensor | None = None
    c2_source_cos: torch.Tensor | None = None
    c2_source_sin: torch.Tensor | None = None
    c2_metadata_group_id: int | None = None


@dataclass(frozen=True)
class DeepseekV41CompressorMetadata:
    """V4-shaped cache/state bundle consumed by the compressor stage."""

    cache: DeepseekV41Metadata
    state: DeepseekV41Metadata | None = None


@dataclass(frozen=True)
class DeepseekV41IndexerMetadata:
    """V4-shaped source cache bundle consumed by the indexer stage."""

    cache: DeepseekV41Metadata


@dataclass(frozen=True)
class DeepseekV41LayerMetadata:
    """All metadata consumed by one V4.1 attention layer invocation."""

    attention: DeepseekV41Metadata | None
    swa: DeepseekV41Metadata
    compressor: DeepseekV41CompressorMetadata | None
    indexer: DeepseekV41IndexerMetadata | None

    @property
    def positions(self) -> torch.Tensor:
        if self.swa.positions is None:
            raise RuntimeError("V4.1 SWA metadata does not contain input positions")
        return self.swa.positions

    def rope(self, layer_name: str, num_tokens: int):
        if self.swa.cos is None or self.swa.sin is None:
            raise RuntimeError("V4.1 SWA metadata does not contain RoPE tensors")
        return self.swa.cos[layer_name][:num_tokens], self.swa.sin[layer_name][:num_tokens]


def compressed_slot_mapping(slot_mapping: torch.Tensor, ratio: int) -> torch.Tensor:
    """Convert original-token physical slots to completed compressed slots.

    Logical block sizes must be divisible by ratio. Negative/padded slots and
    incomplete compression groups never produce a write.
    """
    if ratio not in (1, 2):
        raise ValueError("V4.1 only supports ratio 1 or 2")
    valid = (slot_mapping >= 0) & ((slot_mapping + 1) % ratio == 0)
    return torch.where(valid, slot_mapping // ratio, -1)


def _request_counts(common: Any, num_reqs: int):
    """Return V4-shaped request counters without synchronizing the NPU."""
    is_prefilling = getattr(common, "is_prefilling", None)
    query_start_loc_cpu = getattr(common, "query_start_loc_cpu", None)
    if (
        is_prefilling is None
        or query_start_loc_cpu is None
        or getattr(is_prefilling, "device", None) is None
        or is_prefilling.device.type != "cpu"
    ):
        return 0, 0, 0, 0
    flags = is_prefilling[:num_reqs].bool()
    query_lens_cpu = query_start_loc_cpu[1 : num_reqs + 1] - query_start_loc_cpu[:num_reqs]
    num_prefills = int(flags.sum().item())
    num_decodes = num_reqs - num_prefills
    num_prefill_tokens = int(query_lens_cpu[flags].sum().item())
    num_decode_tokens = int(query_lens_cpu[~flags].sum().item())
    return num_decodes, num_decode_tokens, num_prefills, num_prefill_tokens


def scatter_cache_sk(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Store rows using builder-prepared coordinates and V4's Ascend op.

    V4.1 cache planes can be views into a larger layer-outermost slot, so the
    physical page stride is not necessarily the contiguous stride implied by
    the plane shape. ``npu_scatter_nd_update_sk`` preserves that stride and
    treats the builder's ``[-1, -1]`` coordinates as skipped rows, matching V4.
    """
    if slot_mapping.ndim != 2 or slot_mapping.shape[-1] != 2:
        raise ValueError(
            f"V4.1 fused cache store requires builder-prepared [T, 2] slot_mapping, got {tuple(slot_mapping.shape)}"
        )
    cache = cache.squeeze(-2)
    indices = slot_mapping[: values.shape[0]]
    updates = values.to(cache.dtype).contiguous()
    torch.ops._C_ascend.npu_scatter_nd_update_sk(cache, indices, updates)


def pad_sparse_indices(indices: torch.Tensor, topk: int) -> torch.Tensor:
    """Convert V4.1's compact [T, K] selection into SMLA [T, 1, topk]."""
    if indices.ndim != 2:
        raise ValueError(f"V4.1 sparse indices must be rank 2, got {indices.shape}")
    if indices.shape[-1] > topk:
        raise ValueError(f"V4.1 sparse indices width {indices.shape[-1]} exceeds operator topk {topk}")
    if indices.shape[-1] < topk:
        indices = F.pad(indices, (0, topk - indices.shape[-1]), value=-1)
    return indices.unsqueeze(1).contiguous().int()


class DeepseekV41EagerAttentionImpl:
    """V4-shaped execution boundary backed by fused Ascend operators.

    Projection, compressor and indexer modules remain registered by the model,
    while this object resolves the complete per-layer metadata bundle and owns
    their invocation order.  That is the same separation used by ``dsa_v1``:
    model construction is independent from cache-aware attention execution.
    """

    def __init__(self, prefix, role, topology, long_kv_source_prefix, index_k_source_prefix):
        self.prefix = prefix
        self.layer_name = f"{prefix}.attn"
        self.role = role
        self.topology = topology
        self.swa_prefix = f"{prefix}.swa_cache"
        self.long_kv_source_prefix = long_kv_source_prefix
        self.index_k_source_prefix = index_k_source_prefix
        self.compressor_state_prefix = (
            f"{prefix}.compressor.state_cache" if role.is_kv_source and role.compress_ratio == 2 else None
        )

    def _get_layer_metadata(self, metadata) -> DeepseekV41LayerMetadata:
        try:
            swa = metadata[self.swa_prefix]
            long_kv = metadata[self.long_kv_source_prefix] if self.long_kv_source_prefix is not None else None
            index_k = metadata[self.index_k_source_prefix] if self.index_k_source_prefix is not None else None
            compressor_state = (
                metadata[self.compressor_state_prefix] if self.compressor_state_prefix is not None else None
            )
        except KeyError as exc:
            raise RuntimeError(f"Missing V4.1 cache metadata for {exc.args[0]}") from exc
        return DeepseekV41LayerMetadata(
            attention=long_kv,
            swa=swa,
            compressor=(
                DeepseekV41CompressorMetadata(long_kv, compressor_state)
                if self.role.is_kv_source and long_kv is not None
                else None
            ),
            indexer=(DeepseekV41IndexerMetadata(index_k) if index_k is not None else None),
        )

    @staticmethod
    def _project_q_kv(attn, hidden_states, cos, sin):
        q_a = attn.wq_a(hidden_states)
        qr = attn.q_norm(q_a)
        q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
        kv = attn.kv_norm(attn.wkv(hidden_states))
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        kv = kv.view(-1, 1, attn.head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        return q.to(hidden_states.dtype), qr, kv.squeeze(1)

    def preprocess(self, attn, hidden_states, cos, sin, swa_metadata):
        """Project Q/KV and populate this layer's SWA cache on the current stream."""
        q, qr, kv = self._project_q_kv(attn, hidden_states, cos, sin)
        scatter_cache_sk(
            attn.dsa_attn.swa_cache_layer.kv_cache[0],
            swa_metadata.slot_mapping,
            kv,
        )
        return q, qr

    def multistream_preprocess(self, attn, hidden_states, cos, sin, swa_metadata):
        """Overlap Q Vector work with KV Cube work, then reverse their roles.

        Reuse V1's stream and projection wrappers. V4.1 keeps floating-point
        qr for its indexer and has no post-Wq_b Q RMSNorm. Stage events serialize
        the Cube matmuls; the final join makes SWA writes visible to attention.
        """
        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()
        v1_impl = attn.dsa_attn.dsa_attn.impl
        wq_a, wkv, wq_b = v1_impl.cv_wq_a, v1_impl.cv_wkv, v1_impl.cv_wq_b
        share_quant = (
            type(wq_a._quant_method) is type(wkv._quant_method) and wq_a._has_communication == wkv._has_communication
        )

        # Part 1: Q_a matmul (Cube) overlaps independent KV quantization (Vector).
        q_quant, q_scale = wq_a.quantize(hidden_states)
        kv_quant_done = None
        if share_quant:
            kv_quant, kv_scale = q_quant, q_scale
        else:
            q_quant_done = main_stream.record_event()
            with npu_stream_switch(aux_stream, enabled=True):
                aux_stream.wait_event(q_quant_done)
                kv_quant, kv_scale = wkv.quantize(hidden_states)
                kv_quant_done = aux_stream.record_event()
        q_a = wq_a.matmul(q_quant, q_scale, bias=attn.wq_a.bias)

        # Part 2: Q normalization/quantization (Vector) overlaps KV matmul (Cube).
        part2_start = main_stream.record_event()
        if kv_quant_done is not None:
            main_stream.wait_event(kv_quant_done)
        with npu_stream_switch(aux_stream, enabled=True):
            aux_stream.wait_event(part2_start)
            kv = wkv.matmul(kv_quant, kv_scale, bias=attn.wkv.bias)
            kv_matmul_done = aux_stream.record_event()
        qr = attn.q_norm(q_a)
        q_b_quant, q_b_scale = wq_b.quantize(qr)

        # Part 3: Q_b matmul (Cube) overlaps KV norm, RoPE and cache store (Vector).
        part3_start = main_stream.record_event()
        main_stream.wait_event(kv_matmul_done)
        with npu_stream_switch(aux_stream, enabled=True):
            aux_stream.wait_event(part3_start)
            kv = attn.kv_norm(kv).view(-1, 1, attn.head_dim)
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[attn.nope_head_dim, attn.head_dim],
            )
            scatter_cache_sk(
                attn.dsa_attn.swa_cache_layer.kv_cache[0],
                swa_metadata.slot_mapping,
                kv.squeeze(1),
            )
        q = wq_b.matmul(q_b_quant, q_b_scale, bias=attn.wq_b.bias).unflatten(-1, (attn.n_local_heads, attn.head_dim))
        main_stream.wait_stream(aux_stream)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        return q.to(hidden_states.dtype), qr

    def _write_compressed_source(
        self,
        attn,
        hidden_states,
        positions,
        cos,
        sin,
        metadata,
    ):
        compressor = attn.compressor
        if compressor is None or metadata.compressor is None or metadata.indexer is None:
            raise RuntimeError("V4.1 KV source is missing compressor or source metadata")
        compressor_metadata = metadata.compressor
        indexer_metadata = metadata.indexer
        ratio = self.role.compress_ratio
        if ratio == 1:
            latent = compressor(hidden_states)
            # C1 source positions are the current token positions. Reuse the
            # query RoPE selected by the SWA metadata builder instead of
            # indexing the global table a second time.
            source_cos = cos
            source_sin = sin
            index_slots = indexer_metadata.cache.slot_mapping[: positions.shape[0]]
            long_slots = compressor_metadata.cache.slot_mapping[: positions.shape[0]]
        else:
            if compressor_metadata.state is None:
                raise RuntimeError("V4.1 ratio-2 source is missing compressor-state metadata")
            state_metadata = compressor_metadata.state
            if state_metadata.c2_ring_metadata is None or state_metadata.c2_metadata_group_id is None:
                raise RuntimeError("V4.1 ring compressor metadata is missing")
            wait_for_device_metadata(DeviceMetadataStage.COMPRESSOR, state_metadata.c2_metadata_group_id)
            hidden_states_fp32 = hidden_states.float()
            kv = compressor.wkv(hidden_states_fp32)
            score = compressor.wgate(hidden_states_fp32)
            latent = compressor.pool_projected(kv, score, state_metadata)
            source_cos = state_metadata.c2_source_cos
            source_sin = state_metadata.c2_source_sin
            if source_cos is None or source_sin is None:
                fallback_cos, fallback_sin = get_cos_and_sin_dsa(state_metadata.c2_source_positions)
                source_cos = fallback_cos[attn.rotary_emb.layername]
                source_sin = fallback_sin[attn.rotary_emb.layername]
            source_cos = source_cos[: positions.shape[0]]
            source_sin = source_sin[: positions.shape[0]]
            index_slots = indexer_metadata.cache.slot_mapping[: positions.shape[0]]
            long_slots = compressor_metadata.cache.slot_mapping[: positions.shape[0]]

        if attn.indexer is None:
            raise RuntimeError("V4.1 KV source is missing its indexer")
        attn.indexer.update_keys(
            latent,
            index_slots,
            source_cos,
            source_sin,
        )
        latent = latent.view(-1, 1, attn.head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            latent.unsqueeze(1),
            source_cos,
            source_sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        scatter_cache_sk(
            attn.long_kv_cache.kv_cache[0],
            long_slots,
            latent.squeeze(1),
        )

    def _select_sparse_indices(self, attn, hidden_states, qr, positions, cos, sin, metadata):
        if not self.role.has_long_context:
            return None
        shared = attn.shared_state
        if shared is None:
            raise RuntimeError("V4.1 shared attention state is not initialized")
        if not self.role.is_index_source:
            selected = shared.topk_indices[: hidden_states.shape[0]]
            return self._remap_selection(selected, positions)
        if attn.indexer is None or metadata.indexer is None:
            raise RuntimeError("V4.1 index source is missing indexer metadata")

        context = get_forward_context().no_compile_layers
        source_layer = context[self.index_k_source_prefix]
        selected, candidates = attn.indexer.select(
            hidden_states,
            qr,
            positions,
            cos,
            sin,
            source_layer.kv_cache[0],
            metadata.indexer.cache,
            is_candidate_source=self.role.is_candidate_source,
            uses_candidate_filter=self.role.uses_candidate_filter,
            candidate_topk_blocks=self.topology.candidate_topk_blocks,
            candidate_block_size=self.topology.candidate_block_size,
            candidates=shared.candidates[: hidden_states.shape[0]],
        )
        # [V41-DCP] `indexer.select` 的输出是**全局压缩 token 坐标**（因为 indexer K
        # cache 是每个 rank 一份全量复制，所以 8 个 rank 选出的集合逐位相同）。
        # 而本 rank 的 long_kv 只有 1/dcp，所以必须把全局坐标重映射到本 rank 的
        # 本地压缩坐标，并把不属于本 rank 的项置 -1（算子对负索引直接跳过）。
        selected = self._remap_selection(selected, positions)
        shared.topk_indices[: selected.shape[0]].copy_(selected)
        if self.role.is_candidate_source:
            shared.candidates[: candidates.shape[0]].copy_(candidates)
        return shared.topk_indices[: selected.shape[0]]

    def _remap_selection(self, selected, positions):
        """把全局压缩 top-k 索引重映射到本 rank 的本地压缩坐标。"""
        import os as _os

        from vllm_ascend.patch.platform.patch_v41_dcp import replicate_indexer, v41_dcp_active

        if not v41_dcp_active() or not replicate_indexer():
            return selected
        from vllm.distributed import get_dcp_group

        from vllm_ascend.attention.context_parallel.v41_dcp import remap_sparse_indices

        parallel = get_forward_context().vllm_config.parallel_config
        dcp_size = int(getattr(parallel, "decode_context_parallel_size", 1) or 1)
        if dcp_size <= 1:
            return selected
        interleave = int(getattr(parallel, "cp_kv_cache_interleave_size", 1) or 1)
        return remap_sparse_indices(
            selected,
            block_size=int(get_forward_context().vllm_config.cache_config.block_size),
            interleave=interleave,
            ratio=int(self.role.compress_ratio),
            dcp_size=dcp_size,
            dcp_rank=int(get_dcp_group().rank_in_group),
        )

    def _attention(self, attn, q, metadata, compressed_indices):
        source_cache = None
        if self.role.has_long_context:
            source_cache = get_forward_context().no_compile_layers[self.long_kv_source_prefix].kv_cache[0]
        return self._native_attention(
            attn,
            q,
            metadata,
            source_cache=source_cache,
            compressed_indices=compressed_indices,
        )

    def _native_attention(
        self,
        attn,
        q,
        metadata,
        *,
        source_cache,
        compressed_indices,
    ):
        """Run SparseFlashMla with the same PA metadata for both operator stages."""
        if attn.head_dim != 512:
            raise ValueError(f"SparseFlashMla requires head_dim 512, got {attn.head_dim}")
        if attn.window_size != 128:
            raise ValueError(f"A2/A3 SparseFlashMla requires sliding_window 128, got {attn.window_size}")
        if not 1 <= attn.n_local_heads <= 128 or attn.n_local_heads & (attn.n_local_heads - 1):
            raise ValueError(
                "A2/A3 SparseFlashMla requires the local query-head count to be "
                f"a power of two in [1, 128], got {attn.n_local_heads}"
            )
        has_compressed = self.role.compress_ratio in (1, 2)
        ratio = self.role.compress_ratio if has_compressed else 0
        num_reqs = metadata.swa.num_reqs
        query_start_loc = metadata.swa.query_start_loc[: num_reqs + 1]
        seq_lens = metadata.swa.seq_lens[:num_reqs]
        ori_block_table = metadata.swa.block_table[:num_reqs]
        cmp_block_table = None
        cmp_seq_lens = None
        cmp_residual = None
        cmp_indices = None
        cmp_topk = 0
        if has_compressed:
            if source_cache is None or metadata.attention is None or compressed_indices is None:
                raise RuntimeError("V4.1 compressed attention is missing KV or TopK metadata")
            cmp_block_table = metadata.attention.block_table[:num_reqs]
            cmp_seq_lens = metadata.attention.cache_seq_lens[:num_reqs]
            cmp_residual = metadata.attention.cmp_residual
            cmp_topk = self.topology.index_topk
            if cmp_topk not in (512, 1024):
                raise ValueError(f"SparseFlashMla only supports TopK 512 or 1024, got {cmp_topk}")
            cmp_indices = pad_sparse_indices(compressed_indices, cmp_topk)

        operator_metadata = metadata.attention if has_compressed else metadata.swa
        op_metadata = operator_metadata.smla_metadata
        if op_metadata is None:
            raise RuntimeError(f"V4.1 ratio-{ratio} SMLA metadata was not built")
        wait_for_device_metadata(
            DeviceMetadataStage.ATTENTION,
            id(op_metadata),
        )
        # =====================================================================
        # [V41-DCP 2026-09-29] 合并路径的开关与参数。
        #
        # 只在「有压缩（ratio∈{1,2}）」的层上走跨 rank 合并：
        #   · ratio=0 的层只有 ori，滑窗平面每 rank 全量复制 ⇒ 本地算出的就是
        #     完整正确的滑窗注意力，**不能**再去合并（否则滑窗被计 dcp 次）。
        #   · ratio>0 的层：ori 由 rank 0 独占（见 metadata 的 `has_ori_kv`），
        #     各 rank 只带自己的 cmp 分片，合并后 A 恰好计一次。
        #
        # sink 同理必须**只计一次**：只在 rank 0 传真值，其余 rank 传一个
        # 在 fp32 里 exp() 下溢到 0 的大负数（离线夹具回归：每 rank 都传真值
        # 时 relout 0.63，只计一次时 9.5e-16）。
        # =====================================================================
        dcp_active = _v41_dcp_on() and has_compressed
        _owner = _v41_dcp_ori_owner() if dcp_active else "all"
        _is_ori_owner = (not dcp_active) or _owner == "seqused0" or _v41_dcp_rank() == 0
        sinks = attn.attn_sink
        if dcp_active and sinks is not None and _v41_dcp_rank() != 0:
            sinks = torch.full_like(sinks, -1e30)
        _ori_seqused = seq_lens
        if dcp_active and not _is_ori_owner:
            # 退路：不摘掉 ori_kv，但把可见长度设 0 ⇒ ori 贡献为空。
            _ori_seqused = torch.zeros_like(seq_lens)
        output, softmax_lse = torch.ops._C_ascend.npu_sparse_flash_mla(
            q,
            ori_kv=(
                attn.dsa_attn.swa_cache_layer.kv_cache[0] if _is_ori_owner or _owner == "seqused0" else None
            ),
            cmp_kv=source_cache,
            cmp_sparse_indices=cmp_indices,
            ori_block_table=(ori_block_table if _is_ori_owner or _owner == "seqused0" else None),
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=query_start_loc,
            seqused_ori_kv=_ori_seqused if (_is_ori_owner or _owner == "seqused0") else None,
            seqused_cmp_kv=cmp_seq_lens,
            cmp_residual_kv=cmp_residual,
            sinks=sinks,
            metadata=op_metadata,
            softmax_scale=attn.softmax_scale,
            cmp_ratio=ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3 if has_compressed else 0,
            ori_win_left=attn.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
            topk_value_mode=1,
            return_softmax_lse=dcp_active,
        )
        if dcp_active:
            output = _v41_dcp_merge_attention(output, softmax_lse)
        return output

    @staticmethod
    def update_graph_params(*args, **kwargs):
        """V4.1 owns stable metadata buffers; no backend pointer patch is needed."""
        return None

    def forward(self, attn, positions, hidden_states, output: torch.Tensor | None = None):
        # The custom-op caller provides a graph-stable output buffer.  Write
        # O-projection results into it directly instead of materializing a
        # second full hidden-state tensor and copying it at the graph boundary.
        if output is None:
            output = torch.empty_like(hidden_states)
        forward_context = get_forward_context()
        if forward_context.attn_metadata is None:
            output.zero_()
            return output
        metadata = self._get_layer_metadata(forward_context.attn_metadata)
        positions = metadata.positions[: hidden_states.shape[0]]
        cos, sin = metadata.rope(attn.rotary_emb.layername, hidden_states.shape[0])
        v1_impl = attn.dsa_attn.dsa_attn.impl
        preprocess = self.multistream_preprocess if v1_impl.multistream_dsv4_dsa_overlap else self.preprocess
        q, qr = preprocess(attn, hidden_states, cos, sin, metadata.swa)
        if self.role.is_kv_source:
            self._write_compressed_source(
                attn,
                hidden_states,
                positions,
                cos,
                sin,
                metadata,
            )
        compressed_indices = self._select_sparse_indices(attn, hidden_states, qr, positions, cos, sin, metadata)
        attention_output = self._attention(attn, q, metadata, compressed_indices)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            attention_output.unsqueeze(1),
            cos,
            -sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        attn.dsa_attn.dsa_attn.impl._forward_o_proj(attention_output, output)
        return output


class DeepseekV41MetadataBuilder(AttentionMetadataBuilder[DeepseekV41Metadata]):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_tokens = getattr(vllm_config.scheduler_config, "max_num_batched_tokens", 4096)
        max_reqs = getattr(vllm_config.scheduler_config, "max_num_seqs", 256)
        self._supports_device_ops = getattr(device, "type", "cpu") != "cpu"
        self._slot_mapping = torch.full((max_tokens,), -1, dtype=torch.int64, device=device)
        self._slot_mapping_2d = torch.full((max_tokens, 2), -1, dtype=torch.int32, device=device)
        self._seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._cache_seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._cmp_residual = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._smla_metadata = torch.zeros(V41_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device)
        self._qli_metadata = torch.zeros(V41_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device)
        self._c2_ring_metadata = torch.zeros(5 * max_reqs, dtype=torch.int32, device=device)
        self._c2_complete_mask = torch.zeros(max_tokens, dtype=torch.bool, device=device)
        self._c2_source_positions = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        text_config = vllm_config.model_config.hf_text_config
        rope_dim = int(
            _config_value(
                text_config,
                "qk_rope_head_dim",
                _config_value(text_config, "head_dim"),
            )
        )
        c2_rope_rows = (
            max_tokens if self._supports_device_ops and isinstance(kv_cache_spec, DeepseekV41CompressorStateSpec) else 0
        )
        self._c2_source_cos = torch.ones(
            (c2_rope_rows, 1, 1, rope_dim),
            dtype=torch.float32,
            device=device,
        )
        self._c2_source_sin = torch.zeros_like(self._c2_source_cos)
        self._c2_rope_layer_names = tuple(
            name.removesuffix(".compressor.state_cache") + ".attn"
            for name in layer_names
            if name.endswith(".compressor.state_cache")
        )
        self._c2_full_source_rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self._device_metadata_enabled = False
        self._device_metadata_tasks: tuple[DeviceMetadataTask, ...] = ()

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata,
        **kwargs,
    ) -> DeepseekV41Metadata:
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            **kwargs,
        )

    def enable_device_metadata(self) -> None:
        self._device_metadata_enabled = True
        if isinstance(self.kv_cache_spec, DeepseekV41CompressorStateSpec):
            if not self._c2_rope_layer_names:
                raise RuntimeError("V4.1 compressor-state builder has no source RoPE layer")
            source_rope = get_full_cos_and_sin_dsa_for_layer(self._c2_rope_layer_names[0])
            for rope_layer_name in self._c2_rope_layer_names[1:]:
                other_rope = get_full_cos_and_sin_dsa_for_layer(rope_layer_name)
                if any(other.data_ptr() != source.data_ptr() for other, source in zip(other_rope, source_rope)):
                    raise RuntimeError("V4.1 ratio-2 source layers must share one RoPE table")
            self._c2_full_source_rope = source_rope

    def take_device_metadata_tasks(self) -> tuple[DeviceMetadataTask, ...]:
        tasks = self._device_metadata_tasks
        self._device_metadata_tasks = ()
        return tasks

    def _publish_task(
        self,
        shared: dict[str, Any],
        key: str,
        buffer: torch.Tensor,
        stage: DeviceMetadataStage,
        run,
    ) -> torch.Tensor:
        existing = shared.get(key)
        if existing is not None:
            return existing
        shared[key] = buffer
        if self._device_metadata_enabled:
            self._device_metadata_tasks = (
                *self._device_metadata_tasks,
                DeviceMetadataTask(stage, run, id(buffer)),
            )
        else:
            run()
        return buffer

    def _build_batch_metadata(self, common, num_reqs, num_actual_reqs, num_input_tokens):
        self._seq_lens[:num_reqs].copy_(common.seq_lens[:num_reqs])
        if num_actual_reqs < num_reqs:
            self._seq_lens[num_actual_reqs:num_reqs].zero_()
        seq_lens_cpu = getattr(common, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            seq_lens_cpu = getattr(common, "_seq_lens_cpu", None)
        max_seq_len = int(getattr(common, "max_seq_len", 0))
        if seq_lens_cpu is not None:
            max_seq_len = int(seq_lens_cpu[:num_actual_reqs].max().item()) if num_actual_reqs else 0
        num_decodes, num_decode_tokens, num_prefills, num_prefill_tokens = _request_counts(common, num_reqs)
        positions = common.positions
        if positions is not None:
            positions = positions[:num_input_tokens].long()
        return dict(
            query_start_loc=common.query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=getattr(common, "query_start_loc_cpu", None),
            seq_lens=self._seq_lens[:num_reqs],
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            max_cache_seq_len=max_seq_len,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
        )

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
        **kwargs,
    ):
        if common_prefix_len:
            raise NotImplementedError("V4.1 prefix caching is not implemented")
        self._device_metadata_tasks = ()
        spec = self.kv_cache_spec
        common = common_attn_metadata
        is_compressor_state = isinstance(spec, DeepseekV41CompressorStateSpec)
        ratio = getattr(spec, "compress_ratio", 1)
        if isinstance(spec, DeepseekV41SWASpec):
            cache_kind = "swa"
        elif isinstance(spec, DeepseekV41FullSpec):
            cache_kind = "long_kv"
        elif isinstance(spec, DeepseekV41IndexerSpec):
            cache_kind = "index_k"
        elif is_compressor_state:
            cache_kind = "compressor_state"
        else:
            raise TypeError(f"Unsupported V4.1 cache spec: {type(spec).__name__}")

        num_reqs = int(getattr(common, "num_reqs", common.seq_lens.shape[0]))
        num_actual_reqs = int(kwargs.get("num_actual_reqs", num_reqs))
        num_actual_reqs = min(num_actual_reqs, num_reqs)
        num_input_tokens = int(getattr(common, "num_input_tokens", common.slot_mapping.shape[0]))
        num_actual_tokens = int(getattr(common, "num_actual_tokens", num_input_tokens))
        shared = kwargs.get("common_v41_metadata")
        if shared is None:
            shared = {}
        batch_shared = kwargs.get("common_v41_batch_metadata")
        if batch_shared is None:
            batch_shared = shared

        # The runner resets both dictionaries on each build. Batch values do
        # not depend on physical block IDs; slot mappings remain group-local.
        batch_metadata = batch_shared.get("batch")
        if batch_metadata is None:
            batch_metadata = self._build_batch_metadata(common, num_reqs, num_actual_reqs, num_input_tokens)
            batch_shared["batch"] = batch_metadata
        coordinates = dict(batch_metadata)
        seq_lens = coordinates["seq_lens"]
        positions = coordinates["positions"]

        # =====================================================================
        # [V41-DCP 2026-09-29] 复制态 indexer 的 `[T, 2]` 槽位映射。
        #
        # 与 long_kv 的关键区别：indexer 要覆盖**全量**序列（每个 rank 一份），
        # 而 long_kv 只覆盖本 rank 的 1/dcp。物理布局：
        #   同一个物理块内，index 面被放成 `dcp` 个「子块」，
        #   全局压缩 token `g` → 块列 `g // (dcp*B')`、面内偏移 `((g % (dcp*B'))//B')*B' + g % B'`
        #   其中 `B' = storage_block_size`（已按 ratio 缩过）。
        # 块列是**本 rank 的局部列**：所有 rank 的调度器确定性一致 ⇒ 同一列拿到同一
        # 个物理块 ID ⇒ 8 份 indexer 副本内容逐位相同（这是"全局 top-k 零通信一致"
        # 的前提）。`g` 用全局位置算，不做 rank 过滤。
        # =====================================================================
        dcp_size = int(getattr(self.vllm_config.parallel_config, "decode_context_parallel_size", 1) or 1)
        from vllm_ascend.patch.platform.patch_v41_dcp import replicate_indexer

        index_is_replicated = (
            cache_kind == "index_k"
            and dcp_size > 1
            and replicate_indexer()
            and max(1, int(getattr(spec, "dcp_world_size", 1) or 1)) > 1
        )

        # SWA uses original-token coordinates; circular state has no token slots.
        # Long KV and index K are addressed in completed compression groups.
        compressed = cache_kind in {"long_kv", "index_k"}
        if is_compressor_state:
            # State writes use ring ownership metadata; this buffer stays PAD.
            slots = self._slot_mapping[:num_input_tokens]
        else:
            # Scope ``shared`` to one framework KV cache group in the model
            # runner. Long KV and Indexer builders with the same physical
            # layout then share one persistent [T, 2] mapping, while every SWA
            # group owns a distinct mapping buffer.
            slot_key = f"slot:c{ratio}:b{spec.storage_block_size}"
            if index_is_replicated:
                slot_key += f":dcp{dcp_size}"
            prepared_slots = shared.get(slot_key)
            if prepared_slots is None and index_is_replicated:
                # ---- 复制态：块列与面内偏移都从**全局压缩位置**算，不做 rank 过滤 ----
                storage = int(spec.storage_block_size)
                block_table = getattr(common, "block_table_tensor", None)
                if block_table is None or positions is None:
                    raise RuntimeError("[V41-DCP] 复制态 indexer 需要 block_table_tensor 与 positions")
                pos = positions[:num_input_tokens]
                global_g = torch.div(pos, ratio, rounding_mode="floor")
                span = dcp_size * storage
                column = torch.div(global_g, span, rounding_mode="floor")
                within = global_g.remainder(span)
                face_offset = (
                    torch.div(within, storage, rounding_mode="floor") * storage + within.remainder(storage)
                )
                query_lens = (
                    common.query_start_loc[1 : num_reqs + 1] - common.query_start_loc[:num_reqs]
                ).to(torch.int64)
                req_indices = torch.repeat_interleave(
                    torch.arange(num_reqs, dtype=torch.int64, device=block_table.device),
                    query_lens,
                    output_size=pos.shape[0],
                )[: num_input_tokens]
                columns = column.clamp(0, block_table.shape[1] - 1)
                block_numbers = block_table[req_indices, columns]
                group_complete = (
                    pos.remainder(ratio) == (ratio - 1)
                    if ratio > 1
                    else torch.ones_like(pos, dtype=torch.bool)
                )
                repl_valid = group_complete & (block_numbers > 0)
                self._slot_mapping_2d[:num_input_tokens, 0].copy_(
                    torch.where(repl_valid, block_numbers, -1)
                )
                self._slot_mapping_2d[:num_input_tokens, 1].copy_(
                    torch.where(repl_valid, face_offset, -1)
                )
                prepared_slots = self._slot_mapping_2d[:num_input_tokens]
                shared[slot_key] = prepared_slots
            if prepared_slots is None:
                active_slots = common.slot_mapping[:num_input_tokens]
                if compressed and ratio != 1:
                    active_slots = compressed_slot_mapping(active_slots, ratio)
                valid = active_slots >= 0
                if compressed and ratio == 2:
                    # Prepare the C2 store mask once per cache group, before
                    # forward. Match the ring compressor's completion policy.
                    if kwargs.get("skip_ring_state_update", False):
                        valid.zero_()
                    else:
                        valid_end = common.query_start_loc[num_actual_reqs].clamp_max(num_actual_tokens)
                        valid &= torch.arange(num_input_tokens, device=active_slots.device) < valid_end
                        if positions is not None:
                            valid &= positions.remainder(2) == 1
                physical = active_slots.clamp_min(0)
                self._slot_mapping_2d[:num_input_tokens, 0].copy_(
                    torch.where(
                        valid,
                        torch.div(
                            physical,
                            spec.storage_block_size,
                            rounding_mode="floor",
                        ),
                        -1,
                    )
                )
                self._slot_mapping_2d[:num_input_tokens, 1].copy_(
                    torch.where(
                        valid,
                        physical.remainder(spec.storage_block_size),
                        -1,
                    )
                )
                prepared_slots = self._slot_mapping_2d[:num_input_tokens]
                shared[slot_key] = prepared_slots
            slots = prepared_slots
        plane_ratio = ratio if compressed else 1
        # =====================================================================
        # [V41-DCP 2026-09-29] ★ 压缩平面的可见长度必须是**本 rank 的本地长度**，
        # 不是全局长度。
        #
        # 原因：本 rank 的 long_kv / index_k 物理块只装 1/dcp 的序列。若把全局
        # 压缩长度当 `seqused_cmp_kv` / `seqused_k` 传下去，算子会去读本 rank
        # 块表里不存在的行 ⇒ 读到 null/邻块数据 ⇒ 返回一个**有限的** LSE ⇒
        # 在 `Σ e^{L_r}·O_r / Σ e^{L_r}` 里按错误权重污染结果。
        #
        # ★ 极端且已复现：prompt 只有 17 个 token（I=32, dcp=8）时
        # `owner(pos)=(pos//32)%8` ⇒ **只有 rank 0 有 KV**，rank 1-7 本地长度为 0。
        # 全局长度会告诉它们"你有 8 行"，于是它们读空块、报有限 LSE、把结果搞乱。
        # 这与实测的输出错误形态吻合。
        #
        # `local_compressed_len` 与写侧 `compressed_slot_mapping` 完全同源，
        # 已用 600+ 个长度 × 8 个 rank 与"逐 g 模拟"对拍，零 mismatch。
        # =====================================================================
        _dcp = int(getattr(self.vllm_config.parallel_config, "decode_context_parallel_size", 1) or 1)
        _local_len_ratio = 1
        from vllm_ascend.patch.platform.patch_v41_dcp import v41_dcp_active as _dcp_active

        if _dcp > 1 and compressed and _dcp_active() and not index_is_replicated:
            from vllm.distributed import get_dcp_group

            from vllm_ascend.attention.context_parallel.v41_dcp import local_compressed_len

            _interleave = int(getattr(self.vllm_config.parallel_config, "cp_kv_cache_interleave_size", 1) or 1)
            _rank = int(get_dcp_group().rank_in_group)
            local_lengths = local_compressed_len(
                seq_lens,
                interleave=_interleave,
                ratio=ratio,
                dcp_size=_dcp,
                dcp_rank=_rank,
            )
            local_lengths = local_lengths.to(self._cache_seq_lens.dtype)
            local_max = local_compressed_len(
                torch.tensor([coordinates["max_cache_seq_len"]], dtype=seq_lens.dtype, device=seq_lens.device),
                interleave=_interleave,
                ratio=ratio,
                dcp_size=_dcp,
                dcp_rank=_rank,
            ).item()
            _local_len_ratio = max(1, int(local_max)) if local_max > 0 else 1
        else:
            local_lengths = None
        coordinates["cache_seq_lens"] = seq_lens
        cmp_residual_buffer = None
        if compressed and ratio == 2:
            compressed_lengths = batch_shared.get("lengths:c2")
            if compressed_lengths is None:
                if local_lengths is not None:
                    self._cache_seq_lens[:num_reqs].copy_(local_lengths[:num_reqs])
                else:
                    torch.div(seq_lens, ratio, rounding_mode="floor", out=self._cache_seq_lens[:num_reqs])
                torch.remainder(seq_lens, ratio, out=self._cmp_residual[:num_reqs])
                compressed_lengths = (self._cache_seq_lens[:num_reqs], self._cmp_residual[:num_reqs])
                batch_shared["lengths:c2"] = compressed_lengths
            coordinates["cache_seq_lens"], cmp_residual_buffer = compressed_lengths
        elif local_lengths is not None:
            self._cache_seq_lens[:num_reqs].copy_(local_lengths[:num_reqs])
            coordinates["cache_seq_lens"] = self._cache_seq_lens[:num_reqs]
        coordinates["max_cache_seq_len"] = (
            _local_len_ratio if local_lengths is not None else coordinates["max_cache_seq_len"] // plane_ratio
        )
        cos = sin = None
        if cache_kind == "swa" and positions is not None:
            rope = batch_shared.get("rope")
            if rope is None:
                rope = get_cos_and_sin_dsa(positions, use_cache=coordinates["num_prefills"] == 0)
                batch_shared["rope"] = rope
            cos, sin = rope
        text_config = self.vllm_config.model_config.hf_text_config
        window_size = int(_config_value(text_config, "sliding_window", 0))
        n_local_heads = (
            int(_config_value(text_config, "num_attention_heads"))
            // self.vllm_config.parallel_config.tensor_parallel_size
        )
        head_dim = int(_config_value(text_config, "head_dim"))
        index_topk = int(_config_value(text_config, "index_topk"))
        operator_ratio = 0 if cache_kind == "swa" else ratio
        smla_metadata = None
        qli_metadata = None

        if self._supports_device_ops and cache_kind in {"swa", "long_kv"}:
            has_compressed = operator_ratio in (1, 2)
            cmp_seq_lens = coordinates["cache_seq_lens"] if has_compressed else None
            cmp_residual = cmp_residual_buffer

            def build_smla_metadata() -> None:
                value = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
                    n_local_heads,
                    1,
                    head_dim,
                    cu_seqlens_q=common.query_start_loc[: num_reqs + 1].int(),
                    seqused_ori_kv=seq_lens,
                    seqused_cmp_kv=cmp_seq_lens,
                    cmp_residual_kv=cmp_residual,
                    batch_size=num_reqs,
                    max_seqlen_q=int(getattr(common, "max_query_len", 0)),
                    max_seqlen_ori_kv=int(getattr(common, "max_seq_len", 0)),
                    max_seqlen_cmp_kv=(coordinates["max_cache_seq_len"] if has_compressed else 0),
                    ori_topk=0,
                    cmp_topk=index_topk if has_compressed else 0,
                    cmp_ratio=operator_ratio,
                    ori_mask_mode=4,
                    cmp_mask_mode=3 if has_compressed else 0,
                    ori_win_left=max(0, window_size - 1),
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_BBND",
                    # ==========================================================
                    # [V41-DCP 2026-09-29] ori（滑窗）平面**只由一个 rank 携带**。
                    #
                    # 为什么：DCP 切 KV，滑窗是**每 rank 全量复制**的。合并式
                    #     O = Σ_r w_r·O_r / Σ_r w_r,  w_r = exp(L_r)
                    # 里若 8 个 rank 都算进了同一份 ori 窗口 A，则分子分母都把 A
                    # 计了 8 次（其它 7 次是与不同 B_r 混在一起的，无法事后剥离）。
                    # 让 **rank 0 独占 A**、其余 rank 只带自己的 cmp 分片 B_r：
                    #     L_0      = logsumexp(A ∪ B_0)
                    #     L_{r>0}  = logsumexp(B_r)
                    # ⇒ 合并后的分子/分母里 A 恰好各出现一次
                    #   ⇒ 与全局 `A ∪ B_0 ∪ … ∪ B_7` 精确一致。
                    #
                    # 这条等价性是纯组合性质的（各 rank 的可见集互不重叠地覆盖
                    # 全局集），已被离线夹具用真算子验证（分段覆盖 relout=0.0）。
                    #
                    # 注意只对 **有压缩（ratio∈{1,2}）** 的层生效：ratio=0 的层
                    # 只有 ori、且每 rank 本地就能算出完整正确的滑窗注意力，
                    # 不需要、也不应该走合并。
                    # ==========================================================
                    has_ori_kv=(
                        True
                        if not _v41_dcp_on()
                        else (_v41_dcp_ori_owner() != "rank0" or _v41_dcp_rank() == 0)
                    ),
                    has_cmp_kv=has_compressed,
                )
                self._smla_metadata.copy_(value)

            smla_metadata = self._publish_task(
                batch_shared,
                f"smla:c{operator_ratio}",
                self._smla_metadata,
                DeviceMetadataStage.ATTENTION,
                build_smla_metadata,
            )

        if self._supports_device_ops and cache_kind == "index_k":
            residual = cmp_residual_buffer

            def build_qli_metadata() -> None:
                value = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
                    int(_config_value(text_config, "index_n_heads")),
                    1,
                    int(_config_value(text_config, "index_head_dim")),
                    index_topk,
                    2,
                    cu_seqlens_q=common.query_start_loc[: num_reqs + 1].int(),
                    seqused_k=coordinates["cache_seq_lens"],
                    cmp_residual_k=residual,
                    batch_size=num_reqs,
                    max_seqlen_q=int(getattr(common, "max_query_len", 0)),
                    max_seqlen_k=coordinates["max_cache_seq_len"],
                    layout_q="TND",
                    layout_k="PA_BBND",
                    mask_mode=3,
                    cmp_ratio=ratio,
                )
                self._qli_metadata.copy_(value)

            qli_metadata = self._publish_task(
                batch_shared,
                f"qli:c{ratio}",
                self._qli_metadata,
                DeviceMetadataStage.INDEXER,
                build_qli_metadata,
            )

        c2_ring_metadata = None
        c2_complete_mask = None
        c2_source_positions = None
        c2_source_cos = None
        c2_source_sin = None
        c2_metadata_group_id = None
        if cache_kind == "compressor_state" and positions is not None:
            ring_meta = self._c2_ring_metadata[: 5 * num_reqs].view(5, num_reqs)
            input_positions = positions
            if self._supports_device_ops:
                if self._c2_full_source_rope is None:
                    raise RuntimeError("V4.1 source RoPE buffers were not initialized")
                full_source_cos, full_source_sin = self._c2_full_source_rope
            else:
                full_source_cos = full_source_sin = None

            def build_c2_metadata() -> None:
                starts = common.query_start_loc[:num_reqs].int()
                ends = common.query_start_loc[1 : num_reqs + 1].int()
                query_lens = ends - starts
                live = torch.arange(num_reqs, device=starts.device) < num_actual_reqs
                used = (ends.clamp_max(num_actual_tokens) - starts).clamp_min(0)
                used = torch.where(live, used, 0)
                if kwargs.get("skip_ring_state_update", False):
                    used = torch.zeros_like(used)
                ring_meta[0].copy_((seq_lens - query_lens).clamp_min(0))
                ring_meta[1].copy_(used)
                ring_meta[2].copy_(starts)
                ring_meta[3].copy_(starts)
                ring_meta[4].copy_(torch.where(used > 0, common.block_table_tensor[:num_reqs, 0], 0))
                valid_end = common.query_start_loc[num_actual_reqs].clamp_max(num_actual_tokens)
                valid = torch.arange(num_input_tokens, device=input_positions.device) < valid_end
                complete = (input_positions.remainder(2) == 1) & valid
                if kwargs.get("skip_ring_state_update", False):
                    complete = torch.zeros_like(complete)
                self._c2_complete_mask[:num_input_tokens].copy_(complete)
                self._c2_source_positions[:num_input_tokens].copy_(
                    torch.where(
                        complete,
                        input_positions - 1,
                        torch.zeros_like(input_positions),
                    )
                )
                if full_source_cos is not None and full_source_sin is not None:
                    gather_idx = (
                        self._c2_source_positions[:num_input_tokens]
                        .reshape(-1, 1, 1, 1)
                        .expand(
                            num_input_tokens,
                            1,
                            1,
                            full_source_cos.shape[-1],
                        )
                    )
                    torch.gather(
                        full_source_cos,
                        0,
                        gather_idx,
                        out=self._c2_source_cos[:num_input_tokens],
                    )
                    torch.gather(
                        full_source_sin,
                        0,
                        gather_idx,
                        out=self._c2_source_sin[:num_input_tokens],
                    )

            compressor_group = self._publish_task(
                shared,
                "c2:compressor",
                self._c2_complete_mask,
                DeviceMetadataStage.COMPRESSOR,
                build_c2_metadata,
            )
            if compressor_group is not self._c2_complete_mask:
                raise RuntimeError("V4.1 compressor metadata must have one owner")
            c2_complete_mask = self._c2_complete_mask[:num_input_tokens]
            c2_ring_metadata = ring_meta
            c2_source_positions = self._c2_source_positions[:num_input_tokens]
            if self._supports_device_ops:
                c2_source_cos = self._c2_source_cos[:num_input_tokens]
                c2_source_sin = self._c2_source_sin[:num_input_tokens]
            c2_metadata_group_id = id(self._c2_complete_mask)
        return DeepseekV41Metadata(
            block_table=common.block_table_tensor[:num_reqs],
            slot_mapping=slots,
            compress_ratio=ratio,
            storage_block_size=spec.storage_block_size,
            is_compressor_state=is_compressor_state,
            cache_kind=cache_kind,
            cos=cos,
            sin=sin,
            num_actual_tokens=num_actual_tokens,
            num_input_tokens=num_input_tokens,
            num_reqs=num_reqs,
            num_actual_reqs=num_actual_reqs,
            logical_block_size=spec.block_size,
            max_query_len=int(getattr(common, "max_query_len", 0)),
            max_seq_len=int(getattr(common, "max_seq_len", 0)),
            attn_state=getattr(common, "attn_state", None),
            is_prefilling=getattr(common, "is_prefilling", None),
            causal=getattr(common, "causal", True),
            ori_win_left=max(0, window_size - 1),
            ori_win_right=0,
            smla_metadata=smla_metadata,
            qli_metadata=qli_metadata,
            cmp_residual=cmp_residual_buffer,
            c2_ring_metadata=c2_ring_metadata,
            c2_complete_mask=c2_complete_mask,
            c2_source_positions=c2_source_positions,
            c2_source_cos=c2_source_cos,
            c2_source_sin=c2_source_sin,
            c2_metadata_group_id=c2_metadata_group_id,
            **coordinates,
        )


class DeepseekV41CacheBackend(AttentionBackend):
    """Cache-only backend: supplies layout and metadata, not an AttentionImpl."""

    @staticmethod
    def get_name():
        return "ASCEND_DSA_V41_CACHE"

    @staticmethod
    def get_impl_cls():
        return DeepseekV41EagerAttentionImpl

    @staticmethod
    def get_builder_cls():
        return DeepseekV41MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        return num_blocks, block_size, num_kv_heads, head_size


class DeepseekV41CacheLayer(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(self, vllm_config, prefix, spec):
        super().__init__()
        self.prefix = prefix
        self.spec = spec
        self.kv_cache = [torch.empty(0)]
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate V4.1 cache prefix: {prefix}")
        context[prefix] = self

    def get_kv_cache_spec(self, vllm_config):
        return self.spec

    def get_attn_backend(self):
        return DeepseekV41CacheBackend
