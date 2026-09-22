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

import sys
import math

import torch
import torch.nn.functional as F
import torch_npu
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
    KV8_SCALE_DIM,
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


# --------------------------------------------------------------------- KV8
# The shared long-KV plane can be stored as INT8 payload + FP16 per-group
# scales inside one cache page (``DeepseekV41FullSpec`` with
# ``scale_dim=KV8_SCALE_DIM``).  SparseFlashMla only consumes BF16 KV, so the
# read side dequantises *on demand*: it gathers the rows the selection points
# at, rebuilds them into a contiguous PA_BBND scratch plane and renumbers the
# sparse indices onto that plane.  Only the rows a step really reads are
# dequantised, which keeps the HBM win (520 B/token instead of 1024 B/token).
_KV8_SCRATCH: dict[tuple, torch.Tensor] = {}


def kv8_scratch_plane(
    blocks: int,
    block_size: int,
    dim: int,
    dtype,
    device,
    role: str | None = None,
) -> torch.Tensor:
    """Return a reusable PA_BBND scratch plane (one allocation per geometry)."""
    # [X_integrate] The window rebuild (``kv8_ori_plane``) and the compressed
    # rebuild (``_kv8_cmp_plane``) both allocate through here.  Keying only on
    # ``(blocks, block_size, dim)`` makes them share one buffer whenever the two
    # page counts coincide, and the second write silently clobbers the first
    # (logs/033 §3.4: operator output drift 1.13 at a 512-token chunk).
    # Keying by caller is O(1) and needs no call-site change; the prefill
    # kernels already pass an explicit ``role``.
    # ``X_LEGACY_SCRATCH=1`` restores the old key for the diagnostic arm.
    if role is None and os.environ.get("X_LEGACY_SCRATCH", "0") != "1":
        try:
            role = sys._getframe(1).f_code.co_name
        except Exception:  # noqa: BLE001
            role = "?"
    key = (role, blocks, block_size, dim, dtype, str(device))
    plane = _KV8_SCRATCH.get(key)
    if plane is None:
        plane = torch.empty((blocks, block_size, 1, dim), dtype=dtype, device=device)
        _KV8_SCRATCH[key] = plane
    return plane


def kv8_dequant_rows(kv_i8: torch.Tensor, kv_scale: torch.Tensor) -> torch.Tensor:
    """[N, D] INT8 rows plus [N, G] scales -> BF16 rows (group width D // G)."""
    rows, dim = kv_i8.shape
    groups = kv_scale.shape[-1]
    if not groups or dim % groups:
        raise ValueError(f"KV8 group count {groups} does not divide head_dim {dim}")
    payload = kv_i8.reshape(rows, groups, dim // groups).to(torch.float32)
    scales = kv_scale.reshape(rows, groups).to(torch.float32)
    return (payload * scales.unsqueeze(-1)).reshape(rows, dim).to(torch.bfloat16)


def kv8_quantize_latent(latent: torch.Tensor, groups: int):
    """Per-group dynamic INT8 quantisation of the long-KV rows being stored."""
    rows, dim = latent.shape
    payload, scale = torch_npu.npu_dynamic_quant(
        latent.reshape(rows * groups, dim // groups),
        dst_type=torch.int8,
    )
    return payload.reshape(rows, dim), scale.reshape(rows, groups).to(torch.float16)


def kv8_store_rows(store, slot_mapping: torch.Tensor, values: torch.Tensor) -> bool:
    """Publish long-KV rows into their cache plane.

    An INT8 plane is a ``(payload, scale)`` pair, so the rows are quantised here
    - at the store, after the indexer has already scored the full-precision
    latent - and written with two scatters into the same page.  Returns whether
    the plane was quantised.
    """
    if not isinstance(store, (tuple, list)):
        scatter_cache_sk(store, slot_mapping, values)
        return False
    key_cache, scale_cache = store
    if values.shape[0]:
        payload, scale = kv8_quantize_latent(values, KV8_SCALE_DIM)
        scatter_cache_sk(key_cache, slot_mapping, payload)
        scatter_cache_sk(scale_cache, slot_mapping, scale)
    return True


def kv8_swa_store(attn, slot_mapping: torch.Tensor, values: torch.Tensor) -> None:
    """Store projected SWA rows, quantising them when the window plane is INT8.

    C1 is satisfied structurally: the SWA row is projected, RoPE'd and then
    quantised in the same statement that writes it, so nothing downstream of
    the store (indexer selection included) ever sees the quantised row.
    """
    kv8_store_rows(attn.dsa_attn.swa_cache_layer.kv_cache[0], slot_mapping, values)


# ------------------------------------------------------------------ KV8 gather
# A KV8 plane is one region of a hybrid slot page, so its page stride is the
# *slot capacity* and is larger than its own payload: ``kv.view(-1, width)``
# fails with "view size is not compatible with input tensor's size and stride".
# The old read side therefore used a two-dimensional advanced index
# (``kv_i8[phys]`` / ``kv_i8[phys, offset]``), which builds an index tensor the
# size of the *output* (R*512 int64) and ran at 16-33 GB/s (logs/020 6,
# logs/023 5).  Neither the layout nor the semantics have to change to avoid
# it: **every row (and every page) is a contiguous chunk at a linear byte
# offset**, so the same rows can be fetched with a one-dimensional
# ``index_select`` whose index vector has R entries instead of R*512.
def kv8_page_view(plane: torch.Tensor) -> torch.Tensor:
    """``[pages, page_payload_elements]``: one whole page payload per row."""
    page = int(plane.stride(0))
    per_page = int(plane.shape[1]) * int(plane.stride(1))
    return plane.as_strided((int(plane.shape[0]), per_page), (page, 1), plane.storage_offset())


def kv8_chunk_view(plane: torch.Tensor, chunk: int) -> torch.Tensor:
    """``[chunks, chunk]`` over the slot storage, chunk = gcd(page, row stride)."""
    page = int(plane.stride(0))
    row = int(plane.stride(1))
    rows = ((int(plane.shape[0]) - 1) * page + int(plane.shape[1]) * row) // chunk
    return plane.as_strided((rows, chunk), (chunk, 1), plane.storage_offset())


def kv8_gather_pages(plane: torch.Tensor, pages: torch.Tensor) -> torch.Tensor:
    """``plane[pages]`` via one 1-D ``index_select`` (index = the block table)."""
    out = torch.index_select(kv8_page_view(plane), 0, pages.reshape(-1).to(torch.int64))
    return out.reshape(*pages.shape, *plane.shape[1:])


def kv8_gather_rows(plane: torch.Tensor, phys: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """``plane[phys, offs]`` via one 1-D ``index_select`` over storage chunks."""
    chunk = math.gcd(int(plane.stride(0)), int(plane.stride(1)))
    per_page = int(plane.stride(0)) // chunk
    per_row = int(plane.stride(1)) // chunk
    ids = phys.to(torch.int64) * per_page + offs.to(torch.int64) * per_row
    if per_row == 1:
        ids = ids.reshape(-1)
    else:
        steps = torch.arange(per_row, device=ids.device, dtype=torch.int64)
        ids = (ids.unsqueeze(-1) + steps).reshape(-1)
    out = torch.index_select(kv8_chunk_view(plane, chunk), 0, ids)
    return out.reshape(*phys.shape, *plane.shape[2:-1], int(plane.shape[-1]))


def _kv8_graph_safe_enabled() -> bool:
    """[S_graphfix] env gate: default off so the legacy paths run unchanged."""
    import os as _os

    return _os.environ.get("VLLM_V41_KV8_GRAPH_SAFE", "0").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def _kv8_cmp_legacy() -> bool:
    """[S_graphfix] diagnostic arm: keep the ``cmp`` face on the legacy path.

    ``SG_CMP_LEGACY=1`` leaves the window (SWA) fix in place but sends the
    compressed long-KV face back through the capture-era path.  That isolates
    "what does tier D actually do when only the window face is fixed" - i.e.
    whether a page count frozen at capture time fails loudly or silently.
    Default 0 => no effect.
    """
    import os as _os

    return _os.environ.get("SG_CMP_LEGACY", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


_SG_PPR_SEEN: dict = {}


def _sg_ppr_trace(tag: str, **fields) -> None:
    """[S_graphfix] Hot-path probe for the capture-vs-replay page counts.

    ``SG_TRACE_PPR=1`` prints the *host* scalars that decide the rebuild's scratch
    size, labelled with ``capturing=<bool>`` so a single log shows both sides:

      * capture time: ``mcs=6`` (the dummy batch's seq_len) ⇒ legacy ``ppr=1``;
      * replay time : the real compressed prefix ⇒ hundreds of pages.

    It never reads a device value (that is the defect being measured), and it is
    rate limited so a 40-layer graph cannot flood the log.
    """
    import os as _os

    if _os.environ.get("SG_TRACE_PPR", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return
    try:
        import torch as _t

        capturing = bool(_t.npu.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        capturing = None
    key = (tag, capturing, tuple(sorted((k, str(v)) for k, v in fields.items())))
    if key in _SG_PPR_SEEN or len(_SG_PPR_SEEN) >= 24:
        return
    _SG_PPR_SEEN[key] = 1
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[SG-PPR] {tag} capturing={capturing} {payload}", flush=True)


def _kv8_graph_rows_bound(swa, query_rows: int):
    """[S_graphfix] Host-only per-request query-row bound; ``(False, None)`` = legacy.

    ``graph_safe`` says "this batch's rows are decode rows, so the rebuild must not
    sync the stream"; ``rows_bound`` upper-bounds *every* request's query length and
    is what sizes the scratch.  Both scalars come from CPU-side bookkeeping the
    engine already computed: ``num_prefills`` / ``max_query_len`` are plain ints on
    the layer metadata (the builder copies ``max_query_len`` out of the common
    metadata, which vLLM derives from ``query_start_loc_cpu``), and the fallback is
    ``query_rows`` (the batch's padded row count, a *shape*), which upper-bounds any
    single request's query length.  Prefill batches return the legacy path on
    purpose: prefill is eager, its ``.item()`` was measured cheap (logs/033), and
    this task forbids changing its geometry.
    """
    if not _kv8_graph_safe_enabled():
        return False, None
    if int(getattr(swa, "num_prefills", 0) or 0) > 0:
        return False, None
    bound = int(getattr(swa, "max_query_len", 0) or 0)
    if bound <= 0 or bound > int(query_rows):
        bound = int(query_rows)
    return True, max(1, bound)


def kv8_ori_plane(
    kv_i8: torch.Tensor,
    kv_scale: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_reqs: int,
    query_rows: int,
    window: int,
    rows_bound: int | None = None,
):
    """Rebuild the rows a sliding-window pass actually reads into a BF16 scratch.

    ``npu_sparse_flash_mla`` addresses ``ori_kv`` in *absolute* logical token
    coordinates -- ``block = pos // storage_block_size``,
    ``offset = pos % storage_block_size``,
    ``page = ori_block_table[b, block]`` (``GetOriSparseKeyGmOffset`` in
    ``sparse_flash_mla_swa_block_vector.h``) -- and the band mask
    (``ori_mask_mode=4``) selects, for a query row ``i`` of request ``b``,
    ``[seq_len[b] - q_len[b] + i - window + 1, seq_len[b] - q_len[b] + i]``.

    So the union over the request's query rows is
    ``[seq_len - min(seq_len, q_len + window), seq_len - 1]``.  We keep
    ``seqused_ori_kv`` untouched (the compressed plane's causal mask is derived
    from it), gather exactly those rows, and re-point the *block table* at a
    scratch plane while preserving every row's in-page offset.  Unread rows of
    the table point at page 0; unread pages of the scratch are never fetched.

    The rebuild is *page granular* on purpose: whole logical blocks are copied,
    which keeps every row at its original in-page offset (so a page simply maps
    to a scratch page) and lets the copy run as a contiguous ``copy_`` instead
    of a scattered ``index_copy_``.  At most two pages per request are fetched
    for a decode step; a chunked prefill fetches ``ceil(chunk + window) +
    block_size`` rows, which is what 015 measured as the affordable shape.
    """
    block_size = kv_i8.shape[1]
    dim = kv_i8.shape[-1]
    groups = kv_scale.shape[-1]
    device = kv_i8.device
    lens = seq_lens[:num_reqs].to(torch.int64)
    width = block_table.shape[1]

    if rows_bound is not None:
        # [S_graphfix] Capture-safe rebuild for decode-shaped batches, including
        # speculative decoding, where every request carries ``1 + num_spec_tokens``
        # query rows instead of one.  The old predicate (``query_rows == num_reqs``)
        # only recognised the pure-decode shape, so a spec-decode batch fell into
        # the eager prefill branch below, whose ``.max().item()`` syncs the stream
        # and aborts graph capture (Not_Supported(EE1016), logs/048).
        #
        # Nothing here may read a device value: the band start uses the *real*
        # per-request query length (a device-side diff of ``query_start_loc``, no
        # host round trip) and the page count uses the host-side bound
        # ``rows_bound >= max(q_len)``.  The widest band any request can need is
        # ``rows_bound + window`` rows, and a span of ``s`` rows spans at most
        # ``((s - 2) // block_size) + 2 <= (s - 1) // block_size + 2`` pages, so the
        # bound holds for every in-page alignment.  Extra pages are inert: the block
        # table only points at blocks the mask reads, and the operator ignores rows
        # outside the band.
        q_len = (query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]).to(torch.int64)
        span = torch.minimum(lens, q_len + window)
        window_start = lens - span
        pages_per_req = min(width, max(1, (int(rows_bound) + window - 1) // block_size + 2))
    elif query_rows == num_reqs:
        # Decode: one query row per request, so the span is exactly one window
        # and covers at most two pages.  Fully device-side (capture safe).
        pages_per_req = 2
        window_start = (lens - window).clamp_min(0)
    else:
        # Prefill: every query row carries its own window, so the rebuilt span
        # is the chunk plus one window.  Eager only, hence the host syncs.
        q_len = (query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]).to(torch.int64)
        span = torch.minimum(lens, q_len + window)
        window_start = lens - span
        pages_per_req = int(
            ((lens - 1) // block_size - window_start // block_size + 1).max().item()
        )

    first_block = window_start // block_size
    blocks_per_req = (lens - 1) // block_size - first_block + 1
    base = torch.arange(num_reqs, device=device, dtype=torch.int64) * pages_per_req
    steps = torch.arange(pages_per_req, device=device, dtype=torch.int64).view(1, pages_per_req)
    # Clamping only ever duplicates a column that the mask does not read.
    columns = torch.minimum(
        first_block.view(num_reqs, 1) + steps,
        torch.full((num_reqs, 1), width - 1, dtype=torch.int64, device=device),
    )
    phys = torch.gather(block_table[:num_reqs].to(torch.int64), 1, columns)
    sel_i8 = kv8_gather_pages(kv_i8, phys)     # page-granular, flat index_select
    sel_scale = kv8_gather_pages(kv_scale, phys)
    deq = kv8_dequant_rows(sel_i8.reshape(-1, dim), sel_scale.reshape(-1, groups))
    scratch = kv8_scratch_plane(num_reqs * pages_per_req, block_size, dim, deq.dtype, deq.device)
    scratch.view(-1, 1, dim).copy_(deq.view(-1, 1, dim))

    # Logical block -> scratch page, preserving in-page offsets.  Reads only
    # ever touch blocks inside the window, so the rest of the table is inert.
    columns = torch.arange(width, device=device, dtype=torch.int64).view(1, width)
    delta = columns - first_block.view(num_reqs, 1)
    table = torch.where(
        (delta >= 0) & (delta < blocks_per_req.view(num_reqs, 1)),
        base.view(num_reqs, 1) + delta,
        torch.zeros(1, dtype=torch.int64, device=device),
    )
    return scratch, table.to(torch.int32)


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
        kv8_swa_store(attn, swa_metadata.slot_mapping, kv)
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
            kv8_swa_store(attn, swa_metadata.slot_mapping, kv.squeeze(1))
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
        store = attn.long_kv_cache.kv_cache[0]
        # KV8: quantise the RoPE'd latent exactly where it is published.  The
        # quantiser sits *after* ``indexer.update_keys`` on purpose - the indexer
        # must keep scoring the unquantised latent, or block selection changes.
        kv8_store_rows(store, long_slots, latent.squeeze(1))

    def _select_sparse_indices(self, attn, hidden_states, qr, positions, cos, sin, metadata):
        if not self.role.has_long_context:
            return None
        shared = attn.shared_state
        if shared is None:
            raise RuntimeError("V4.1 shared attention state is not initialized")
        if not self.role.is_index_source:
            return shared.topk_indices[: hidden_states.shape[0]]
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
        shared.topk_indices[: selected.shape[0]].copy_(selected)
        if self.role.is_candidate_source:
            shared.candidates[: candidates.shape[0]].copy_(candidates)
        return shared.topk_indices[: selected.shape[0]]

    def _attention(self, attn, q, metadata, compressed_indices):
        source_cache = None
        source_scale = None
        if self.role.has_long_context:
            store = get_forward_context().no_compile_layers[self.long_kv_source_prefix].kv_cache[0]
            if isinstance(store, (tuple, list)):
                source_cache, source_scale = store
            else:
                source_cache = store
        return self._native_attention(
            attn,
            q,
            metadata,
            source_cache=source_cache,
            source_scale=source_scale,
            compressed_indices=compressed_indices,
        )

    def _kv8_cmp_plane(
        self,
        kv_i8,
        kv_scale,
        indices,
        block_table,
        num_reqs,
        cache_seq_lens,
        graph_safe: bool = False,
    ):
        """Gather + dequantise the selected long-KV rows into a PA_BBND scratch.

        Returns ``(scratch, identity_block_table, indices)``.  The operator keeps
        reading a plain BF16 paged plane; only its addressing changes.
        """
        block_size = kv_i8.shape[1]
        dim = kv_i8.shape[-1]
        groups = kv_scale.shape[-1]
        rows, _, topk = indices.shape
        per_req = (topk + block_size - 1) // block_size
        if rows == num_reqs and per_req * block_size == topk:
            # Decode: one query token per request, so the topk rows a request
            # selected fit in one private scratch segment (index t -> slot t).
            idx = indices[:, 0].to(torch.int64)
            valid = idx >= 0
            safe = torch.where(valid, idx, torch.zeros_like(idx))
            block = safe // block_size
            offset = safe % block_size
            # ``block`` indexes the *real* block table (logical compressed block),
            # while ``per_req`` only describes the scratch segment.
            phys = torch.gather(block_table[:rows].to(torch.int64), 1, block)
            keys = kv8_gather_rows(kv_i8, phys, offset)
            scales = kv8_gather_rows(kv_scale, phys, offset)
            deq = kv8_dequant_rows(keys.reshape(rows * topk, dim), scales.reshape(rows * topk, groups))
            scratch = kv8_scratch_plane(rows * per_req, block_size, dim, deq.dtype, deq.device)
            scratch.view(rows * per_req * block_size, 1, dim)[: rows * topk].copy_(deq.view(rows * topk, 1, dim))
            columns = torch.arange(topk, dtype=torch.int32, device=indices.device).view(1, topk).expand(rows, topk)
            renumbered = torch.where(valid, columns, torch.full_like(columns, -1))
            table = torch.arange(rows * per_req, dtype=torch.int32, device=indices.device).view(rows, per_req)
            return scratch, table, renumbered.unsqueeze(1)
        if graph_safe and per_req * block_size == topk and rows % num_reqs == 0:
            # [S_graphfix] Speculative decode: ``rows = num_reqs * (1 + num_spec)``
            # query rows, and the topk sets differ per row.  The fallback below
            # rebuilds each request's whole compressed prefix, which needs
            # ``cache_seq_lens.max()`` -- a D2H sync (capture-fatal) *and*, if the
            # page count were frozen at capture time instead, a silent replay bug:
            # the capture dummy batch reports seq_len == max_query_len (6 here), so a
            # capture-time page count would cover 1 block while replay addresses
            # hundreds.
            #
            # Instead give every query row a private scratch segment: row ``i``'s
            # ``t``-th selection lands at synthetic index ``i * topk + t``, i.e.
            # scratch page ``i * per_req + t // block_size`` at in-page offset
            # ``t % block_size`` -- exact because ``per_req * block_size == topk``
            # (the same identity the single-row fast path above relies on).  The
            # scratch table is then the identity over ``rows * per_req`` pages, and
            # every scalar involved is a shape or a config constant, so the branch is
            # capture-safe and replay-safe.
            reps = rows // num_reqs
            idx = indices[:, 0].to(torch.int64)
            valid = idx >= 0
            safe = torch.where(valid, idx, torch.zeros_like(idx))
            block = safe // block_size
            offset = safe % block_size
            # Row ``i`` belongs to request ``i // reps`` in the padded batch (uniform
            # spec-decode query length).  ``repeat_interleave`` is a device op with a
            # static output shape: no host round trip.
            table_rows = block_table[:num_reqs].to(torch.int64).repeat_interleave(reps, dim=0)
            phys = torch.gather(table_rows, 1, block)
            keys = kv8_gather_rows(kv_i8, phys, offset)
            scales = kv8_gather_rows(kv_scale, phys, offset)
            deq = kv8_dequant_rows(
                keys.reshape(rows * topk, dim),
                scales.reshape(rows * topk, groups),
            )
            segments = rows * per_req
            _sg_ppr_trace(
                "cmp_graph_safe",
                num_reqs=num_reqs,
                rows=rows,
                per_req=per_req,
                segments=segments,
                topk=topk,
            )
            scratch = kv8_scratch_plane(segments, block_size, dim, deq.dtype, deq.device)
            scratch.view(segments * block_size, 1, dim).copy_(deq.view(rows * topk, 1, dim))
            row_base = (
                torch.arange(rows, dtype=torch.int32, device=indices.device) * topk
            ).view(rows, 1)
            columns = torch.arange(topk, dtype=torch.int32, device=indices.device).view(1, topk)
            renumbered = torch.where(valid, row_base + columns, torch.full_like(columns, -1))
            table = (
                torch.arange(segments, dtype=torch.int32, device=indices.device)
                .view(1, segments)
                .repeat(num_reqs, 1)
            )
            return scratch, table, renumbered.unsqueeze(1)
        # Prefill / multi query-row batch: the topk sets differ per query row, so
        # the whole compressed prefix of every request is rebuilt instead and the
        # operator keeps the original logical indices.
        used = int(cache_seq_lens[:num_reqs].max().item())
        nblocks = max(1, -(-used // block_size))
        pages = block_table[:num_reqs, :nblocks].to(torch.int64).clamp_(0, kv_i8.shape[0] - 1)
        keys = kv8_gather_pages(kv_i8, pages)
        scales = kv8_gather_pages(kv_scale, pages)
        deq = kv8_dequant_rows(keys.reshape(-1, dim), scales.reshape(-1, groups))
        scratch = kv8_scratch_plane(num_reqs * nblocks, block_size, dim, deq.dtype, deq.device)
        scratch.view(-1, 1, dim).copy_(deq.view(-1, 1, dim))
        table = torch.arange(num_reqs * nblocks, dtype=torch.int32, device=indices.device).view(num_reqs, nblocks)
        return scratch, table, indices

    def _native_attention(
        self,
        attn,
        q,
        metadata,
        *,
        source_cache,
        source_scale,
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
        ori_kv = attn.dsa_attn.swa_cache_layer.kv_cache[0]
        # [S_graphfix] Host-only gate + per-request row bound (no D2H, no
        # capture-time *value*): decode-shaped batches take the bound branch
        # above, real prefill batches keep the legacy eager branch.
        graph_safe, rows_bound = _kv8_graph_rows_bound(metadata.swa, q.shape[0])
        _sg_ppr_trace(
            "native_attention",
            num_reqs=num_reqs,
            query_rows=int(q.shape[0]),
            num_prefills=int(getattr(metadata.swa, "num_prefills", 0) or 0),
            max_query_len=int(getattr(metadata.swa, "max_query_len", 0) or 0),
            swa_mcs=int(getattr(metadata.swa, "max_cache_seq_len", 0) or 0),
            cmp_mcs=int(
                0 if getattr(metadata, "attention", None) is None
                else (getattr(metadata.attention, "max_cache_seq_len", 0) or 0)
            ),
            graph_safe=graph_safe,
            rows_bound=rows_bound,
        )
        if isinstance(ori_kv, (tuple, list)):
            # KV8-SWA: the window plane is INT8, so rebuild the rows this pass
            # reads and re-point the table at the scratch plane.  ``seqused_ori_kv``
            # stays in real sequence coordinates (the compressed plane's causal
            # mask is derived from the same value).
            ori_kv, ori_block_table = kv8_ori_plane(
                ori_kv[0],
                ori_kv[1],
                query_start_loc,
                seq_lens,
                ori_block_table,
                num_reqs,
                q.shape[0],
                attn.window_size,
                rows_bound=rows_bound,
            )
        cmp_kv = source_cache
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
            if source_scale is not None:
                cmp_kv, cmp_block_table, cmp_indices = self._kv8_cmp_plane(
                    source_cache,
                    source_scale,
                    cmp_indices,
                    cmp_block_table,
                    num_reqs,
                    cmp_seq_lens,
                    graph_safe=graph_safe and not _kv8_cmp_legacy(),
                )

        operator_metadata = metadata.attention if has_compressed else metadata.swa
        op_metadata = operator_metadata.smla_metadata
        if op_metadata is None:
            raise RuntimeError(f"V4.1 ratio-{ratio} SMLA metadata was not built")
        wait_for_device_metadata(
            DeviceMetadataStage.ATTENTION,
            id(op_metadata),
        )
        output, _ = torch.ops._C_ascend.npu_sparse_flash_mla(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_indices,
            ori_block_table=ori_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=query_start_loc,
            seqused_ori_kv=seq_lens,
            seqused_cmp_kv=cmp_seq_lens,
            cmp_residual_kv=cmp_residual,
            sinks=attn.attn_sink,
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
            return_softmax_lse=False,
        )
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
            prepared_slots = shared.get(slot_key)
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
        coordinates["cache_seq_lens"] = seq_lens
        cmp_residual_buffer = None
        if compressed and ratio == 2:
            compressed_lengths = batch_shared.get("lengths:c2")
            if compressed_lengths is None:
                torch.div(seq_lens, ratio, rounding_mode="floor", out=self._cache_seq_lens[:num_reqs])
                torch.remainder(seq_lens, ratio, out=self._cmp_residual[:num_reqs])
                compressed_lengths = (self._cache_seq_lens[:num_reqs], self._cmp_residual[:num_reqs])
                batch_shared["lengths:c2"] = compressed_lengths
            coordinates["cache_seq_lens"], cmp_residual_buffer = compressed_lengths
        coordinates["max_cache_seq_len"] //= plane_ratio
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
                    has_ori_kv=True,
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


# --- KV8_prefill wiring (logs/033) ------------------------------------------
import os


def _kv8_prefill_enabled() -> bool:
    return os.environ.get("VLLM_V41_KV8_PREFILL", "0").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )


if _kv8_prefill_enabled():
    from vllm_ascend.attention.kv8_prefill_triton import fused_cmp_plane3 as _kv8_pf_cmp
    from vllm_ascend.attention.kv8_prefill_triton import fused_ori_plane2 as _kv8_pf_ori

    _kv8_ori_plane_decode = kv8_ori_plane

    def kv8_ori_plane(
        kv_i8,
        kv_scale,
        query_start_loc,
        seq_lens,
        block_table,
        num_reqs,
        query_rows,
        window,
        rows_bound=None,
    ):
        """Chunked prefill: one Triton launch rebuilds the whole window union."""
        if rows_bound is not None:
            # [S_graphfix] Decode-shaped batch (incl. spec decode): hand the host
            # bound to the plain-torch rebuild.  The Triton prefill kernel sizes its
            # scratch from ``max_q_len=query_rows`` and is only needed for eager
            # prefill, where a chunk carries far more than one row per request.
            return _kv8_ori_plane_decode(
                kv_i8,
                kv_scale,
                query_start_loc,
                seq_lens,
                block_table,
                num_reqs,
                query_rows,
                window,
                rows_bound=rows_bound,
            )
        if query_rows == num_reqs:
            return _kv8_ori_plane_decode(
                kv_i8,
                kv_scale,
                query_start_loc,
                seq_lens,
                block_table,
                num_reqs,
                query_rows,
                window,
            )
        # ``query_rows`` is the step's total query rows (a shape, so no D2H
        # round trip): it upper-bounds every request's query length, which is
        # all the kernel needs to size the window union.  Reading the true max
        # per-request length instead costs a host sync per layer, which in eager
        # prefill lands directly on the step time (measured: +12.2 ms/step).
        return _kv8_pf_ori(
            kv_i8,
            kv_scale,
            query_start_loc,
            seq_lens,
            block_table,
            num_reqs,
            query_rows,
            window,
            max_q_len=query_rows,
        )

    # The compressed scratch needs a page count *before* the launch, and it has
    # to be a host int.  ``_native_attention`` already carries it: the layer
    # metadata's ``max_cache_seq_len`` is a plain Python int computed by the
    # builder from the CPU-side sequence lengths, so this costs no D2H sync -
    # unlike ``cache_seq_lens.max().item()``, which stalls the eager prefill
    # step for ~0.4 ms per layer (measured: +1.6 ms/step, see logs/033).
    _kv8_model_meta = {"max_cache_seq_len": 0}
    _kv8_native_attention = DeepseekV41EagerAttentionImpl._native_attention

    def _native_attention_prefill(self, attn, q, metadata, **kwargs):
        attn_md = getattr(metadata, "attention", None)
        mcs = getattr(attn_md, "max_cache_seq_len", 0) if attn_md is not None else 0
        _kv8_model_meta["max_cache_seq_len"] = int(mcs or 0)
        return _kv8_native_attention(self, attn, q, metadata, **kwargs)

    DeepseekV41EagerAttentionImpl._native_attention = _native_attention_prefill

    _kv8_cmp_plane_decode = DeepseekV41EagerAttentionImpl._kv8_cmp_plane

    def _kv8_cmp_plane_prefill(
        self,
        kv_i8,
        kv_scale,
        indices,
        block_table,
        num_reqs,
        cache_seq_lens,
        graph_safe=False,
    ):
        """Chunked prefill: page-granular rebuild, one Triton launch."""
        rows, _, topk = indices.shape
        block_size = kv_i8.shape[1]
        per_req = (topk + block_size - 1) // block_size
        if graph_safe and per_req * block_size == topk and rows % num_reqs == 0:
            # [S_graphfix] Uniform multi-row batch (spec decode): the class method's
            # selection-based branch is both capture- and replay-safe.  The Triton
            # prefill kernel below sizes its scratch from ``max_cache_seq_len``, a
            # *value* the capture dummy batch reports as 6 ⇒ never use it on a
            # graph-replayed step.
            return _kv8_cmp_plane_decode(
                self,
                kv_i8,
                kv_scale,
                indices,
                block_table,
                num_reqs,
                cache_seq_lens,
                graph_safe=True,
            )
        if rows == num_reqs and per_req * block_size == topk:
            return _kv8_cmp_plane_decode(
                self,
                kv_i8,
                kv_scale,
                indices,
                block_table,
                num_reqs,
                cache_seq_lens,
                graph_safe=graph_safe,
            )
        mcs = _kv8_model_meta["max_cache_seq_len"]
        ppr = max(1, -(-mcs // block_size)) if mcs else None
        _sg_ppr_trace(
            "cmp_legacy_triton",
            num_reqs=num_reqs,
            rows=rows,
            topk=topk,
            per_req=per_req,
            max_cache_seq_len=mcs,
            legacy_ppr=(ppr if ppr is not None else -1),
            block_size=block_size,
        )
        return _kv8_pf_cmp(
            self,
            kv_i8,
            kv_scale,
            indices,
            block_table,
            num_reqs,
            cache_seq_lens,
            ppr=ppr,
        )

    DeepseekV41EagerAttentionImpl._kv8_cmp_plane = _kv8_cmp_plane_prefill

# --- end KV8_prefill wiring -------------------------------------------------
