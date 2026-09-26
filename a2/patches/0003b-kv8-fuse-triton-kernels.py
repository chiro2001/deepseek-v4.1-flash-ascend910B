"""KV8 read side fused into one kernel per rebuild path (Triton-Ascend).

Why: `logs/023` measured the rebuild cost as **operator count x ~5 us**, not
bandwidth.  The SWA rebuild is ~30 device ops and the cmp read path ~30 more, so
the layer increment is ~350 us while the same bytes moved at 1.2 TB/s would cost
~10 us.  This module replaces both paths with

  SWA rebuild : 2 launches (row gather+dequant, block-table renumber)
  cmp rebuild : 1 launch  (index math + row gather + dequant + renumber)

Everything the torch version did (index arithmetic, the two-plane gather, the
group-128 dequant, the scratch write, the block-table / sparse-index renumbering)
happens inside the kernels, so no intermediate tensor is ever materialised.

Bit-exactness: the arithmetic is the same as `kv8_dequant_rows`
(``int8 -> fp32``, ``fp16 -> fp32``, one fp32 multiply, ``-> bf16``).
"""
from __future__ import annotations

import os

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

_READY = False
_TRACE = os.environ.get("KV8_FUSE_TRACE") == "1"


def _trace(msg: str) -> None:
    if _TRACE:
        print(f"[kv8fuse] {msg}", flush=True)


def _ready() -> None:
    global _READY
    if not _READY:
        init_device_properties_triton()
        _READY = True


# The dequant tile is **[ROWS, GROUP, DIM//GROUP]**, not [ROWS, DIM] with a
# ``cols // GROUP`` scale index: on Triton-Ascend the latter compiles to an
# unbounded-width index (measured: 468k wrong elements + NaN on a 2048x512
# tile, 297 us) while the 3-D form is bit-exact and 12 us (raw/026-micro.json).
# Scale broadcast is the multi-dimensional multiply, so no division is needed.


# ------------------------------------------------------------------ SWA plane
@triton.jit
def _kv8_swa_rows_kernel(
    payload_ptr,          # int8 plane  [P, BS, 1, DIM] (own page/row strides)
    scale_ptr,            # fp16 plane  [P, BS, 1, G]
    lens_ptr,             # int32/int64 [num_reqs]
    table_ptr,            # int32       [num_reqs, width]  (real block table)
    scratch_ptr,          # bf16        [num_reqs*PP*BS, DIM]
    window,
    page_stride,
    row_stride,
    s_page_stride,
    s_row_stride,
    width,
    BS: tl.constexpr,
    PP: tl.constexpr,
    LBS: tl.constexpr,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    ROWS: tl.constexpr,
    RB: tl.constexpr,
):
    """One program per (scratch page, row block): the whole window rebuild."""
    GD: tl.constexpr = DIM // GROUP
    pid = tl.program_id(0)
    slot = pid // RB
    rb = pid % RB
    req = slot // PP
    p = slot % PP
    length = tl.load(lens_ptr + req)
    first = tl.maximum(length - window, 0) >> LBS
    blk = tl.minimum(first + p, width - 1)
    page = tl.load(table_ptr + req * width + blk)
    rows = rb * ROWS + tl.arange(0, ROWS)
    g = tl.arange(0, GROUP)
    d = tl.arange(0, GD)
    poff = (page * page_stride + rows[:, None, None] * row_stride
            + g[None, :, None] * GD + d[None, None, :])
    codes = tl.load(payload_ptr + poff).to(tl.float32)
    goff = page * s_page_stride + rows[:, None] * s_row_stride + g[None, :]
    scales = tl.load(scale_ptr + goff).to(tl.float32)
    out = (codes * scales[:, :, None]).to(tl.bfloat16)
    ooff = ((slot * BS + rows)[:, None, None] * DIM + g[None, :, None] * GD
            + d[None, None, :])
    tl.store(scratch_ptr + ooff, out)


@triton.jit
def _kv8_swa_table_kernel(
    out_ptr,              # int32 [num_reqs, width]
    lens_ptr,
    width,
    window,
    BS: tl.constexpr,
    PP: tl.constexpr,
    LBS: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Logical block -> scratch page, one program per request (whole row)."""
    req = tl.program_id(0)
    length = tl.load(lens_ptr + req)
    first = tl.maximum(length - window, 0) >> LBS
    bpr = ((length - 1) >> LBS) - first + 1
    base = req * PP
    for start in range(0, width, BLOCK_W):
        col = start + tl.arange(0, BLOCK_W)
        d = col - first
        val = tl.where((d >= 0) & (d < bpr), base + d, 0)
        tl.store(out_ptr + req * width + col, val.to(tl.int32), mask=col < width)


# ------------------------------------------------------------------ cmp plane
@triton.jit
def _kv8_cmp_rows_kernel(
    payload_ptr,          # int8 plane [P, BS, 1, DIM]
    scale_ptr,            # fp16 plane [P, BS, 1, G]
    idx_ptr,              # int32 [rows*TOPK] (sparse indices, -1 = unused)
    table_ptr,            # int32 [rows, width] (real block table)
    scratch_ptr,          # bf16  [rows*SEG, DIM]
    renum_ptr,            # int32 [rows*TOPK] out
    page_stride,
    row_stride,
    s_page_stride,
    s_row_stride,
    width,
    total,
    SEG: tl.constexpr,    # per-request scratch rows
    BS: tl.constexpr,
    TOPK: tl.constexpr,
    LBS: tl.constexpr,    # log2(BS)      -- div/mod by powers of two is done with
    LTOPK: tl.constexpr,  # log2(TOPK)       shifts: Triton-Ascend's address pass
                          #                rejects `%` / `/` in any address term
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Index math + gather + dequant + renumber for the selected cmp rows."""
    GD: tl.constexpr = DIM // GROUP
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    mask = rows < total
    req = rows >> LTOPK
    t = rows & (TOPK - 1)
    idx = tl.load(idx_ptr + rows, mask=mask, other=0)
    valid = idx >= 0
    safe = tl.where(valid, idx, 0)
    blk = safe >> LBS
    off = safe & (BS - 1)
    page = tl.load(table_ptr + req * width + blk, mask=mask, other=0)
    g = tl.arange(0, GROUP)
    d = tl.arange(0, GD)
    poff = (page[:, None, None] * page_stride + off[:, None, None] * row_stride
            + g[None, :, None] * GD + d[None, None, :])
    codes = tl.load(payload_ptr + poff, mask=mask[:, None, None], other=0).to(tl.float32)
    goff = page[:, None] * s_page_stride + off[:, None] * s_row_stride + g[None, :]
    scales = tl.load(scale_ptr + goff, mask=mask[:, None], other=0).to(tl.float32)
    out = (codes * scales[:, :, None]).to(tl.bfloat16)
    # SEG == TOPK on this branch (the wrapper refuses any other geometry), so
    # the per-request scratch row of selection ``t`` is simply ``rows``.
    ooff = (rows[:, None, None] * DIM + g[None, :, None] * GD + d[None, None, :])
    tl.store(scratch_ptr + ooff, out, mask=mask[:, None, None])
    renum = tl.where(valid, t, -1)
    tl.store(renum_ptr + rows, renum.to(tl.int32), mask=mask)


# ------------------------------------------------------------------- wrappers
_TABLE_CACHE: dict[tuple, torch.Tensor] = {}

# Tile knobs (rows per program).  Measured optima: SWA 2 rows (net 6.5 us vs
# 8.8 / 9.0 / 10.6 for 8 / 4 / 16), cmp 16 rows (net 11.8 us vs 17.7 / 27.3 /
# 32.9 for 8 / 4 / 2).  In the 40-layer graph (2, 16) gives 80.8 us/layer
# absolute vs 98.2 for (8, 4) -- raw/026-fuse-run6.json.
# Never 1: a shape-1 ``tl.arange`` makes Triton-Ascend's BlockPtrAnalysis bail
# out with "AddPtrOp produced by unsupported operation" (hard abort, run3/run4).
SWA_ROWS = 2
CMP_ROWS = 16


def _static_table(rows: int, per_req: int, device) -> torch.Tensor:
    """``arange(rows*per_req).view(rows, per_req)`` -- input independent."""
    key = (rows, per_req, str(device))
    table = _TABLE_CACHE.get(key)
    if table is None:
        table = torch.arange(rows * per_req, dtype=torch.int32, device=device).view(rows, per_req)
        _TABLE_CACHE[key] = table
    return table


def _plane_geometry(payload: torch.Tensor, scale: torch.Tensor):
    return (
        int(payload.stride(0)), int(payload.stride(1)),
        int(scale.stride(0)), int(scale.stride(1)),
    )


def _pow2(value: int) -> bool:
    """The kernels use shifts/masks for div/mod, which needs a power of two."""
    return value > 0 and (1 << (value.bit_length() - 1)) == value


def _supported(*values: int) -> bool:
    """``tl.arange`` needs power-of-two extents and the address maths shifts."""
    return all(_pow2(int(v)) for v in values)


def fused_ori_plane(
    kv_i8, kv_scale, query_start_loc, seq_lens, block_table, num_reqs, query_rows, window,
):
    """Drop-in for ``dsa_v41.kv8_ori_plane`` (decode only; prefill falls back)."""
    from vllm_ascend.attention import dsa_v41

    block_size = int(kv_i8.shape[1])
    dim = int(kv_i8.shape[-1])
    groups = int(kv_scale.shape[-1])
    if _ORIG_ORI is None:
        capture_originals()
    if query_rows != num_reqs or not _supported(block_size, dim // groups, groups):
        return _ORIG_ORI(kv_i8, kv_scale, query_start_loc, seq_lens, block_table,
                         num_reqs, query_rows, window)
    _ready()
    device = kv_i8.device
    width = int(block_table.shape[1])
    pages_per_req = 2
    rows = SWA_ROWS if (SWA_ROWS > 1 and block_size % SWA_ROWS == 0) else 2
    rblock = block_size // rows
    lens = seq_lens[:num_reqs]
    scratch = dsa_v41.kv8_scratch_plane(
        num_reqs * pages_per_req, block_size, dim, torch.bfloat16, device)
    table = torch.empty((num_reqs, width), dtype=torch.int32, device=device)
    ps, rs, sps, srs = _plane_geometry(kv_i8, kv_scale)
    _trace(f"swa rows grid={num_reqs * pages_per_req * rblock} rows={rows} geom={ps},{rs},{sps},{srs}")
    _kv8_swa_rows_kernel[(num_reqs * pages_per_req * rblock,)](
        kv_i8, kv_scale, lens, block_table, scratch,
        int(window), ps, rs, sps, srs, width,
        BS=block_size, PP=pages_per_req, LBS=int(block_size).bit_length() - 1,
        DIM=dim, GROUP=groups, ROWS=rows, RB=rblock,
        num_warps=4,
    )
    _trace(f"swa table grid={num_reqs} width={width}")
    _kv8_swa_table_kernel[(num_reqs,)](
        table, lens, width, int(window),
        BS=block_size, PP=pages_per_req, LBS=int(block_size).bit_length() - 1,
        BLOCK_W=128, num_warps=1,
    )
    return scratch, table


def fused_cmp_plane(self, kv_i8, kv_scale, indices, block_table, num_reqs, cache_seq_lens):
    """Drop-in for ``DeepseekV41EagerAttentionImpl._kv8_cmp_plane`` (decode)."""
    from vllm_ascend.attention import dsa_v41

    block_size = int(kv_i8.shape[1])
    dim = int(kv_i8.shape[-1])
    groups = int(kv_scale.shape[-1])
    rows, _, topk = indices.shape
    per_req = (topk + block_size - 1) // block_size
    if _ORIG_CMP is None:
        capture_originals()
    if (rows != num_reqs or per_req * block_size != topk
            or not _supported(block_size, dim // groups, groups, topk)):
        return _ORIG_CMP(self, kv_i8, kv_scale, indices, block_table, num_reqs, cache_seq_lens)
    _ready()
    device = kv_i8.device
    width = int(block_table.shape[1])
    seg = per_req * block_size
    total = rows * topk
    rrows = CMP_ROWS if (CMP_ROWS > 1 and total % CMP_ROWS == 0) else 2
    scratch = dsa_v41.kv8_scratch_plane(rows * per_req, block_size, dim, torch.bfloat16, device)
    renum = torch.empty((rows, 1, topk), dtype=torch.int32, device=device)
    ps, rs, sps, srs = _plane_geometry(kv_i8, kv_scale)
    _trace(f"cmp rows grid={total // rrows} rows={rrows} geom={ps},{rs},{sps},{srs} width={width}")
    _kv8_cmp_rows_kernel[(total // rrows,)](
        kv_i8, kv_scale, indices, block_table, scratch, renum,
        ps, rs, sps, srs, width, total,
        SEG=seg, BS=block_size, TOPK=topk,
        LBS=int(block_size).bit_length() - 1, LTOPK=int(topk).bit_length() - 1,
        DIM=dim, GROUP=groups, ROWS=rrows,
        num_warps=4,
    )
    return scratch, _static_table(rows, per_req, device), renum


# Captured before any monkeypatching so the fallbacks stay reachable.  Both the
# explicit entry point (`capture_originals`) and the lazy path inside the
# wrappers go through this so a patch that replaces ``dsa_v41.kv8_ori_plane``
# before importing this module still leaves the torch implementation reachable.
_ORIG_ORI = None
_ORIG_CMP = None


def capture_originals() -> None:
    """Remember the torch implementations this module falls back to."""
    global _ORIG_ORI, _ORIG_CMP
    from vllm_ascend.attention import dsa_v41

    if _ORIG_ORI is None and dsa_v41.kv8_ori_plane is not fused_ori_plane:
        _ORIG_ORI = dsa_v41.kv8_ori_plane
    if _ORIG_CMP is None and dsa_v41.DeepseekV41EagerAttentionImpl._kv8_cmp_plane is not fused_cmp_plane:
        _ORIG_CMP = dsa_v41.DeepseekV41EagerAttentionImpl._kv8_cmp_plane
