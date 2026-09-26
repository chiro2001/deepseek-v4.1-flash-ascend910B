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

--------------------------------------------------------------------------
[FUSE_MULTIROW] 多行推广（本文件相对 `agents/KV8_fuse/kv8_fuse_triton.py`
md5 `6ce00b8f6fdd9ba4ad5935876601f8d6` 的唯一改动）

背景：生产是 **spec-decode**（`query_rows = num_reqs x (1+sptok)`，本臂 6 行 / 1 请求，
`sptok=5`）。原版两道 guard —— `query_rows != num_reqs`（原 `:226`）与
`rows != num_reqs`（原 `:268`）—— 把生产形状整体退回 torch 版 ⇒ **收益 0**；
而且原版回退调用只有 8 / 6 个位置实参，**丢 `rows_bound` / `graph_safe`** ⇒ 接到
S_graphfix 的 graphsafe 版上会让 spec 形状重新落进 legacy prefill 支的
`cache_seq_lens.max().item()` ⇒ 捕获期 `Not_Supported(EE1016)`。

推广后的"多行"在 **kernel 里怎么表达**（这是本次唯一要回答的机制问题）：

  * SWA：行映射**完全不动** —— 每个 program 仍然是「一个 (scratch 页, 页内行块)」。
    "多行"只改两件事：① scratch 页数/请求 `PP` 从常量 2 变成 host 上界算出的
    `pages_per_req`（生产 = `min(width, (rows_bound+window-1)//block_size + 2)` = 3）；
    ② band 起点从纯 decode 的 `max(len-window,0)>>LBS` 换成 graphsafe 的
    `(len - min(len, q_len+window))>>LBS`，其中 `q_len` 由 `query_start_loc` 差分
    **在 kernel 内**算（device 侧、无 D2H）。栅格因此是
    `(num_reqs, PP x (block_size // ROWS))`，**不是**"每 block 多处理 R 行"。
  * cmp：`rows` 在这个 kernel 里**本来就是**扁平选择索引 `q*TOPK + t`
    （`req = rows >> LTOPK` 得到的是 **query 行号**，不是请求号）。多行只需要
    ① `block_table` 的行索引由 `b = q // reps` 的**预计算 int32 偏移**给出
    （kernel 里不能出现非 2 幂的 `//`：Triton-Ascend 的地址 pass 拒绝）；
    ② 重编号写回扁平行号 `q*topk+t`（单行时写 `t`，两者由各自的表补偿），
    ③ 段表改成"全局 identity 段表"（每行相同）。

★ 只放宽 guard 是**错的**：`page = table_ptr[req*width + blk]` 在
`num_reqs=1, rows=6` 时会去读 `block_table` 的第 q 行（越界）。

★ 两个新分支都与 `S_graphfix` 的 graphsafe torch 支**语义逐位等价**
（SWA：同一 PP 公式、同一 `first_block` 公式、同一页内容；
cmp：同一私有段映射 `i*topk+t`、同一 identity 段表）—— 等价性由
`out/verify_rowmap.py`（纯 Python 整数字典序对拍）离线证明。
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
    qsl_ptr,              # int32/int64 [num_reqs+1] query_start_loc
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
    USE_QLEN: tl.constexpr,
):
    """One program per (scratch page, row block): the whole window rebuild.

    [FUSE_MULTIROW] 栅格 = `(num_reqs, PP * RB)`：
    `req = program_id(0)`（请求号），`pq = program_id(1)` ⇒
    `p = pq // RB`（scratch 页）、`rb = pq % RB`（页内行块）。
    这与单行版的 1-D 线性化 `((req*PP)+p)*RB + rb` 逐字相同；两条 div/mod 的除数
    都是常量幂次 `RB`（生产 64），`PP=3` 不参与 div/mod —— 避开 Triton-Ascend
    地址 pass 对非幂次除法的拒绝（单行版 `slot//PP` 能过只是因为 PP=2）。
    """
    GD: tl.constexpr = DIM // GROUP
    req = tl.program_id(0)
    pq = tl.program_id(1)
    p = pq // RB
    rb = pq % RB
    length = tl.load(lens_ptr + req)
    if USE_QLEN:
        # graphsafe / spec-decode：band 起点用**真实** per-request query 长度
        # （`query_start_loc` 的差分，device 侧，无 host 同步、无 `.item()`）。
        q_len = tl.load(qsl_ptr + req + 1) - tl.load(qsl_ptr + req)
        span = tl.minimum(length, q_len + window)
    else:
        # 纯 decode（1 行/请求）：window_start = clamp(len - window, 0)。
        span = tl.minimum(length, window)
    first = tl.maximum(length - span, 0) >> LBS
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
    ooff = ((req * PP + p) * BS + rows)[:, None, None] * DIM + g[None, :, None] * GD + d[None, None, :]
    tl.store(scratch_ptr + ooff, out)


@triton.jit
def _kv8_swa_table_kernel(
    out_ptr,              # int32 [num_reqs, width]
    lens_ptr,
    qsl_ptr,
    width,
    window,
    BS: tl.constexpr,
    PP: tl.constexpr,
    LBS: tl.constexpr,
    BLOCK_W: tl.constexpr,
    USE_QLEN: tl.constexpr,
):
    """Logical block -> scratch page, one program per request (whole row)."""
    req = tl.program_id(0)
    length = tl.load(lens_ptr + req)
    if USE_QLEN:
        q_len = tl.load(qsl_ptr + req + 1) - tl.load(qsl_ptr + req)
        span = tl.minimum(length, q_len + window)
    else:
        span = tl.minimum(length, window)
    first = tl.maximum(length - span, 0) >> LBS
    bpr = ((length - 1) >> LBS) - first + 1
    base = req * PP
    for start in range(0, width, BLOCK_W):
        col = start + tl.arange(0, BLOCK_W)
        d = col - first
        # `d < PP` 是**上界的显式保护**：`rows_bound` 成立时 `bpr <= PP`（bound 论证见
        # S_graphfix `kv8_ori_plane` docstring），此时与 torch 支逐字相同；万一调用方
        # 给的上界偏小，越界页会被指向 page 0（惰性）而不是读到 scratch 之外。
        val = tl.where((d >= 0) & (d < bpr) & (d < PP), base + d, 0)
        tl.store(out_ptr + req * width + col, val.to(tl.int32), mask=col < width)


# ------------------------------------------------------------------ cmp plane
@triton.jit
def _kv8_cmp_rows_kernel(
    payload_ptr,          # int8 plane [P, BS, 1, DIM]
    scale_ptr,            # fp16 plane [P, BS, 1, G]
    idx_ptr,              # int32 [rows*TOPK] (sparse indices, -1 = unused)
    roff_ptr,             # int32 [rows]  (row offset into block_table; MULTI only)
    table_ptr,            # int32 [num_reqs, width] (real block table)
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
    MULTI: tl.constexpr,
):
    """Index math + gather + dequant + renumber for the selected cmp rows.

    [FUSE_MULTIROW] `rows` 是扁平选择索引 `q*TOPK + t`；`req = rows >> LTOPK`
    因此是 **query 行号 q**（不是请求号）。多行时：

      * `block_table` 的行由 `roff_ptr` 给出（`(q // reps) * width`，host 侧预计算、
        与输入无关，缓存）—— 单行时仍是原来的 `req * width`（编译期选择，那段
        IR 与 026 实测版逐字节相同）；
      * 重编号写回 `q*TOPK + t`（= 本 program 写的 scratch 扁平行号），段表用
        全局 identity —— 与 S_graphfix graphsafe 支的
        `renumbered = i*topk+t` + identity 段表逐位一致。

    单行时（rows == num_reqs）保持 026 原语义：重编号写 `t`，段表
    `table[b, c] = b*per_req + c` ⇒ 算子算出的全局行仍是 `b*topk + t`，与本 kernel
    的写行号相同（两种约定的等价性见 `out/verify_rowmap.py` Part B）。
    """
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
    if MULTI:
        roff = tl.load(roff_ptr + req, mask=mask, other=0)
        page = tl.load(table_ptr + roff + blk, mask=mask, other=0)
    else:
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
    if MULTI:
        renum = tl.where(valid, rows, -1)
    else:
        renum = tl.where(valid, t, -1)
    tl.store(renum_ptr + rows, renum.to(tl.int32), mask=mask)


# ------------------------------------------------------------------- wrappers
_TABLE_CACHE: dict[tuple, torch.Tensor] = {}
_IDENT_CACHE: dict[tuple, torch.Tensor] = {}
_ROFF_CACHE: dict[tuple, torch.Tensor] = {}

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


def _static_identity_table(num_reqs: int, segments: int, device) -> torch.Tensor:
    """``[num_reqs, segments]`` 的全局 identity 段表（多行 cmp 用，输入无关）。

    与 graphsafe torch 支的
    ``arange(segments).view(1, segments).expand(num_reqs, segments).contiguous()``
    **逐位相同**，只是缓存起来 ⇒ 热路径 0 个 torch 算子。
    """
    key = (num_reqs, segments, str(device))
    table = _IDENT_CACHE.get(key)
    if table is None:
        table = (
            torch.arange(segments, dtype=torch.int32, device=device)
            .view(1, segments)
            .expand(num_reqs, segments)
            .contiguous()
        )
        _IDENT_CACHE[key] = table
    return table


def _row_offsets(rows: int, reps: int, width: int, device) -> torch.Tensor:
    """``(q // reps) * width`` -- query 行号 -> block_table 行偏移（输入无关）。"""
    key = (rows, reps, width, str(device))
    off = _ROFF_CACHE.get(key)
    if off is None:
        off = (torch.arange(rows, dtype=torch.int32, device=device) // reps) * width
        _ROFF_CACHE[key] = off
    return off


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
    rows_bound=None,
):
    """Drop-in for ``dsa_v41.kv8_ori_plane`` -- **decode 形状（含 spec-decode）**。

    [FUSE_MULTIROW] 两条融合支：

      * ``rows_bound`` 非 None（graphsafe 的 decode 支，spec-decode 走这条）：
        ``PP = min(width, max(1, (rows_bound + window - 1)//block_size + 2))``
        —— 与 S_graphfix 逐字同式（生产 ``rows_bound=6`` ⇒ **PP=3**）；
        band 起点在 kernel 内按 ``query_start_loc`` 差分算。
      * ``rows_bound`` None 且 ``query_rows == num_reqs``（纯 decode，026 实测路径）：
        ``PP = 2``、band 起点 ``max(len-window,0)``，与 026 的 2-kernel 路径相同。

    其余形状（真 eager prefill 等）**安全回退**：`rows_bound` 原样转发给 torch 版，
    绝不丢（丢了会让 spec 形状落进 legacy prefill 支的 ``.item()`` ⇒ 捕获 EE1016）。
    """
    from vllm_ascend.attention import dsa_v41

    block_size = int(kv_i8.shape[1])
    dim = int(kv_i8.shape[-1])
    groups = int(kv_scale.shape[-1])
    if _ORIG_ORI is None:
        capture_originals()

    def _fallback():
        return _ORIG_ORI(
            kv_i8, kv_scale, query_start_loc, seq_lens, block_table,
            num_reqs, query_rows, window, rows_bound=rows_bound,
        )

    if (
        not _supported(block_size, dim // groups, groups)
        or SWA_ROWS < 2
        or block_size % SWA_ROWS != 0
        or int(num_reqs) < 1
    ):
        return _fallback()
    if rows_bound is not None:
        bound = int(rows_bound)
        if bound < 1:
            return _fallback()
        use_qlen = True
        pages_per_req = min(
            int(block_table.shape[1]),
            max(1, (bound + int(window) - 1) // block_size + 2),
        )
    elif query_rows == num_reqs:
        use_qlen = False
        pages_per_req = 2
    else:
        # 真 eager prefill（chunk >> 1 行/请求）：几何/格子未测 ⇒ 一律走 torch/预填充线。
        return _fallback()

    _ready()
    device = kv_i8.device
    width = int(block_table.shape[1])
    rows = SWA_ROWS
    rblock = block_size // rows
    lens = seq_lens[:num_reqs]
    qsl = query_start_loc[: num_reqs + 1]
    scratch = dsa_v41.kv8_scratch_plane(
        num_reqs * pages_per_req, block_size, dim, torch.bfloat16, device)
    table = torch.empty((num_reqs, width), dtype=torch.int32, device=device)
    ps, rs, sps, srs = _plane_geometry(kv_i8, kv_scale)
    _trace(
        f"swa rows grid=({num_reqs},{pages_per_req * rblock}) pp={pages_per_req} "
        f"rows={rows} use_qlen={use_qlen} rows_bound={rows_bound} query_rows={query_rows} "
        f"scratch_pages={num_reqs * pages_per_req} geom={ps},{rs},{sps},{srs} width={width}"
    )
    _kv8_swa_rows_kernel[(num_reqs, pages_per_req * rblock)](
        kv_i8, kv_scale, lens, qsl, block_table, scratch,
        int(window), ps, rs, sps, srs, width,
        BS=block_size, PP=pages_per_req, LBS=int(block_size).bit_length() - 1,
        DIM=dim, GROUP=groups, ROWS=rows, RB=rblock, USE_QLEN=use_qlen,
        num_warps=4,
    )
    _trace(f"swa table grid=({num_reqs},) width={width} pp={pages_per_req} use_qlen={use_qlen}")
    _kv8_swa_table_kernel[(num_reqs,)](
        table, lens, qsl, width, int(window),
        BS=block_size, PP=pages_per_req, LBS=int(block_size).bit_length() - 1,
        BLOCK_W=128, USE_QLEN=use_qlen, num_warps=1,
    )
    return scratch, table


def fused_cmp_plane(
    self, kv_i8, kv_scale, indices, block_table, num_reqs, cache_seq_lens, graph_safe=False,
):
    """Drop-in for ``DeepseekV41EagerAttentionImpl._kv8_cmp_plane``.

    [FUSE_MULTIROW] 两条融合支：

      * ``rows == num_reqs``（纯 decode）：026 原语义（单行约定），无论 `graph_safe`。
      * ``rows % num_reqs == 0`` 且 ``graph_safe``（spec-decode）：每个 query 行一段私有
        scratch（`per_req*block_size == topk` 使 `i*topk+t` 恰好落在
        `(i*per_req + t//BS, t%BS)`），段表用 identity —— 与 S_graphfix graphsafe 支
        逐位一致，且**不读** ``cache_seq_lens.max()``（无 D2H ⇒ 捕获安全）。

    其余形状（eager prefill、非 2 幂 topk、非 uniform 多行但 graph_safe=False）
    **安全回退**：`graph_safe` 原样转发（丢了等于把 spec 形状塞回 legacy prefill 支）。
    """
    from vllm_ascend.attention import dsa_v41

    block_size = int(kv_i8.shape[1])
    dim = int(kv_i8.shape[-1])
    groups = int(kv_scale.shape[-1])
    rows, _, topk = indices.shape
    per_req = (topk + block_size - 1) // block_size
    if _ORIG_CMP is None:
        capture_originals()

    def _fallback():
        return _ORIG_CMP(
            self, kv_i8, kv_scale, indices, block_table, num_reqs, cache_seq_lens,
            graph_safe=graph_safe,
        )

    if (
        not _supported(block_size, dim // groups, groups, topk)
        or per_req * block_size != topk
        or int(num_reqs) < 1
        or int(rows) % int(num_reqs) != 0
        or CMP_ROWS < 2
    ):
        return _fallback()
    if rows == num_reqs:
        multi = False
    elif graph_safe:
        # ★ uniform spec-decode 前提（与 S_graphfix graphsafe 支同一前提）：批内每个请求的
        # query 行数都等于 `reps = rows // num_reqs`，否则 `q // reps` 的 row->request
        # 映射不成立。vLLM 的 spec-decode 图按 uniform query 长度捕获/重放，这个前提由
        # 图本身保证；非 uniform 的批必须走 torch 的 prefill 支（graph_safe=False）。
        multi = True
    else:
        return _fallback()

    _ready()
    device = kv_i8.device
    width = int(block_table.shape[1])
    reps = int(rows) // int(num_reqs)
    seg = per_req * block_size
    total = int(rows) * int(topk)
    rrows = CMP_ROWS if (CMP_ROWS > 1 and total % CMP_ROWS == 0) else 2
    scratch = dsa_v41.kv8_scratch_plane(rows * per_req, block_size, dim, torch.bfloat16, device)
    renum = torch.empty((rows, 1, topk), dtype=torch.int32, device=device)
    roff = _row_offsets(int(rows), reps, width, device) if multi else None
    ps, rs, sps, srs = _plane_geometry(kv_i8, kv_scale)
    _trace(
        f"cmp rows grid=({total // rrows},) rows={rrows} multi={multi} num_reqs={num_reqs} "
        f"query_rows={rows} reps={reps} segments={int(rows) * per_req} topk={topk} "
        f"geom={ps},{rs},{sps},{srs} width={width}"
    )
    _kv8_cmp_rows_kernel[(total // rrows,)](
        kv_i8, kv_scale, indices, roff, block_table, scratch, renum,
        ps, rs, sps, srs, width, total,
        SEG=seg, BS=block_size, TOPK=topk,
        LBS=int(block_size).bit_length() - 1, LTOPK=int(topk).bit_length() - 1,
        DIM=dim, GROUP=groups, ROWS=rrows, MULTI=multi,
        num_warps=4,
    )
    table = (
        _static_identity_table(int(num_reqs), int(rows) * per_req, device)
        if multi
        else _static_table(int(rows), per_req, device)
    )
    return scratch, table, renum


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
