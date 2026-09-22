"""KV8 *prefill* rebuild as Triton-Ascend kernels.

logs/015 5.3 rejected the per-query-row rebuild (Q_T x topk rows) for chunked
prefill.  logs/018 replaced it with a page-granular whole-prefix rebuild that
keeps the real logical indices, but left it in torch.  Measured (logs/033):
that torch chain costs ~1.4 ms *per layer* in eager, and the number is flat in
context length (8 -> 136 pages) => it is host-dispatch bound, exactly like the
decode rebuild was before logs/026/028 fused it.

These kernels are the prefill counterparts of ``kv8_fuse_triton`` / logs/028:
one launch rebuilds a whole run of pages (``PPR`` pages per request) into the
scratch plane and writes the scratch block table in the same launch.

  * ``kv8_prefill_swa_kernel`` - the sliding-window plane.  The rebuilt span is
    ``query_len + window`` rows starting at ``window_start``; source pages come
    from the *real* ``ori_block_table`` (absolute logical block -> physical
    page), destination pages are ``b * PPR + j``.
  * ``kv8_prefill_cmp_kernel`` - the compressed long-KV plane.  The whole
    compressed prefix is rebuilt, source pages are ``cmp_block_table[b, j]``
    and the scratch table is the identity.

Both keep every row at its original in-page offset, so the operator keeps
addressing rows with the real logical indices (``_kv8_cmp_plane`` prefill
branch) - no renumbering, no causal-mask change.
"""
import torch
import triton  # noqa: F401
import triton.language as tl


@triton.jit
def kv8_prefill_swa_kernel(
    src_ptr, sc_ptr, out_ptr, src_tab_ptr, dst_tab_ptr, fb_ptr, nb_ptr,
    PAGE_S: tl.constexpr, ROW_S: tl.constexpr,
    PAGE_C: tl.constexpr, ROW_C: tl.constexpr,
    BLOCK: tl.constexpr, DIM: tl.constexpr, GROUPS: tl.constexpr,
    WIDTH: tl.constexpr, WPOW: tl.constexpr, PPR: tl.constexpr,
    BR: tl.constexpr,
):
    b = tl.program_id(0)
    j = tl.program_id(1)
    rb = tl.program_id(2)
    fb = tl.load(fb_ptr + b)
    nb = tl.load(nb_ptr + b)
    phys = tl.load(src_tab_ptr + b * WIDTH + tl.minimum(fb + j, WIDTH - 1))
    rows = rb * BR + tl.arange(0, BR)
    cols = tl.arange(0, DIM)
    x = tl.load(src_ptr + phys * PAGE_S + rows[:, None] * ROW_S + cols[None, :])
    s = tl.load(sc_ptr + phys * PAGE_C + rows[:, None] * ROW_C
                + (cols // (DIM // GROUPS))[None, :])
    y = (x.to(tl.float32) * s.to(tl.float32)).to(tl.bfloat16)
    dst = ((b * PPR + j) * BLOCK + rows)[:, None] * DIM + cols[None, :]
    tl.store(out_ptr + dst, y)
    tcols = tl.arange(0, WPOW)
    delta = tcols - fb
    newt = tl.where((delta >= 0) & (delta < nb), b * PPR + delta, 0)
    tl.store(dst_tab_ptr + b * WIDTH + tcols, newt.to(tl.int32),
             mask=(tcols < WIDTH) & (rb == 0) & (j == 0))


@triton.jit
def kv8_prefill_cmp_kernel(
    src_ptr, sc_ptr, out_ptr, src_tab_ptr, dst_tab_ptr, nb_ptr,
    PAGE_S: tl.constexpr, ROW_S: tl.constexpr,
    PAGE_C: tl.constexpr, ROW_C: tl.constexpr,
    BLOCK: tl.constexpr, DIM: tl.constexpr, GROUPS: tl.constexpr,
    WIDTH: tl.constexpr, PPR: tl.constexpr, PPOW: tl.constexpr,
    TSTRIDE: tl.constexpr,
    BR: tl.constexpr,
):
    b = tl.program_id(0)
    j = tl.program_id(1)
    rb = tl.program_id(2)
    nb = tl.load(nb_ptr + b)
    phys = tl.load(src_tab_ptr + b * WIDTH + tl.minimum(j, nb - 1))
    rows = rb * BR + tl.arange(0, BR)
    cols = tl.arange(0, DIM)
    x = tl.load(src_ptr + phys * PAGE_S + rows[:, None] * ROW_S + cols[None, :])
    s = tl.load(sc_ptr + phys * PAGE_C + rows[:, None] * ROW_C
                + (cols // (DIM // GROUPS))[None, :])
    y = (x.to(tl.float32) * s.to(tl.float32)).to(tl.bfloat16)
    dst = ((b * PPR + j) * BLOCK + rows)[:, None] * DIM + cols[None, :]
    tl.store(out_ptr + dst, y)
    tcols = tl.arange(0, PPOW)
    tl.store(dst_tab_ptr + b * TSTRIDE + tcols, (b * PPR + tcols).to(tl.int32),
             mask=(tcols < PPR) & (rb == 0) & (j == 0))


PF_BR = 32
LAUNCH = [True]          # harness switch: measure the wrapper without the launch
_BUF: dict = {}
_SCRATCH: dict = {}


def _buf(key, shape, dtype, device):
    t = _BUF.get(key)
    if t is None or tuple(t.shape) != tuple(shape) or t.dtype != dtype:
        t = torch.empty(shape, dtype=dtype, device=device)
        _BUF[key] = t
    return t


def _scratch(blocks, block_size, dim, device, role):
    """Reusable BF16 scratch plane.

    ``role`` is part of the key on purpose.  The shipped
    ``dsa_v41.kv8_scratch_plane`` keys only on ``(blocks, block_size, dim)``, so
    the window rebuild and the compressed rebuild **share one buffer whenever
    their page counts happen to be equal** - and since both are written before
    the attention operator reads them, the second rebuild silently overwrites
    the first.  It is invisible at 32K (18 vs 136 pages) and reproducible by
    shrinking the context: with a 512-row chunk both need 6 pages and the
    operator output drifts by ~1.1 (see logs/033 p10 C arm).
    """
    key = (role, blocks, block_size, dim)
    t = _SCRATCH.get(key)
    if t is None:
        t = torch.empty((blocks, block_size, 1, dim), dtype=torch.bfloat16, device=device)
        _SCRATCH[key] = t
    return t


def fused_ori_plane(kv_i8, kv_scale, query_start_loc, seq_lens, block_table,
                    num_reqs, query_rows, window):
    """Prefill counterpart of ``dsa_v41.kv8_ori_plane`` (page granular, eager)."""
    if query_rows == num_reqs:
        raise RuntimeError("decode shape: use the logs/028 SWA kernel")
    bs, dim = kv_i8.shape[1], kv_i8.shape[-1]
    groups = kv_scale.shape[-1]
    width = block_table.shape[1]
    lens = seq_lens[:num_reqs].to(torch.int64)
    q_len = (query_start_loc[1: num_reqs + 1] - query_start_loc[:num_reqs]).to(torch.int64)
    span = torch.minimum(lens, q_len + window)
    window_start = lens - span
    first_block = (window_start // bs).to(torch.int32)
    nb = ((lens - 1) // bs - first_block.long() + 1).to(torch.int32)
    ppr = int(nb.max().item())
    scratch = _scratch(num_reqs * ppr, bs, dim, kv_i8.device, "swa")
    table = _buf("pf_swa_tab", (num_reqs, width), torch.int32, kv_i8.device)
    if LAUNCH[0]:
        kv8_prefill_swa_kernel[(num_reqs, ppr, bs // PF_BR)](
            kv_i8, kv_scale, scratch, block_table, table, first_block, nb,
            PAGE_S=int(kv_i8.stride(0)), ROW_S=int(kv_i8.stride(1)),
            PAGE_C=int(kv_scale.stride(0)), ROW_C=int(kv_scale.stride(1)),
            BLOCK=bs, DIM=dim, GROUPS=groups, WIDTH=width,
            WPOW=triton.next_power_of_2(width), PPR=ppr, BR=PF_BR)
    return scratch, table


def fused_cmp_plane(self, kv_i8, kv_scale, indices, block_table, num_reqs, cache_seq_lens):
    """Prefill counterpart of ``_kv8_cmp_plane``'s page-granular branch."""
    bs, dim = kv_i8.shape[1], kv_i8.shape[-1]
    groups = kv_scale.shape[-1]
    used = int(cache_seq_lens[:num_reqs].max().item())
    nblocks = max(1, -(-used // bs))
    scratch = _scratch(num_reqs * nblocks, bs, dim, kv_i8.device, "cmp")
    table = _buf("pf_cmp_tab", (num_reqs, nblocks), torch.int32, kv_i8.device)
    nb = _buf("pf_cmp_nb", (num_reqs,), torch.int32, kv_i8.device)
    nb.fill_(nblocks)
    if LAUNCH[0]:
        kv8_prefill_cmp_kernel[(num_reqs, nblocks, bs // PF_BR)](
            kv_i8, kv_scale, scratch, block_table, table, nb,
            PAGE_S=int(kv_i8.stride(0)), ROW_S=int(kv_i8.stride(1)),
            PAGE_C=int(kv_scale.stride(0)), ROW_C=int(kv_scale.stride(1)),
            BLOCK=bs, DIM=dim, GROUPS=groups, WIDTH=int(block_table.shape[1]),
            PPR=nblocks, PPOW=triton.next_power_of_2(nblocks), TSTRIDE=nblocks, BR=PF_BR)
    return scratch, table, indices


# --------------------------------------------------------------------- v2
# v1 pays a host-device round trip (``nb.max().item()``) plus ~8 tensor ops per
# layer, which lands on the critical path of an eager prefill step.  v2 moves
# the index arithmetic into the kernel (it reads ``seq_lens`` and
# ``query_start_loc`` directly, exactly like the decode kernels) and takes the
# page count as a host-side upper bound, so a layer issues **one** launch.


@triton.jit
def kv8_prefill_swa_kernel2(
    src_ptr, sc_ptr, out_ptr, src_tab_ptr, dst_tab_ptr, lens_ptr, qsl_ptr,
    PAGE_S: tl.constexpr, ROW_S: tl.constexpr,
    PAGE_C: tl.constexpr, ROW_C: tl.constexpr,
    BLOCK: tl.constexpr, DIM: tl.constexpr, GROUPS: tl.constexpr,
    WIDTH: tl.constexpr, WPOW: tl.constexpr, PPR: tl.constexpr,
    WINDOW: tl.constexpr, BR: tl.constexpr,
):
    b = tl.program_id(0)
    j = tl.program_id(1)
    rb = tl.program_id(2)
    length = tl.load(lens_ptr + b)
    q_len = tl.load(qsl_ptr + b + 1) - tl.load(qsl_ptr + b)
    span = tl.minimum(length, q_len + WINDOW)
    first_block = (length - span) // BLOCK
    nb = (length - 1) // BLOCK - first_block + 1
    phys = tl.load(src_tab_ptr + b * WIDTH + tl.minimum(first_block + j, WIDTH - 1))
    rows = rb * BR + tl.arange(0, BR)
    cols = tl.arange(0, DIM)
    x = tl.load(src_ptr + phys * PAGE_S + rows[:, None] * ROW_S + cols[None, :])
    s = tl.load(sc_ptr + phys * PAGE_C + rows[:, None] * ROW_C
                + (cols // (DIM // GROUPS))[None, :])
    y = (x.to(tl.float32) * s.to(tl.float32)).to(tl.bfloat16)
    dst = ((b * PPR + j) * BLOCK + rows)[:, None] * DIM + cols[None, :]
    tl.store(out_ptr + dst, y)
    tcols = tl.arange(0, WPOW)
    delta = tcols - first_block
    newt = tl.where((delta >= 0) & (delta < nb), b * PPR + delta, 0)
    tl.store(dst_tab_ptr + b * WIDTH + tcols, newt.to(tl.int32),
             mask=(tcols < WIDTH) & (rb == 0) & (j == 0))


def fused_ori_plane2(kv_i8, kv_scale, query_start_loc, seq_lens, block_table,
                     num_reqs, query_rows, window, max_q_len=None):
    """v2: no host sync, no tensor arithmetic - one launch per layer."""
    bs, dim = kv_i8.shape[1], kv_i8.shape[-1]
    groups = kv_scale.shape[-1]
    width = block_table.shape[1]
    if max_q_len is None:
        q_len = (query_start_loc[1: num_reqs + 1] - query_start_loc[:num_reqs])
        max_q_len = int(q_len.max().item())
    ppr = -(-(max_q_len + window) // bs) + 1
    scratch = _scratch(num_reqs * ppr, bs, dim, kv_i8.device, "swa")
    table = _buf("pf_swa_tab2", (num_reqs, width), torch.int32, kv_i8.device)
    if LAUNCH[0]:
        kv8_prefill_swa_kernel2[(num_reqs, ppr, bs // PF_BR)](
            kv_i8, kv_scale, scratch, block_table, table, seq_lens, query_start_loc,
            PAGE_S=int(kv_i8.stride(0)), ROW_S=int(kv_i8.stride(1)),
            PAGE_C=int(kv_scale.stride(0)), ROW_C=int(kv_scale.stride(1)),
            BLOCK=bs, DIM=dim, GROUPS=groups, WIDTH=width,
            WPOW=triton.next_power_of_2(width), PPR=ppr, WINDOW=window, BR=PF_BR)
    return scratch, table


@triton.jit
def kv8_prefill_cmp_kernel3(
    src_ptr, sc_ptr, out_ptr, src_tab_ptr, dst_tab_ptr, nb_ptr,
    PAGE_S: tl.constexpr, ROW_S: tl.constexpr,
    PAGE_C: tl.constexpr, ROW_C: tl.constexpr,
    BLOCK: tl.constexpr, DIM: tl.constexpr, GROUPS: tl.constexpr,
    WIDTH: tl.constexpr, PPR: tl.constexpr, PPOW: tl.constexpr,
    BR: tl.constexpr,
):
    b = tl.program_id(0)
    j = tl.program_id(1)
    rb = tl.program_id(2)
    nb = tl.load(nb_ptr + b)                      # live pages of *this* request
    phys = tl.load(src_tab_ptr + b * WIDTH + tl.minimum(j, nb - 1))
    rows = rb * BR + tl.arange(0, BR)
    cols = tl.arange(0, DIM)
    # NOTE: no per-program mask here.  A scalar-predicated load/store turns the
    # vector ops into predicated ones and cost 5x (measured 351 -> 1655 us per
    # visit), so the wrapper keeps ``ppr`` equal to the live page count instead
    # and duplicates nothing.
    x = tl.load(src_ptr + phys * PAGE_S + rows[:, None] * ROW_S + cols[None, :])
    s = tl.load(sc_ptr + phys * PAGE_C + rows[:, None] * ROW_C
                + (cols // (DIM // GROUPS))[None, :])
    y = (x.to(tl.float32) * s.to(tl.float32)).to(tl.bfloat16)
    dst = ((b * PPR + j) * BLOCK + rows)[:, None] * DIM + cols[None, :]
    tl.store(out_ptr + dst, y)
    tcols = tl.arange(0, PPOW)
    newt = tl.where(tcols < nb, b * PPR + tcols, 0)
    tl.store(dst_tab_ptr + b * PPR + tcols, newt.to(tl.int32),
             mask=(tcols < PPR) & (rb == 0) & (j == 0))


def fused_cmp_plane3(self, kv_i8, kv_scale, indices, block_table, num_reqs, cache_seq_lens,
                     ppr=None):
    """v3: page count comes from the host (scheduler-side lengths), no ``.item()``.

    ``ppr`` is the number of compressed pages the scratch holds per request.  It
    only has to be an *upper bound* on the live prefix (``cache_seq_lens/128``);
    the kernel derives the live count from the device tensor, so a too-large
    ``ppr`` costs bandwidth, never correctness.
    """
    bs, dim = kv_i8.shape[1], kv_i8.shape[-1]
    groups = kv_scale.shape[-1]
    if ppr is None:
        # ``block_table.shape[1]`` is the pages *allocated* for the session, a
        # host-visible shape (no D2H): a valid upper bound on the live prefix,
        # and the kernel masks the pages past the live count, so an over-large
        # bound costs nothing but idle programs.  Reading the live count with
        # ``.item()`` instead costs a host sync per layer, which in eager
        # prefill is on the step's critical path (measured: +1.6 ms/step).
        ppr = max(1, int(block_table.shape[1]))
    nb = _buf("pf_cmp_nb3", (num_reqs,), torch.int32, kv_i8.device)
    nb.copy_((cache_seq_lens[:num_reqs].to(torch.int32) + (bs - 1)) // bs)
    scratch = _scratch(num_reqs * ppr, bs, dim, kv_i8.device, "cmp")
    table = _buf("pf_cmp_tab3", (num_reqs, ppr), torch.int32, kv_i8.device)
    if LAUNCH[0]:
        kv8_prefill_cmp_kernel3[(num_reqs, ppr, bs // PF_BR)](
            kv_i8, kv_scale, scratch, block_table, table, nb,
            PAGE_S=int(kv_i8.stride(0)), ROW_S=int(kv_i8.stride(1)),
            PAGE_C=int(kv_scale.stride(0)), ROW_C=int(kv_scale.stride(1)),
            BLOCK=bs, DIM=dim, GROUPS=groups, WIDTH=int(block_table.shape[1]),
            PPR=ppr, PPOW=triton.next_power_of_2(ppr), BR=PF_BR)
    return scratch, table, indices
