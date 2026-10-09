"""Group-local V4.1 slot preparation; dynamic coordinates are read every call."""
import triton
import triton.language as tl


@triton.jit
def slot_mapping_kernel(slots, positions, query_start, output,
                        n, actual_reqs, actual_tokens,
                        BLOCK_SIZE: tl.constexpr, RATIO: tl.constexpr,
                        COMPRESSED: tl.constexpr, HAS_POSITIONS: tl.constexpr,
                        SKIP: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    raw = tl.load(slots + i, i < n, -1)
    valid = raw >= 0
    physical = tl.maximum(raw, 0)
    if COMPRESSED and RATIO != 1:
        valid = valid & ((physical + 1) % RATIO == 0)
        physical = physical // RATIO
    if COMPRESSED and RATIO == 2:
        if SKIP:
            valid = tl.full((B,), False, tl.int1)
        else:
            end = tl.minimum(tl.load(query_start + actual_reqs), actual_tokens)
            valid = valid & (i < end)
            if HAS_POSITIONS:
                pos = tl.load(positions + i, i < n, 0)
                valid = valid & (pos % 2 == 1)
    tl.store(output + 2 * i, tl.where(valid, physical // BLOCK_SIZE, -1), i < n)
    tl.store(output + 2 * i + 1, tl.where(valid, physical % BLOCK_SIZE, -1), i < n)


def prepare_slots(builder, common, positions, n, actual_reqs, actual_tokens,
                  compressed, ratio, block_size, skip):
    assert ratio in (1, 2) and block_size > 0
    slots = common.slot_mapping
    assert slots.is_contiguous() and builder._slot_mapping_2d.is_contiguous()
    slot_mapping_kernel[(triton.cdiv(n, 128),)](
        slots, positions if positions is not None else slots,
        common.query_start_loc, builder._slot_mapping_2d,
        n, actual_reqs, actual_tokens, block_size, ratio, compressed,
        positions is not None, skip, 128)
    return builder._slot_mapping_2d[:n]


@triton.jit
def ring_counts_kernel(query, seq_lens, blocks, ring, nr, actual_reqs, actual_tokens,
                       BLOCK_STRIDE: tl.constexpr, SKIP: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0) * B + tl.arange(0, B)
    start = tl.load(query + r, r < nr, 0)
    end = tl.load(query + r + 1, r < nr, 0)
    length = tl.load(seq_lens + r, r < nr, 0)
    used = tl.maximum(tl.minimum(end, actual_tokens) - start, 0)
    used = tl.where(r < actual_reqs, used, 0)
    if SKIP: used = tl.full((B,), 0, tl.int32)
    owner = tl.load(blocks + r * BLOCK_STRIDE, r < nr, 0)
    tl.store(ring + r, tl.maximum(length - (end - start), 0), r < nr)
    tl.store(ring + nr + r, used, r < nr)
    tl.store(ring + 2 * nr + r, start, r < nr)
    tl.store(ring + 3 * nr + r, start, r < nr)
    tl.store(ring + 4 * nr + r, tl.where(used > 0, owner, 0), r < nr)


@triton.jit
def ring_sources_kernel(positions, query, full_cos, full_sin, complete_out, source_out,
                        cos_out, sin_out, n, actual_reqs, actual_tokens,
                        D: tl.constexpr, HAS_ROPE: tl.constexpr, SKIP: tl.constexpr,
                        BT: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    pos = tl.load(positions + t, t < n, 0)
    end = tl.minimum(tl.load(query + actual_reqs), actual_tokens)
    complete = (pos % 2 == 1) & (t < end)
    if SKIP: complete = tl.full((BT,), False, tl.int1)
    source = tl.where(complete, pos - 1, 0)
    tl.store(complete_out + t, complete, t < n)
    tl.store(source_out + t, source, t < n)
    if HAS_ROPE:
        d = tl.arange(0, BD)
        cos = tl.load(full_cos + source[:, None] * D + d[None, :],
                      (t[:, None] < n) & (d[None, :] < D), 0)
        sin = tl.load(full_sin + source[:, None] * D + d[None, :],
                      (t[:, None] < n) & (d[None, :] < D), 0)
        tl.store(cos_out + t[:, None] * D + d[None, :], cos,
                 (t[:, None] < n) & (d[None, :] < D))
        tl.store(sin_out + t[:, None] * D + d[None, :], sin,
                 (t[:, None] < n) & (d[None, :] < D))


def prepare_ring(builder, common, positions, seq_lens, nr, actual_reqs, actual_tokens,
                 n, skip, full_cos, full_sin):
    ring_counts_kernel[(triton.cdiv(nr, 128),)](
        common.query_start_loc, seq_lens, common.block_table_tensor, builder._c2_ring_metadata,
        nr, actual_reqs, actual_tokens, common.block_table_tensor.stride(0), skip, 128)
    has_rope = full_cos is not None and full_sin is not None
    if has_rope:
        assert full_cos.is_contiguous() and full_sin.is_contiguous()
        assert full_cos.ndim == 4 and full_cos.shape[1:3] == (1, 1)
        assert full_cos.shape == full_sin.shape
        dim = full_cos.shape[-1]
    else: dim = 1
    ring_sources_kernel[(triton.cdiv(n, 8),)](
        positions, common.query_start_loc, full_cos if has_rope else positions,
        full_sin if has_rope else positions, builder._c2_complete_mask, builder._c2_source_positions,
        builder._c2_source_cos, builder._c2_source_sin, n, actual_reqs, actual_tokens,
        dim, has_rope, skip, 8, triton.next_power_of_2(dim))
