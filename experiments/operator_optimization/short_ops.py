"""Semantics-preserving small-token routing and HC post candidates."""
import torch
import triton
import triton.language as tl


@triton.jit
def route_init_kernel(x, ids, expanded, reverse, counts, D: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    d = block * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + d, d < D, 0)
    tl.store(expanded + d, value, d < D)
    tl.store(expanded + D + d, value, d < D)
    if block == 0:
        a = tl.load(ids); b = tl.load(ids + 1)
        tl.store(reverse, tl.where(a <= b, 0, 1))
        tl.store(reverse + 1, tl.where(a <= b, 1, 0))
        e = tl.arange(0, 8)
        number = (e == a).to(tl.int64) + (e == b).to(tl.int64)
        tl.store(counts + e, number)


def route_init(x, ids, block=512):
    assert x.shape == (1, 5120) and ids.shape == (1, 2)
    expanded = torch.empty((2, 5120), dtype=x.dtype, device=x.device)
    reverse = torch.empty((2,), dtype=torch.int32, device=x.device)
    counts = torch.empty((8,), dtype=torch.int64, device=x.device)
    route_init_kernel[(triton.cdiv(5120, block),)](x, ids, expanded, reverse, counts,
                                                5120, block, enable_fp_fusion=False)
    return expanded, reverse, counts, None


@triton.jit
def route_combine_kernel(x, reverse, probs, y, D: tl.constexpr, BLOCK: tl.constexpr):
    d = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    row0 = tl.load(reverse); row1 = tl.load(reverse + 1)
    p0 = tl.load(probs).to(tl.float32); p1 = tl.load(probs + 1).to(tl.float32)
    a = tl.load(x + row0*D + d, d < D, 0).to(tl.float32)
    b = tl.load(x + row1*D + d, d < D, 0).to(tl.float32)
    tl.store(y + d, a*p0 + b*p1, d < D)


def route_combine(x, reverse, probs, block=512):
    assert x.shape == (2, 5120) and reverse.shape == (2,) and probs.shape == (1, 2)
    y = torch.empty((1, 5120), dtype=x.dtype, device=x.device)
    route_combine_kernel[(triton.cdiv(5120, block),)](x, reverse, probs, y,
                                                   5120, block, enable_fp_fusion=False)
    return y


@triton.jit
def hc_post_kernel(x, residual, post, comb, y, D: tl.constexpr, BLOCK: tl.constexpr,
                   FMA: tl.constexpr):
    d = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    xv = tl.load(x + d, d < D, 0).to(tl.float32)
    r0 = tl.load(residual + d, d < D, 0).to(tl.float32)
    r1 = tl.load(residual + D + d, d < D, 0).to(tl.float32)
    r2 = tl.load(residual + 2*D + d, d < D, 0).to(tl.float32)
    r3 = tl.load(residual + 3*D + d, d < D, 0).to(tl.float32)
    for h in tl.static_range(4):
        c0 = tl.load(comb + h); c1 = tl.load(comb + 4 + h)
        c2 = tl.load(comb + 8 + h); c3 = tl.load(comb + 12 + h)
        p = tl.load(post + h)
        # DAV_2201 native starts with x*post, then adds residual rows 0..3.
        value = xv*p
        if FMA:
            value = tl.fma(r0, c0, value)
            value = tl.fma(r1, c1, value)
            value = tl.fma(r2, c2, value)
            value = tl.fma(r3, c3, value)
        else:
            value = value + r0*c0
            value = value + r1*c1
            value = value + r2*c2
            value = value + r3*c3
        tl.store(y + h*D + d, value, d < D)


def hc_post(x, residual, post, comb, block=512, fma=False):
    assert x.numel() == 5120 and residual.numel() == 20480
    assert post.numel() == 4 and comb.numel() == 16
    y = torch.empty_like(residual)
    hc_post_kernel[(triton.cdiv(5120, block),)](x, residual, post, comb, y, 5120, block,
                                              fma, enable_fp_fusion=False)
    return y
