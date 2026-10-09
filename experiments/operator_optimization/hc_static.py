"""HC candidate: pre-round static fn weights; BF16 x is already HF32 exact."""
import torch
import triton
import triton.language as tl

from hc_vector import finish_hc


def round_hf32(fn):
    assert fn.dtype == torch.float32 and fn.is_contiguous()
    return ((fn.view(torch.int32) + 2048) & -4096).view(torch.float32)


@triton.jit
def project_static(x, fn, dots, squares, K: tl.constexpr, PARTS: tl.constexpr,
                   BK: tl.constexpr):
    n = tl.program_id(0)
    part = tl.program_id(1)
    k = tl.arange(0, BK)
    size = tl.cdiv(K, PARTS)
    dot = tl.full((), 0, tl.float32)
    sq = tl.full((), 0, tl.float32)
    for start in range(tl.cdiv(size, BK)):
        offset = part * size + start * BK + k
        valid = (offset < K) & (start * BK + k < size)
        xv = tl.load(x + offset, valid, 0).to(tl.float32)
        fv = tl.load(fn + n * K + offset, valid, 0)
        dot += tl.sum(xv * fv, 0)
        if n == 0:
            sq += tl.sum(xv * xv, 0)
    tl.store(dots + n * PARTS + part, dot)
    if n == 0:
        tl.store(squares + part, sq)


def hc_static(x, rounded_fn, scale, base, pre_mix=None, *, parts=1, bk=4096,
              by=1024, hc_sinkhorn_iters=20, norm_eps=1e-20, hc_eps=1e-6):
    assert x.shape == (1, 4, 5120) and x.dtype == torch.bfloat16
    assert rounded_fn.shape == (24, 20480) and rounded_fn.dtype == torch.float32
    dots = torch.empty((24, parts), dtype=torch.float32, device=x.device)
    squares = torch.empty((parts,), dtype=torch.float32, device=x.device)
    y = torch.empty((1, 5120), dtype=x.dtype, device=x.device)
    post = torch.empty((1, 4), dtype=torch.float32, device=x.device)
    comb = torch.empty((1, 4, 4), dtype=torch.float32, device=x.device)
    pre = torch.empty((1, 4), dtype=torch.float32, device=x.device)
    project_static[(24, parts)](x, rounded_fn, dots, squares, 20480, parts, bk,
                               enable_fp_fusion=False)
    finish_hc[(triton.cdiv(5120, by),)](
        x, dots, squares, scale, base, pre_mix if pre_mix is not None else pre,
        y, post, comb, pre, 5120, parts, by, hc_sinkhorn_iters, pre_mix is not None,
        norm_eps, hc_eps, enable_fp_fusion=False)
    return y, post, comb, pre
