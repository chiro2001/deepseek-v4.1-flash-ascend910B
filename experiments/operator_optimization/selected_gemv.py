"""Small-M Cube/SIMD candidates for grouped BF16 projections.

Weights are explicit physical ND; layout conversion belongs to model loading,
never the captured decode graph. Group IDs are read on device from GMM counts.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def cube_gemv(x, w, counts, y, K: tl.constexpr, N: tl.constexpr,
              GROUPED: tl.constexpr, COUNT_TYPE: tl.constexpr,
              BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    slot = tl.program_id(1)
    expert = slot
    if GROUPED:
        ids = tl.arange(0, 8)
        ends = tl.load(counts + ids)
        if COUNT_TYPE == 1:
            ends = tl.cumsum(ends, 0)
        expert = tl.sum((ends <= slot).to(tl.int32), 0)
    m = tl.arange(0, 16)
    n = tile * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(x + (slot + m[:, None]) * K + kk[None, :],
                    (m[:, None] == 0) & (kk[None, :] < K), 0)
        b = tl.load(w + expert * K * N + kk[:, None] * N + n[None, :],
                    (kk[:, None] < K) & (n[None, :] < N), 0)
        acc = tl.dot(a, b, acc)
    tl.store(y + (slot + m[:, None]) * N + n[None, :], acc.to(tl.bfloat16),
             (m[:, None] == 0) & (n[None, :] < N))


@triton.jit
def vector_gemv(x, w, counts, y, K: tl.constexpr, N: tl.constexpr,
                GROUPED: tl.constexpr, COUNT_TYPE: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr, NK_LAYOUT: tl.constexpr):
    tile = tl.program_id(0)
    slot = tl.program_id(1)
    expert = slot
    if GROUPED:
        ids = tl.arange(0, 8)
        ends = tl.load(counts + ids)
        if COUNT_TYPE == 1:
            ends = tl.cumsum(ends, 0)
        expert = tl.sum((ends <= slot).to(tl.int32), 0)
    n = tile * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BN,), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(x + slot * K + kk, kk < K, 0).to(tl.float32)
        if NK_LAYOUT:
            b = tl.load(w + expert * K * N + n[:, None] * K + kk[None, :],
                        (n[:, None] < N) & (kk[None, :] < K), 0).to(tl.float32)
            acc += tl.sum(b * a[None, :], 1)
        else:
            b = tl.load(w + expert * K * N + kk[:, None] * N + n[None, :],
                        (kk[:, None] < K) & (n[None, :] < N), 0).to(tl.float32)
            acc += tl.sum(b * a[:, None], 0)
    tl.store(y + slot * N + n, acc, n < N)


def gemv(x, w, *, counts=None, count_type=1, kind='cube', bn=64, bk=256,
          nk_layout=False):
    assert x.ndim == 2 and x.dtype == w.dtype == torch.bfloat16
    assert x.is_contiguous() and w.is_contiguous()
    k = x.shape[1]
    n = w.shape[-2] if nk_layout else w.shape[-1]
    assert w.shape[-1 if nk_layout else -2] == k
    y = torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)
    group = counts is not None
    if group:
        assert counts.shape == (8,) and x.shape[0] == 2 and count_type in [0, 1]
    fn = cube_gemv if kind == 'cube' else vector_gemv
    extra = {} if kind == 'cube' else {'NK_LAYOUT': nk_layout}
    assert kind != 'cube' or not nk_layout
    fn[(triton.cdiv(n, bn), x.shape[0])](
        x, w, counts if group else x, y, k, n, group, count_type, bn, bk,
        **extra, enable_fp_fusion=False)
    return y
