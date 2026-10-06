#!/usr/bin/env python3
"""HcPre: 融合算子 vs 原生 PyTorch 实现（T=48, K=40 批量）。

原生实现（cann-recipes hc_pre_native）：
  x: [T, hc, d] -> flatten -> float
  rsqrt = rsqrt(mean(x^2,-1)+eps); mixes = linear(x, hc_fn) * rsqrt
  pre/post/comb = sinkhorn(mixes)
  y = sum(pre[...,None] * x_view, dim=-2) -> bf16
"""
import statistics
import time

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

DEV = "npu:0"
HM, H = 4, 5120
MIX_HC = (HM + 2) * HM
K = 40
EPS = 1e-6


def bench_batch(call, k=K, warmup=2, iters=15):
    for _ in range(warmup):
        for _ in range(k):
            call()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for _ in range(k):
            call()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6 / k)
    return statistics.median(ts), min(ts)


def native(x, hc_fn, hc_scale, hc_base, iters=20):
    dtype = x.dtype
    xf = x.reshape(x.shape[0], -1).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    mixes = F.linear(xf, hc_fn) * rsqrt
    pre, post, comb = mixes.split([HM, HM, HM * HM], dim=-1)
    comb = comb.unflatten(-1, (HM, HM))
    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:HM]) + EPS
    post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[HM:2 * HM])
    comb = comb * hc_scale[2] + hc_base[2 * HM:].view(HM, HM)
    comb = torch.softmax(comb, -1) + EPS
    comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    y = (pre.unsqueeze(-1) * x.float()).sum(1).to(dtype)
    return y, post, comb


for T in (48, 96, 512):
    x = torch.randn((T, HM, H), dtype=torch.bfloat16, device=DEV) * 0.1
    fn_w = torch.randn((MIX_HC, HM * H), dtype=torch.float32, device=DEV) * 0.02
    scale = torch.randn((3,), dtype=torch.float32, device=DEV)
    base = torch.randn((MIX_HC,), dtype=torch.float32, device=DEV) * 0.1
    pre_mix = torch.zeros((T, HM), dtype=torch.float32, device=DEV)
    KW = dict(hc_mult=HM, hc_sinkhorn_iters=20, norm_eps=EPS, hc_eps=EPS)

    fused = lambda: torch.ops._C_ascend.npu_hc_pre_v2(x, fn_w, scale, base, pre_mix, **KW)
    nat = lambda: native(x, fn_w, scale, base)
    m1, _ = bench_batch(fused)
    m2, _ = bench_batch(nat)
    print("T=%3d  fused %7.1f us (x40=%5.2f ms) | native %7.1f us (x40=%5.2f ms)  speedup %.2fx"
          % (T, m1, m1 * 40 / 1000, m2, m2 * 40 / 1000, m1 / m2))

# 数值一致性（相对误差）
T = 48
x = torch.randn((T, HM, H), dtype=torch.bfloat16, device=DEV) * 0.1
fn_w = torch.randn((MIX_HC, HM * H), dtype=torch.float32, device=DEV) * 0.02
scale = torch.randn((3,), dtype=torch.float32, device=DEV)
base = torch.randn((MIX_HC,), dtype=torch.float32, device=DEV) * 0.1
pre_mix = torch.zeros((T, HM), dtype=torch.float32, device=DEV)
y1, p1, c1 = torch.ops._C_ascend.npu_hc_pre_v2(x, fn_w, scale, base, pre_mix,
                                               hc_mult=HM, hc_sinkhorn_iters=20, norm_eps=EPS, hc_eps=EPS)
y2, p2, c2 = native(x, fn_w, scale, base)
print()
print("shape check: fused y=%s post=%s comb=%s | native y=%s post=%s comb=%s"
      % (tuple(y1.shape), tuple(p1.shape), tuple(c1.shape), tuple(y2.shape), tuple(p2.shape), tuple(c2.shape)))
den = y2.float().abs().mean().clamp_min(1e-6)
print("rel err y = %.3e   post = %.3e   comb = %.3e"
      % ((y1.float() - y2.float()).abs().mean() / den,
         (p1.float() - p2.float()).abs().mean() / p2.float().abs().mean().clamp_min(1e-6),
         (c1.float() - c2.float()).abs().mean() / c2.float().abs().mean().clamp_min(1e-6)))
