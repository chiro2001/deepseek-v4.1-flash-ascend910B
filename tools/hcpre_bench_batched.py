#!/usr/bin/env python3
"""HcPre 批量微基准：**去掉每调用同步**，量真实的单次 device 成本。

动机：profile（tp8-class，decode 档）显示 HcPre 每步 40 次、合计 3.279 ms(profile)
⇒ 单次约 82 µs。但 decode 档 T≈48，数据量只有 ~1.6 MB ⇒ 按 1182 GB/s 只需 ~1.4 µs。
若实测单次仍 ~80 µs，说明**固定开销主导**，40 层/步 ≈ 3.2 ms 是纯浪费。
做法：一次提交 K 次调用再统一同步，总时间 / K。
"""
import statistics
import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

DEV = "npu:0"
HM = 4
H = 5120
MIX_HC = (HM + 2) * HM
K = 40          # 模拟一步的 40 层调用


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


op = torch.ops._C_ascend.npu_hc_pre_v2
print("HcPre batched: per-call us (K=%d calls per measurement, no per-call sync)" % K)
print("%6s %6s %12s %12s %14s" % ("T", "iters", "med_us", "min_us", "MB/call"))
for T in (48, 96, 512):
    x = torch.randn((T, HM, H), dtype=torch.bfloat16, device=DEV) * 0.1
    fn_w = torch.randn((MIX_HC, HM * H), dtype=torch.float32, device=DEV) * 0.02
    scale = torch.randn((3,), dtype=torch.float32, device=DEV)
    base = torch.randn((MIX_HC,), dtype=torch.float32, device=DEV) * 0.1
    pre_mix = torch.zeros((T, HM), dtype=torch.float32, device=DEV)
    mb = (T * HM * H * 2 + HM * H * 4) / 1e6
    for it in (1, 20):
        def call(x=x, fn_w=fn_w, scale=scale, base=base, pre_mix=pre_mix, it=it):
            return op(x, fn_w, scale, base, pre_mix,
                      hc_mult=HM, hc_sinkhorn_iters=it,
                      norm_eps=1e-6, hc_eps=1e-6)

        med, mn = bench_batch(call)
        print("%6d %6d %12.1f %12.1f %14.2f" % (T, it, med, mn, mb))

print()
print("=> 若 T=48 的单次 ~几十 us，则 40 层/步 ≈ 数 ms，属固定开销主导")
