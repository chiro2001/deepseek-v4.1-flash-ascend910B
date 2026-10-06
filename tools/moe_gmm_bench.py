#!/usr/bin/env python3
"""MoE grouped_matmul：耗时是否随**活跃专家数**线性增长？（= 权重带宽受限的判据）

背景（本轮要解决的未知量）：
  · 每步 MoE 权重若按"48 个本地专家全读"算 = 34 GB ⇒ 纯流式下界 26 ms，
    与 conc=1 实测 24.79 ms 相符；但 profile 里 MoE 只有 4.87 ms ⇒ 需 7 TB/s，不可能。
  · 若按"只读命中专家"算 ≈ 4.5/层/rank ⇒ 3.2 GB/步 ⇒ 2.5 ms @1300GB/s，**与 profile 自洽**。
⇒ 用固定 token 数、只改**活跃组数 g** 的 grouped_matmul 微基准来判定。

形状（真实）：H=5120, I=2304 ⇒ w1=[E,H,2I], w2=[E,I,H]；E=48（EP=8 下的本地专家数）。
"""
import statistics
import time

import torch
import torch_npu  # noqa: F401

DEV = "npu:0"
E = 48
H = 5120
I = 2304


def bench(fn, warmup=3, iters=12):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts), min(ts)


print("=== grouped_matmul(w1) : 固定 1 token/组，只改活跃组数 g ===")
print("w1=[%d,%d,%d] bf16 ⇒ 每专家 %.1f MB   （int4 实际为 %.1f MB）"
      % (E, H, 2 * I, H * 2 * I * 2 / 1e6, H * 2 * I * 0.5 / 1e6))
print("%8s %10s %12s %12s %14s" % ("g", "tokens", "med_us", "min_us", "有效GB/s"))
w1 = torch.randn((E, H, 2 * I), dtype=torch.bfloat16, device=DEV) * 0.02
for g in (1, 2, 4, 8, 16, 48):
    counts = torch.zeros(E, dtype=torch.int64, device=DEV)
    counts[:g] = 1
    x = torch.randn((g, H), dtype=torch.bfloat16, device=DEV) * 0.1
    try:
        def call(x=x, counts=counts, w1=w1):
            return torch_npu.npu_grouped_matmul(
                x=[x], weight=[w1], split_item=2, group_type=0,
                group_list_type=1, group_list=counts,
            )[0]

        call()
        torch.npu.synchronize()
        med, mn = bench(call)
        by = g * H * 2 * I * 2
        print("%8d %10d %12.1f %12.1f %14.1f" % (g, g, med, mn, by / med / 1e3))
    except Exception as exc:  # noqa: BLE001
        print("%8d  失败: %s" % (g, repr(exc)[:140]))

print()
print("=== 对照：g=4 但把 token 数从 4 提到 64（同活跃组、更多 token）===")
w1b = w1
for T in (4, 16, 64, 256):
    counts = torch.zeros(E, dtype=torch.int64, device=DEV)
    counts[:4] = T // 4
    x = torch.randn((T, H), dtype=torch.bfloat16, device=DEV) * 0.1

    def call2(x=x, counts=counts, w1=w1b):
        return torch_npu.npu_grouped_matmul(
            x=[x], weight=[w1], split_item=2, group_type=0,
            group_list_type=1, group_list=counts,
        )[0]

    med, mn = bench(call2)
    print("  T=%4d med %8.1f us  min %8.1f us" % (T, med, mn))
