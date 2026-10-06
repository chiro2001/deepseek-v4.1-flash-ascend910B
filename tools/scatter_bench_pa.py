#!/usr/bin/env python3
"""Scatter 对照微基准 #3：正确调用签名的 paged-KV 写算子。

`npu::npu_scatter_pa_kv_cache_functional(key, value, key_cache, value_cache,
                                          slot_mapping, *, cache_mode="PA_NZ")`
"""
import statistics
import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

DEV = "npu:0"
D = 512
PAGE = 128
NB = 4096


def bench(fn, warmup=5, iters=40):
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


def pa_bench(T, mode, dk=D, dv=D):
    kc = torch.zeros((NB, PAGE, 1, dk), dtype=torch.bfloat16, device=DEV)
    vc = torch.zeros((NB, PAGE, 1, dv), dtype=torch.bfloat16, device=DEV)
    key = torch.ones((T, 1, dk), dtype=torch.bfloat16, device=DEV)
    val = torch.ones((T, 1, dv), dtype=torch.bfloat16, device=DEV)
    slots = torch.arange(T, dtype=torch.int32, device=DEV)
    fn = torch.ops.npu.npu_scatter_pa_kv_cache_functional
    try:
        def call():
            return fn(key, val, kc, vc, slots, cache_mode=mode)
        med, mn = bench(call)
        return med, mn, None
    except Exception as exc:  # noqa: BLE001
        return None, None, repr(exc)[:180]


print("=== npu_scatter_pa_kv_cache_functional ===")
for mode in ("PA_NZ", "PA", "PA_BNSD", "PA_BLK_BNSD"):
    for T in (8, 64):
        med, mn, err = pa_bench(T, mode)
        if err:
            print("  mode=%-12s T=%3d  FAILED: %s" % (mode, T, err))
            break
        print("  mode=%-12s T=%3d  med %7.1f us  min %7.1f us" % (mode, T, med, mn))
    else:
        continue
    continue
