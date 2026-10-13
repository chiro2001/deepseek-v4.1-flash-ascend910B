#!/usr/bin/env python3
"""8 卡 HCCL **小消息 allreduce 延迟**微基准（torchrun 启动）。

动机（2026-10-05 实测）：decode 主图的关键路径上**每层有 2 次 `hcom_allReduce_` 完全暴露**
（`Add → HcPost` 的空隙 = allreduce 自身时长，事件开销仅 ~1 µs），
40 层 × 2 = 80 次/步 ≈ **2.3 ms/步（9%）**。
本脚本用来回答："这 29 µs 还能不能压下去" —— 换 HCCL 环境变量重跑同一负载。

消息形状取自生产实测：
  * `[6, 5120]` bf16 = **61 KB**（每层的 row-parallel 归约，占 80 次/步）
  * `[6, 129280]` bf16 = 1.5 MB（LM head 的 vocab-parallel 归约，1 次/步）

用法（在 8 张空闲卡上）：
  torchrun --nproc_per_node=8 hccl_allreduce_latency.py [iters]
每个 rank 打印自己的中位/均值；rank0 再打印全局汇总。
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
WARM = 40


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world)

    if rank == 0:
        keys = ("HCCL_BUFFSIZE", "HCCL_OP_EXPANSION_MODE", "HCCL_ALGO", "HCCL_INTRA_PCIE_ENABLE",
                "HCCL_INTRA_ROCE_ENABLE", "HCCL_EXEC_TIMEOUT")
        print("[env] " + " ".join(f"{k}={os.environ.get(k, '<unset>')}" for k in keys), flush=True)

    shapes = [(6, 5120), (6, 129280)]
    for shape in shapes:
        x = torch.randn(*shape, dtype=torch.bfloat16, device="npu")
        for _ in range(WARM):
            dist.all_reduce(x)
        torch.npu.synchronize()
        samples = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            dist.all_reduce(x)
            torch.npu.synchronize()
            samples.append((time.perf_counter() - t0) * 1e6)
        p50 = statistics.median(samples)
        p10 = sorted(samples)[int(len(samples) * 0.1)]
        p90 = sorted(samples)[int(len(samples) * 0.9)]
        nbytes = shape[0] * shape[1] * 2
        # 流水化吞吐：连续 50 次不同步，再统一次同步
        for _ in range(10):
            dist.all_reduce(x)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            dist.all_reduce(x)
        torch.npu.synchronize()
        pipelined = (time.perf_counter() - t0) / 50 * 1e6
        print(
            f"[rank{rank}] shape={shape} {nbytes/1024:7.1f} KB  "
            f"p50={p50:7.2f} us  p10={p10:7.2f}  p90={p90:7.2f}  流水化={pipelined:7.2f} us",
            flush=True,
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
