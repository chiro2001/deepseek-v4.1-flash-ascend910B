#!/usr/bin/env python3
"""8 卡 HCCL allreduce **带宽/延迟扫描**（torchrun 启动）。

为什么需要：本仓实测 decode 的 allreduce 每步 81 次、合计 2.2 ms（N=1）/ **4.7 ms（N=8）**，
而单次时长在服务里是 **32–73 µs**。要判断这是"链路带宽差"还是"小消息延迟"，
必须看**同一集合通信在不同消息大小下的耗时曲线**：
  * 若大消息带宽正常（~100+ GB/s）而小消息慢 ⇒ **延迟问题**（只能减少调用次数）；
  * 若大消息带宽也只有 ~10 GB/s ⇒ **链路/算法问题**（可能可配置修复）。

注意：本脚本测的是**主机下发 + 设备执行**的总时间。要分离两者，看 `pipelined` 列：
连续 N 次不结巴地入队，若每次仍 ~100 µs，说明**设备侧**就是这个速度；
若 pipelined 明显低于 per-call，说明单次测的是主机开销。

用法（8 张空闲卡）：
  torchrun --nproc_per_node=8 hccl_bw_sweep.py [iters]
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
WARM = 5
SHAPES = [
    ("60 KB",  (30, 1024)),      # 30×1024×2 = 61 KB
    ("491 KB", (240, 1024)),     # 240×1024×2 = 491 KB
    ("2 MB",   (1024, 1024)),
    ("8 MB",   (4096, 1024)),
    ("32 MB",  (16384, 1024)),
    ("128 MB", (65536, 1024)),
]


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(backend="hccl", rank=rank, world_size=world)
    if rank == 0:
        print("[env] " + " ".join(
            f"{k}={os.environ.get(k, '<unset>')}" for k in
            ("HCCL_BUFFSIZE", "HCCL_OP_EXPANSION_MODE", "HCCL_INTRA_PCIE_ENABLE", "HCCL_INTRA_ROCE_ENABLE")
        ), flush=True)
        print("%-8s %10s %12s %12s %10s %12s" % ("size", "nbytes", "per_call_us", "pipelined_us", "BW_GB/s", "inBW_GB/s"))
    for label, shape in SHAPES:
        x = torch.randn(*shape, dtype=torch.bfloat16, device="npu")
        nbytes = x.numel() * 2
        for _ in range(WARM):
            dist.all_reduce(x)
        torch.npu.synchronize()
        for _ in range(10):
            dist.all_reduce(x)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            dist.all_reduce(x)
        torch.npu.synchronize()
        pipelined = (time.perf_counter() - t0) / ITERS
        if rank == 0:
            print("%-8s %10d %12s %12.1f %10.1f %12.1f" % (
                label, nbytes, "-", pipelined * 1e6,
                nbytes / pipelined / 1e9, 2 * (world - 1) / world * nbytes / pipelined / 1e9), flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
