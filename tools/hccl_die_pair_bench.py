#!/usr/bin/env python3
"""SIO（同卡两 die）vs HCCS_SW（跨卡）的集合通信微基准 —— **用图捕获剥掉主机开销**。

背景：A3（910C）一张卡是两个 die 合封，`npu-smi info -t topo` 显示
  * 同卡两 die = **SIO**
  * 跨卡     = **HCCS_SW**（经 HCCS 交换芯片）
本脚本回答"同卡是否更快"，并给出**按 rank 顺序摆位**的对照。

方法：把一个图里放 CAP 次 allreduce，捕获后重放 REPLAY 次 ⇒
per_call = 总时间 / (CAP × REPLAY)，主机只下发一次图重放，所以这是**设备侧**代价。
（对照过 CAP=20/100/400：per_call 稳定，确认不是主机下发受限。）

用法：
  # 2 ranks（同一容器内选两个 die）
  DEVS=0,1 SIZES=30,3072,12288,65536 CAP=50 REPLAY=5 \
    torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
             --master_addr=127.0.0.1 --master_port=29591 hccl_die_pair_bench.py

  # 8 ranks + rank 顺序对照（DEVS 的第 i 项 = 第 i 个 rank 用哪个 die）
  DEVS=0,1,2,3,4,5,6,7     ... --nproc_per_node=8   # 自然：同卡两 die 相邻
  DEVS=0,2,4,6,1,3,5,7     ... --nproc_per_node=8   # 交错：同卡两 die 相隔 4

环境：ASCEND_RT_VISIBLE_DEVICES 决定容器内可见 die 的顺序；容器需挂 /usr/local/Ascend/driver。
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

CAP = int(os.environ.get("CAP", "20"))
REPLAY = int(os.environ.get("REPLAY", "20"))
SIZES = [int(x) for x in os.environ.get(
    "SIZES", "1,16,256,1024,2048,4096,8192,16384,32768,65536").split(",")]


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    devs = os.environ.get("DEVS")
    dev = devs.split(",")[rank] if devs else os.environ.get("DEV0" if rank == 0 else "DEV1", "0")
    label = os.environ.get("PAIR_LABEL", "?")
    torch.npu.set_device(f"npu:{dev}")
    dist.init_process_group(backend="hccl", rank=rank, world_size=world)

    if rank == 0:
        print(f"[pair] label={label} mode=graph cap={CAP} replay={REPLAY}", flush=True)
        print("%12s %12s %14s %12s" % ("elems", "nbytes", "per_call_us", "BW_GB/s"), flush=True)

    for n in SIZES:
        x = torch.randn(n, 1024, dtype=torch.bfloat16, device=f"npu:{dev}")
        nb = x.numel() * 2
        for _ in range(10):          # 预热，触发 HCCL 建链
            dist.all_reduce(x)
        torch.npu.synchronize()
        g = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(g):
                for _ in range(CAP):
                    dist.all_reduce(x)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPLAY):
                g.replay()
            torch.npu.synchronize()
            per = (time.perf_counter() - t0) / (CAP * REPLAY)
            if rank == 0:
                print("%12d %12d %14.2f %12.1f" % (n, nb, per * 1e6, nb / per / 1e9), flush=True)
        except Exception as exc:  # noqa: BLE001
            if rank == 0:
                print("%12d %12d  graph 失败: %s" % (n, nb, str(exc)[:80]), flush=True)
        finally:
            del g
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
