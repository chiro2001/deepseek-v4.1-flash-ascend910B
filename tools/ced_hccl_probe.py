#!/usr/bin/env python3
"""Small TP8 all-reduce check inside the experiment's locked container."""

from __future__ import annotations

import argparse
import datetime
import json
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401


def rank_probe(rank, world_size, port):
    torch.set_num_threads(1)
    torch.npu.set_device(rank)
    dist.init_process_group(
        "hccl", init_method=f"tcp://127.0.0.1:{port}", world_size=world_size,
        rank=rank, timeout=datetime.timedelta(seconds=90),
    )
    value = torch.full((16,), rank + 1, dtype=torch.float32, device=f"npu:{rank}")
    for iteration in range(3):
        value.fill_(rank + 1)
        dist.all_reduce(value)
        expected = world_size * (world_size + 1) / 2
        torch.testing.assert_close(value.cpu(), torch.full((16,), expected), rtol=0, atol=0)
    print(json.dumps({"rank": rank, "hccl_ok": True, "rounds": 3, "sum": expected}), flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(rank_probe, args=(args.count, port), nprocs=args.count, join=True)


if __name__ == "__main__":
    main()
