#!/usr/bin/env python3
"""Run only inside the experiment's locked, device-mapped container."""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import torch
import torch_npu  # noqa: F401


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--mooncake", action="store_true")
    args = parser.parse_args()
    results = []
    for device in range(args.count):
        start = time.perf_counter()
        try:
            torch.npu.set_device(device)
            x = torch.arange(4096, dtype=torch.float32, device=f"npu:{device}")
            y = (x * 2 + 1).cpu()
            torch.testing.assert_close(y, torch.arange(4096, dtype=torch.float32) * 2 + 1)
            matrix = torch.ones((128, 128), dtype=torch.float16, device=f"npu:{device}")
            product = (matrix @ matrix).cpu()
            torch.testing.assert_close(product, torch.full((128, 128), 128, dtype=torch.float16), rtol=0, atol=0)
            results.append({"device": device, "ok": True, "stages": ["vector", "matmul"],
                            "seconds": time.perf_counter() - start})
        except Exception as exc:
            results.append({"device": device, "ok": False, "error": repr(exc)})
        print(json.dumps(results[-1]), flush=True)
    if not all(row["ok"] for row in results):
        return 1
    if args.mooncake:
        from mooncake.engine import TransferEngine

        torch.npu.set_device(0)
        engine = TransferEngine()
        assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "ascend", "") == 0
        source = torch.arange(256, dtype=torch.int32).to("npu:0")
        host = torch.zeros(256, dtype=torch.int32)
        size = source.numel() * source.element_size()
        assert engine.register_memory(source.data_ptr(), size) == 0
        assert engine.register_memory(host.data_ptr(), size) == 0
        ret = engine.batch_transfer_sync_read(
            f"127.0.0.1:{engine.get_rpc_port()}",
            [host.data_ptr()], [source.data_ptr()], [size],
        )
        assert ret >= 0, ret
        torch.testing.assert_close(host, torch.arange(256, dtype=torch.int32))
        print(json.dumps({"mooncake_host_read": True, "bytes": size,
                          "sha256": hashlib.sha256(host.numpy().tobytes()).hexdigest()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
