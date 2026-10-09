#!/usr/bin/env python3
"""Cross-process Ascend TE read, with no model and bounded allocations.

Run in a container whose locked visible devices are [receiver, producer].
This isolates the mock receive path from vLLM and its cache geometry.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import multiprocessing as mp
import time


def producer(queue, finished, nbytes, hostname):
    faulthandler.enable(all_threads=True)
    import torch
    import torch_npu  # noqa: F401
    from mooncake.engine import TransferEngine

    torch.npu.set_device(1)
    engine = TransferEngine()
    assert engine.initialize(hostname, "P2PHANDSHAKE", "ascend", "") == 0
    tensor = torch.arange(nbytes // 4, dtype=torch.int32).to("npu:1")
    torch.npu.synchronize()
    assert engine.register_memory(tensor.data_ptr(), nbytes) == 0
    queue.put({"host": hostname, "rpc_port": engine.get_rpc_port(), "ptr": tensor.data_ptr(), "bytes": nbytes})
    print(json.dumps({"producer_ready": True, "device": 1, "bytes": nbytes}), flush=True)
    if not finished.wait(120):
        raise TimeoutError("Pair consumer did not finish")
    assert engine.unregister_memory(tensor.data_ptr()) == 0
    del engine


def receiver(queue, finished, nbytes, destination, hostname, copy_ops):
    faulthandler.enable(all_threads=True)
    import torch
    import torch_npu  # noqa: F401
    from mooncake.engine import TransferEngine

    metadata = queue.get(timeout=60)
    torch.npu.set_device(0)
    engine = TransferEngine()
    assert engine.initialize(hostname, "P2PHANDSHAKE", "ascend", "") == 0
    tensor = torch.empty(nbytes // 4, dtype=torch.int32,
                         device="cpu" if destination == "host" else "npu:0")
    assert engine.register_memory(tensor.data_ptr(), nbytes) == 0
    chunk = (nbytes // copy_ops // 4) * 4
    offsets = [i * chunk for i in range(copy_ops)]
    lengths = [chunk] * (copy_ops - 1) + [nbytes - offsets[-1]]
    timings = []
    for iteration in range(3):
        started = time.perf_counter()
        print(json.dumps({"before_read": True, "destination": destination, "round": iteration}), flush=True)
        ret = engine.batch_transfer_sync_read(
            f"{metadata['host']}:{metadata['rpc_port']}",
            [tensor.data_ptr() + offset for offset in offsets],
            [metadata["ptr"] + offset for offset in offsets], lengths,
        )
        timings.append(time.perf_counter() - started)
        assert ret >= 0, ret
        host = tensor.cpu()
        torch.testing.assert_close(host, torch.arange(nbytes // 4, dtype=torch.int32), rtol=0, atol=0)
        digest = hashlib.sha256(host.numpy().tobytes()).hexdigest()
        print(json.dumps({"pair_ok": True, "destination": destination, "round": iteration,
                          "bytes": nbytes, "seconds": timings[-1], "sha256": digest}), flush=True)
    finished.set()
    assert engine.unregister_memory(tensor.data_ptr()) == 0
    del engine


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--destination", choices=("host", "npu"), default="host")
    parser.add_argument("--mib", type=int, default=1)
    parser.add_argument("--source-host", default="127.0.0.1")
    parser.add_argument("--receiver-host", default="127.0.0.1")
    parser.add_argument("--copy-ops", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.mib <= 64:
        parser.error("Use a bounded 1–64 MiB buffer")
    if not 1 <= args.copy_ops <= (args.mib << 20) // 4:
        parser.error("Each copy must include at least one int32")
    context = mp.get_context("spawn")
    queue, finished = context.Queue(), context.Event()
    source = context.Process(target=producer, args=(queue, finished, args.mib << 20, args.source_host))
    target = context.Process(target=receiver, args=(queue, finished, args.mib << 20, args.destination,
                                                 args.receiver_host, args.copy_ops))
    source.start()
    target.start()
    try:
        target.join(90)
        if target.is_alive():
            raise TimeoutError("Pair receive exceeded 90 seconds")
        finished.set()
        source.join(10)
        return 0 if target.exitcode == 0 and source.exitcode == 0 else 1
    finally:
        for child in (source, target):
            if child.is_alive():
                child.terminate()
                child.join(10)
        queue.close()


if __name__ == "__main__":
    raise SystemExit(main())
