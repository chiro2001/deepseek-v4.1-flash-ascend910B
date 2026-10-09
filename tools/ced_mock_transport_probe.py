#!/usr/bin/env python3
"""Exercise the actual mock consumer using toy KV and real TE/ZMQ.

Two locked devices: consumer=0, source=1. One source engine provides eight
virtual rank endpoints. This verifies the receiver/protocol, not real P8 KV.
"""

from __future__ import annotations

import faulthandler
import json
import multiprocessing as mp
import socket
import time


def producer(queue, finished, base_port):
    faulthandler.enable(all_threads=True)
    import msgspec
    import torch
    import torch_npu  # noqa: F401
    import zmq
    from mooncake.engine import TransferEngine

    torch.npu.set_device(1)
    engine = TransferEngine()
    assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "ascend", "") == 0
    tensor = torch.arange(1024, dtype=torch.int32).to("npu:1")
    torch.npu.synchronize()
    nbytes = tensor.numel() * tensor.element_size()
    assert engine.register_memory(tensor.data_ptr(), nbytes) == 0
    metadata = {
        "engine_id": "ced-toy", "te_rpc_port": engine.get_rpc_port(),
        "mock_registered_regions": [(tensor.data_ptr(), nbytes)],
        "mock_segments": [{"group": 0, "component": "toy.kv", "base": tensor.data_ptr(),
                           "stride": 512, "page_bytes": 256, "num_blocks": 8,
                           "tokens_per_block": 128, "prefix_cacheable": True}],
    }
    context = zmq.Context()
    poller = zmq.Poller()
    sockets = []
    for rank in range(8):
        sock = context.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://127.0.0.1:{base_port + rank}")
        sockets.append(sock)
        poller.register(sock, zmq.POLLIN)
    queue.put({"port": base_port, "rpc_port": engine.get_rpc_port()})
    ack_count = 0
    deadline = time.monotonic() + 120
    while not finished.is_set() and time.monotonic() < deadline:
        for sock, _ in poller.poll(100):
            frames = sock.recv_multipart()
            msg = msgspec.msgpack.decode(frames[-1])
            if msg[0] == b"get_meta_msg":
                data = msgspec.msgpack.encode(metadata)
            elif msg[0] == b"done_recving_msg":
                ack_count += 1
                data = b"ACK"
            else:
                raise ValueError(f"Unexpected mock protocol message {msg[0]!r}")
            sock.send_multipart([frames[0], b"", data])
    print(json.dumps({"toy_ack_count": ack_count}), flush=True)
    for sock in sockets:
        sock.close()
    context.term()
    assert engine.unregister_memory(tensor.data_ptr()) == 0
    del engine


def main():
    faulthandler.enable(all_threads=True)
    from ced_mock_decode import MockConsumer

    context = mp.get_context("spawn")
    queue, finished = context.Queue(), context.Event()
    # Reserve a contiguous local endpoint range without touching service ports.
    reservations = []
    for base_port in range(19380, 19480, 8):
        try:
            for port in range(base_port, base_port + 8):
                sock = socket.socket()
                sock.bind(("127.0.0.1", port))
                reservations.append(sock)
            break
        except OSError:
            for sock in reservations:
                sock.close()
            reservations = []
    if not reservations:
        raise RuntimeError("No free diagnostic ZMQ endpoint range")
    for sock in reservations:
        sock.close()
    source = context.Process(target=producer, args=(queue, finished, base_port))
    source.start()
    try:
        endpoint = queue.get(timeout=60)
        consumer = MockConsumer("127.0.0.1", 0, 1 << 20, 30000)
        fingerprints = []
        for iteration in range(3):
            result = consumer.consume({
                "remote_ptp_size": 8, "do_remote_prefill": True,
                "ced_replay_tokens": 128, "ced_missing_swa_groups": [7, 8, 9, 10, 11],
                "ced_prefix_tokens": 256, "remote_block_ids": [[1, 2]] + [[] for _ in range(11)],
                "remote_host": "127.0.0.1", "remote_port": endpoint["port"],
                "remote_engine_id": "ced-toy", "remote_request_id": f"toy-{iteration}",
            })
            assert result["bytes"] == 8 * 2 * 256
            fingerprints.append([row["fingerprints"] for row in result["ranks"]])
        assert fingerprints[0] == fingerprints[1] == fingerprints[2]
        print(json.dumps({"mock_transport_ok": True, "rounds": 3,
                          "virtual_ranks": 8, "real_prefill_verified": False}), flush=True)
    finally:
        finished.set()
        source.join(10)
        if source.is_alive():
            source.terminate()
            source.join(10)
        queue.close()


if __name__ == "__main__":
    main()
