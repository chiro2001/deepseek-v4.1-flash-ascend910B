#!/usr/bin/env python3
"""One-device, model-free consumer of real CED Mooncake KV payloads.

Run under ced_npu_lock.py. This process initializes an Ascend context but
loads no model/vLLM engine. It reads all eight producer ranks into a bounded
host buffer and sends DONE_RECVING only after all data have been consumed.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


GET_META = b"get_meta_msg"
DONE_RECVING = b"done_recving_msg"


def page_plan(metadata: dict, params: dict):
    """Return bounded/validated wire spans in stable logical payload order."""
    segments = metadata.get("mock_segments")
    blocks = params["remote_block_ids"]
    prefix = int(params["ced_prefix_tokens"])
    if not segments or len(blocks) != 12 or prefix < 1:
        raise ValueError("Mock requires CED P exact-payload metadata and 12 groups")
    if tuple(params.get("ced_missing_swa_groups", ())) != (7, 8, 9, 10, 11):
        raise ValueError("Incompatible CED missing-group contract")
    for segment in segments:
        group = int(segment["group"])
        if not 0 <= group < len(blocks) or group in (7, 8, 9, 10, 11):
            raise ValueError("Invalid/uncomputed KV group in mock metadata")
        stride, size = int(segment["stride"]), int(segment["page_bytes"])
        if not 0 < size <= stride or int(segment["base"]) <= 0:
            raise ValueError("Invalid KV byte geometry")
        unit = int(segment["tokens_per_block"])
        if unit <= 0:
            raise ValueError("Invalid KV block size")
        logical_start = max(0, math.ceil(prefix / unit) - len(blocks[group]))
        for position, block in enumerate(blocks[group]):
            block = int(block)
            if not 0 <= block < int(segment["num_blocks"]):
                raise ValueError("Producer block ID exceeds registered KV allocation")
            # Circular state and partially written last pages are consumed,
            # but do not enter the reusable-prefix consistency fingerprint.
            compare = bool(segment["prefix_cacheable"]) and (
                logical_start + position < prefix // unit
            )
            yield {
                "remote": int(segment["base"]) + block * stride,
                "size": size,
                "group": group,
                "component": segment["component"],
                "logical_block": logical_start + position,
                "compare": compare,
            }


class MockConsumer:
    def __init__(self, host: str, device: int, buffer_bytes: int, timeout_ms: int):
        import msgspec
        import torch
        import torch_npu  # noqa: F401
        import zmq
        from mooncake.engine import TransferEngine

        torch.npu.set_device(device)
        self.torch = torch
        self.device = device
        self.engine = TransferEngine()
        ret = self.engine.initialize(host, "P2PHANDSHAKE", "ascend", "")
        if ret != 0:
            raise RuntimeError(f"Mooncake mock initialize failed: {ret}")
        self.buffer = torch.empty(buffer_bytes, dtype=torch.uint8, device="cpu")
        ret = self.engine.register_memory(self.buffer.data_ptr(), buffer_bytes)
        if ret != 0:
            raise RuntimeError(f"Mooncake mock host registration failed: {ret}")
        self.buffer_bytes = buffer_bytes
        self.timeout_ms = timeout_ms
        self.lock = threading.Lock()
        self.zmq = zmq
        self.context = zmq.Context()
        self.encode = msgspec.msgpack.encode
        self.decode = msgspec.msgpack.decode
        self.sockets = {}
        print(json.dumps({"mock_ready": True, "device": device,
                          "host_buffer_bytes": buffer_bytes,
                          "rpc_port": self.engine.get_rpc_port()}), flush=True)

    def socket(self, host, port):
        # A request failure leaves a REQ socket in its send/recv state. Drop
        # it on failure instead of retrying DONE_RECVING with an unknown ACK.
        key = (host, port)
        if key not in self.sockets:
            sock = self.context.socket(self.zmq.REQ)
            sock.setsockopt(self.zmq.LINGER, 0)
            sock.setsockopt(self.zmq.SNDTIMEO, self.timeout_ms)
            sock.setsockopt(self.zmq.RCVTIMEO, self.timeout_ms)
            sock.connect(f"tcp://{host}:{port}")
            self.sockets[key] = sock
        return self.sockets[key]

    def exchange(self, host, port, payload):
        sock = self.socket(host, port)
        try:
            sock.send(self.encode(payload))
            return sock.recv()
        except Exception:
            self.sockets.pop((host, port)).close()
            raise

    def consume(self, params, hold_ms=0):
        with self.lock:
            self.torch.npu.set_device(self.device)
            return self._consume(params, hold_ms)

    def _consume(self, params, hold_ms):
        if int(params.get("remote_ptp_size", 0)) != 8:
            raise ValueError("Mock expects eight P tensor-parallel ranks")
        if not params.get("do_remote_prefill") or params.get("ced_replay_tokens") != 128:
            raise ValueError("Mock requires a finished CED P handoff")
        if not 0 <= hold_ms <= 60000:
            raise ValueError("Mock hold must be in [0, 60000] ms")
        start = time.perf_counter()
        rank_results = []
        endpoints = []
        for rank in range(8):
            mapping = (params.get("remote_multi_nodes_meta_mapping") or {}).get(str(rank), {})
            host = mapping.get("host", params["remote_host"])
            port = int(params["remote_port"]) + rank
            metadata = self.decode(self.exchange(host, port, (GET_META, "")))
            expected_engine = mapping.get("engine_id", params["remote_engine_id"])
            if metadata["engine_id"] != expected_engine:
                raise ValueError("Producer engine identity changed during mock consumption")
            session = f"{host}:{metadata['te_rpc_port']}"
            rows = list(page_plan(metadata, params))
            hashes = {}
            nonzero = {}
            byte_count, read_seconds, hash_seconds = 0, 0.0, 0.0
            offset = 0
            batch = []

            def flush():
                nonlocal offset, batch, byte_count, read_seconds, hash_seconds
                if not batch:
                    return
                before = time.perf_counter()
                ret = self.engine.batch_transfer_sync_read(
                    session,
                    [self.buffer.data_ptr() + where for where, row in batch],
                    [row["remote"] for _, row in batch],
                    [row["size"] for _, row in batch],
                )
                read_seconds += time.perf_counter() - before
                if ret < 0:
                    raise RuntimeError(f"Mock Mooncake read failed for rank {rank}: {ret}")
                before = time.perf_counter()
                array = self.buffer.numpy()
                for where, row in batch:
                    payload = array[where:where + row["size"]]
                    group = row["group"]
                    nonzero[group] = nonzero.get(group, False) or bool(payload.any())
                    if row["compare"]:
                        key = f"g{group}/{row['component']}"
                        digest = hashes.setdefault(key, hashlib.sha256())
                        digest.update(row["logical_block"].to_bytes(8, "little"))
                        digest.update(payload)
                    byte_count += row["size"]
                hash_seconds += time.perf_counter() - before
                offset, batch = 0, []

            for row in rows:
                if row["size"] > self.buffer_bytes:
                    raise ValueError("Mock host buffer is smaller than one KV page")
                if offset + row["size"] > self.buffer_bytes:
                    flush()
                batch.append((offset, row))
                offset += row["size"]
            flush()
            if not nonzero.get(0, False):
                raise RuntimeError("Mock read only zeros from global KV; data consumption is unproven")
            rank_results.append({
                "rank": rank, "bytes": byte_count, "read_s": read_seconds,
                "hash_s": hash_seconds, "nonzero_groups": sorted(g for g, hit in nonzero.items() if hit),
                "fingerprints": {key: digest.hexdigest() for key, digest in hashes.items()},
            })
            endpoints.append((host, port))
        if hold_ms:
            time.sleep(hold_ms / 1000)
        # All ranks' reads and consistency fingerprints succeeded. Until this
        # point there is no DONE_RECVING signal, even after a partial failure.
        ack_start = time.perf_counter()
        for host, port in endpoints:
            ack = self.exchange(host, port, (DONE_RECVING, params["remote_request_id"], {}))
            if ack != b"ACK":
                raise RuntimeError("Mock received an invalid consumption ACK")
        result = {
            "ok": True, "request_id": params["remote_request_id"], "ranks": rank_results,
            "bytes": sum(row["bytes"] for row in rank_results),
            "consume_s": ack_start - start, "ack_s": time.perf_counter() - ack_start,
            "total_s": time.perf_counter() - start, "hold_ms": hold_ms,
        }
        print(json.dumps(result), flush=True)
        return result


def serve(consumer, host, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status = 200 if self.path == "/health" else 404
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b"ced-mock-decode" if status == 200 else b"unknown")

        def do_POST(self):
            try:
                if self.path != "/consume":
                    raise ValueError("Only /consume is supported")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ValueError("Invalid request body length")
                body = json.loads(self.rfile.read(length))
                result = consumer.consume(body["kv_transfer_params"], int(body.get("hold_ms", 0)))
                status = 200
            except Exception as exc:
                result, status = {"ok": False, "error": repr(exc)}, 500
                print(json.dumps(result), flush=True)
            encoded = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    # Ascend TE and its device context are initialized on this thread. Keep
    # consumption on the same thread rather than moving calls between fresh
    # HTTP worker contexts. The bounded receiver was already serialized.
    HTTPServer((host, port), Handler).serve_forever()


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--engine-host", default="127.0.0.1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19191)
    parser.add_argument("--buffer-mib", type=int, default=64)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--params", type=Path)
    args = parser.parse_args()
    if not 1 <= args.buffer_mib <= 1024:
        parser.error("--buffer-mib must be in [1, 1024]")
    consumer = MockConsumer(args.engine_host, args.device, args.buffer_mib * 1048576, args.timeout_ms)
    if args.params:
        document = json.loads(args.params.read_text())
        consumer.consume(document.get("kv_transfer_params", document))
    else:
        serve(consumer, args.host, args.port)


if __name__ == "__main__":
    main()
