#!/usr/bin/env python3
"""One-device, model-free consumer of real CED Mooncake KV payloads.

Run under ced_npu_lock.py. This process initializes an Ascend context but
loads no model/vLLM engine. It reads all eight producer ranks into a bounded
host buffer and sends DONE_RECVING only after all data have been consumed.
"""

from __future__ import annotations

import argparse
import base64
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
    regions = metadata.get("mock_registered_regions")
    blocks = params["remote_block_ids"]
    prefix = int(params["ced_prefix_tokens"])
    if not segments or not regions or len(blocks) != 12 or prefix < 1:
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
            # Null pages never prove consistency. A SWA handoff often has a
            # null full page plus the actual partial tail; compare only its
            # initialized slots using worker-verified physical geometry.
            valid_tokens = max(0, min(unit, prefix - (logical_start + position) * unit))
            compare_bytes = size if valid_tokens == unit else 0
            if group in (2, 3, 4, 5, 6) and valid_tokens < unit:
                slot_bytes = int(segment.get("slot_bytes", 0))
                if slot_bytes <= 0 or slot_bytes * unit != size:
                    raise ValueError("Mock SWA partial page lacks verified contiguous slot geometry")
                compare_bytes = valid_tokens * slot_bytes
            compare = bool(segment["prefix_cacheable"]) and block != 0 and compare_bytes > 0
            address = int(segment["base"]) + block * stride
            if not any(int(base) <= address and address + size <= int(base) + int(length)
                       for base, length in regions):
                raise ValueError("Mock payload falls outside producer's registered TE memory")
            yield {
                "remote": address,
                "size": size,
                "group": group,
                "component": segment["component"],
                "logical_block": logical_start + position,
                "compare": compare,
                "compare_bytes": compare_bytes,
            }


class MockConsumer:
    def __init__(self, host: str, device: int, buffer_bytes: int, timeout_ms: int, max_copy_ops: int = 64):
        import msgspec
        import torch
        import torch_npu  # noqa: F401
        import zmq
        from mooncake.engine import TransferEngine

        torch.npu.set_device(device)
        self.torch = torch
        self.device = device
        self.engine_class = TransferEngine
        self.engine = None
        self.engine_host = host
        self.buffer = torch.empty(buffer_bytes, dtype=torch.uint8, device="cpu")
        self.buffer_bytes = buffer_bytes
        self.timeout_ms = timeout_ms
        self.max_copy_ops = max_copy_ops
        self.lock = threading.Lock()
        self.zmq = zmq
        self.context = zmq.Context()
        self.encode = msgspec.msgpack.encode
        self.decode = msgspec.msgpack.decode
        self.sockets = {}
        if host:
            self.initialize_transport(host)
        print(json.dumps({"mock_ready": True, "device": device,
                          "host_buffer_bytes": buffer_bytes,
                          "transport_initialized": self.engine is not None}), flush=True)

    def initialize_transport(self, host):
        engine = self.engine_class()
        ret = engine.initialize(host, "P2PHANDSHAKE", "ascend", "")
        if ret != 0:
            raise RuntimeError(f"Mooncake mock initialize failed: {ret}")
        ret = engine.register_memory(self.buffer.data_ptr(), self.buffer_bytes)
        if ret != 0:
            raise RuntimeError(f"Mooncake mock host registration failed: {ret}")
        self.engine, self.engine_host = engine, host
        print(json.dumps({"mock_transport_ready": True, "engine_host": host,
                          "rpc_port": engine.get_rpc_port()}), flush=True)

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

    def consume(self, params, hold_ms=0, page_fingerprints=False, snapshot_blocks=()):
        with self.lock:
            snapshot_blocks = tuple(int(n) for n in snapshot_blocks)
            if len(snapshot_blocks) > 4 or any(n < 0 for n in snapshot_blocks):
                raise ValueError("Snapshot accepts at most four non-negative global block indices")
            self.torch.npu.set_device(self.device)
            host = params["remote_host"]
            if self.engine_host and self.engine_host != host:
                raise ValueError("Single-host mock TE identity differs from producer host")
            if any(node.get("host", host) != host for node in
                   (params.get("remote_multi_nodes_meta_mapping") or {}).values()):
                raise ValueError("Single-host mock requires one producer TE host identity")
            if self.engine is None:
                # The single-host CED experiment must use the same TE host
                # identity as P. Mixing loopback with the physical host IP
                # crashes this Ascend TE build, even on a model-free probe.
                self.initialize_transport(host)
            return self._consume(params, hold_ms, page_fingerprints, snapshot_blocks)

    def _consume(self, params, hold_ms, page_fingerprints=False, snapshot_blocks=()):
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
            print(json.dumps({"mock_phase": "planned", "request_id": params["remote_request_id"],
                              "rank": rank, "pages": len(rows),
                              "bytes": sum(row["size"] for row in rows)}), flush=True)
            hashes = {}
            page_hashes = {}
            snapshots = {}
            nonzero = {}
            byte_count, read_seconds, hash_seconds = 0, 0.0, 0.0
            offset = 0
            batch = []

            def flush():
                nonlocal offset, batch, byte_count, read_seconds, hash_seconds
                if not batch:
                    return
                before = time.perf_counter()
                print(json.dumps({"mock_phase": "before_read", "rank": rank,
                                  "copies": len(batch), "bytes": offset}), flush=True)
                ret = self.engine.batch_transfer_sync_read(
                    session,
                    [self.buffer.data_ptr() + where for where, row in batch],
                    [row["remote"] for _, row in batch],
                    [row["size"] for _, row in batch],
                )
                read_seconds += time.perf_counter() - before
                if ret < 0:
                    raise RuntimeError(f"Mock Mooncake read failed for rank {rank}: {ret}")
                print(json.dumps({"mock_phase": "read_complete", "rank": rank,
                                  "ret": ret, "bytes": offset}), flush=True)
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
                        initialized = payload[:row["compare_bytes"]]
                        digest.update(initialized)
                        if page_fingerprints:
                            page_hashes[f"{key}/b{row['logical_block']}"] = hashlib.sha256(initialized).hexdigest()
                        if rank == 0 and group == 0 and row["logical_block"] in snapshot_blocks:
                            snapshots[f"{key}/b{row['logical_block']}"] = base64.b64encode(payload).decode("ascii")
                    byte_count += row["size"]
                hash_seconds += time.perf_counter() - before
                offset, batch = 0, []

            for row in rows:
                if row["size"] > self.buffer_bytes:
                    raise ValueError("Mock host buffer is smaller than one KV page")
                if offset + row["size"] > self.buffer_bytes or len(batch) >= self.max_copy_ops:
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
                "page_fingerprints": page_hashes,
                "snapshots_base64": snapshots,
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
        # Page-level diagnostics are returned to the runner's raw artifact;
        # duplicating them in service logs is costly at long context lengths.
        logged = {**result, "ranks": [
            {key: value for key, value in row.items()
             if key not in ("fingerprints", "page_fingerprints", "snapshots_base64")}
            for row in rank_results
        ]}
        print(json.dumps(logged), flush=True)
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
                result = consumer.consume(body["kv_transfer_params"], int(body.get("hold_ms", 0)),
                                          bool(body.get("page_fingerprints", False)),
                                          body.get("snapshot_blocks", ()))
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
    parser.add_argument("--engine-host", help="TE identity; defaults to the first P handoff host")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19191)
    parser.add_argument("--buffer-mib", type=int, default=64)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--max-copy-ops", type=int, default=64)
    parser.add_argument("--params", type=Path)
    args = parser.parse_args()
    if not 1 <= args.buffer_mib <= 1024:
        parser.error("--buffer-mib must be in [1, 1024]")
    if not 1 <= args.max_copy_ops <= 65536:
        parser.error("--max-copy-ops must be in [1, 65536]")
    consumer = MockConsumer(args.engine_host, args.device, args.buffer_mib * 1048576,
                            args.timeout_ms, args.max_copy_ops)
    if args.params:
        document = json.loads(args.params.read_text())
        consumer.consume(document.get("kv_transfer_params", document))
    else:
        serve(consumer, args.host, args.port)


if __name__ == "__main__":
    main()
