"""Read-only MAP_SHARED checkpoint row views, with no whole-table copy.

No Ascend/ACL imports or host registration: this is the device-index=0 path.
Integration into the frozen Engram loader is a separate candidate/quality gate.
"""
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import struct
import stat

MAX_HEADER_BYTES = 16 * 1024 * 1024
ELEMENT_BYTES = {"I8": 1, "F32": 4}


class SharedTensorRows:
    def __init__(self, path, key, expected_shape, expected_dtype, start, stop):
        path = Path(path).resolve(strict=True)
        if (expected_dtype not in ELEMENT_BYTES or len(expected_shape) != 2
                or any(type(dim) is not int or dim <= 0 for dim in expected_shape)
                or type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= expected_shape[0]):
            raise ValueError("Invalid immutable Engram tensor/shard geometry")
        self.mapping = None
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("Shared Engram requires a regular checkpoint file")
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated safetensors length")
            header_bytes = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_bytes <= MAX_HEADER_BYTES or 8 + header_bytes > before.st_size:
                raise ValueError("Invalid bounded safetensors header length")
            raw_header = source.read(header_bytes)
            header = json.loads(raw_header)
            # Keep the safetensors nonoverlapping, gap-free payload invariant.
            # Otherwise a plausible selected dtype/shape could alias bytes of
            # another tensor (e.g. an I8 payload interpreted as FP32 scales).
            intervals = []
            if not isinstance(header, dict):
                raise ValueError("Safetensors header must be a tensor dictionary")
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                bounds = entry.get("data_offsets") if isinstance(entry, dict) else None
                if (not isinstance(bounds, list) or len(bounds) != 2
                        or any(type(value) is not int for value in bounds)
                        or not 0 <= bounds[0] <= bounds[1] <= before.st_size - 8 - header_bytes):
                    raise ValueError("Invalid safetensors payload interval")
                if bounds[0] != bounds[1]:
                    intervals.append(tuple(bounds))
            end_of_payload = 0
            for begin_of_tensor, end_of_tensor in sorted(intervals):
                if begin_of_tensor != end_of_payload:
                    raise ValueError("Overlapping or incomplete safetensors payload")
                end_of_payload = end_of_tensor
            if end_of_payload != before.st_size - 8 - header_bytes:
                raise ValueError("Incomplete safetensors payload coverage")
            spec = header.get(key) if isinstance(header, dict) else None
            if (not isinstance(spec, dict) or spec.get("dtype") != expected_dtype
                    or spec.get("shape") != list(expected_shape)):
                raise ValueError("Checkpoint tensor dtype/shape mismatch: " + key)
            offsets = spec.get("data_offsets")
            size = math.prod(expected_shape) * ELEMENT_BYTES[expected_dtype]
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(offset) is not int for offset in offsets)
                    or not 0 <= offsets[0] < offsets[1] <= before.st_size - 8 - header_bytes
                    or offsets[1] - offsets[0] != size):
                raise ValueError("Checkpoint tensor data range/size mismatch: " + key)
            tensor_begin = 8 + header_bytes + offsets[0]
            row_bytes = expected_shape[1] * ELEMENT_BYTES[expected_dtype]
            begin, end = tensor_begin + start * row_bytes, tensor_begin + stop * row_bytes
            if begin % ELEMENT_BYTES[expected_dtype]:
                raise ValueError("Misaligned checkpoint tensor element offset")
            aligned = begin // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
            self.mapping = mmap.mmap(source.fileno(), end - aligned,
                                     flags=mmap.MAP_SHARED, prot=mmap.PROT_READ, offset=aligned)
            after = os.fstat(source.fileno())
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                self.mapping.close()
                raise ValueError("Checkpoint changed during shared mapping")
        self.delta, self.byte_count = begin - aligned, end - begin
        self.shape, self.dtype = (stop - start, expected_shape[1]), expected_dtype
        self.receipt = {"path": str(path), "key": key, "device": before.st_dev, "inode": before.st_ino,
                        "file_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                        "header_sha256": hashlib.sha256(raw_header).hexdigest(),
                        "dtype": expected_dtype, "table_shape": list(expected_shape),
                        "row_start": start, "row_stop": stop, "tensor_begin": tensor_begin,
                        "view_begin": begin, "view_bytes": self.byte_count,
                        "map_begin": aligned, "map_bytes": end - aligned,
                        "MAP_SHARED": True, "PROT_READ_only": True,
                        "ACL_host_registered": False, "whole_table_private_copy": False}

    def buffer(self):
        return memoryview(self.mapping)[self.delta:self.delta + self.byte_count]

    def torch_view(self):
        import torch
        dtype = torch.int8 if self.dtype == "I8" else torch.float32
        # PyTorch warns about read-only buffers; no caller may mutate this view.
        # mmap PROT_READ additionally enforces that constraint at the OS level.
        return torch.frombuffer(self.buffer(), dtype=dtype).reshape(self.shape)

    def close(self):
        # Fails with BufferError while any exported tensor/view is still alive.
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None


def map_checkpoint_rows(model_path, key, rows, width, start, stop):
    if width % 32:
        raise ValueError("INT8 Engram requires group32 scales")
    root = Path(model_path).resolve(strict=True)
    index = json.loads((root / "quant_model_weights.safetensors.index.json").read_text())
    weights = index.get("weight_map")
    scale_key = key.removesuffix(".weight") + ".scale"
    if (not isinstance(weights, dict) or not isinstance(weights.get(key), str)
            or not isinstance(weights.get(scale_key), str)):
        raise ValueError("Shared host Engram requires prequantized INT8 weight + FP32 scale")
    codes = SharedTensorRows(root / weights[key], key, (rows, width), "I8", start, stop)
    try:
        scales = SharedTensorRows(root / weights[scale_key], scale_key, (rows, width // 32), "F32", start, stop)
    except BaseException:
        codes.close()
        raise
    return codes, scales
