#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a DSpark-ready runtime checkpoint for the W4A8-quantized DeepSeek-V4.1-Flash trunk.

WHAT IT DOES
------------
Given the W4A8 trunk output and the official checkpoint, this script creates an
output directory that contains:

  1. absolute symlinks to every quantized trunk shard, plus copies of
     configuration.json / tokenizer.json / tokenizer_config.json and of the
     optional/ directory (so optional/quarot.safetensors is reachable);
  2. one or more new safetensors shards named mtp-0000N-of-0000M.safetensors
     holding the official mtp.* tensors.  Quantized mtp weights are dequantized
     to BF16 with the exact semantics of
     src/msmodelslim/msmodelslim/model/deepseek_v41/convert.py; their mtp.*.scale
     siblings are intentionally NOT written.  Every other mtp.* tensor is copied
     as-is in its original dtype (norm weights are BF16, hc_*/attn_sink/... are F32);
  3. quant_model_weights.safetensors.index.json = quant weight_map + one entry per
     written mtp tensor (JSON schema {"metadata": ..., "weight_map": ...};
     original metadata keys are preserved and total_size, if present, is set to 0);
  4. quant_model_description.json = quant description with every official mtp.*
     name (all 2401, including the skipped .scale siblings) set to the string
     "FLOAT"; every other key/value is unchanged;
  5. config.json = quant config with text_config.num_nextn_predict_layers = 3 and
     everything else (including quantization_config) untouched.

The build is streaming and memory-frugal: tensors are read one at a time through
safe_open(..., framework="pt").get_tensor(name), dequantized on CPU, and written
directly to the output file with a small inlined safetensors writer.  At most one
source tensor plus its dequantized result is resident at a time; a new output
shard is started whenever the next tensor would push the shard above ~8 GiB.
The build is idempotent: it reuses already-correct mtp shards and rewrites all
metadata deterministically.

USAGE (run *inside* the container; torch + safetensors live in the venv there)
-----------------------------------------------------------------------------
Full build:

  /home/user/projects/dsv41/env/mslim-venv/bin/python /home/user/projects/dsv41/scripts/make_dspark_ckpt.py --quant-dir /home/user/models/out/v41-w4a8-stage1 --official-dir /home/user/models/DeepSeek-V4.1-Flash --out-dir /home/user/models/out/v41-w4a8-dspark

Plan only (writes nothing, does not create --out-dir):

  /home/user/projects/dsv41/env/mslim-venv/bin/python /home/user/projects/dsv41/scripts/make_dspark_ckpt.py --quant-dir /home/user/models/out/v41-w4a8-stage1 --official-dir /home/user/models/DeepSeek-V4.1-Flash --out-dir /home/user/models/out/v41-w4a8-dspark --dry-run

Fast smoke (only the first N mtp tensors sorted by name):

  /home/user/projects/dsv41/env/mslim-venv/bin/python /home/user/projects/dsv41/scripts/make_dspark_ckpt.py --quant-dir /home/user/models/out/v41-w4a8-stage1 --official-dir /home/user/models/DeepSeek-V4.1-Flash --out-dir /home/user/models/out/v41-w4a8-dspark --limit 50

Verify an already produced directory (pass --limit N if the directory was built with --limit N):

  /home/user/projects/dsv41/env/mslim-venv/bin/python /home/user/projects/dsv41/scripts/make_dspark_ckpt.py --quant-dir /home/user/models/out/v41-w4a8-stage1 --official-dir /home/user/models/DeepSeek-V4.1-Flash --out-dir /home/user/models/out/v41-w4a8-dspark --verify-only

The script never writes to --quant-dir or --official-dir and refuses an --out-dir
that is equal to or nested below either input directory.  It imports nothing from
msmodelslim; the small dequant helpers are inlined below.  For a syntax check on
a host without torch, python3 -m py_compile still works because compilation does
not execute the torch/safetensors imports.
"""

import argparse
import contextlib
import copy
import ctypes
import glob
import json
import math
import os
import shutil
import struct
import sys

import torch
from safetensors import safe_open

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
MTP_PREFIX = "mtp."
WEIGHT_SUFFIX = ".weight"
SCALE_SUFFIX = ".scale"

QUANT_INDEX_NAME = "quant_model_weights.safetensors.index.json"
OFFICIAL_INDEX_NAME = "model.safetensors.index.json"
DESCRIPTION_NAME = "quant_model_description.json"

# "roughly 8 GB per file": cap the tensor payload of each new shard at 8 GiB.
TARGET_SHARD_BYTES = 8 * 1024 ** 3
# 64 MiB writes keep the copy from a memory-mapped source tensor streaming.
WRITE_CHUNK_BYTES = 64 * 1024 * 1024

# Exact FP4_TABLE from .../deepseek_v41/convert.py (sign bit included).
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# safetensors dtype string -> bytes per element (only the dtypes we can meet).
DTYPE_NBYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "U16": 2, "U32": 4, "U64": 8, "C64": 8,
    "F8_E4M3": 1, "F8_E4M3FNUZ": 1, "F8_E5M2": 1, "F8_E5M2FNUZ": 1,
    "F8_E8M0": 1,
}

_FLOAT8_E8M0 = getattr(torch, "float8_e8m0fnu", None)
_SCALE_DTYPES = (torch.float32,) + ((_FLOAT8_E8M0,) if _FLOAT8_E8M0 is not None else ())

_ST_TO_TORCH = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F16": torch.float16,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
}
if _FLOAT8_E8M0 is not None:
    _ST_TO_TORCH["F8_E8M0"] = _FLOAT8_E8M0


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
def log_ok(message):
    print("[ok] " + message, flush=True)


def log_info(message):
    print("[info] " + message, flush=True)


def log_error(message):
    print("[error] " + message, flush=True)


# ---------------------------------------------------------------------------
# dequantization helpers, inlined from deepseek_v41/convert.py
# ---------------------------------------------------------------------------
def scale_to_float(scale):
    """UE8M0 / FP32 scale -> FP32. Native cast first, E8M0 exponent fallback."""
    if scale.dtype == torch.float32:
        return scale
    try:
        return scale.to(torch.float32)
    except (RuntimeError, TypeError):
        raw = scale.view(torch.uint8).to(torch.int32) - 127
        return torch.ldexp(torch.ones_like(raw, dtype=torch.float32), raw)


def decode_fp8(weight, scale):
    """weight: fp8 [out, in]; scale: [out/block_out, in/block_in] (32x32 in V4.1)."""
    out_block = weight.size(0) // scale.size(0)
    in_block = weight.size(1) // scale.size(1)
    w = weight.float().unflatten(0, (-1, out_block)).unflatten(-1, (-1, in_block))
    w = w * scale_to_float(scale).to(w.device)[:, None, :, None]
    return w.flatten(2, 3).flatten(0, 1).bfloat16()


def decode_fp4(packed_fp4_data, block_scales):
    """packed int8 [out, in/2] + scale [out, in/fp4_block] (V4.1 fp4_block=32)."""
    uint8 = packed_fp4_data.view(torch.uint8)
    low = uint8 & 0x0F
    high = (uint8 >> 4) & 0x0F
    # Official layout: low nibble first, then high nibble.
    indices = torch.stack([low, high], dim=-1).flatten(-2)
    values = FP4_TABLE.to(packed_fp4_data.device)[indices.long()]
    in_dim = packed_fp4_data.size(1) * 2
    fp4_block = in_dim // block_scales.size(1)
    scales = scale_to_float(block_scales).to(values.device).repeat_interleave(fp4_block, dim=-1)
    return (values * scales).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# small generic helpers
# ---------------------------------------------------------------------------
def _numel(shape):
    return math.prod(shape) if len(shape) else 1


def _weight_sibling(scale_key):
    return scale_key[: -len(SCALE_SUFFIX)] + WEIGHT_SUFFIX


def _scale_sibling(weight_key):
    return weight_key[: -len(WEIGHT_SUFFIX)] + SCALE_SUFFIX


def load_json(path):
    if not os.path.isfile(path):
        raise RuntimeError("missing required JSON file: %s" % path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, obj, indent=None):
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            if indent is None:
                json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"))
            else:
                json.dump(obj, fh, ensure_ascii=False, indent=indent)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def read_safetensors_header(path):
    """Stdlib-only safetensors header reader -> (header_len, header_without_metadata)."""
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise RuntimeError("%s: too short to be a safetensors file" % path)
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 0 or header_len > (1 << 32):
            raise RuntimeError("%s: implausible safetensors header length %d" % (path, header_len))
        raw = fh.read(header_len)
        if len(raw) != header_len:
            raise RuntimeError("%s: truncated safetensors header" % path)
    try:
        header = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("%s: invalid safetensors JSON header: %s" % (path, exc)) from exc
    if not isinstance(header, dict):
        raise RuntimeError("%s: safetensors header is not a JSON object" % path)
    header.pop("__metadata__", None)
    return header_len, header


def _header_offsets_cover_file(header_len, header, file_size):
    offset = 0
    for info in header.values():
        offsets = info.get("data_offsets")
        if not (isinstance(offsets, list) and len(offsets) == 2):
            return False
        start, end = offsets
        if not isinstance(start, int) or not isinstance(end, int):
            return False
        if start != offset or end < start:
            return False
        offset = end
    return file_size == 8 + header_len + offset


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def build_plan(official_dir, official_index, limit=None):
    """Return (plan, stats); reads only headers, never tensor data."""
    weight_map = official_index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("official index has no 'weight_map' object")
    all_mtp_keys = sorted(key for key in weight_map if key.startswith(MTP_PREFIX))
    selected_keys = all_mtp_keys if limit is None else all_mtp_keys[:limit]

    header_cache = {}

    def meta_for(key):
        file_name = weight_map[key]
        if file_name not in header_cache:
            path = os.path.join(official_dir, file_name)
            header_cache[file_name] = read_safetensors_header(path)[1]
        header = header_cache[file_name]
        if key not in header:
            raise RuntimeError(
                "official index/header mismatch: %r is not present in %s" % (key, file_name)
            )
        return header[key]

    plan = []
    skipped_scales = []
    for key in selected_keys:
        # A .scale sibling of a quantized weight is represented by the
        # dequantized .weight and must not be written to the output.
        if key.endswith(SCALE_SUFFIX) and _weight_sibling(key) in weight_map:
            skipped_scales.append(key)
            continue

        meta = meta_for(key)
        dtype = meta.get("dtype")
        shape = tuple(meta.get("shape") or ())
        scale_key = _scale_sibling(key) if key.endswith(WEIGHT_SUFFIX) else None
        is_quant_weight = scale_key is not None and scale_key in weight_map

        if is_quant_weight:
            if len(shape) != 2:
                raise RuntimeError(
                    "mtp tensor %r: quantized weight must be 2-D, got shape %s" % (key, shape)
                )
            if dtype == "F8_E4M3":
                action = "dequant_fp8"
                out_shape = shape
                expanded_in_dim = shape[1]
            elif dtype == "I8":
                # packed FP4: [out, in/2] -> [out, in]
                action = "dequant_fp4"
                out_shape = (shape[0], shape[1] * 2)
                expanded_in_dim = shape[1] * 2
            else:
                raise RuntimeError(
                    "unsupported quantized dtype %r for mtp tensor %r (sibling %r exists; "
                    "expected 'F8_E4M3' for fp8 or 'I8' for packed fp4)"
                    % (dtype, key, scale_key)
                )
            out_st_dtype = "BF16"
            out_nbytes = _numel(out_shape) * 2
        else:
            if dtype not in DTYPE_NBYTES:
                raise RuntimeError(
                    "unsupported dtype %r for copied mtp tensor %r" % (dtype, key)
                )
            action = "copy"
            scale_key = None
            out_shape = shape
            out_st_dtype = dtype
            expanded_in_dim = None
            out_nbytes = _numel(out_shape) * DTYPE_NBYTES[dtype]

        plan.append({
            "name": key,
            "action": action,
            "src_file": weight_map[key],
            "src_dtype": dtype,
            "src_shape": shape,
            "scale_name": scale_key,
            "scale_file": weight_map[scale_key] if scale_key is not None else None,
            "out_st_dtype": out_st_dtype,
            "out_shape": out_shape,
            "out_nbytes": out_nbytes,
            "expanded_in_dim": expanded_in_dim,
        })

    stats = {
        "all_mtp": len(all_mtp_keys),
        "selected_mtp": len(selected_keys),
        "skipped_scales": len(skipped_scales),
        "skipped_keys": skipped_scales,
        "dequant_fp8": 0,
        "dequant_fp4": 0,
        "copy": 0,
        "copy_by_dtype": {},
        "dequant_bytes": 0,
        "copy_bytes": 0,
    }
    for entry in plan:
        if entry["action"] in ("dequant_fp8", "dequant_fp4"):
            stats[entry["action"]] += 1
            stats["dequant_bytes"] += entry["out_nbytes"]
        else:
            stats["copy"] += 1
            stats["copy_bytes"] += entry["out_nbytes"]
            dtype = entry["src_dtype"]
            stats["copy_by_dtype"][dtype] = stats["copy_by_dtype"].get(dtype, 0) + 1
    stats["total_bytes"] = stats["dequant_bytes"] + stats["copy_bytes"]
    return plan, stats


def pack_shards(plan, target_bytes=TARGET_SHARD_BYTES):
    """Greedy split into ~target_bytes shards; returns [(file_name, entries), ...]."""
    groups = []
    current = []
    current_bytes = 0
    for entry in plan:
        nbytes = entry["out_nbytes"]
        if current and current_bytes + nbytes > target_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(entry)
        current_bytes += nbytes
    if current:
        groups.append(current)
    total = len(groups)
    return [
        ("mtp-%05d-of-%05d.safetensors" % (index, total), group)
        for index, group in enumerate(groups, 1)
    ]


# ---------------------------------------------------------------------------
# streaming safetensors writer
# ---------------------------------------------------------------------------
def _build_header(entries):
    offset = 0
    header = {}
    for entry in entries:
        nbytes = entry["out_nbytes"]
        header[entry["name"]] = {
            "dtype": entry["out_st_dtype"],
            "shape": list(entry["out_shape"]),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    # The safetensors header JSON is space-padded so the data section starts
    # on an 8-byte boundary (the length prefix is 8 bytes).
    raw += b" " * ((8 - (len(raw) % 8)) % 8)
    return struct.pack("<Q", len(raw)) + raw, offset


def _write_tensor_bytes(fh, tensor):
    """Stream the raw little-endian bytes of a contiguous CPU tensor, chunk by chunk."""
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    nbytes = int(tensor.numel()) * int(tensor.element_size())
    if nbytes == 0:
        return
    raw = (ctypes.c_ubyte * nbytes).from_address(tensor.data_ptr())
    view = memoryview(raw)
    try:
        offset = 0
        while offset < nbytes:
            end = min(offset + WRITE_CHUNK_BYTES, nbytes)
            fh.write(view[offset:end])
            offset = end
    finally:
        view.release()


def _shard_is_reusable(path, entries):
    if not os.path.isfile(path):
        return False
    try:
        header_len, header = read_safetensors_header(path)
    except Exception:
        return False
    expected_bytes = sum(entry["out_nbytes"] for entry in entries)
    if os.path.getsize(path) != 8 + header_len + expected_bytes:
        return False
    if set(header) != {entry["name"] for entry in entries}:
        return False
    offset = 0
    for entry in entries:
        info = header[entry["name"]]
        if info.get("dtype") != entry["out_st_dtype"]:
            return False
        if list(info.get("shape") or []) != list(entry["out_shape"]):
            return False
        if info.get("data_offsets") != [offset, offset + entry["out_nbytes"]]:
            return False
        offset += entry["out_nbytes"]
    return True


def _check_loaded_copy(tensor, entry):
    if tuple(tensor.shape) != entry["src_shape"]:
        raise RuntimeError(
            "official tensor %r: header shape %s but loaded shape %s"
            % (entry["name"], entry["src_shape"], tuple(tensor.shape))
        )
    expected_dtype = _ST_TO_TORCH.get(entry["src_dtype"])
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise RuntimeError(
            "official tensor %r: header dtype %r but torch loaded %s"
            % (entry["name"], entry["src_dtype"], tensor.dtype)
        )


def _validate_quant_source(entry, weight, scale):
    name = entry["name"]
    if weight.dim() != 2:
        raise RuntimeError("mtp tensor %r: weight must be 2-D, got %s" % (name, tuple(weight.shape)))
    if scale.dim() != 2:
        raise RuntimeError("mtp tensor %r: scale must be 2-D, got %s" % (name, tuple(scale.shape)))
    if scale.dtype not in _SCALE_DTYPES:
        raise RuntimeError(
            "mtp tensor %r: unsupported scale dtype %s (expected float32 or float8_e8m0fnu)"
            % (name, scale.dtype)
        )
    if entry["action"] == "dequant_fp8":
        if weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                "mtp tensor %r: expected torch.float8_e4m3fn, got %s" % (name, weight.dtype)
            )
        if (
            scale.size(0) == 0
            or scale.size(1) == 0
            or weight.size(0) % scale.size(0) != 0
            or weight.size(1) % scale.size(1) != 0
        ):
            raise RuntimeError(
                "mtp tensor %r: fp8 scale shape %s does not evenly divide weight shape %s"
                % (name, tuple(scale.shape), tuple(weight.shape))
            )
    else:
        if weight.dtype != torch.int8:
            raise RuntimeError(
                "mtp tensor %r: expected torch.int8 packed FP4, got %s" % (name, weight.dtype)
            )
        expanded_in_dim = weight.size(1) * 2
        if (
            scale.size(0) != weight.size(0)
            or scale.size(1) == 0
            or expanded_in_dim % scale.size(1) != 0
        ):
            raise RuntimeError(
                "mtp tensor %r: packed shape %s / scale shape %s do not describe "
                "per-row FP4 blocks on the expanded input dimension %d"
                % (name, tuple(weight.shape), tuple(scale.shape), expanded_in_dim)
            )


def _check_dequant_result(tensor, entry):
    name = entry["name"]
    if tensor.dtype != torch.bfloat16:
        raise RuntimeError(
            "mtp tensor %r: dequantized dtype is %s, expected torch.bfloat16" % (name, tensor.dtype)
        )
    if tuple(tensor.shape) != tuple(entry["out_shape"]):
        raise RuntimeError(
            "mtp tensor %r: dequantized shape %s, expected %s"
            % (name, tuple(tensor.shape), tuple(entry["out_shape"]))
        )
    if entry["expanded_in_dim"] is not None and tensor.shape[-1] != entry["expanded_in_dim"]:
        raise RuntimeError(
            "mtp tensor %r: result input dim %d does not equal the full expanded dim %d"
            % (name, tensor.shape[-1], entry["expanded_in_dim"])
        )
    if entry["action"] == "dequant_fp4":
        if tensor.shape[-1] != entry["src_shape"][1] * 2:
            raise RuntimeError(
                "mtp tensor %r: packed [out, in/2] -> [out, in] expansion failed (%s -> %s)"
                % (name, entry["src_shape"], tuple(tensor.shape))
            )
    if entry["action"] == "dequant_fp8" and tuple(tensor.shape) != tuple(entry["src_shape"]):
        raise RuntimeError(
            "mtp tensor %r: fp8 dequant must keep the shape %s, got %s"
            % (name, tuple(entry["src_shape"]), tuple(tensor.shape))
        )


_MAIN_PROJ_ROTATION = None


def _maybe_rotate_main_proj(name, tensor):
    """QuaRot 约定：main_proj 必须右乘全局旋转 R（输入维按 hidden 分块）。

    残差流在量化模型里是 x' = R^T x，draft 自己的权重在原始基下，
    所以 main_proj' = W @ R 才能把 x' 映射回模型空间（与运行时对 V4 的 fc
    做的是同一件事；V4.1 的 main_proj 运行时不会自动旋转，必须在 checkpoint
    里预先旋转）。输入维 = len(target_layer_ids) * hidden，按 hidden 分块。
    """
    if _MAIN_PROJ_ROTATION is None or not name.endswith("main_proj.weight"):
        return tensor
    rotation = _MAIN_PROJ_ROTATION.to(torch.float32)
    r_dim = rotation.shape[0]
    in_dim = tensor.shape[-1]
    if in_dim % r_dim:
        raise RuntimeError(
            "%s: input dim %d is not a multiple of rotation dim %d" % (name, in_dim, r_dim)
        )
    blocks = in_dim // r_dim
    work = tensor.to(torch.float32)
    out = torch.empty_like(work)
    for k in range(blocks):
        sl = slice(k * r_dim, (k + 1) * r_dim)
        out[:, sl] = work[:, sl] @ rotation
    delta = (out - work).abs().max().item()
    log_info(
        "rotated %s: W @ blockdiag(R x %d), max|delta|=%.4g" % (name, blocks, delta)
    )
    return out.to(tensor.dtype)


def _write_entry(fh, entry, official_dir):
    src_path = os.path.join(official_dir, entry["src_file"])
    if entry["action"] == "copy":
        with safe_open(src_path, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(entry["name"])
            _check_loaded_copy(tensor, entry)
            tensor = _maybe_rotate_main_proj(entry["name"], tensor)
            _write_tensor_bytes(fh, tensor)
        return

    scale_path = os.path.join(official_dir, entry["scale_file"])
    with contextlib.ExitStack() as stack:
        weight_handle = stack.enter_context(safe_open(src_path, framework="pt", device="cpu"))
        weight = weight_handle.get_tensor(entry["name"])
        if os.path.abspath(src_path) == os.path.abspath(scale_path):
            scale = weight_handle.get_tensor(entry["scale_name"])
        else:
            scale_handle = stack.enter_context(safe_open(scale_path, framework="pt", device="cpu"))
            scale = scale_handle.get_tensor(entry["scale_name"])
        _validate_quant_source(entry, weight, scale)
        with torch.no_grad():
            if entry["action"] == "dequant_fp8":
                tensor = decode_fp8(weight, scale)
            else:
                tensor = decode_fp4(weight, scale)
        del weight, scale
        _check_dequant_result(tensor, entry)
        tensor = _maybe_rotate_main_proj(entry["name"], tensor)
        _write_tensor_bytes(fh, tensor)
        del tensor


def _write_or_reuse_shard(out_dir, shard_name, entries, official_dir):
    path = os.path.join(out_dir, shard_name)
    total_bytes = sum(entry["out_nbytes"] for entry in entries)
    if _shard_is_reusable(path, entries):
        log_ok(
            "reuse %s: %d mtp tensors, %.3f GiB"
            % (shard_name, len(entries), total_bytes / (1024 ** 3))
        )
        return

    header_bytes, data_bytes = _build_header(entries)
    if data_bytes != total_bytes:
        raise RuntimeError("%s: internal byte accounting mismatch" % shard_name)
    tmp_path = path + ".tmp"
    log_info(
        "writing %s: %d mtp tensors, %.3f GiB"
        % (shard_name, len(entries), total_bytes / (1024 ** 3))
    )
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(header_bytes)
            for index, entry in enumerate(entries, 1):
                _write_entry(fh, entry, official_dir)
                if index % 100 == 0 or index == len(entries):
                    log_info("  %s: %d/%d tensors" % (shard_name, index, len(entries)))
            fh.flush()
        # header_bytes 已包含 8 字节长度前缀，不能再加一次（曾把前缀数两遍）
        expected = len(header_bytes) + data_bytes
        written = os.path.getsize(tmp_path)
        if written != expected:
            raise RuntimeError(
                "%s: wrote %d bytes, expected %d" % (shard_name, written, expected)
            )
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    log_ok(
        "wrote %s: %d mtp tensors, %.3f GiB"
        % (shard_name, len(entries), total_bytes / (1024 ** 3))
    )


# ---------------------------------------------------------------------------
# output assembly
# ---------------------------------------------------------------------------
def _require_inputs(quant_dir, official_dir):
    for name in (
        QUANT_INDEX_NAME,
        DESCRIPTION_NAME,
        "config.json",
        "configuration.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        path = os.path.join(quant_dir, name)
        if not os.path.isfile(path):
            raise RuntimeError("missing required quant input: %s" % path)
    optional_dir = os.path.join(quant_dir, "optional")
    if not os.path.isdir(optional_dir):
        raise RuntimeError("missing required quant input directory: %s" % optional_dir)
    official_index = os.path.join(official_dir, OFFICIAL_INDEX_NAME)
    if not os.path.isfile(official_index):
        raise RuntimeError("missing required official input: %s" % official_index)


def _check_out_dir_guard(out_dir, quant_dir, official_dir):
    out_real = os.path.realpath(out_dir)
    for source in (quant_dir, official_dir):
        source_real = os.path.realpath(source)
        if out_real == source_real or out_real.startswith(source_real + os.sep):
            raise RuntimeError(
                "--out-dir %r must not be equal to or nested inside the input dir %r"
                % (out_dir, source)
            )


def _cleanup_stale_shards(out_dir, planned_names):
    for path in sorted(glob.glob(os.path.join(out_dir, "mtp-*-of-*.safetensors"))):
        name = os.path.basename(path)
        if name not in planned_names:
            os.unlink(path)
            log_info("removed stale mtp shard %s" % name)
    for path in sorted(glob.glob(os.path.join(out_dir, "mtp-*.safetensors.tmp"))):
        try:
            os.unlink(path)
        except OSError:
            pass


def _link_quant_shards(quant_dir, out_dir, quant_index):
    weight_map = quant_index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    for name in shard_names:
        src = os.path.abspath(os.path.join(quant_dir, name))
        if not os.path.isfile(src):
            raise RuntimeError("quant shard referenced by index is missing: %s" % src)
        dst = os.path.join(out_dir, name)
        parent = os.path.dirname(dst)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        if os.path.islink(dst):
            if os.readlink(dst) == src:
                continue
            os.unlink(dst)
        elif os.path.exists(dst):
            raise RuntimeError("refusing to replace non-symlink file in output dir: %s" % dst)
        os.symlink(src, dst)
    log_ok(
        "absolute symlinks to %d quant shards are present (source: %s)"
        % (len(shard_names), quant_dir)
    )


def _copy_file(src, dst):
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = dst + ".tmp"
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _copy_tree(src, dst):
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        src_path = os.path.join(src, name)
        dst_path = os.path.join(dst, name)
        if os.path.islink(src_path):
            target = os.path.realpath(src_path)
            if os.path.isdir(target):
                _copy_tree(target, dst_path)
            else:
                _copy_file(target, dst_path)
        elif os.path.isdir(src_path):
            _copy_tree(src_path, dst_path)
        else:
            _copy_file(src_path, dst_path)


def _copy_aux_files(quant_dir, out_dir):
    for name in ("configuration.json", "tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(quant_dir, name)
        if not os.path.isfile(src):
            raise RuntimeError("missing required quant input: %s" % src)
        _copy_file(src, os.path.join(out_dir, name))
    optional_src = os.path.join(quant_dir, "optional")
    optional_dst = os.path.join(out_dir, "optional")
    _copy_tree(optional_src, optional_dst)
    if not os.path.isfile(os.path.join(optional_dst, "quarot.safetensors")):
        raise RuntimeError(
            "optional/quarot.safetensors is not reachable under the output dir"
        )
    log_ok(
        "copied configuration.json, tokenizer.json, tokenizer_config.json and optional/"
    )


def _write_description(quant_dir, out_dir, official_index):
    src_path = os.path.join(quant_dir, DESCRIPTION_NAME)
    description = load_json(src_path)
    if not isinstance(description, dict):
        raise RuntimeError("%s: expected a JSON object" % src_path)
    mtp_keys = sorted(
        key for key in official_index["weight_map"] if key.startswith(MTP_PREFIX)
    )
    for key in mtp_keys:
        description[key] = "FLOAT"
    _write_json(os.path.join(out_dir, DESCRIPTION_NAME), description)
    log_ok("wrote %s (+%d mtp.* -> FLOAT)" % (DESCRIPTION_NAME, len(mtp_keys)))


def _write_config(quant_dir, out_dir):
    src_path = os.path.join(quant_dir, "config.json")
    config = load_json(src_path)
    if not isinstance(config, dict):
        raise RuntimeError("%s: expected a JSON object" % src_path)
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise RuntimeError("%s: missing text_config object" % src_path)
    text_config["num_nextn_predict_layers"] = 3
    _write_json(os.path.join(out_dir, "config.json"), config, indent=2)
    log_ok("wrote config.json: text_config.num_nextn_predict_layers = 3")


def _write_index(quant_dir, out_dir, quant_index, plan, shard_of):
    quant_weight_map = quant_index["weight_map"]
    weight_map = dict(quant_weight_map)
    for entry in plan:
        weight_map[entry["name"]] = shard_of[entry["name"]]
    metadata = dict(quant_index.get("metadata") or {})
    if "total_size" in metadata:
        metadata["total_size"] = 0
    out_index = {"metadata": metadata, "weight_map": weight_map}
    _write_json(os.path.join(out_dir, QUANT_INDEX_NAME), out_index)
    log_ok(
        "wrote %s: %d keys (%d quant + %d mtp)"
        % (QUANT_INDEX_NAME, len(weight_map), len(quant_weight_map), len(plan))
    )


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def _print_plan(quant_dir, official_dir, out_dir, quant_index, plan, stats, shard_groups):
    log_info("dry-run: nothing will be written, --out-dir is not created")
    log_info("quant dir    : %s" % quant_dir)
    log_info("official dir : %s" % official_dir)
    log_info("out dir      : %s" % out_dir)
    log_info(
        "official mtp tensors: %d selected of %d total"
        % (stats["selected_mtp"], stats["all_mtp"])
    )
    log_info(
        "mtp tensors to write: %d = dequant_fp8 %d + dequant_fp4 %d + copy %d"
        % (len(plan), stats["dequant_fp8"], stats["dequant_fp4"], stats["copy"])
    )
    if stats["copy_by_dtype"]:
        log_info(
            "copied dtypes: %s"
            % json.dumps(stats["copy_by_dtype"], sort_keys=True)
        )
    log_info(
        "skipped quant .scale siblings: %d (not written)"
        % stats["skipped_scales"]
    )
    log_info(
        "dequantized byte total: %d bytes (%.3f GiB)"
        % (stats["dequant_bytes"], stats["dequant_bytes"] / (1024 ** 3))
    )
    log_info(
        "copied byte total     : %d bytes (%.3f GiB)"
        % (stats["copy_bytes"], stats["copy_bytes"] / (1024 ** 3))
    )
    log_info(
        "new tensor byte total : %d bytes (%.3f GiB)"
        % (stats["total_bytes"], stats["total_bytes"] / (1024 ** 3))
    )
    log_info("new safetensors files: %d (target <= 8 GiB each)" % len(shard_groups))
    for shard_name, entries in shard_groups:
        nbytes = sum(entry["out_nbytes"] for entry in entries)
        log_info(
            "  %s: %d tensors, %d bytes (%.3f GiB)"
            % (shard_name, len(entries), nbytes, nbytes / (1024 ** 3))
        )
    log_info("files that would be created in %s:" % out_dir)
    log_info(
        "  %s: quant keys + %d mtp keys" % (QUANT_INDEX_NAME, len(plan))
    )
    log_info(
        "  %s: quant description + %d mtp.* names = FLOAT"
        % (DESCRIPTION_NAME, stats["all_mtp"])
    )
    log_info("  config.json: text_config.num_nextn_predict_layers = 3")
    log_info(
        "  %d absolute symlinks to quant shards from %s"
        % (len(set(quant_index["weight_map"].values())), quant_dir)
    )
    log_info("  copies of configuration.json, tokenizer.json, tokenizer_config.json, optional/")


def cmd_dry(args):
    quant_dir = os.path.abspath(args.quant_dir)
    official_dir = os.path.abspath(args.official_dir)
    out_dir = os.path.abspath(args.out_dir)
    _require_inputs(quant_dir, official_dir)
    quant_index = load_json(os.path.join(quant_dir, QUANT_INDEX_NAME))
    official_index = load_json(os.path.join(official_dir, OFFICIAL_INDEX_NAME))
    plan, stats = build_plan(official_dir, official_index, args.limit)
    shard_groups = pack_shards(plan)
    _print_plan(quant_dir, official_dir, out_dir, quant_index, plan, stats, shard_groups)
    log_ok("dry-run complete")
    return 0


def cmd_build(args):
    quant_dir = os.path.abspath(args.quant_dir)
    official_dir = os.path.abspath(args.official_dir)
    out_dir = os.path.abspath(args.out_dir)
    _require_inputs(quant_dir, official_dir)
    _check_out_dir_guard(out_dir, quant_dir, official_dir)
    quant_index = load_json(os.path.join(quant_dir, QUANT_INDEX_NAME))
    official_index = load_json(os.path.join(official_dir, OFFICIAL_INDEX_NAME))
    plan, stats = build_plan(official_dir, official_index, args.limit)
    shard_groups = pack_shards(plan)

    quant_weight_map = quant_index.get("weight_map")
    if not isinstance(quant_weight_map, dict):
        raise RuntimeError("quant index has no 'weight_map' object")
    overlap = set(entry["name"] for entry in plan) & set(quant_weight_map)
    if overlap:
        raise RuntimeError(
            "refusing to build: mtp keys already exist in the quant index: %s"
            % sorted(overlap)[:5]
        )

    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(out_dir):
        raise RuntimeError("could not create output dir: %s" % out_dir)

    planned_names = set(name for name, _ in shard_groups)
    shard_of = {}
    for shard_name, entries in shard_groups:
        for entry in entries:
            shard_of[entry["name"]] = shard_name
        _write_or_reuse_shard(out_dir, shard_name, entries, official_dir)

    _cleanup_stale_shards(out_dir, planned_names)
    _link_quant_shards(quant_dir, out_dir, quant_index)
    _copy_aux_files(quant_dir, out_dir)
    _write_description(quant_dir, out_dir, official_index)
    _write_config(quant_dir, out_dir)
    _write_index(quant_dir, out_dir, quant_index, plan, shard_of)

    log_ok(
        "built %s: %d new mtp tensors in %d shard(s), %.3f GiB"
        % (out_dir, len(plan), len(shard_groups), stats["total_bytes"] / (1024 ** 3))
    )
    log_info("now run the same command with --verify-only to validate the result")
    return 0


def cmd_verify(args):
    quant_dir = os.path.abspath(args.quant_dir)
    official_dir = os.path.abspath(args.official_dir)
    out_dir = os.path.abspath(args.out_dir)
    _require_inputs(quant_dir, official_dir)
    if not os.path.isdir(out_dir):
        raise RuntimeError("output dir does not exist: %s" % out_dir)

    quant_index = load_json(os.path.join(quant_dir, QUANT_INDEX_NAME))
    official_index = load_json(os.path.join(official_dir, OFFICIAL_INDEX_NAME))
    plan, stats = build_plan(official_dir, official_index, args.limit)

    quant_weight_map = quant_index.get("weight_map")
    if not isinstance(quant_weight_map, dict):
        raise RuntimeError("quant index has no 'weight_map' object")

    out_index = load_json(os.path.join(out_dir, QUANT_INDEX_NAME))
    out_weight_map = out_index.get("weight_map")
    if not isinstance(out_weight_map, dict):
        raise RuntimeError("%s: missing 'weight_map' object" % QUANT_INDEX_NAME)
    if "metadata" not in out_index:
        raise RuntimeError("%s: missing top-level 'metadata' key" % QUANT_INDEX_NAME)

    plan_name_set = set(entry["name"] for entry in plan)
    if plan_name_set & set(quant_weight_map):
        raise RuntimeError("plan contains mtp keys that already exist in the quant index")
    expected_keys = set(quant_weight_map) | plan_name_set
    missing = sorted(expected_keys - set(out_weight_map))
    extra = sorted(set(out_weight_map) - expected_keys)
    if missing or extra:
        raise RuntimeError(
            "index key mismatch: missing %d (%s), extra %d (%s); if this dir was built "
            "with --limit, pass the same --limit to --verify-only"
            % (len(missing), missing[:3], len(extra), extra[:3])
        )
    if len(out_weight_map) != len(quant_weight_map) + len(plan):
        raise RuntimeError(
            "index key count %d != quant %d + new mtp %d"
            % (len(out_weight_map), len(quant_weight_map), len(plan))
        )
    log_ok(
        "index: %d keys = quant %d + new mtp %d (selected mtp %d - skipped .scale %d; "
        "official mtp total %d)"
        % (
            len(out_weight_map),
            len(quant_weight_map),
            len(plan),
            stats["selected_mtp"],
            stats["skipped_scales"],
            stats["all_mtp"],
        )
    )
    for key in stats["skipped_keys"]:
        if key in out_weight_map:
            raise RuntimeError(
                "skipped quant .scale sibling %r is present in the output index" % key
            )
    log_ok(
        "all %d skipped quant .scale siblings are absent from the index"
        % stats["skipped_scales"]
    )

    mtp_files = sorted(set(out_weight_map[entry["name"]] for entry in plan))
    headers_by_file = {}
    for file_name in mtp_files:
        path = os.path.join(out_dir, file_name)
        if not os.path.isfile(path):
            raise RuntimeError("mtp shard referenced by index is missing: %s" % path)
        header_len, header = read_safetensors_header(path)
        if not _header_offsets_cover_file(header_len, header, os.path.getsize(path)):
            raise RuntimeError(
                "%s: tensor data_offsets do not contiguously cover the file" % file_name
            )
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
        except Exception as exc:
            raise RuntimeError(
                "%s: safetensors parser rejected the file: %s" % (file_name, exc)
            ) from exc
        if set(keys) != set(header):
            raise RuntimeError(
                "%s: safe_open keys differ from parsed header keys" % file_name
            )
        headers_by_file[file_name] = header
        log_ok(
            "shard %s: parses, %d tensors, %.3f GiB"
            % (file_name, len(header), os.path.getsize(path) / (1024 ** 3))
        )

    dequant_count = 0
    for entry in plan:
        file_name = out_weight_map[entry["name"]]
        if file_name not in headers_by_file:
            raise RuntimeError(
                "%r: index points at unexpected shard %r" % (entry["name"], file_name)
            )
        info = headers_by_file[file_name].get(entry["name"])
        if info is None:
            raise RuntimeError(
                "%r: missing from shard %s" % (entry["name"], file_name)
            )
        if info.get("dtype") != entry["out_st_dtype"]:
            raise RuntimeError(
                "%r: dtype %r, expected %r"
                % (entry["name"], info.get("dtype"), entry["out_st_dtype"])
            )
        if list(info.get("shape") or []) != list(entry["out_shape"]):
            raise RuntimeError(
                "%r: shape %s, expected %s"
                % (entry["name"], info.get("shape"), list(entry["out_shape"]))
            )
        if entry["action"].startswith("dequant"):
            dequant_count += 1
            if info.get("dtype") != "BF16":
                raise RuntimeError(
                    "%r: dequantized from a quantized source but dtype is %r, expected 'BF16'"
                    % (entry["name"], info.get("dtype"))
                )
    for file_name, header in headers_by_file.items():
        expected_in_file = set(
            entry["name"] for entry in plan if out_weight_map[entry["name"]] == file_name
        )
        if set(header) != expected_in_file:
            raise RuntimeError(
                "%s: unexpected tensor set (extra %s, missing %s)"
                % (
                    file_name,
                    sorted(set(header) - expected_in_file)[:3],
                    sorted(expected_in_file - set(header))[:3],
                )
            )
    log_ok(
        "%d dequantized mtp weights are BF16; all %d written mtp tensors match the plan"
        % (dequant_count, len(plan))
    )

    quant_shard_names = sorted(set(quant_weight_map.values()))
    for name in quant_shard_names:
        path = os.path.join(out_dir, name)
        if not os.path.islink(path):
            raise RuntimeError("%s must be an absolute symlink to the quant shard" % path)
        target = os.readlink(path)
        expected_target = os.path.abspath(os.path.join(quant_dir, name))
        if not os.path.isabs(target) or target != expected_target:
            raise RuntimeError(
                "%s: symlink target %r, expected absolute %r" % (path, target, expected_target)
            )
        if not os.path.isfile(path):
            raise RuntimeError("%s: dangling symlink" % path)
    log_ok(
        "%d quant shards are absolute symlinks to %s" % (len(quant_shard_names), quant_dir)
    )

    for relative in (
        "configuration.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "optional/quarot.safetensors",
    ):
        path = os.path.join(out_dir, relative)
        if not os.path.isfile(path):
            raise RuntimeError("missing auxiliary output file: %s" % path)
    log_ok(
        "auxiliary files present: configuration.json, tokenizer.json, "
        "tokenizer_config.json, optional/quarot.safetensors"
    )

    src_config = load_json(os.path.join(quant_dir, "config.json"))
    out_config = load_json(os.path.join(out_dir, "config.json"))
    expected_config = copy.deepcopy(src_config)
    text_config = expected_config.get("text_config")
    if not isinstance(text_config, dict):
        raise RuntimeError("quant config.json has no text_config object")
    text_config["num_nextn_predict_layers"] = 3
    if out_config != expected_config:
        raise RuntimeError(
            "config.json differs from the quant config outside "
            "text_config.num_nextn_predict_layers"
        )
    if out_config["text_config"]["num_nextn_predict_layers"] != 3:
        raise RuntimeError("config.json: text_config.num_nextn_predict_layers != 3")
    if out_config.get("quantization_config") != src_config.get("quantization_config"):
        raise RuntimeError("config.json: quantization_config block was modified")
    log_ok(
        "config.json: text_config.num_nextn_predict_layers == 3; "
        "quantization_config and all other fields preserved"
    )

    src_description = load_json(os.path.join(quant_dir, DESCRIPTION_NAME))
    out_description = load_json(os.path.join(out_dir, DESCRIPTION_NAME))
    if not isinstance(src_description, dict) or not isinstance(out_description, dict):
        raise RuntimeError("quant description must be a JSON object")
    all_mtp_keys = sorted(
        key for key in official_index["weight_map"] if key.startswith(MTP_PREFIX)
    )
    expected_description = dict(src_description)
    for key in all_mtp_keys:
        expected_description[key] = "FLOAT"
    if out_description != expected_description:
        changed = sorted(
            key for key in expected_description
            if out_description.get(key) != expected_description[key]
        )
        missing_keys = sorted(set(expected_description) - set(out_description))
        extra_keys = sorted(set(out_description) - set(expected_description))
        raise RuntimeError(
            "description mismatch: changed %d (%s), missing %d, extra %d"
            % (len(changed), changed[:3], len(missing_keys), len(extra_keys))
        )
    mtp_float_count = sum(
        1
        for key, value in out_description.items()
        if key.startswith(MTP_PREFIX) and value == "FLOAT"
    )
    if mtp_float_count != len(all_mtp_keys):
        raise RuntimeError(
            "description has %d mtp.* FLOAT entries, expected %d"
            % (mtp_float_count, len(all_mtp_keys))
        )
    log_ok(
        "description: all %d mtp.* names are FLOAT; non-mtp entries unchanged"
        % mtp_float_count
    )

    log_ok("verify-only passed")
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="make_dspark_ckpt.py",
        description=(
            "Build a DSpark-ready runtime checkpoint from a W4A8 quantized trunk "
            "plus the official mtp.* tensors."
        ),
    )
    parser.add_argument(
        "--quant-dir",
        required=True,
        help="W4A8 quantized trunk dir (read-only input)",
    )
    parser.add_argument(
        "--official-dir",
        required=True,
        help="official DeepSeek-V4.1-Flash dir (read-only input)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="output dir to create/populate (never one of the inputs)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and exit without writing anything",
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an already produced output directory",
    )
    parser.add_argument(
        "--rotation-file",
        default=None,
        help="全局旋转矩阵 safetensors；默认 <quant-dir>/optional/quarot.safetensors",
    )
    parser.add_argument(
        "--no-rotate-main-proj",
        action="store_true",
        help="不旋转 mtp.*.main_proj（默认旋转，符合 QuaRot 约定）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="debug: only consider the first N mtp tensors (sorted by name)",
    )
    return parser


def main(argv=None):
    global _MAIN_PROJ_ROTATION
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        log_error("--limit must be a positive integer")
        return 2
    if not args.no_rotate_main_proj and not args.verify_only:
        rot_path = args.rotation_file or os.path.join(
            args.quant_dir, "optional", "quarot.safetensors"
        )
        if not os.path.exists(rot_path):
            log_error(
                "rotation file %s not found; pass --rotation-file or --no-rotate-main-proj"
                % rot_path
            )
            return 2
        with safe_open(rot_path, framework="pt", device="cpu") as rot_handle:
            _MAIN_PROJ_ROTATION = rot_handle.get_tensor("global_rotation")
        log_info(
            "loaded global rotation %s from %s"
            % (tuple(_MAIN_PROJ_ROTATION.shape), rot_path)
        )
    try:
        if args.verify_only:
            return cmd_verify(args)
        if args.dry_run:
            return cmd_dry(args)
        return cmd_build(args)
    except KeyboardInterrupt:
        log_error("interrupted")
        return 130
    except Exception as exc:
        log_error("%s: %s" % (type(exc).__name__, exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
