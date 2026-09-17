#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the Engram-on-DRAM derived model directory.

This is the reproducible builder for
``models/out/v41-w4a8-engram-dr`` (see docs/ENGRAM_DRAM_PLACEMENT.md §6.2.1
and logs/perf/engram_dr_prep.md).

Key contract handled here (P17 §6.2.1 correction):
  * int8 embed/scale files live in a sub-directory ``engram_int8/`` so the
    default loader's top-level ``*.safetensors`` glob never materialises the
    98 GB tensors.
  * ``engram_extra.safetensors`` top-level shard holds the real BF16
    ``wkv``/``q_weight``/``k_weight`` plus [1,1] embed placeholders that only
    trigger ``NodeShardedEngram.load_checkpoint``.
  * Official FP8 (E4M3 + E8M0 32x32 block) ``wkv`` is dequantised to BF16.
    The LAST ``hidden_size`` rows (the V projection) are folded with the
    QuaRot global rotation: ``V_folded = Q.T @ V``.  The K rows and the
    q/k gate weights stay in the original basis.  This matches the official
    W8A8 export contract (value_basis=quarot_global,
    key_and_gate_basis=original) that vllm-ascend's ``engram_gate`` expects.

No NPU is required.  Reading the root-only W4A8 optional/quarot.safetensors
file requires root (run this script with sudo -n), or run it inside the
serving container.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

P_DEFAULT = Path("/home/user/projects/dsv41")
BLOCK = 32
WIDTH = 256
HIDDEN = 5120
OUT_FEATURES = 25600
VALUE_ROWS_START = OUT_FEATURES - HIDDEN  # 20480, the last 5120 rows are V
ENGRAM_LAYERS = (1, 14)
ENGRAM_ROTATION_CONFIG = {
    "value_projection_rotated": True,
    "value_basis": "quarot_global",
    "key_and_gate_basis": "original",
    "runtime_delta_rotation": False,
}


def log(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


def e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Convert F8_E8M0 (or a uint8 view) to FP32 exactly."""
    if scale.dtype == torch.float32:
        return scale
    try:
        return scale.to(torch.float32)
    except (RuntimeError, TypeError):
        raw = scale.view(torch.uint8).to(torch.int32) - 127
        return torch.ldexp(torch.ones_like(raw, dtype=torch.float32), raw)


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 weight [R, C] x E8M0 scale [R//32, C//32] -> FP32 [R, C]."""
    rows, cols = weight.shape
    if rows % BLOCK or cols % BLOCK:
        raise ValueError(f"unsupported block shape {tuple(weight.shape)}")
    s = e8m0_to_fp32(scale)
    if tuple(s.shape) != (rows // BLOCK, cols // BLOCK):
        raise ValueError(f"scale shape {tuple(s.shape)} != {(rows // BLOCK, cols // BLOCK)}")
    w = weight.float().unflatten(0, (-1, BLOCK)).unflatten(-1, (-1, BLOCK))
    return (w * s[:, None, :, None]).reshape(rows, cols)


def load_global_rotation(path: Path) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if "global_rotation" not in keys:
            raise ValueError(f"{path}: no global_rotation tensor (keys={keys})")
        rot = handle.get_tensor("global_rotation")
    if rot.dtype != torch.float32 or rot.ndim != 2 or rot.shape[0] != rot.shape[1]:
        raise ValueError(f"{path}: bad global_rotation {rot.dtype} {tuple(rot.shape)}")
    if rot.shape[0] % BLOCK:
        raise ValueError(f"{path}: global_rotation dim not divisible by {BLOCK}")
    block = rot[:BLOCK, :BLOCK]
    expect = torch.block_diag(*([block] * (rot.shape[0] // BLOCK)))
    if not torch.equal(rot, expect):
        raise ValueError(f"{path}: global_rotation is not repeated {BLOCK}x{BLOCK} block diag")
    if not bool(torch.isfinite(rot).all()):
        raise ValueError(f"{path}: global_rotation contains NaN/Inf")
    log(f"global_rotation {tuple(rot.shape)} {rot.dtype}, extra keys={keys}")
    return rot.contiguous()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", default=str(P_DEFAULT), help="project root P")
    parser.add_argument("--src", default=None, help="W4A8 serve checkpoint dir")
    parser.add_argument("--out", default=None, help="derived output dir")
    parser.add_argument("--official", default=None, help="official FP8 checkpoint dir")
    parser.add_argument("--int8", default=None, help="converted engram-int8 dir")
    parser.add_argument("--force", action="store_true", help="rebuild an existing output dir")
    args = parser.parse_args()

    p = Path(args.p)
    src = Path(args.src) if args.src else p / "models/out/v41-w4a8-dspark"
    out = Path(args.out) if args.out else p / "models/out/v41-w4a8-engram-dr"
    official = Path(args.official) if args.official else p / "models/DeepSeek-V4.1-Flash"
    int8 = Path(args.int8) if args.int8 else p / "models/out/engram-int8"

    for name, path in (("src", src), ("official", official), ("int8", int8)):
        if not path.is_dir():
            raise SystemExit(f"{name} dir not found: {path}")

    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} already exists; pass --force to rebuild")
        if out.name != "v41-w4a8-engram-dr":
            raise SystemExit(f"refusing to delete unexpected dir {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    log(f"created {out}")

    # ---- 1. symlink the W4A8 backbone except config/index -----------------
    for entry in sorted(src.iterdir(), key=lambda item: item.name):
        if entry.name in ("config.json", "quant_model_weights.safetensors.index.json"):
            continue
        os.symlink(str(entry.absolute()), str(out / entry.name))
    log(f"symlinked {len(list(src.iterdir())) - 2} backbone entries")

    # ---- 2. real int8 files in a sub-directory (top-level glob cannot see)
    engram_int8 = out / "engram_int8"
    engram_int8.mkdir()
    for layer in ENGRAM_LAYERS:
        for kind in ("weight", "scale"):
            name = f"layers_{layer}_engram_embed.{kind}.safetensors"
            source = int8 / name
            if not source.is_file():
                raise SystemExit(f"missing int8 file: {source}")
            os.symlink(str(source.absolute()), str(engram_int8 / name))
    log(f"engram_int8/: {sorted(item.name for item in engram_int8.iterdir())}")

    # ---- 3. config.json ---------------------------------------------------
    config = json.loads((src / "config.json").read_text())
    text_config = config["text_config"]
    original_layer_ids = list(text_config.get("engram_layer_ids", []))
    text_config["engram_layer_ids"] = list(ENGRAM_LAYERS)
    text_config["max_position_embeddings"] = 1048576
    config["engram_rotation_config"] = dict(ENGRAM_ROTATION_CONFIG)
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    log(f"config.json: engram_layer_ids {original_layer_ids} -> {list(ENGRAM_LAYERS)}, "
        f"max_position_embeddings -> 1048576, engram_rotation_config added")

    # ---- 4. quant index ---------------------------------------------------
    index = json.loads((src / "quant_model_weights.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    source_keys = len(weight_map)
    for layer in ENGRAM_LAYERS:
        weight_map[f"layers.{layer}.engram.embed.weight"] = (
            f"engram_int8/layers_{layer}_engram_embed.weight.safetensors"
        )
        weight_map[f"layers.{layer}.engram.embed.scale"] = (
            f"engram_int8/layers_{layer}_engram_embed.scale.safetensors"
        )
        for name in ("wkv.weight", "q_weight", "k_weight"):
            weight_map[f"layers.{layer}.engram.{name}"] = "engram_extra.safetensors"
    (out / "quant_model_weights.safetensors.index.json").write_text(json.dumps(index))
    log(f"quant index: {source_keys} -> {len(weight_map)} keys (+{len(weight_map) - source_keys} engram)")

    # ---- 5. engram_extra.safetensors -------------------------------------
    official_index = json.loads((official / "model.safetensors.index.json").read_text())["weight_map"]
    rotation = load_global_rotation(out / "optional" / "quarot.safetensors")
    if rotation.shape[0] != HIDDEN:
        raise SystemExit(f"global_rotation dim {rotation.shape[0]} != hidden_size {HIDDEN}")

    tensors: dict[str, torch.Tensor] = {}
    for layer in ENGRAM_LAYERS:
        wkv_key = f"layers.{layer}.engram.wkv.weight"
        scale_key = f"layers.{layer}.engram.wkv.scale"
        shard = official / official_index[wkv_key]
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            wkv_fp8 = handle.get_tensor(wkv_key)
            wkv_scale = handle.get_tensor(scale_key)
            q_weight = handle.get_tensor(f"layers.{layer}.engram.q_weight")
            k_weight = handle.get_tensor(f"layers.{layer}.engram.k_weight")
        if tuple(wkv_fp8.shape) != (OUT_FEATURES, HIDDEN + 1024):
            raise SystemExit(f"{wkv_key}: unexpected shape {tuple(wkv_fp8.shape)}")
        if tuple(q_weight.shape) != (4, HIDDEN) or q_weight.dtype != torch.bfloat16:
            raise SystemExit(f"layers.{layer}.engram.q_weight: {q_weight.dtype} {tuple(q_weight.shape)}")
        if tuple(k_weight.shape) != (4, HIDDEN) or k_weight.dtype != torch.bfloat16:
            raise SystemExit(f"layers.{layer}.engram.k_weight: {k_weight.dtype} {tuple(k_weight.shape)}")

        w32 = dequant_fp8_block(wkv_fp8, wkv_scale)
        if not bool(torch.isfinite(w32).all()):
            raise SystemExit(f"{wkv_key}: NaN/Inf after FP8 dequant")
        key_rows = w32[:VALUE_ROWS_START].to(torch.bfloat16)
        value_rows = (rotation.T @ w32[VALUE_ROWS_START:]).to(torch.bfloat16)
        wkv_bf16 = torch.cat([key_rows, value_rows], dim=0).contiguous()

        # evidence numbers: naive P17 output is *not* in the rotated basis
        naive = w32[VALUE_ROWS_START:].to(torch.bfloat16).float()
        fold_delta = float((value_rows.float() - naive).abs().max().item())
        alt_fold = (rotation.T @ w32[VALUE_ROWS_START:].to(torch.bfloat16).float()).to(torch.bfloat16)
        alt_delta = float((value_rows.float() - alt_fold.float()).abs().max().item())
        log(f"layer {layer}: wkv {tuple(wkv_bf16.shape)} bf16, "
            f"max|V_folded - V_unfolded|={fold_delta:.6g}, "
            f"max|fp32_fold - bf16_fold|={alt_delta:.6g}")

        tensors[wkv_key] = wkv_bf16
        tensors[f"layers.{layer}.engram.q_weight"] = q_weight.to(torch.bfloat16).contiguous()
        tensors[f"layers.{layer}.engram.k_weight"] = k_weight.to(torch.bfloat16).contiguous()
        # placeholders: only exist so the outer loader yields the name and calls
        # NodeShardedEngram.load_checkpoint (real data lives in engram_int8/).
        tensors[f"layers.{layer}.engram.embed.weight"] = torch.zeros(1, 1, dtype=torch.int8)
        tensors[f"layers.{layer}.engram.embed.scale"] = torch.zeros(1, 1, dtype=torch.float32)

    extra_path = out / "engram_extra.safetensors"
    save_file(tensors, str(extra_path))
    log(f"wrote {extra_path} ({extra_path.stat().st_size} bytes, {len(tensors)} tensors)")
    log("ENGRAM_DR_BUILD_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
