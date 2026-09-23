#!/usr/bin/env python3
"""Locate the first CED encoder or Engram divergence at one token position."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metrics(left: np.ndarray, right: np.ndarray) -> dict:
    if left.shape != right.shape:
        return {"shape_match": False, "left_shape": left.shape, "right_shape": right.shape}
    a, b = left.astype(np.float64).ravel(), right.astype(np.float64).ravel()
    delta = a - b
    base_norm = np.linalg.norm(b)
    norms = np.linalg.norm(a) * base_norm
    return {
        "shape_match": True,
        "shape": left.shape,
        "exact": bool(np.array_equal(left, right)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "rel_l2": float(np.linalg.norm(delta) / base_norm) if base_norm else None,
        "cosine": float(np.dot(a, b) / norms) if norms else None,
    }


def load(root: Path, rank: int, layer: int, stage: str, position: int) -> dict[str, np.ndarray]:
    path = root / f"rank{rank}" / f"layer{layer:02d}_{stage}_pos{position}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        for key, value in (("position", position), ("layer", layer), ("tp_rank", rank)):
            if int(data[key]) != value:
                raise ValueError(f"{path}: {key}={int(data[key])}, expected {value}")
        if str(data["stage"]) != stage:
            raise ValueError(f"{path}: wrong stage")
        return {key: data[key].copy() for key in data.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", type=Path, required=True)
    parser.add_argument("--decode", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--layers", default="0,1,2,13,14,15,19,20")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    if args.tp < 1 or not layers:
        parser.error("--tp must be positive and --layers must not be empty")
    rows = []
    for rank in range(args.tp):
        for layer in layers:
            stages = ("pre", "after_engram", "post") if layer in (1, 14) else ("pre", "post")
            for stage in stages:
                roles = {
                    "baseline": load(args.baseline, rank, layer, stage, args.position),
                    "decode": load(args.decode, rank, layer, stage, args.position),
                }
                if layer < 20 or stage == "pre":
                    roles["prefill"] = load(args.prefill, rank, layer, stage, args.position)
                for left, right in (
                    ("prefill", "baseline"),
                    ("decode", "prefill"),
                    ("decode", "baseline"),
                ):
                    if left not in roles or right not in roles:
                        continue
                    for plane in ("input_id", "hidden_states", "pre_mix", "engram_lookup", "engram_token_mask"):
                        if plane not in roles[left] and plane not in roles[right]:
                            continue
                        if plane not in roles[left] or plane not in roles[right]:
                            raise ValueError(f"rank{rank} layer{layer} {stage}: missing {plane} in {left}/{right}")
                        rows.append({
                            "rank": rank,
                            "layer": layer,
                            "stage": stage,
                            "plane": plane,
                            "pair": f"{left}_vs_{right}",
                            **metrics(roles[left][plane], roles[right][plane]),
                        })
    summary = {
        "position": args.position,
        "tp": args.tp,
        "layers": layers,
        "rows": len(rows),
        "input_id_mismatches": sum(
            1 for row in rows if row["plane"] == "input_id" and not row.get("exact", False)
        ),
        "first_hidden_difference": {
            pair: next(
                ({"rank": row["rank"], "layer": row["layer"], "stage": row["stage"]}
                 for row in rows if row["pair"] == pair and row["plane"] == "hidden_states"
                 and not row.get("exact", False)),
                None,
            )
            for pair in ("prefill_vs_baseline", "decode_vs_prefill", "decode_vs_baseline")
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
