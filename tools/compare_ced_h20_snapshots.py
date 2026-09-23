#!/usr/bin/env python3
"""Compare the encoder output entering layer 20 across P, D, and baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(left: np.ndarray, right: np.ndarray) -> dict:
    if left.shape != right.shape:
        return {"shape_match": False, "left_shape": left.shape, "right_shape": right.shape}
    a = left.astype(np.float64).ravel()
    b = right.astype(np.float64).ravel()
    delta = a - b
    norm_b = np.linalg.norm(b)
    norm_a = np.linalg.norm(a)
    return {
        "shape_match": True,
        "shape": left.shape,
        "exact": bool(np.array_equal(left, right)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "rel_l2": float(np.linalg.norm(delta) / norm_b) if norm_b else None,
        "cosine": float(np.dot(a, b) / (norm_a * norm_b)) if norm_a and norm_b else None,
    }


def load(directory: Path, rank: int, position: int) -> dict[str, np.ndarray]:
    path = directory / f"rank{rank}_pos{position}.npz"
    with np.load(path) as archive:
        if int(archive["position"]) != position or int(archive["rank"]) != rank:
            raise ValueError(f"snapshot metadata mismatch: {path}")
        return {name: archive[name].copy() for name in ("hidden_states", "pre_mix")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", type=Path, required=True)
    parser.add_argument("--decode", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for rank in range(args.tp):
        snapshots = {
            role: load(directory, rank, args.position)
            for role, directory in (
                ("prefill", args.prefill),
                ("decode", args.decode),
                ("baseline", args.baseline),
            )
        }
        for plane in ("hidden_states", "pre_mix"):
            for left, right in (
                ("prefill", "baseline"),
                ("decode", "prefill"),
                ("decode", "baseline"),
            ):
                row = {
                    "rank": rank,
                    "position": args.position,
                    "plane": plane,
                    "pair": f"{left}_vs_{right}",
                    **compare(snapshots[left][plane], snapshots[right][plane]),
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
    summary = {
        "position": args.position,
        "tp": args.tp,
        "rows": len(rows),
        "exact_rows": sum(bool(row.get("exact")) for row in rows),
        "max_abs": max((row.get("max_abs", 0.0) for row in rows), default=0.0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
