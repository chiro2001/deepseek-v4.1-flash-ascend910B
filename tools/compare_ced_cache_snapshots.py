#!/usr/bin/env python3
"""Compare CED D replay cache rows with a full-40-layer tiny baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metrics(pd: np.ndarray, baseline: np.ndarray) -> dict:
    if pd.shape != baseline.shape:
        return {"shape_pd": pd.shape, "shape_baseline": baseline.shape, "shape_match": False}
    left = pd.astype(np.float64).ravel()
    right = baseline.astype(np.float64).ravel()
    delta = left - right
    denom = np.linalg.norm(right)
    norms = np.linalg.norm(left) * denom
    return {
        "shape_match": True,
        "shape": pd.shape,
        "exact": bool(np.array_equal(pd, baseline)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "rel_l2": float(np.linalg.norm(delta) / denom) if denom else None,
        "cosine": float(np.dot(left, right) / norms) if norms else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pd", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for layer in range(40):
        name = f"layer{layer:02d}_pos{args.position}.npz"
        pd_path, base_path = args.pd / name, args.baseline / name
        if not pd_path.is_file() or not base_path.is_file():
            raise SystemExit(f"missing layer {layer} snapshot: {pd_path} / {base_path}")
        with np.load(pd_path) as pd_data, np.load(base_path) as base_data:
            planes = ("swa", "long_kv", "index_k", "index_scale") if layer == 20 else ("swa",)
            for plane in planes:
                row = {"layer": layer, "plane": plane, **metrics(pd_data[plane], base_data[plane])}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
    summary = {
        "position": args.position,
        "rows": len(rows),
        "exact_rows": sum(row.get("exact", False) for row in rows),
        "max_abs": max(row.get("max_abs", 0.0) for row in rows),
        "max_rel_l2": max((row.get("rel_l2") or 0.0) for row in rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
