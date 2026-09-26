#!/usr/bin/env python3
"""Compare one-chip CED PD and full-model top-logprob probe artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def indexed(path: Path) -> tuple[dict, dict[tuple[int, int], dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, {(row["length"], row["repeat"]): row for row in data["rows"]}


def max_common_delta(left: dict, right: dict) -> float | None:
    a, b = left.get("top_logprobs") or {}, right.get("top_logprobs") or {}
    common = a.keys() & b.keys()
    return max((abs(a[token] - b[token]) for token in common), default=None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pd", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pd, pd_rows = indexed(args.pd)
    baseline, base_rows = indexed(args.baseline)
    if pd["token_pattern"] != baseline["token_pattern"] or pd_rows.keys() != base_rows.keys():
        raise SystemExit("token pattern or (length, repeat) coverage differs")

    result = []
    for length, repeat in sorted(pd_rows):
        left, right = pd_rows[(length, repeat)], base_rows[(length, repeat)]
        a, b = left.get("top_logprobs") or {}, right.get("top_logprobs") or {}
        common = a.keys() & b.keys()
        union = a.keys() | b.keys()
        row = {
            "length": length,
            "repeat": repeat,
            "pd_http": left["http"],
            "baseline_http": right["http"],
            "selected_equal": left.get("text") == right.get("text"),
            "pd_text": left.get("text"),
            "baseline_text": right.get("text"),
            "selected_logprob_abs_delta": abs(left["token_logprob"] - right["token_logprob"])
            if left.get("token_logprob") is not None and right.get("token_logprob") is not None else None,
            "top20_common": len(common),
            "top20_union": len(union),
            "top20_jaccard": len(common) / len(union) if union else None,
            "top20_max_common_logprob_abs_delta": max_common_delta(left, right),
        }
        result.append(row)
        print(json.dumps(row, ensure_ascii=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": result}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
