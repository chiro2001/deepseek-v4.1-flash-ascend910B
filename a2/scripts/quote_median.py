#!/usr/bin/env python3
"""从 quote 探针的 jsonl 里取中位数（ms/step、A、tok/s）。

为什么要单独一个文件：`bneck_suite_8card.sh` 里内联 python 会被补丁工具的缩进规则绊住，
而且这段逻辑要重复用 —— 单独成文件更可靠（本仓纪律：能代码痕迹就不靠手抄）。

用法： python3 quote_median.py <file.jsonl> [ms|a|ts]
  ms → 中位 ms_per_step；a → 中位 accept_length；ts → 中位 decode_tok_s
没数据时打印空串（调用方用 `?` 兜底）。
"""
from __future__ import annotations

import json
import statistics
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print("", end="")
        return 2
    path, what = sys.argv[1], sys.argv[2]
    key = {"ms": "ms_per_step", "a": "accept_length", "ts": "decode_tok_s"}.get(what)
    if key is None:
        print("", end="")
        return 2
    rows = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("_meta"):
                    continue
                if not obj.get("metrics_ok") or obj.get("error"):
                    continue
                rows.append(obj)
    except OSError:
        print("", end="")
        return 2
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        print("", end="")
        return 1
    med = statistics.median(vals)
    print(f"{med:.3f}" if key == "ms_per_step" else (f"{med:.3f}" if key == "accept_length" else f"{med:.1f}"), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
