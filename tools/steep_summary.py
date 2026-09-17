#!/usr/bin/env python3
"""steep_summary.py -- 从 p42 jsonl 汇总「steep / flat / shallow」run 分布。

steep  : A>=3.3 且 pos 严格单调（deep draft 接受）
flat   : A>=3.3 但 pos 平坦（重复循环）
shallow: A<3.3

用法: python3 steep_summary.py <glob> [...]
"""
import glob
import json
import sys

STEEP_A = 3.3


def classify(r):
    pp = r.get("accepted_per_pos") or {}
    try:
        v = [float(pp[str(i)]) for i in range(5)]
    except (KeyError, TypeError, ValueError):
        return "unknown", None
    A = r.get("accept_length") or 0
    mono = all(v[i] >= v[i + 1] - 0.02 for i in range(4))
    decay = (v[1] - v[4]) / v[0] if v[0] > 1e-9 else float("nan")
    if A >= STEEP_A:
        return ("steep" if (mono and decay >= 0.45) else "flat"), v
    return "shallow", v


def main(patterns):
    files = []
    for p in patterns:
        files += glob.glob(p)
    rows = []
    for f in sorted(set(files)):
        for line in open(f, encoding="utf-8", errors="replace"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("accept_length"):
                continue
            cls, v = classify(r)
            rows.append((f.split("/")[-1], cls, r["accept_length"], r.get("ms_per_step"), v))
    if not rows:
        print("(no rows)")
        return
    from collections import defaultdict
    by = defaultdict(list)
    for name, cls, A, ms, v in rows:
        by[cls].append((A, ms, v, name))
    print(f"total={len(rows)}  steep={len(by['steep'])}  flat={len(by['flat'])}  shallow={len(by['shallow'])}")
    for cls in ("steep", "flat", "shallow"):
        rs = by.get(cls) or []
        if not rs:
            continue
        As = sorted(x[0] for x in rs)
        ms = sorted(x[1] for x in rs if x[1])
        print(f"  {cls:8s} n={len(rs):3d} A_med={As[len(As)//2]:.3f} A_max={As[-1]:.3f} "
              f"ms_med={(ms[len(ms)//2] if ms else float('nan')):.2f}")
        for A, ms, v, name in sorted(rs, key=lambda x: -x[0])[:4]:
            print(f"      A={A:.3f} ms={ms:.2f} pos={[round(x,3) for x in (v or [])]} {name[:52]}")
    # 按 arm（文件名里的 _r<N>）分组，便于 A/B
    print("\nper-arm (文件名去掉 r<N>):")
    ag = defaultdict(list)
    for name, cls, A, ms, v in rows:
        import re
        key = re.sub(r"_r\d+\.jsonl$", "", name)
        ag[key].append((cls, A, ms))
    for k in sorted(ag):
        rs = ag[k]
        steep = sum(1 for c, _, _ in rs if c == "steep")
        As = sorted(a for _, a, _ in rs)
        ms = sorted(m for _, _, m in rs if m)
        print(f"  {k[:56]:56s} n={len(rs):3d} steep={steep} A_med={As[len(As)//2]:.3f} "
              f"A_max={As[-1]:.3f} ms_med={(ms[len(ms)//2] if ms else 0):.2f}")


if __name__ == "__main__":
    import os
    _here = os.path.dirname(os.path.abspath(__file__))
    _pkg = os.path.dirname(_here)
    main(sys.argv[1:] or [os.path.join(_pkg, "results", "*", "p42_t4_quote_131072_*.jsonl"),
                          os.path.join(_pkg, "logs_meta", "samples", "*_131072_*.jsonl")])
