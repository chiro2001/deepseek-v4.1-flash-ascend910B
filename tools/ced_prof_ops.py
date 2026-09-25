#!/usr/bin/env python3
"""从 kernel_details.csv 里切出 **decode 窗口**，给出算子级耗时排行。

为什么不用 `op_statistic.csv`：它是**全窗口**聚合，会把 prefill（几百毫秒的
大 kernel）和 decode（每步几十毫秒）混在一起，而我们要优化的是 decode。

用法：
  python3 tools/ced_prof_ops.py kernel_details.csv --t0 11.87 --t1 13.42
  python3 tools/ced_prof_ops.py kernel_details.csv --t0 11.87 --t1 13.42 --by-stream
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def load(path: str, t0: float | None, t1: float | None):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            try:
                r["_a"] = float(r["Start Time(us)"])
                r["_d"] = float(r["Duration(us)"])
                r["_s"] = int(r["Stream ID"])
            except (TypeError, ValueError, KeyError):
                continue
            rows.append(r)
    if not rows:
        return rows
    origin = min(r["_a"] for r in rows)
    for r in rows:
        r["_t"] = (r["_a"] - origin) / 1e6
    if t0 is not None:
        rows = [r for r in rows if r["_t"] >= t0]
    if t1 is not None:
        rows = [r for r in rows if r["_t"] <= t1]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernel_details_csv")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--by-stream", action="store_true", help="另外按流汇总")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows = load(args.kernel_details_csv, args.t0, args.t1)
    if not rows:
        print("窗口内没有 kernel")
        return 1
    t_min = min(r["_a"] for r in rows)
    t_max = max(r["_a"] + r["_d"] for r in rows)
    wall_ms = (t_max - t_min) / 1000.0

    agg: dict[str, dict] = defaultdict(lambda: {"n": 0, "us": 0.0, "core": set()})
    for r in rows:
        key = r.get("Name") or r.get("Type") or "?"
        # 名字尾部带 hash / 版本后缀，归一化掉才看得出模式
        short = key.split("_")[0] if "_" in key else key
        a = agg[short]
        a["n"] += 1
        a["us"] += r["_d"]
        a["core"].add((r.get("Accelerator Core") or "").strip())

    tot = sum(v["us"] for v in agg.values()) / 1000.0
    print(f"== {args.label or args.kernel_details_csv}")
    print(f"   窗口 {wall_ms/1000:.3f} s   kernel {len(rows)}   累加 core 时 {tot/1000:.3f} s "
          f"(并行度 {tot/wall_ms:.2f}×)")
    print()
    print(f"{'op':<44} {'n':>7} {'total_ms':>10} {'ms/step':>8} {'avg_us':>9} {'core':>12}")
    # 40 步是这份捕获里的 decode step 数（见 ced_prof_steps.py）
    nstep = 40
    for name, v in sorted(agg.items(), key=lambda kv: -kv[1]["us"])[: args.top]:
        print(f"{name[:44]:<44} {v['n']:>7} {v['us']/1000:>10.2f} "
              f"{v['us']/1000/nstep:>8.3f} {v['us']/max(1,v['n']):>9.1f} "
              f"{'/'.join(sorted(v['core']))[:12]:>12}")
    if args.by_stream:
        print()
        per: dict[int, float] = defaultdict(float)
        cnt: dict[int, int] = defaultdict(int)
        for r in rows:
            per[r["_s"]] += r["_d"]
            cnt[r["_s"]] += 1
        print(f"{'stream':>7} {'n':>8} {'total_ms':>10} {'duty':>7}")
        for sid, us in sorted(per.items(), key=lambda kv: -kv[1]):
            print(f"{sid:>7} {cnt[sid]:>8} {us/1000:>10.2f} {100*us/(wall_ms*1000):>6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
