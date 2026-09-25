#!/usr/bin/env python3
"""把一次 profiler 捕获切成 **decode step**，给出每步的时延构成。

为什么需要它：开了推测解码后，**step 才是引擎的工作量单位**
（每步 = 一次 40 层 target forward + 一次 3 层 draft forward + 验证）。
ms/token 会被接受长度 A 稀释（A≈1 时看着很快，其实每步只出 1 个 token），
所以要降时延必须看 step。

切分原理：主计算流（任务最多的 AI_VECTOR_CORE/MIX_AIC 流）上一个 decode
step 是一段**连续的 task 串**；步与步之间会有一个等待间隙（等通信/等调度）。
用 `--gap-ms` 作为间隙阈值把计算流切成 block，再对每个 block 统计各流忙时。

用法：
  python3 tools/ced_prof_steps.py task_time.csv --gap-ms 0.5
  python3 tools/ced_prof_steps.py task_time.csv --t0 11.8 --t1 13.4 --gap-ms 0.3
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict


def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("task_start(us)") or "").strip():
                continue
            try:
                r["_s"] = int(r["stream_id"])
                r["_a"] = float(r["task_start(us)"])
                r["_d"] = float(r["task_time(us)"])
            except (TypeError, ValueError):
                continue
            r["_e"] = r["_a"] + r["_d"]
            rows.append(r)
    return rows


def pick_compute_stream(rows: list[dict]) -> int:
    """任务数最多的、以 AI_VECTOR_CORE / MIX_AIC 为主的流（= 主计算流）。"""
    per: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        per[r["_s"]][r.get("kernel_type") or ""] += 1
    best, best_n = None, -1
    for sid, kinds in per.items():
        n = kinds.get("AI_VECTOR_CORE", 0) + kinds.get("MIX_AIC", 0)
        if n > best_n:
            best, best_n = sid, n
    assert best is not None
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_time_csv")
    ap.add_argument("--gap-ms", type=float, default=0.5,
                    help="计算流上判定为 step 边界的间隙阈值（默认 0.5 ms）")
    ap.add_argument("--t0", type=float, default=None, help="只看这个起始时刻（秒，相对窗口）")
    ap.add_argument("--t1", type=float, default=None, help="只看这个结束时刻（秒）")
    ap.add_argument("--min-step-ms", type=float, default=5.0,
                    help="短于这个的 block 当噪声丢掉（默认 5 ms）")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows = load(args.task_time_csv)
    if not rows:
        print("没有有效 task")
        return 1
    origin = min(r["_a"] for r in rows)
    compute = pick_compute_stream(rows)

    def keep(r):
        t = (r["_a"] - origin) / 1000.0
        if args.t0 is not None and t < args.t0:
            return False
        if args.t1 is not None and t > args.t1:
            return False
        return True

    rows = [r for r in rows if keep(r)]
    if not rows:
        print("时间窗内没有 task")
        return 1
    t_min = min(r["_a"] for r in rows)
    t_max = max(r["_e"] for r in rows)
    win_ms = (t_max - t_min) / 1000.0
    print(f"== {args.label or args.task_time_csv}")
    print(f"   窗口 {win_ms/1000:.3f} s   任务 {len(rows)}   主计算流 {compute}")

    # --- 在计算流上切 block ---
    comp = sorted((r for r in rows if r["_s"] == compute), key=lambda r: r["_a"])
    gap_us = args.gap_ms * 1000.0
    blocks: list[list[dict]] = []
    cur = [comp[0]]
    for r in comp[1:]:
        if r["_a"] - cur[-1]["_e"] > gap_us:
            blocks.append(cur)
            cur = [r]
        else:
            cur.append(r)
    blocks.append(cur)
    blocks = [b for b in blocks if (b[-1]["_e"] - b[0]["_a"]) / 1000.0 >= args.min_step_ms]
    if not blocks:
        print("  没有超过 --min-step-ms 的 block")
        return 1

    # --- 按 block 统计各流忙时 ---
    busy_by_stream: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        busy_by_stream[r["_s"]].append((r["_a"], r["_e"]))
    for v in busy_by_stream.values():
        v.sort()

    def overlap_ms(sid: int, a: float, b: float) -> float:
        """stream sid 在 [a,b] 内的忙时（毫秒，合并区间）。"""
        iv = busy_by_stream.get(sid, [])
        tot = 0.0
        for s, e in iv:
            if e <= a:
                continue
            if s >= b:
                break
            tot += min(e, b) - max(s, a)
        return tot / 1000.0

    comm_streams = [
        sid for sid in busy_by_stream
        if sid != compute
        and any(
            (r.get("kernel_type") or "").startswith("COMMUNICATION")
            for r in rows if r["_s"] == sid
        )
    ]
    print(f"   通信流 {sorted(comm_streams)}   block(gap>{args.gap_ms}ms) = {len(blocks)}")
    print()
    print(f"{'step':>5} {'start_s':>9} {'dur_ms':>8} {'compute_ms':>11} "
          f"{'comm_ms':>8} {'comp%':>6} {'gap_ms':>7}")

    durs, comps = [], []
    for i, b in enumerate(blocks):
        a, z = b[0]["_a"], b[-1]["_e"]
        dur = (z - a) / 1000.0
        c = overlap_ms(compute, a, z)
        cm = sum(overlap_ms(s, a, z) for s in comm_streams)
        durs.append(dur)
        comps.append(c)
        print(f"{i:>5} {(a-t_min)/1e6:>9.3f} {dur:>8.3f} {c:>11.3f} {cm:>8.3f} "
              f"{100.0*c/dur:>5.1f}% {dur-c:>7.3f}")
    if len(durs) > 2:
        mid = durs[1:-1] if len(durs) > 4 else durs
        midc = comps[1:-1] if len(comps) > 4 else comps
        print()
        print(f"   中位 step = {statistics.median(mid):.2f} ms   "
              f"中位 compute = {statistics.median(midc):.2f} ms   "
              f"compute 占比 {100.0*sum(midc)/sum(mid):.1f}%")
        print(f"   首步 {durs[0]:.2f} ms   末步 {durs[-1]:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
