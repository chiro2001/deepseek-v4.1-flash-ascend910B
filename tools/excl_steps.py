#!/usr/bin/env python3
"""多步「独占贡献」分析。

独占贡献 = union(该步全部任务) − union(去掉某 stream 后的任务)
即：把这个 stream 的活儿全删掉，这一步的墙钟最多能缩多少（上界）。

切步：用主模型 stream 上 `GroupedMatmulSwigluQuantV2`（每步 40 个）的锚点，
每 40 个锚点 = 1 步。比"手填时间窗"可靠（手填窗口可能跨步）。

用法:
  excl_steps.py <mindstudio_profiler_output> [anchor_stream] [n_steps]
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd


def union(pairs: np.ndarray) -> float:
    if len(pairs) == 0:
        return 0.0
    order = pairs[np.argsort(pairs[:, 0])]
    total = 0.0
    cs, ce = order[0]
    for s, e in order[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    return total + ce - cs


def idle_gaps(pairs: np.ndarray, thresh_ms: float = 0.020) -> list[float]:
    order = pairs[np.argsort(pairs[:, 0])]
    gaps: list[float] = []
    _, ce = order[0]
    for s, e in order[1:]:
        if s > ce:
            if s - ce >= thresh_ms:
                gaps.append(s - ce)
            ce = e
        else:
            ce = max(ce, e)
    return gaps


def main() -> int:
    prof_dir = sys.argv[1]
    anchor_stream = float(sys.argv[2]) if len(sys.argv) > 2 else 158.0
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    cols = ["OP Type", "Stream ID", "Task Start Time(us)", "Task Duration(us)", "Task Type"]
    frames = [
        pd.read_csv(f, usecols=cols, low_memory=False)
        for f in sorted(glob.glob(prof_dir + "/op_summary*.csv"))
    ]
    df = pd.concat(frames, ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
    origin = df["Task Start Time(us)"].min()
    df["s"] = (df["Task Start Time(us)"] - origin) / 1000.0
    df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0

    anchors = df[
        (df["OP Type"] == "GroupedMatmulSwigluQuantV2") & (df["Stream ID"] == anchor_stream)
    ].sort_values("s")
    starts = anchors["s"].to_numpy()[::40]
    if len(starts) < 2:
        print("锚点不足，无法切步")
        return 1
    mid = len(starts) // 3
    bounds = starts[mid : mid + n_steps + 1]
    print(f"锚点 {len(anchors)} 个 / 步 {len(starts)} 步；本次取 {len(bounds)-1} 个步区间")

    rows: list[dict] = []
    all_gaps: list[float] = []
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        sub = df[(df["s"] < hi) & (df["e"] > lo)].copy()
        if len(sub) < 500:
            continue
        sub["s"] = sub["s"].clip(lo, hi)
        sub["e"] = sub["e"].clip(lo, hi)
        span = hi - lo
        u_all = union(sub[["s", "e"]].to_numpy())
        all_gaps += idle_gaps(sub[["s", "e"]].to_numpy())
        rec = {"step": k, "len": span, "union": u_all, "n": len(sub)}
        for sid, grp in sub.groupby("Stream ID"):
            rest = sub[sub["Stream ID"] != sid]
            rec[f"excl_{sid}"] = u_all - union(rest[["s", "e"]].to_numpy())
            rec[f"busy_{sid}"] = (grp["e"] - grp["s"]).sum()
            rec[f"n_{sid}"] = len(grp)
        rows.append(rec)

    res = pd.DataFrame(rows)
    mean_len = res["len"].mean()
    print(f"平均步长 {mean_len:.2f} ms   平均任务数 {res['n'].mean():.0f}")
    print(
        f"平均并集 {res['union'].mean():.3f} ms "
        f"({res['union'].mean()/mean_len*100:.1f}%)   "
        f"平均空隙 {mean_len-res['union'].mean():.3f} ms"
    )
    print(
        f"真空闲(>=20us) 合计 {sum(all_gaps):.3f} ms / {res['len'].sum():.1f} ms "
        f"=> 每步 {sum(all_gaps)/len(res):.3f} ms（{sum(all_gaps)/res['len'].sum()*100:.2f}%）"
    )
    print()
    sids = sorted(
        (c[len("excl_") :] for c in res.columns if c.startswith("excl_")),
        key=lambda s: -res[f"excl_{s}"].mean(),
    )
    print(f"{'stream':>8}{'n':>7}{'busy_ms':>10}{'excl_ms':>10}{'excl%':>8}")
    for sid in sids:
        print(
            f"{float(sid):>8.0f}{res[f'n_{sid}'].mean():>7.0f}"
            f"{res[f'busy_{sid}'].mean():>10.3f}{res[f'excl_{sid}'].mean():>10.3f}"
            f"{res[f'excl_{sid}'].mean()/mean_len*100:>7.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
