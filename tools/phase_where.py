#!/usr/bin/env python3
"""某个 stream（或算子族）在 decode 步内的**时间位置分布**。

用途：判断"这个阶段是早发晚执行（被排队），还是本来就晚"。
如果它稳定出现在步尾，就说明它的**提交**被上游依赖卡住了，
这时"提前提交以与前面的阶段重叠"才有意义（否则早已重叠）。

用法: phase_where.py <mindstudio_profiler_output> <选择器> [n_steps] [bins]
选择器：`s<id>` 表示 stream（例 `s35`）；`op:<OP Type>` 表示算子族。
例:  phase_where.py $PROF s35 40 10                 # stream 35 落在步内哪个时间段
     phase_where.py $PROF op:SparseFlashMlaMetadata 40 10
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd


def main() -> int:
    prof_dir = sys.argv[1]
    spec = sys.argv[2]
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    bins = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    cols = ["OP Type", "Stream ID", "Task Start Time(us)", "Task Duration(us)"]
    frames = [
        pd.read_csv(f, usecols=cols, low_memory=False)
        for f in sorted(glob.glob(prof_dir + "/op_summary*.csv")
        or glob.glob(prof_dir + "/kernel_details.csv"))
    ]
    df = pd.concat(frames, ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
    origin = df["Task Start Time(us)"].min()
    df["s"] = (df["Task Start Time(us)"] - origin) / 1000.0
    df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0

    anchors = df[(df["OP Type"] == "GroupedMatmulSwigluQuantV2") & (df["Stream ID"] == 158.0)].sort_values("s")
    starts = anchors["s"].to_numpy()[::40]
    mid = len(starts) // 3

    if spec.startswith("op:"):
        kind, val = "op", spec[3:]
    elif spec.startswith("s"):
        kind, val = "stream", spec[1:]
    else:
        raise SystemExit(f"无法识别的选择器 {spec!r}（用 s35 或 op:名字）")
    hist = np.zeros(bins)
    rel_all: list[float] = []
    lens: list[float] = []
    for k in range(mid, mid + n_steps):
        lo, hi = starts[k], starts[k + 1]
        span = hi - lo
        lens.append(span)
        w = df[(df["s"] < hi) & (df["e"] > lo)].copy()
        if kind == "op":
            w = w[w["OP Type"] == val]
        else:
            w = w[w["Stream ID"] == float(val)]
        if len(w) == 0:
            continue
        rel = ((w["s"] - lo) / span).clip(0, 1)
        rel_all += list(rel)
        for r in rel:
            hist[min(int(r * bins), bins - 1)] += 1

    total = hist.sum()
    print(f"{spec}  步数={len(lens)}  任务数={total}  步长中位={np.median(lens):.2f} ms")
    if total == 0:
        return 0
    print("步内相对位置分布（1 = 步尾）:")
    for i, v in enumerate(hist):
        bar = "#" * int(round(v / total * 60))
        print(f"  {i/bins:.1f}-{(i+1)/bins:.1f}  {int(v):5d} ({v/total*100:5.1f}%) {bar}")
    q = np.percentile(rel_all, [10, 50, 90])
    print(f"相对位置 p10/p50/p90 = {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
