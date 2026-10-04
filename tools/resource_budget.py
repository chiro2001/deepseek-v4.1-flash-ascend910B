#!/usr/bin/env python3
"""主图一步的**硬件单元预算账**：把每个算子的各资源分量按步求和。

为什么需要：`Task Duration` 只告诉你"这个算子多久"，
但多个单元是并行工作的（mac/aiv/scalar/mte2 重叠），只看 duration 会误判该优化谁。
把每个单元**各自的占用时间**加起来，才能回答"这一步到底被哪个单元卡住"。

口径注意：各分量是**并集意义上的占用**，不同算子之间也可能重叠，
所以"总和"是**上界**；如果某个单元的总和已经接近步长，它就是瓶颈。

用法: resource_budget.py <mindstudio_profiler_output> [stream] [n_steps]
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

COUNTERS = [
    ("aicore_time(us)", "AICore 总"),
    ("aic_mac_time(us)", "AIC MAC"),
    ("aic_scalar_time(us)", "AIC scalar"),
    ("aic_mte1_time(us)", "AIC MTE1"),
    ("aic_mte2_time(us)", "AIC MTE2(load)"),
    ("aic_fixpipe_time(us)", "AIC fixpipe"),
    ("aiv_time(us)", "AIV 总"),
    ("aiv_vec_time(us)", "AIV vec"),
    ("aiv_scalar_time(us)", "AIV scalar"),
    ("aiv_mte2_time(us)", "AIV MTE2"),
    ("aiv_mte3_time(us)", "AIV MTE3(store)"),
]


def main() -> int:
    prof_dir = sys.argv[1]
    stream = float(sys.argv[2]) if len(sys.argv) > 2 else 158.0
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 626

    cols = ["OP Type", "Stream ID", "Task Duration(us)"] + [c for c, _ in COUNTERS]
    frames = [
        pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False)
        for f in sorted(glob.glob(prof_dir + "/op_summary*.csv")
        or glob.glob(prof_dir + "/kernel_details.csv"))
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Stream ID"] == stream]
    dur = df["Task Duration(us)"].sum() / n_steps / 1000.0
    print(f"stream {stream}  任务 {len(df)}  时长合计/步 = {dur:.3f} ms（步数按 {n_steps}）")
    print()
    print(f"{'单元':22s}{'ms/step':>10}{'占时长%':>9}")
    for col, name in COUNTERS:
        if col not in df.columns:
            continue
        # 该列在部分算子（AIV-only / AIC-only）上是 NaN，按 5.5 节口径用 0 填
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        ms = vals.sum() / n_steps / 1000.0
        print(f"{name:22s}{ms:>10.3f}{ms/dur*100 if dur else 0:>8.1f}%")
    print()
    # 按算子族给 top
    print("按算子族的 AIV / MTE2 / scalar（ms/step，取最大的 12 个族）")
    agg = df.assign(
        aiv=pd.to_numeric(df.get("aiv_time(us)"), errors="coerce").fillna(0.0),
        mte2=pd.to_numeric(df.get("aic_mte2_time(us)"), errors="coerce").fillna(0.0),
        scal=pd.to_numeric(df.get("aic_scalar_time(us)"), errors="coerce").fillna(0.0),
        mac=pd.to_numeric(df.get("aic_mac_time(us)"), errors="coerce").fillna(0.0),
    ).groupby("OP Type")[["aiv", "mte2", "scal", "mac"]].sum() / n_steps / 1000.0
    agg["sum"] = agg.sum(axis=1)
    print(agg.sort_values("sum", ascending=False).head(12).round(3).to_string())
    print()
    print("★ 按算子族的 AIV 分解（ms/step）：'未归因' ≈ 跨核同步等待")
    a = df.assign(
        aiv=pd.to_numeric(df.get("aiv_time(us)"), errors="coerce").fillna(0.0),
        vec=pd.to_numeric(df.get("aiv_vec_time(us)"), errors="coerce").fillna(0.0),
        asc=pd.to_numeric(df.get("aiv_scalar_time(us)"), errors="coerce").fillna(0.0),
        am2=pd.to_numeric(df.get("aiv_mte2_time(us)"), errors="coerce").fillna(0.0),
        am3=pd.to_numeric(df.get("aiv_mte3_time(us)"), errors="coerce").fillna(0.0),
        cnt=pd.to_numeric(df.get("Task Duration(us)"), errors="coerce").fillna(0.0),
    ).groupby("OP Type")[["aiv", "vec", "asc", "am2", "am3", "cnt"]].agg(["sum"])
    a.columns = ["aiv", "vec", "asc", "am2", "am3", "dur"]
    a = a / n_steps / 1000.0
    # 子分量之间是**并行**的（MTE2 载入与 vec 计算重叠），所以"同步/其它"用
    # aiv_total − max(子分量) 估计，而不是减它们的和（减和会得到负数）。
    a["同步/其它"] = a["aiv"] - a[["vec", "asc", "am2", "am3"]].max(axis=1)
    a["同步%"] = (a["同步/其它"] / a["aiv"].replace(0, np.nan)) * 100
    a = a[a["aiv"] > 0.02].sort_values("同步/其它", ascending=False)
    print(a.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
