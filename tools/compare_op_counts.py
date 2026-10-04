#!/usr/bin/env python3
"""对比两个 run 的**逐算子族次数与总时长**（用于确认某个 flag 的机制是否真的生效）。

为什么需要它：单跑 A/B 的噪声底（±0.16 ms/步）常常大于被测改动的期望收益
（例如 armH 的两个开关合计约 0.2 ms/步）。这时**只看端到端 ms/step 无法判决**，
但"该消失的算子是否真的消失了"是**零噪声**的证据 —— 机制生效 + 收益方向一致
就能形成完整证据链。

用法: compare_op_counts.py <runA> <runB> [--results-dir DIR] [--ops a,b,c]
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

RENAME = {
    "Name": "Op Name",
    "Type": "OP Type",
    "Start Time(us)": "Task Start Time(us)",
    "Duration(us)": "Task Duration(us)",
    "Accelerator Core": "Task Type",
}


def load(results_dir: str, run: str) -> pd.DataFrame | None:
    files = sorted(
        glob.glob(f"{results_dir}/{run}/prof/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv")
        + glob.glob(f"{results_dir}/{run}/prof/*/mindstudio_profiler_output/op_summary*.csv")
    )
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    return df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--results-dir", default=os.path.expanduser("~/cedpd-repo/results"))
    ap.add_argument("--ops", default="ZerosLike,Fill,Cast,ViewCopy,IndexCheck,FloorDiv,FloorMod,SelectV2")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    da = load(a.results_dir, a.run_a)
    db = load(a.results_dir, a.run_b)
    if da is None or db is None:
        print(f"缺 profile：A={da is not None} B={db is not None}")
        return 1

    for name, d in ((a.run_a, da), (a.run_b, db)):
        span = (d["Task Start Time(us)"].max() - d["Task Start Time(us)"].min()) / 1e6
        print(f"{name:22s} 任务={len(d):6d} 窗口={span:6.2f}s")

    print()
    print(f"{'OP Type':26s}{'nA':>9}{'nB':>9}{'Δn':>8}{'msA':>10}{'msB':>10}{'Δms':>9}")
    ops = [x for x in a.ops.split(",") if x]
    for t in ops:
        ga, gb = da[da["OP Type"] == t], db[db["OP Type"] == t]
        if not len(ga) and not len(gb):
            continue
        ma = ga["Task Duration(us)"].sum() / 1000.0
        mb = gb["Task Duration(us)"].sum() / 1000.0
        print(f"{t:26s}{len(ga):>9d}{len(gb):>9d}{len(gb)-len(ga):>8d}{ma:>10.1f}{mb:>10.1f}{mb-ma:>9.1f}")

    print()
    print("按 |Δms| 排序的 top 算子族（B − A）:")
    pivot_a = da.groupby("OP Type")["Task Duration(us)"].agg(["size", "sum"])
    pivot_b = db.groupby("OP Type")["Task Duration(us)"].agg(["size", "sum"])
    joined = pivot_a.join(pivot_b, lsuffix="_a", rsuffix="_b", how="outer").fillna(0)
    joined["d_ms"] = (joined["sum_b"] - joined["sum_a"]) / 1000.0
    joined["d_n"] = joined["size_b"] - joined["size_a"]
    print(joined.reindex(joined["d_ms"].abs().sort_values(ascending=False).index).head(a.top)[
        ["size_a", "size_b", "d_n", "d_ms"]
    ].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
