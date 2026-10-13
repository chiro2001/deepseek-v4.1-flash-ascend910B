#!/usr/bin/env python3
"""**prefill** 画像：按 chunk 切分，给出每 chunk 的墙钟、并集、资源账与吞吐。

与 decode 的区别（决定了为什么不能复用 `excl_steps.py`）：
  * decode 步的 `GroupedMatmulSwigluQuantV2` 每步恰好 40 个（40 层）；
  * prefill 一个 chunk（BAT_TOKENS 个 token）里同样 40 层，但**M 很大**、
    且 chunk 之间的间隔远大于层间间隔 ⇒ 用"大间隙"切 chunk 更稳。

用法: prefill_report.py <profdir> [--main-stream N] [--min-gap-ms X]
输出：每 chunk 一行（墙钟/并集/AIC/AIV/MAC/MTE2），以及按 token 数折算的吞吐。
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd


def load(prof_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(prof_dir + "/op_summary*.csv")) or [
        prof_dir + "/kernel_details.csv"
    ]
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        ren = {
            "Name": "Op Name",
            "Type": "OP Type",
            "Start Time(us)": "Task Start Time(us)",
            "Duration(us)": "Task Duration(us)",
            "Accelerator Core": "Task Type",
        }
        frames.append(df.rename(columns={k: v for k, v in ren.items() if k in df.columns}))
    out = pd.concat(frames, ignore_index=True).sort_values("Task Start Time(us)").reset_index(drop=True)
    origin = out["Task Start Time(us)"].min()
    out["s"] = (out["Task Start Time(us)"] - origin) / 1000.0
    out["e"] = out["s"] + out["Task Duration(us)"] / 1000.0
    return out


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prof_dir")
    ap.add_argument("--main-stream", type=float, default=None,
                    help="默认自动挑 gmm1 最多的那个 stream")
    ap.add_argument("--min-gap-ms", type=float, default=1.0)
    a = ap.parse_args()

    df = load(a.prof_dir)
    gmm1 = df[df["OP Type"] == "GroupedMatmulSwigluQuantV2"]
    if gmm1.empty:
        print("没有 GroupedMatmulSwigluQuantV2，无法切 chunk")
        return 1
    main_stream = a.main_stream
    if main_stream is None:
        main_stream = float(gmm1["Stream ID"].value_counts().idxmax())
    main = gmm1[gmm1["Stream ID"] == main_stream].sort_values("s")
    print(f"主 stream = {main_stream:.0f}；该流 gmm1 {len(main)} 个")

    # 按"大间隙"切 chunk
    ts = main["s"].to_numpy()
    bounds = [ts[0]]
    for prev, cur in zip(ts, ts[1:]):
        if cur - prev >= a.min_gap_ms:
            bounds.append(cur)
    bounds = np.array(bounds)
    n_chunk = len(bounds)
    print(f"识别到 {n_chunk} 个 chunk（阈值 {a.min_gap_ms} ms）")

    # 每个 chunk 的 M（该 chunk 内 gmm1 的最大 M，≈ 该 chunk 的 token 数）
    ms = []
    for i, b in enumerate(bounds):
        hi = bounds[i + 1] if i + 1 < n_chunk else df["s"].max() + 1
        sub = main[(main["s"] >= b) & (main["s"] < hi)]
        m = pd.to_numeric(sub["Input Shapes"].astype(str).str.extract(r"^\"?(\d+),")[0], errors="coerce")
        ms.append(int(m.max()) if len(m.dropna()) else 0)

    cols = [
        "aicore_time(us)", "aic_mac_time(us)", "aic_scalar_time(us)", "aic_mte2_time(us)",
        "aiv_time(us)", "aiv_vec_time(us)", "aiv_scalar_time(us)",
    ]
    print()
    head = (f"{'chunk':>5}{'M~':>7}{'wall_ms':>9}{'union_ms':>10}{'busy%':>7}"
            f"{'mac':>8}{'scalar':>8}{'mte2':>8}{'aiv':>8}{'avector':>8}")
    print(head)
    tot_wall = 0.0
    tot_tok = 0
    rows = []
    for i, b in enumerate(bounds):
        hi = bounds[i + 1] if i + 1 < n_chunk else df["s"].max() + 1
        sub = df[(df["s"] < hi) & (df["e"] > b)].copy()
        sub["s"] = sub["s"].clip(b, hi)
        sub["e"] = sub["e"].clip(b, hi)
        wall = hi - b
        u = union(sub[["s", "e"]].to_numpy())
        agg = {}
        for c in cols:
            col = sub[c] if c in sub.columns else None
            agg[c] = float(pd.to_numeric(col, errors="coerce").fillna(0).sum() / 1000.0) if col is not None else 0.0
        tot_wall += wall
        tot_tok += ms[i]
        rows.append((i, ms[i], wall, u, agg))
        print(
            f"{i:>5}{ms[i]:>7}{wall:>9.2f}{u:>10.2f}{u/wall*100 if wall else 0:>6.1f}%"
            f"{agg['aic_mac_time(us)']:>8.2f}{agg['aic_scalar_time(us)']:>8.2f}"
            f"{agg['aic_mte2_time(us)']:>8.2f}{agg['aiv_time(us)']:>8.2f}"
            f"{agg['aiv_vec_time(us)']:>8.2f}"
        )
    print()
    print(f"合计墙钟 {tot_wall:.1f} ms；累计 M ≈ {tot_tok} token ⇒ **{tot_tok/tot_wall*1000:.0f} tok/s**")
    # 全窗口资源账（按占比）
    allsub = df[(df["s"] >= bounds[0])]
    print()
    print("全窗口资源占比（分母 = 主 stream 的任务时长合计）:")
    ms_ = allsub[allsub["Stream ID"] == main_stream]
    denom = float(ms_["Task Duration(us)"].sum() / 1000.0)
    for c, name in (
        ("aic_mac_time(us)", "AIC MAC"), ("aic_scalar_time(us)", "AIC scalar"),
        ("aic_mte1_time(us)", "AIC MTE1"), ("aic_mte2_time(us)", "AIC MTE2"),
        ("aic_fixpipe_time(us)", "AIC fixpipe"), ("aiv_time(us)", "AIV 总"),
        ("aiv_vec_time(us)", "AIV vec"),
    ):
        if c in ms_.columns:
            v = float(pd.to_numeric(ms_[c], errors="coerce").fillna(0).sum() / 1000.0)
            print(f"  {name:12s}{v:8.2f} ms  {v/denom*100 if denom else 0:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
