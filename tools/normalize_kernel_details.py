#!/usr/bin/env python3
"""把新版 msprof 导出的 `kernel_details.csv` 归一化成 `op_summary*.csv` 的列名。

背景：`torch_npu` 的导出产物有两代布局 ——
  * 旧：`mindstudio_profiler_output/op_summary_slice_*.csv`
  * 新：`ASCEND_PROFILER_OUTPUT/kernel_details.csv`
两者列名不同（见 `docs/DELIVERY-PROFILE-R6-20261005.md` §5）。
本仓库的分析工具优先找 `op_summary*.csv`，找不到就退回 `kernel_details.csv`；
但**列名归一**仍需一次转换（或让调用方自己 rename）。

用法: normalize_kernel_details.py <kernel_details.csv> <输出目录>
产出: <输出目录>/op_summary_normalized.csv
"""

from __future__ import annotations

import os
import sys

import pandas as pd

RENAME = {
    "Name": "Op Name",
    "Type": "OP Type",
    "Start Time(us)": "Task Start Time(us)",
    "Duration(us)": "Task Duration(us)",
    "Wait Time(us)": "Task Wait Time(us)",
    "Accelerator Core": "Task Type",
}


def main() -> int:
    src = sys.argv[1]
    dst_dir = sys.argv[2]
    df = pd.read_csv(src, low_memory=False)
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})
    os.makedirs(dst_dir, exist_ok=True)
    out = os.path.join(dst_dir, "op_summary_normalized.csv")
    df.to_csv(out, index=False)
    cols = [
        c
        for c in (
            "OP Type", "Stream ID", "Task Start Time(us)", "Task Duration(us)",
            "Task Type", "Input Shapes", "aic_mac_time(us)", "aiv_time(us)",
        )
        if c in df.columns
    ]
    print(f"rows={len(df)} -> {out}")
    print("关键列齐备:", cols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
