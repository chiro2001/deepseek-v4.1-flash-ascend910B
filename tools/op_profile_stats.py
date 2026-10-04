#!/usr/bin/env python3
"""按 (OP Type, Stream) 给出**服务内实测**的每步次数、时长与资源计数中位。

用途：把"隔离容器里的地板"与"服务里的实际值"放在一起比，判断还有没有空间。
注意：本工具的 `合计/步` 是"该算子族在一步里的设备时间之和"，
它**不等于**墙钟收益（可能被重叠），只用于排序与找异常。

用法: op_profile_stats.py <mindstudio_profiler_output> [n_steps] [stream]
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

OPS = [
    "GroupedMatmulSwigluQuantV2",
    "GroupedMatmul",
    "MatMulV2",
    "MatMulV3",
    "SparseFlashMla",
    "SparseAttnSharedkv",
    "HcPre",
    "HcPost",
    "QuantBatchMatmulV3",
    "RmsNorm",
    "RmsNormCast",
    "HcPost",
    "DynamicQuant",
    "MoeInitRoutingV3",
    "MoeTokenUnpermute",
    "MoeGatingTopKHash",
    "DequantSwigluQuant",
    "ScatterNdUpdateSk",
    "InplacePartialRotaryMul",
    "ViewCopy",
    "Cast",
    "Index",
    "FloorDiv",
    "FloorMod",
    "Fill",
    "ZerosLike",
    "hcom_allReduce_",
]


def main() -> int:
    prof_dir = sys.argv[1]
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 626
    only = float(sys.argv[3]) if len(sys.argv) > 3 else None

    cols = [
        "OP Type", "Stream ID", "Task Duration(us)", "Input Shapes", "Block Num",
        "aic_mac_time(us)", "aiv_time(us)", "aic_scalar_time(us)", "aic_mte2_time(us)",
    ]
    frames = [
        pd.read_csv(f, usecols=cols, low_memory=False)
        for f in sorted(glob.glob(prof_dir + "/op_summary*.csv"))
    ]
    df = pd.concat(frames, ignore_index=True)
    if only is not None:
        df = df[df["Stream ID"] == only]

    print(f"步数按 {n_steps} 计；stream 过滤 = {only}")
    print(f"{'OP Type':32s}{'s':>4}{'n/step':>9}{'p50us':>9}{'p90us':>9}{'ms/step':>10}{'mac':>8}{'aiv':>9}{'scalar':>8}{'mte2':>8}")
    for t in OPS:
        s = df[df["OP Type"] == t]
        if not len(s):
            continue
        sid = int(np.median(s["Stream ID"]))
        print(
            f"{t:32s}{sid:>4}{len(s)/n_steps:>9.2f}"
            f"{np.median(s['Task Duration(us)']):>9.2f}"
            f"{np.percentile(s['Task Duration(us)'], 90):>9.2f}"
            f"{s['Task Duration(us)'].sum()/n_steps/1000:>10.3f}"
            f"{np.median(s['aic_mac_time(us)']):>8.3f}"
            f"{np.median(s['aiv_time(us)']):>9.3f}"
            f"{np.median(s['aic_scalar_time(us)']):>8.3f}"
            f"{np.median(s['aic_mte2_time(us)']):>8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
