#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 Input Shapes 聚合 op_summary 的资源计数指纹（中位数），可选附 Stream 分布。

用法:
    python3 tools/prof_shape_stats.py <op_summary.csv> [--op HcPre] [--top 12] [--streams]

背景见 docs/A1-CACHE-PARTIAL-HIT-20261005.md：换 kernel 后必须**逐形状**核验
aic_mac_time / aic_total_cycles 等资源计数（判定"新内核是否真的在跑"）。
只查一个形状可能漏掉"static kernel 缓存部分命中"（半新半旧）。
"""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict

FIELDS = [
    "Task Duration(us)", "Task Wait Time(us)", "Block Num", "Mix Block Num",
    "aicore_time(us)", "aic_total_cycles", "aic_mac_time(us)", "aic_scalar_time(us)",
    "aiv_time(us)", "aiv_total_cycles", "aiv_scalar_time(us)",
]


def q(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p))]


def median(vals):
    return statistics.median(vals) if vals else None


def process(path, ops, want_streams):
    data = defaultdict(lambda: defaultdict(list))
    streams = defaultdict(lambda: defaultdict(int))
    nrows = 0
    with open(path, newline="", errors="replace") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {n: i for i, n in enumerate(header)}
        op_i, shp_i = idx["Op Name"], idx["Input Shapes"]
        st_i = idx.get("Stream ID")
        for row in r:
            nrows += 1
            if len(row) <= shp_i:
                continue
            if ops and row[op_i] not in ops:
                continue
            shape = row[shp_i].strip('"')
            if want_streams and st_i is not None:
                streams[shape][row[st_i]] += 1
            for fld in FIELDS:
                try:
                    data[shape][fld].append(float(row[idx[fld]]))
                except Exception:
                    pass
    out = {}
    for shape, cols in data.items():
        entry = {}
        for fld, v in cols.items():
            entry[fld] = {"n": len(v), "med": median(v), "p10": q(v, 0.10), "p90": q(v, 0.90)}
        if want_streams and shape in streams:
            entry["streams"] = dict(sorted(streams[shape].items(), key=lambda kv: -kv[1]))
        out[shape] = entry
    return {"file": path, "nrows": nrows, "shapes": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--op", action="append", default=[], help="按 Op Name 精确过滤（可多次）")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--streams", action="store_true", help="附加 Stream ID 分布")
    a = ap.parse_args()
    res = process(a.csv, set(a.op) if a.op else None, a.streams)
    shp = res.pop("shapes")
    top = sorted(shp.items(), key=lambda kv: -kv[1].get("Task Duration(us)", {}).get("n", 0))[: a.top]
    res["top_shapes"] = dict(top)
    json.dump(res, sys.stdout, indent=1, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
