#!/usr/bin/env python3
"""按 (算子名, 调用栈尾) 聚合 operator_details.csv —— 直接给出"哪一行代码产生多少算子/多少设备时间"。

这是定位固定成本的最直接工具：kernel_details 只有形状，operator_details 带 **Call Stack**。

用法: prof_callsite_agg.py <ASCEND_PROFILER_OUTPUT> [topN] [--filter 子串]
"""
from __future__ import annotations
import csv, sys, re
from collections import defaultdict

SRC = sys.argv[1]
D = None if SRC == "-" else SRC
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 40
FILT = None
if "--filter" in sys.argv:
    FILT = sys.argv[sys.argv.index("--filter") + 1]

agg = defaultdict(lambda: [0, 0.0])   # (name, site) -> [count, device_self_us]
tot_rows = 0
PATH = "/dev/stdin" if D is None else D + "/operator_details.csv"
with open(PATH, newline="") as fh:
    rd = csv.DictReader(fh)
    for r in rd:
        tot_rows += 1
        nm = r.get("Name") or ""
        if FILT and FILT not in nm:
            continue
        cs = (r.get("Call Stack") or "").strip()
        # 调用栈里取最后 2 个"我们自己"的帧（跳过 torch 内部）
        frames = [f for f in re.split(r"[;\n]", cs) if f.strip()]
        keep = []
        for f in reversed(frames):
            f = f.strip()
            if any(k in f for k in ("vllm_ascend", "vllm/v1", "deepseek_v41", "dsa_v41", "dsa_v1", "vllm/")):
                keep.append(f.split("/")[-1][:70])
            if len(keep) >= 2:
                break
        site = " <= ".join(reversed(keep)) if keep else (frames[-1][:60] if frames else "?")
        try:
            dur = float(r.get("Device Self Duration(us)") or 0.0)
        except ValueError:
            dur = 0.0
        a = agg[(nm[:44], site)]
        a[0] += 1
        a[1] += dur

print(f"总行数 {tot_rows}｜聚合键 {len(agg)}")
print(f"\n{'#':>4} {'算子':<44}{'次数':>8}{'设备自身ms':>11}  调用栈尾")
ranked = sorted(agg.items(), key=lambda kv: -kv[1][1])[:TOPN]
for i, ((nm, site), (c, d)) in enumerate(ranked, 1):
    print(f"{i:>4} {nm:<44}{c:>8}{d/1000:>11.3f}  {site}")
