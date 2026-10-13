#!/usr/bin/env python3
"""按固定间隔采样 npu-smi 的 HBM Bandwidth Usage Rate(%)，同时记录时间戳。

输出 CSV：t,card,chip,hbm_usage,hbm_bw_pct,aicore_pct
"""
import re
import subprocess
import sys
import time

CARDS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "8").split(",")]
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
INT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25

t0 = time.monotonic()
print("t,card,chip,hbm_pct,bw_pct,aicore_pct", flush=True)
n = 0
while time.monotonic() - t0 < DUR:
    for c in CARDS:
        try:
            out = subprocess.run(["npu-smi", "info", "-t", "usages", "-i", str(c)],
                                 capture_output=True, text=True, timeout=10).stdout
        except Exception:
            continue
        blocks = out.split("Chip ID")
        for i, b in enumerate(blocks[:-1]):
            def g(key):
                m = re.search(rf"{key}\s*:\s*([0-9.]+)", b)
                return m.group(1) if m else "NA"
            print("%.2f,%d,%d,%s,%s,%s" % (time.monotonic() - t0, c, c * 2 + i,
                                           g("HBM Usage Rate\\(%\\)"),
                                           g("HBM Bandwidth Usage Rate\\(%\\)"),
                                           g("Aicore Usage Rate\\(%\\)")), flush=True)
    n += 1
print(f"# samples={n} interval={INT}", file=sys.stderr, flush=True)
