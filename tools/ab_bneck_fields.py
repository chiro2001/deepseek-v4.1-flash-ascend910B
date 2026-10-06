#!/usr/bin/env python3
"""对比各臂 bneck 行里的所有子字段（不只 hp）。"""
import re, collections, statistics as st, subprocess

RUNS = [
    ("A1 base 03:20", "/home/l00886679/cedpd-repo/results/armHANDOVER_1007_015350/serve.log"),
    ("A2 base 05:18", "/home/l00886679/cedpd-repo/results/armBASE2_1007_051011/serve.log"),
    ("B  fuse 04:10", "/home/l00886679/cedpd-repo/results/armHCFUSE_1007_035324/serve.log"),
    ("C  ctrl 05:07", "/home/l00886679/cedpd-repo/results/armHCLIMIT8_1007_045838/serve.log"),
    ("D  comment 05:30", "/home/l00886679/cedpd-repo/results/armABCTRL_1007_052202/serve.log"),
]
FIELDS = ["hp", "d2h", "hash", "meta", "pad", "route", "total"]

def collect(p):
    t = subprocess.run(["sudo","-n","cat",p], capture_output=True, text=True).stdout
    out = collections.defaultdict(list)
    for line in t.splitlines():
        if "[bneck]" not in line or "mode-change" in line: continue
        for f in FIELDS:
            m = re.search(rf"\b{f}=([0-9.]+)", line)
            if m: out[f].append(float(m.group(1)))
    return out

print(("{:<16}" + "{:>12}"*4).format("arm", *["n", "hp p50", "route p50", "pad p50"]))
for name, p in RUNS:
    d = collect(p)
    n = len(d.get("hp", []))
    if n == 0:
        print(("{:<16}{:>12}").format(name, 0)); continue
    hp = [v for v in d["hp"] if 20 <= v < 40]
    print(("{:<16}{:>12}{:>12.3f}{:>12.4f}{:>12.4f}").format(
        name, len(hp), st.median(hp) if hp else -1,
        st.median(d["route"]) if d.get("route") else -1,
        st.median(d["pad"]) if d.get("pad") else -1))
print()
print("各臂 bneck 原始行样例：")
for name, p in RUNS:
    t = subprocess.run(["sudo","-n","cat",p], capture_output=True, text=True).stdout
    lines = [l for l in t.splitlines() if "[bneck]" in l and "mode-change" not in l]
    if lines:
        print(f"  {name}: {lines[len(lines)//2].strip()[:150]}")
