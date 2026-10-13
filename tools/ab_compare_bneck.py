#!/usr/bin/env python3
"""尾部等长对齐比较（消除"窗口长短不一"的混淆）。"""
import re, statistics as st, subprocess

RUNS = [
    ("A1 base 03:20", "/home/l00886679/cedpd-repo/results/armHANDOVER_1007_015350/serve.log"),
    ("A2 base 05:18", "/home/l00886679/cedpd-repo/results/armBASE2_1007_051011/serve.log"),
    ("D  comment 05:30", "/home/l00886679/cedpd-repo/results/armABCTRL_1007_052202/serve.log"),
    ("C  ctrl 05:07", "/home/l00886679/cedpd-repo/results/armHCLIMIT8_1007_045838/serve.log"),
    ("B  fuse 04:10", "/home/l00886679/cedpd-repo/results/armHCFUSE_1007_035324/serve.log"),
]
N = 1500
def tail_vals(p, n=N):
    t = subprocess.run(["sudo","-n","cat",p], capture_output=True, text=True).stdout
    v = [float(x) for x in re.findall(r"hp=([0-9.]+)", t)]
    v = [x for x in v if 20.0 <= x < 40.0]
    return v[-n:]

print(f"每臂取**尾部 {N} 个**样本（同窗口长度）：")
print(("{:<18}{:>7}{:>9}{:>9}{:>9}{:>13}").format("arm","n","p10","p50","p90","vs A1"))
a1=None
for name,p in RUNS:
    v=sorted(tail_vals(p))
    if not v: print(("{:<18}{:>7}  none").format(name,0)); continue
    q=lambda x: v[int(x*(len(v)-1))]
    if a1 is None: a1=st.median(v)
    d=st.median(v)-a1
    print(("{:<18}{:>7}{:>9.3f}{:>9.3f}{:>9.3f}{:>+8.3f} ({:+.2f}%)").format(
        name,len(v),q(.1),st.median(v),q(.9),d,100*d/a1))

print("\n逐轮（每 300 样本）中位，看是否有漂移：")
for name,p in RUNS:
    v=tail_vals(p)
    if not v: continue
    chunks=[st.median(v[i:i+300]) for i in range(0,len(v),300)]
    print(("  {:<18}" + "".join("{:>8.3f}" for _ in chunks)).format(name, *chunks))
