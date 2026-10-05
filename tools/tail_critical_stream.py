import sys, numpy as np, pandas as pd
F, SID, S47 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
NSTEP = int(sys.argv[4]) if len(sys.argv) > 4 else 3
df = pd.read_csv(F, low_memory=False).rename(columns={
    "Name": "OP Type", "Start Time(us)": "Task Start Time(us)", "Duration(us)": "Task Duration(us)"})
t0 = df["Task Start Time(us)"].min()
df["s"] = (df["Task Start Time(us)"] - t0) / 1000.0
df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0
sub = df["OP Type"].astype(str)
g = df[sub.str.contains("GroupedMatmulSwigluQuantV2", regex=False, na=False) & (df["Stream ID"] == SID)].sort_values("s")
st = g["s"].to_numpy()[::40]
mid = len(st)//3 + NSTEP
lo, hi = st[mid], st[mid+1]
span = hi - lo
s47 = df[(df["Stream ID"] == S47) & (df["s"] >= lo) & (df["s"] < hi)].sort_values("s").reset_index(drop=True)
print("步长 %.2f ms；s47 任务 %d 个，busy %.2f ms" % (span, len(s47), (s47["e"]-s47["s"]).sum()))
# s47 内部空隙
agg = {}
prev = None
tot = 0.0
for _, r in s47.iterrows():
    if prev is not None and (r["s"]-prev["e"]) >= 0.05:
        k = (str(prev["OP Type"])[:26], str(r["OP Type"])[:26])
        agg[k] = agg.get(k, 0.0) + (r["s"]-prev["e"]); tot += r["s"]-prev["e"]
    prev = r
print("s47 内部 >=50us 空隙合计 %.2f ms" % tot)
for k, v in sorted(agg.items(), key=lambda x: -x[1])[:6]:
    print("   %7.3f ms  %-28s → %s" % (v, k[0], k[1]))
# 关键：s47 在尾部（>79% 相位）的算子与空隙
tail = s47[s47["s"] >= lo + 0.75*span]
print("\n=== s47 在 75%% 之后的算子（>=50us 或首尾）===")
prev = None
for _, r in tail.iterrows():
    gap = 0.0 if prev is None else r["s"]-prev
    if (r["e"]-r["s"]) >= 0.05 or gap >= 0.05:
        print("  t=%6.2f%% %7.3f ms dur=%7.3f  %-34s %s" % ((r["s"]-lo)/span*100, r["s"]-lo, r["e"]-r["s"], str(r["OP Type"])[:34], str(r.get("Input Shapes"))[:26]))
    prev = r["e"]
