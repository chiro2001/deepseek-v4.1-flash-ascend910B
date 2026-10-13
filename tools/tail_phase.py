import sys, numpy as np, pandas as pd
F, SID = sys.argv[1], float(sys.argv[2])
NSTEP = int(sys.argv[3]) if len(sys.argv) > 3 else 8
df = pd.read_csv(F, low_memory=False).rename(columns={
    "Name": "OP Type", "Start Time(us)": "Task Start Time(us)", "Duration(us)": "Task Duration(us)"})
t0 = df["Task Start Time(us)"].min()
df["s"] = (df["Task Start Time(us)"] - t0) / 1000.0
df["e"] = df["s"] + df["Task Duration(us)"] / 1000.0
sub = df["OP Type"].astype(str)
g = df[sub.str.contains("GroupedMatmulSwigluQuantV2", regex=False, na=False) & (df["Stream ID"] == SID)].sort_values("s")
st = g["s"].to_numpy()[::40]
mid = len(st)//3
lo, hi = st[mid], st[mid+NSTEP]
w = df[(df["s"] < hi) & (df["e"] > lo)].copy()
w["s"] = w["s"].clip(lo, hi); w["e"] = w["e"].clip(lo, hi)
# 找 s47（采样链）和 s105(draft)/s35(metadata) 的时间跨度：取所有步的并集区间
for sid in (47.0, 105.0, 142.0, 35.0, 110.0, 108.0, 99.0):
    s = w[w["Stream ID"] == sid]
    if not len(s):
        continue
    # 相对每个步的相位：用 gmm1 锚点算
    ph = []
    for k in range(mid, mid+NSTEP):
        a, b = st[k], st[k+1]
        ss = s[(s["s"] >= a) & (s["s"] < b)]
        if len(ss):
            ph.append(((ss["s"].min()-a)/(b-a)*100, (ss["e"].max()-a)/(b-a)*100))
    if ph:
        p0 = np.median([x[0] for x in ph]); p1 = np.median([x[1] for x in ph])
        print("  s%-5d 步内相位 %5.1f%% → %5.1f%%（跨度 %.1f%%）" % (sid, p0, p1, p1-p0))
