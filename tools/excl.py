import glob, sys, numpy as np, pandas as pd
P = sys.argv[1]; lo, hi = float(sys.argv[2]), float(sys.argv[3])
fs = sorted(glob.glob(P + "/op_summary*.csv"))
cols = ["OP Type","Stream ID","Task Start Time(us)","Task Duration(us)","Task Type"]
df = pd.concat([pd.read_csv(f, usecols=cols, low_memory=False) for f in fs], ignore_index=True)
t0 = df["Task Start Time(us)"].min(); df["s"] = (df["Task Start Time(us)"] - t0)/1000.0; df["e"] = df["s"] + df["Task Duration(us)"]/1000.0
sub = df[(df["s"] < hi) & (df["e"] > lo)].copy()
sub["s"] = sub["s"].clip(lo, hi); sub["e"] = sub["e"].clip(lo, hi)
def union(a):
    if len(a) == 0: return 0.0
    a = a[np.argsort(a[:,0])]; tot = 0.0; cs, ce = a[0]
    for s, e in a[1:]:
        if s <= ce: ce = max(ce, e)
        else: tot += ce - cs; cs, ce = s, e
    return tot + ce - cs
dur = hi - lo
all_u = union(sub[["s","e"]].to_numpy())
print(f"step {dur:.2f} ms  tasks={len(sub)}  union={all_u:.3f} ms  gap={dur-all_u:.3f} ms")
rows = []
for sid, gg in sub.groupby("Stream ID"):
    rest = sub[sub["Stream ID"] != sid]
    u2 = union(rest[["s","e"]].to_numpy())
    excl = all_u - u2          # 该 stream 对并集的独占贡献
    rows.append((sid, len(gg), (gg["e"]-gg["s"]).sum(), u2, excl))
r = pd.DataFrame(rows, columns=["stream","n","busy_sum","union_wo","exclusive_contrib"]).sort_values("exclusive_contrib", ascending=False)
r["excl_pct"] = r["exclusive_contrib"]/dur*100
print(r.to_string(index=False))
