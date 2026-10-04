import glob, sys, numpy as np, pandas as pd
P = sys.argv[1]; lo, hi = float(sys.argv[2]), float(sys.argv[3]); SID = float(sys.argv[4])
fs = sorted(glob.glob(P + "/op_summary*.csv"))
cols = ["Op Name","OP Type","Stream ID","Task Start Time(us)","Task Duration(us)","Input Shapes"]
df = pd.concat([pd.read_csv(f, usecols=cols, low_memory=False) for f in fs], ignore_index=True)
t0 = df["Task Start Time(us)"].min(); df["s"] = (df["Task Start Time(us)"]-t0)/1000.0; df["e"] = df["s"]+df["Task Duration(us)"]/1000.0
sub = df[(df["s"]<hi)&(df["e"]>lo)].copy(); sub["s"]=sub["s"].clip(lo,hi); sub["e"]=sub["e"].clip(lo,hi)
def union(a):
    if len(a)==0: return 0.0
    a=a[np.argsort(a[:,0])]; tot=0.0; cs,ce=a[0]
    for s,e in a[1:]:
        if s<=ce: ce=max(ce,e)
        else: tot+=ce-cs; cs,ce=s,e
    return tot+ce-cs
u_all = union(sub[["s","e"]].to_numpy())
ss = sub[sub["Stream ID"]==SID]
print(f"stream {SID}: union_all={u_all:.3f}")
rows=[]
for op, gg in ss.groupby("OP Type"):
    rest = sub.drop(gg.index)
    u2 = union(rest[["s","e"]].to_numpy())
    rows.append((op, len(gg), (gg["e"]-gg["s"]).sum(), u_all-u2))
r=pd.DataFrame(rows, columns=["op","n","busy_ms","excl_ms"]).sort_values("excl_ms", ascending=False)
print(r.head(22).to_string(index=False))
