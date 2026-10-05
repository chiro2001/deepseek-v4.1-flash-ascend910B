import json, sys, numpy as np, pandas as pd
D, SID = sys.argv[1], float(sys.argv[2])
NSTEP = int(sys.argv[3]) if len(sys.argv) > 3 else 8

comm = json.load(open(D + "/communication.json"))["step"]["collective"]
recs = []
for k, v in comm.items():
    if "allReduce" not in k:
        continue
    t = v["Communication Time Info"]
    recs.append((float(t["Start Timestamp(us)"]), float(t["Elapse Time(ms)"]) * 1000,
                 float(t.get("Wait Time(ms)", 0)) * 1000, float(t.get("Synchronization Time(ms)", 0)) * 1000,
                 float(t.get("Idle Time(ms)", 0)) * 1000, k))
R = pd.DataFrame(recs, columns=["ts_us", "us", "wait_us", "sync_us", "idle_us", "name"]).sort_values("ts_us").reset_index(drop=True)
print("communication.json 里 allReduce 记录 %d 条；耗时中位 %.1f us" % (len(R), R["us"].median()))

df = pd.read_csv(D + "/kernel_details.csv", low_memory=False).rename(columns={
    "Name": "OP Type", "Start Time(us)": "Task Start Time(us)", "Duration(us)": "Task Duration(us)"})
sub = df["OP Type"].astype(str)
g = df[sub.str.contains("GroupedMatmulSwigluQuantV2", regex=False, na=False) & (df["Stream ID"] == SID)].sort_values("Task Start Time(us)")
st = g["Task Start Time(us)"].to_numpy()[::40]
mid = len(st)//3
lo, hi = st[mid], st[mid+NSTEP]
m = df[(df["Stream ID"] == SID) & (df["Task Start Time(us)"] >= lo) & (df["Task Start Time(us)"] < hi)].sort_values("Task Start Time(us)").reset_index(drop=True)
rows = []
for i in range(len(m)-1):
    p, q = str(m.loc[i, "OP Type"]), str(m.loc[i+1, "OP Type"])
    g0 = m.loc[i, "Task Start Time(us)"] + m.loc[i, "Task Duration(us)"]
    g1 = m.loc[i+1, "Task Start Time(us)"]
    if g1 - g0 < 20 or ("Add" not in p and "Matmul" not in p):
        continue
    cand = R[(R["ts_us"] >= g0 - 30) & (R["ts_us"] <= g1 + 30)]
    kind = "MoE(Add)" if "Add" in p else "attn(Matmul)"
    if len(cand):
        c = cand.iloc[(cand["ts_us"] - g0).abs().argsort()[:1]].iloc[0]
        rows.append((kind, (g1-g0), c["us"], c["wait_us"], c["sync_us"], c["idle_us"]))
    else:
        rows.append((kind, (g1-g0), np.nan, np.nan, np.nan, np.nan))
G = pd.DataFrame(rows, columns=["kind", "gap_us", "comm_us", "wait_us", "sync_us", "idle_us"])
print("\n配对 %d 个 allreduce（稳态 %d 步）" % (len(G), NSTEP))
print(G.groupby("kind").agg(n=("gap_us","size"), gap_med=("gap_us","median"), comm_med=("comm_us","median"),
                            wait_med=("wait_us","median"), sync_med=("sync_us","median"),
                            idle_med=("idle_us","median"), comm_sum=("comm_us","sum")).round(1).to_string())
