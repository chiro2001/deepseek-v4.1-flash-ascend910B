import glob, os, pandas as pd, numpy as np, sys

def load(run):
    rows = []
    for d in sorted(glob.glob("/home/l00886679/cedpd-repo/results/%s/prof/dp0_pp0_tp0_*/" % run)):
        f = os.path.join(d, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, low_memory=False).rename(
            columns={"Name": "OP Type", "Duration(us)": "Task Duration(us)"})
        steps = (df["OP Type"] == "HcPre").sum() / 80.0
        if steps < 10:
            continue
        g = df.groupby("OP Type")["Task Duration(us)"].agg(["size", "sum"]) / steps
        g.columns = ["n_ps", "us_ps"]
        rows.append(g)
    return pd.concat(rows).groupby(level=0).mean() if rows else None

a = load(sys.argv[1]); b = load(sys.argv[2])
if a is None or b is None:
    print("缺 profile"); sys.exit(1)
j = a.join(b, lsuffix="_A", rsuffix="_B", how="outer").fillna(0)
j["d_us"] = j["us_ps_B"] - j["us_ps_A"]
j["d_n"] = j["n_ps_B"] - j["n_ps_A"]
print("A=%s  B=%s   （单位：每步）" % (sys.argv[1], sys.argv[2]))
print("A 步总时长 %.2f ms   B 步总时长 %.2f ms" % (j["us_ps_A"].sum()/1000, j["us_ps_B"].sum()/1000))
print()
print("=== |Δ| 最大的 16 个算子族（µs/步）===")
top = j.reindex(j["d_us"].abs().sort_values(ascending=False).index).head(16)
for name, r in top.iterrows():
    print("  %-40s nA=%7.2f nB=%7.2f  usA=%9.1f usB=%9.1f  Δ=%+9.1f" % (
        name[:40], r["n_ps_A"], r["n_ps_B"], r["us_ps_A"], r["us_ps_B"], r["d_us"]))
