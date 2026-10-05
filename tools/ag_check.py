import pandas as pd, numpy as np, sys
df = pd.read_csv(sys.argv[1], low_memory=False).rename(columns={"Name":"OP Type","Duration(us)":"Task Duration(us)"})
steps = (df["OP Type"] == "HcPre").sum() / 80.0
ag = df[df["OP Type"].astype(str).str.contains("allGather|allgather", regex=True, na=False)]
print("步数 %.1f，allGather 任务总数 %d（%.2f/步）" % (steps, len(ag), len(ag)/steps))
print("总时长 %.2f ms（%.3f ms/步）" % (ag["Task Duration(us)"].sum()/1000, ag["Task Duration(us)"].sum()/1000/steps))
g = ag.groupby("OP Type")["Task Duration(us)"].agg(["size","sum"]).sort_values("size", ascending=False)
g["per_step"] = g["size"]/steps
print("\n按名字（per_step>0.3 的）:")
print(g[g["per_step"] > 0.3].head(10).to_string())
print("\n按 (名字前缀) 归并:")
ag2 = ag.copy(); ag2["grp"] = ag2["OP Type"].astype(str).str.replace(r"__\d+_\d+_\d+$", "", regex=True)
g2 = ag2.groupby("grp")["Task Duration(us)"].agg(["size","sum"])
g2["per_step"] = g2["size"]/steps; g2["ms_ps"] = g2["sum"]/1000/steps
print(g2.sort_values("size", ascending=False).head(8).round(3).to_string())
