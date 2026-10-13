import glob, sys
import pandas as pd
M = sys.argv[1]
files = sorted(glob.glob(M + "/op_summary*.csv"))
cols = ["OP Type","Task Start Time(us)","Task Duration(us)","Input Shapes",
        "aicore_time(us)","aic_scalar_time(us)","aic_mte1_time(us)","aic_mte2_time(us)","aic_mac_time(us)"]
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in files], ignore_index=True)
sub = df[df["OP Type"]=="GroupedMatmulSwigluQuantV2"].sort_values("Task Start Time(us)")
print("n=", len(sub))
print(sub[["Task Duration(us)","aicore_time(us)","aic_scalar_time(us)","aic_mte1_time(us)","aic_mte2_time(us)","aic_mac_time(us)"]].median().to_string())
