#!/usr/bin/env python3
"""profile 的算子级时间归因：稳态 decode 窗口内，按算子名聚合总时长与占比。"""
import sys
import pandas as pd

D = sys.argv[1]
NSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 12
df = pd.read_csv(D + "/kernel_details.csv", low_memory=False)
df = df.rename(columns={"Name": "name", "Start Time(us)": "st", "Duration(us)": "dur",
                        "Accelerator Core": "core", "Stream ID": "sid"})
df = df.sort_values("st").reset_index(drop=True)

# 用主计算流的步边界切稳态窗口：找相邻任务间隔 > 5 ms 的位置作为步间隔
main = df[df["dur"] > 0].copy()
starts = main["st"].values
gaps = []
for i in range(1, len(starts)):
    d = starts[i] - starts[i - 1]
    if d > 5000:
        gaps.append((i, d))
print("检测到 %d 个 >5ms 的步间大间隔（前 3 个：%s）" % (len(gaps), [f"{g[1]/1000:.1f}ms" for g in gaps[:3]]))

if len(gaps) >= 3:
    lo = gaps[0][0]
    hi = gaps[min(len(gaps) - 1, NSTEP)][0]
    win = main.iloc[lo:hi]
    nst = min(len(gaps) - 1, NSTEP) - 1 + 1
else:
    win = main
    nst = 1

nb = win["dur"].sum() / 1000.0
span = (win["st"].max() + win.iloc[-1]["dur"] - win["st"].min()) / 1000.0
print("窗口：%d 步，算子 %d 个，算子时长合计 %.1f ms，窗口跨度 %.1f ms ⇒ 每步 %.2f ms"
      % (nst, len(win), nb, span, span / max(nst, 1)))

g = win.groupby("name").agg(total_ms=("dur", lambda s: s.sum() / 1000),
                            n=("dur", "size"), med_us=("dur", "median"))
g["ms_per_step"] = g["total_ms"] / max(nst, 1)
g = g.sort_values("total_ms", ascending=False)
print("\n%-58s %10s %6s %10s %8s" % ("算子", "总ms", "次数", "ms/步", "占比%"))
for name, r in g.head(22).iterrows():
    print("%-58s %10.2f %6d %10.3f %7.1f%%" % (str(name)[:58], r["total_ms"], r["n"], r["ms_per_step"],
                                               100 * r["total_ms"] / nb))
