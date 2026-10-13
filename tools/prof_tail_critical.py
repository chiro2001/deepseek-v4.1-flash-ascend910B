#!/usr/bin/env python3
"""尾链是否在关键路径：比较"整步跨度"与"主流 s109 跨度"。单次遍历，无 O(步×任务)。

若 整步末任务 - 主流末任务 > 0，且这两者之间的时间段里主流已无任务 ⇒ 尾链**延长了步长**。
"""
from __future__ import annotations
import bisect, csv, os, sys, statistics as st

D = sys.argv[1]
ANCHOR = os.environ.get("ANCHOR", "HcPre")
PER = int(os.environ.get("ANCHOR_PER_STEP", "86"))

marks, tasks = [], []
with open(D + "/kernel_details.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        nm = r.get("Name") or ""
        if ANCHOR in nm:
            try: marks.append(float(r["Start Time(us)"]))
            except Exception: pass
        try:
            a = float(r["Start Time(us)"]); du = float(r["Duration(us)"])
        except Exception: continue
        tasks.append((a, a + du, str(r.get("Stream ID") or "")))
marks.sort()
if PER > 1: marks = marks[::PER]
LO, HI = marks[2], marks[-3]
starts = [m for m in marks if LO <= m < HI]
nst = len(starts) - 1
tasks.sort()

# 自动识别主流 = busy 最大的那条（不同 run 的流编号会变：109 / 146 …）
from collections import defaultdict as _dd
_busy = _dd(float)
for _a, _e, _s in tasks:
    _busy[_s] += _e - _a
MAIN = max(_busy.items(), key=lambda kv: kv[1])[0]
print(f"步数 {nst}｜步长 {(HI-LO)/1000/(nst+1):.3f} ms｜任务 {len(tasks)/nst:.0f}/步")
print(f"主流（自动识别，busy 最大）= s{MAIN}  busy={_busy[MAIN]/1000/nst:.3f} ms/步")

# 单次遍历：用 bisect 定位每步的任务区间
diff_all, diff_main, main_last_rel, all_last_rel = [], [], [], []
for i in range(nst):
    a, b = starts[i], starts[i + 1]
    lo = bisect.bisect_left([t[0] for t in tasks], a) if False else None
# 上面的写法会 O(n^2)，改用一次扫描
idx = 0
cur = 0
acc = [[] for _ in range(nst)]
ts = [t[0] for t in tasks]
import array
# 用 bisect 在 ts 上做，但 ts 只算一次
for i in range(nst):
    a, b = starts[i], starts[i + 1]
    lo = bisect.bisect_left(ts, a); hi = bisect.bisect_left(ts, b)
    if hi <= lo: continue
    seg = tasks[lo:hi]
    last_end = max(e for _, e, _ in seg)
    main = [e for _, e, s in seg if s == MAIN]
    last_main = max(main) if main else None
    diff_all.append(last_end - a)
    if last_main is not None:
        diff_main.append(last_main - a)

print()
print(f"  整步：最后任务结束(相对步起点) 中位 = {st.median(diff_all)/1000:.3f} ms")
if diff_main:
    print(f"  主流 s{MAIN}：最后任务结束         中位 = {st.median(diff_main)/1000:.3f} ms")
    delta = st.median(diff_all) - st.median(diff_main)
    print(f"  ★ 差 = {delta/1000:.3f} ms  ⇒ 尾链在主流结束后还在跑约 {delta/1000:.3f} ms")
    if delta > 200:
        print("  ⇒ 【关键路径】尾链延长了步长（这段没有主流在跑）")
    elif delta < 50:
        print("  ⇒ 【非关键路径】尾链基本与主流收尾重叠")
    else:
        print("  ⇒ 部分重叠，需结合空闲分析判断")
