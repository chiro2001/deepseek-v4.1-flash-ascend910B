import json, sys, numpy as np, glob, os
base = "/home/l00886679/cedpd-repo/results/armF_r7_wkvtp/prof"
tgt = sys.argv[1]
for p in sorted(glob.glob(base + "/*%s*/ASCEND_PROFILER_OUTPUT/communication.json" % tgt)):
    d = json.load(open(p))
    c = d["step"]["collective"]
    us = np.array([v["Communication Time Info"]["Elapse Time(ms)"]*1000 for k, v in c.items() if "allReduce" in k])
    us.sort()
    # 取稳态段（后半）避免预热
    st = us[int(len(us)*0.4):]
    print("%-16s n=%5d  稳态中位 %6.1f us  p25 %6.1f  p75 %6.1f  p95 %6.1f" % (
        os.path.basename(os.path.dirname(os.path.dirname(p)))[-16:], len(st),
        np.median(st), np.percentile(st,25), np.percentile(st,75), np.percentile(st,95)))
