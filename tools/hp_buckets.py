import re, sys, statistics as st
for run in sys.argv[1:]:
    txt = open(f"/home/l00886679/cedpd-repo/results/{run}/serve.log", encoding="utf-8", errors="ignore").read()
    d = {}
    for m in re.finditer(r"\[bneck\].*?dec=(\d+) hp=([0-9.]+).*?n=(\d+)", txt):
        n = int(m.group(3)); hp = float(m.group(2))
        if n in (6,12,18,24,48):
            d.setdefault(n, []).append(hp)
    print("===", run)
    for n in (6,12,18,24,48):
        v = d.get(n, [])
        if v:
            print("  n=%-3d 样本=%4d 中位=%7.2f p10=%.2f p90=%.2f" % (n, len(v), st.median(v), sorted(v)[int(len(v)*0.1)], sorted(v)[int(len(v)*0.9)]))
        else:
            print("  n=%-3d 无样本" % n)
