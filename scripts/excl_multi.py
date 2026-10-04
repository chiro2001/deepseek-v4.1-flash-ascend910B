#!/usr/bin/env python3
"""多 slice 稳态独占分析。用法: excl_multi.py <mindstudio_profiler_output> [tail_frac] [topN]"""
import csv, sys, glob
M = sys.argv[1]
TAIL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
TOPN = int(sys.argv[3]) if len(sys.argv) > 3 else 15

ev = []; hc = 0
for f in sorted(glob.glob(M + "/op_summary*.csv")):
    with open(f, encoding="utf-8", errors="ignore") as fh:
        rd = csv.reader(fh); h = next(rd)
        iN, iT, iD, iS = h.index("Op Name"), h.index("OP Type"), h.index("Task Duration(us)"), h.index("Task Start Time(us)")
        for r in rd:
            if len(r) < 20: continue
            try: d = float(r[iD]); st = float(r[iS])
            except ValueError: continue
            if d <= 0: continue
            if r[iT] == "HcPre": hc += 1
            ev.append((st, st + d, r[iN] if r[iT].startswith("hcom_") else r[iT]))
seen = set(); uniq = []
for st, en, k in sorted(ev):
    sig = (round(st, 1), round(en, 1))
    if sig in seen: continue
    seen.add(sig); uniq.append((st, en, k))
T0 = min(e[0] for e in uniq); T1 = max(e[1] for e in uniq)
cut = T1 - (T1 - T0) * TAIL
ss = [e for e in uniq if e[0] >= cut]
def union_ms(items):
    tot = 0.0; cs = ce = None
    for st, en in sorted((s, e) for s, e, _ in items):
        if cs is None: cs, ce = st, en
        elif st <= ce:
            if en > ce: ce = en
        else:
            tot += ce - cs; cs, ce = st, en
    tot += ce - cs
    return tot / 1000.0
U = union_ms(ss)
print(f"uniq_events={len(uniq)} steady_events={len(ss)} U={U:.1f}ms tail={TAIL}")
sums = {}
for st, en, k in ss:
    sums[k] = sums.get(k, 0.0) + (en - st)
top = sorted(sums.items(), key=lambda x: -x[1])[:TOPN]
print(f"{'op':52s} {'sum_ms':>9s} {'excl_ms':>9s} {'excl%':>7s}")
tot_excl = 0.0
for k, s in top:
    keep = [e for e in ss if e[2] != k]
    e = U - union_ms(keep)
    tot_excl += e
    print(f"{k[:52]:52s} {s/1000:9.1f} {e:9.1f} {e/U*100:6.2f}%")
print(f"{'TOP-'+str(TOPN)+' excl sum':52s} {'':9s} {tot_excl:9.1f} {tot_excl/U*100:6.2f}%")
