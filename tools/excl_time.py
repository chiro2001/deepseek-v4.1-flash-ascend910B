#!/usr/bin/env python3
"""算每个算子类型在**设备并集**里的"独占贡献"：union(all) - union(all \ type)。
这比 sum 占比更能回答"砍掉它能省多少墙钟"——被重叠掉的部分贡献为 0。
用法: python3 excl.py <op_summary.csv> [topN]
"""
import csv, sys, collections

f = sys.argv[1]
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 12

ev = []          # (start, end, key)
hc = 0
with open(f, encoding="utf-8", errors="ignore") as fh:
    rd = csv.reader(fh); h = next(rd)
    iN, iT, iD, iS, iSt = h.index("Op Name"), h.index("OP Type"), h.index("Task Duration(us)"), h.index("Stream ID"), h.index("Task Start Time(us)")
    for r in rd:
        if len(r) < 20: continue
        try:
            st = float(r[iSt]); d = float(r[iD])
        except ValueError:
            continue
        if d <= 0: continue
        if r[iT] == "HcPre": hc += 1
        key = r[iN] if r[iT].startswith("hcom_") else r[iT]
        ev.append((st, st + d, key))

# 去重：hcom_ 与 AivKernel 是同一份工作的两次记账 ⇒ 只留一种
def dedupe(events):
    seen, out = set(), []
    for st, en, k in sorted(events):
        sig = (round(st, 1), round(en, 1))
        if sig in seen: continue
        seen.add(sig); out.append((st, en, k))
    return out

ev = dedupe(ev)
def union_ms(items):
    if not items: return 0.0
    tot, cs, ce = 0.0, None, None
    for st, en in sorted(items):
        if cs is None: cs, ce = st, en
        elif st <= ce:
            if en > ce: ce = en
        else:
            tot += ce - cs; cs, ce = st, en
    tot += ce - cs
    return tot / 1000.0

U = union_ms([(s, e) for s, e, _ in ev])
by = collections.defaultdict(list)
for s, e, k in ev: by[k].append((s, e))
sums = {k: sum(e - s for s, e in v) / 1000.0 for k, v in by.items()}
# 独占贡献
excl = {}
for k, v in by.items():
    rest = [(s, e) for s, e, kk in ev if kk != k]
    excl[k] = U - union_ms(rest)

print("union = %.1f ms   (HcPre=%d)" % (U, hc))
print("%-42s %9s %9s %9s" % ("OP", "sum_ms", "excl_ms", "excl/U"))
for k, _ in sorted(excl.items(), key=lambda kv: -kv[1])[:TOPN]:
    print("%-42s %9.1f %9.1f %8.1f%%" % (k[:42], sums[k], excl[k], 100.0 * excl[k] / U if U else 0))
