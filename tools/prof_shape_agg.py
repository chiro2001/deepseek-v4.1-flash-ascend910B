
import csv, sys, re, statistics as st
from collections import defaultdict
D = sys.argv[1]
PAT = re.compile(sys.argv[2])
rows, marks = [], []
with open(D + "/kernel_details.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        nm = r.get("Name") or ""; ty = r.get("OP Type") or ""
        if "allgatherAicpu" in nm or "allgatherAicpu" in ty:
            try: marks.append(float(r["Start Time(us)"]))
            except Exception: pass
        if not PAT.search(nm): continue
        try:
            a = float(r["Start Time(us)"]); d = float(r["Duration(us)"])
        except Exception: continue
        rows.append((a, d, nm, r.get("Input Shapes") or "", r.get("Input Data Types") or ""))
marks.sort(); LO, HI = marks[2], marks[-3]
starts = [m for m in marks if LO <= m < HI]; nst = len(starts)
sel = [x for x in rows if LO <= x[0] < HI]
agg = defaultdict(lambda: [0, 0.0, []])
for a, d, nm, ish, idt in sel:
    k = (nm[:44], ish[:34], idt[:18])
    agg[k][0] += 1; agg[k][1] += d
    for i in range(len(starts)-1):
        if starts[i] <= a < starts[i+1]:
            agg[k][2].append((a - starts[i]) / (starts[i+1] - starts[i])); break
print(f"步数 {nst}｜总命中 {len(sel)/nst:.1f} 个/步")
print(f"{'算子':<44}{'形状':<36}{'入dtype':<20}{'次/步':>7}{'中位us':>8}{'合计ms':>8}{'位置':>7}")
for (nm, ish, idt), (c, d, pos) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:18]:
    med = d / c
    print(f"{nm:<44}{ish:<36}{idt:<20}{c/nst:>7.1f}{med:>8.2f}{d/1000/nst:>8.3f}{st.median(pos) if pos else -1:>7.2f}")
