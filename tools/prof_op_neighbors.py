#!/usr/bin/env python3
"""打印指定算子的邻居（定位它在层结构中的位置）。"""
import csv, sys, re
D, PAT, SHAPE = sys.argv[1], sys.argv[2], sys.argv[3]
rows, marks = [], []
with open(D + "/kernel_details.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        nm = r.get("Name") or ""; ty = r.get("OP Type") or ""
        if "allgatherAicpu" in nm or "allgatherAicpu" in ty:
            try: marks.append(float(r["Start Time(us)"]))
            except Exception: pass
        try:
            a = float(r["Start Time(us)"]); d = float(r["Duration(us)"])
        except Exception: continue
        rows.append((a, d, nm, r.get("Input Shapes") or "", r.get("Input Data Types") or "",
                     r.get("Stream ID") or ""))
marks.sort(); LO, HI = marks[2], marks[4]
starts = [m for m in marks if LO <= m < HI]
sel = sorted([x for x in rows if LO <= x[0] < HI])
rx = re.compile(PAT)
hits = [x for x in sel if rx.search(x[2]) and SHAPE in x[3]]
print(f"命中 {len(hits)} 个（窗口 {len(starts)-1} 步）")
if not hits: raise SystemExit
# 取第 2 个命中，打印前后邻居
h = hits[min(2, len(hits)-1)]
i = sel.index(h)
print(f"\n目标: {h[2][:50]} shape={h[3]} in={h[4]} s{h[5]}\n")
print(f"{'偏移ms':>8} {'流':>5} {'算子':<40}{'时长us':>9} {'形状':<28}")
for x in sel[max(0, i-12): i+8]:
    mark = "  <<<" if x is h else ""
    print(f"{(x[0]-h[0])/1000:>+8.3f} {str(x[5])[:4]:>5} {x[2][:40]:<40}{x[1]:>9.1f} {x[3][:28]:<28}{mark}")
