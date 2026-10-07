#!/usr/bin/env python3
import csv, sys, statistics
D=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 130
rows=[]
with open(D,newline="") as fh:
    for r in csv.DictReader(fh):
        try: st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
        except Exception: continue
        rows.append(dict(st=st, du=du, en=st+du,
                         core=(r.get("Accelerator Core") or "").strip(),
                         sid=(r.get("Stream ID") or "").strip(),
                         nm=(r.get("Name") or ""),
                         blk=(r.get("Block Num") or ""),
                         shp=(r.get("Input Shapes") or "")[:70]))
rows.sort(key=lambda x: x["st"])
anchors=sorted(r["st"] for r in rows if r["nm"].startswith("HcPre")); PER=86
STEP=statistics.median([anchors[i+PER]-anchors[i] for i in range(len(anchors)-PER)])
k0=PER*5; S,E=anchors[k0],anchors[k0+PER]
main=[r for r in rows if r["sid"]=="109" and r["st"]<E and r["en"]>S]
print("step %.1f us  主流算子 %d 个/步"%(STEP,len(main)))
print("%4s %8s %-15s %-40s %7s %-5s"%("idx","t(us)","core","name","dur","blk"))
for i,r in enumerate(main[:N]):
    print("%4d %8.1f %-15s %-40s %7.1f %-5s"%(i,r["st"]-S,r["core"][:15],r["nm"][:40],r["du"],r["blk"]))
