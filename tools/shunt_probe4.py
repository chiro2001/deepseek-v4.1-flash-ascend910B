#!/usr/bin/env python3
"""大 gap（主流空洞）里到底在跑什么。"""
import csv, sys, statistics
from collections import Counter, defaultdict
D=sys.argv[1]
rows=[]
with open(D,newline="") as fh:
    for r in csv.DictReader(fh):
        try: st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
        except Exception: continue
        rows.append((st,du,st+du,(r.get("Accelerator Core") or "").strip(),
                     (r.get("Stream ID") or "").strip(),(r.get("Name") or "")[:40]))
rows.sort()
anchors=sorted(r[0] for r in rows if r[5].startswith("HcPre")); PER=86
STEP=statistics.median([anchors[i+PER]-anchors[i] for i in range(len(anchors)-PER)])
k0=PER*3; S,E=anchors[k0],anchors[k0+3*PER]; N=3
sel=[r for r in rows if r[0]<E and r[2]>S]
def merge(iv):
    if not iv: return []
    iv=sorted(iv); out=[list(iv[0])]
    for s,e in iv[1:]:
        if s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return out
MAIN="109"
for j in range(N):
    s0=anchors[k0+j*PER]; s1=anchors[k0+(j+1)*PER]
    mb=merge([(r[0],r[2]) for r in sel if r[4]==MAIN and r[0]<s1 and r[2]>s0])
    gaps=[]; cur=s0
    for s,e in mb:
        if s>cur: gaps.append((cur,s))
        cur=max(cur,e)
    big=[g for g in gaps if g[1]-g[0]>=200]
    for gs,ge in big:
        print("="*66)
        print("大 gap: %.3f ms  位置 %.0f%%  span=[%.0f,%.0f]"%((ge-gs)/1000,(gs-s0)/(s1-s0)*100,gs,ge))
        other=[r for r in sel if r[4]!=MAIN and r[0]<ge and r[2]>gs]
        print("  期间侧流算子 %d 个，合计 %.3f ms"%(len(other),sum(r[1] for r in other)/1000))
        agg=defaultdict(lambda:[0.0,0])
        for r in other: agg[(r[4],r[3],r[5])][0]+=r[1]; agg[(r[4],r[3],r[5])][1]+=1
        for (st,core,nm),(t,c) in sorted(agg.items(),key=lambda x:-x[1][0])[:12]:
            print("    s%-4s %-15s %-40s %8.3f ms n=%d"%(st,core,nm,t/1000,c))
        # 覆盖度
        cov=merge([(r[0],r[2]) for r in other])
        covs=sum(min(e,ge)-max(s,gs) for s,e in cov)
        print("  → 侧流覆盖 %.3f ms / gap %.3f ms (%.0f%%)"%(covs/1000,(ge-gs)/1000,covs/(ge-gs)*100))
        break
    break
