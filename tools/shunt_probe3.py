#!/usr/bin/env python3
import csv, sys, statistics
from collections import Counter, defaultdict
D=sys.argv[1]
AIC={"AI_CORE","MIX_AIC"}; AIV={"AI_VECTOR_CORE","MIX_AIV"}
rows=[]
with open(D,newline="") as fh:
    for r in csv.DictReader(fh):
        try: st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
        except Exception: continue
        rows.append((st,du,st+du,(r.get("Accelerator Core") or "").strip(),
                     (r.get("Stream ID") or "").strip(),(r.get("Name") or "")[:34]))
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
# 每步单独算 gap 分布，按"落在步内第几成"
per_step=[]
for j in range(N):
    s0=anchors[k0+j*PER]; s1=anchors[k0+(j+1)*PER]
    mb=merge([(r[0],r[2]) for r in sel if r[4]==MAIN and r[0]<s1 and r[2]>s0])
    cur=s0
    for s,e in mb:
        if s>cur and s-cur>1.0: per_step.append((cur-s0,(s-cur),s1-s0))
        cur=max(cur,e)
    if cur<s1: per_step.append((cur-s0,(s1-cur),s1-s0))
print("步长 %.1f us | 主流 gap 块数 %.1f/步, 合计 %.3f ms/步"%(
    STEP, len(per_step)/N, sum(g for _,g,_ in per_step)/N/1000))
gaps=sorted([g for _,g,_ in per_step], reverse=True)
print("gap 大小分布: p50=%.1fus p75=%.1f p90=%.1f max=%.1f us"%(
    statistics.median(gaps), gaps[len(gaps)//4], gaps[int(len(gaps)*0.1)], gaps[0]))
big=[g for g in gaps if g>=200]
print("  ≥200us 的 gap: %.1f 个/步, 合计 %.3f ms/步"%(len(big)/N, sum(big)/N/1000))
mid=[g for g in gaps if 50<=g<200]
print("  50~200us     : %.1f 个/步, 合计 %.3f ms/步"%(len(mid)/N, sum(mid)/N/1000))
sm=[g for g in gaps if g<50]
print("  <50us        : %.1f 个/步, 合计 %.3f ms/步"%(len(sm)/N, sum(sm)/N/1000))
# 按步内位置（百分比）分桶
buck=defaultdict(float)
for off,g,tot in per_step:
    b=min(9,int(off/tot*10)); buck[b]+=g
print("\n主流 gap 按步内位置分布（ms/步）:")
for b in range(10):
    print("  %3d-%3d%%  %7.3f"%(b*10,(b+1)*10,buck[b]/N/1000))
# 大 gap 里谁在跑
bigiv=[(off,off+g) for off,g,_ in per_step if g>=200]
bigabs=[]
for j in range(N):
    s0=anchors[k0+j*PER]
    for off,g,_ in per_step:
        pass
