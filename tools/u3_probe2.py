#!/usr/bin/env python3
"""U3 探针2：主流(109)空闲窗口里谁在跑；AIV 块是否"紧跟AIC"（=真依赖的形态）。"""
import csv, sys, statistics
from collections import defaultdict, Counter

D = sys.argv[1]
AIC={"AI_CORE","MIX_AIC"}; AIV={"AI_VECTOR_CORE","MIX_AIV"}
COMM={"COMMUNICATION"}; AICPU={"AI_CPU"}
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
def inter(A,B):
    i=j=0;t=0.0
    while i<len(A) and j<len(B):
        s=max(A[i][0],B[j][0]); e=min(A[i][1],B[j][1])
        if e>s: t+=e-s
        if A[i][1]<B[j][1]: i+=1
        else: j+=1
    return t
def span(iv): return sum(e-s for s,e in iv)
MAIN="109"
main=merge([(r[0],r[2]) for r in sel if r[4]==MAIN])
gap=[]; cur=S
for s,e in main:
    if s>cur: gap.append((cur,min(s,E)))
    cur=max(cur,e)
if cur<E: gap.append((cur,E))
gap=merge(gap)
print("步长 %.3f ms | 主流busy %.3f (%.1f%%) | 主流gap %.3f (%.1f%%)"%(
    STEP/1000, span(main)/N/1000, span(main)/N/STEP*100, span(gap)/N/1000, span(gap)/N/STEP*100))
print("\n=== 主流 gap 里谁在跑（gap=%.3f ms/步）==="%(span(gap)/N/1000))
for nm,cs,st in [("AIC",AIC,None),("AIV",AIV,None),("COMM",COMM,None),("AICPU",AICPU,None),
                 ("AIC@侧流",AIC,"side"),("AIV@侧流",AIV,"side")]:
    iv=merge([(r[0],r[2]) for r in sel if r[3] in cs and (st is None or r[4]!=MAIN)])
    o=inter(gap,iv)/N
    print("  ∩ %-10s %7.3f ms/步  (%5.1f%% of gap)"%(nm,o/1000,o/span(gap)*100))
allother=merge([(r[0],r[2]) for r in sel if r[4]!=MAIN])
print("  → 侧流全空 = %.3f ms/步"%( (span(gap)-inter(gap,allother))/N/1000 ))

# AIV 块后面紧跟谁
print("\n=== 主流上 AIV 块(合并后)后面紧跟的算子（前 3 个）===")
mb=[r for r in sel if r[4]==MAIN]
bym=defaultdict(list)
for r in mb: bym[r[3]].append(r)
aivsel=[r for r in mb if r[3] in AIV]
# 合并 AIV 块
aiviv=merge([(r[0],r[2]) for r in aivsel])
# 找每个 AIV 块结束后 3us 内开始的算子
after=Counter(); delay=[]
starts=sorted([(r[0],r[3],r[5]) for r in mb])
import bisect
ks=[x[0] for x in starts]
for s,e in aiviv:
    i=bisect.bisect_left(ks,e)
    if i<len(starts):
        d=starts[i][0]-e
        delay.append(d)
        for j in range(i,min(i+3,len(starts))):
            after[(starts[j][1],starts[j][2])]+=1
print("  AIV 块数=%.1f/步  结束后→下一个主流算子的间隔: 中位 %.1f us, p25 %.1f, p75 %.1f"%(
    len(aiviv)/N, statistics.median(delay), sorted(delay)[len(delay)//4], sorted(delay)[3*len(delay)//4]))
for (core,nm),c in after.most_common(8):
    print("   %-16s %-34s %5.1f 次/步"%(core,nm,c/N))
