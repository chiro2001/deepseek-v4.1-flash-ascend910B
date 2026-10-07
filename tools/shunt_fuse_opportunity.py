#!/usr/bin/env python3
"""量化"纯AIC算子与纯AIV算子互相填坑"的机会与天花板。

思路：纯单引擎算子的时间段里，另一个引擎 100% 闲置。
      若能把纯 AIC 工作挪进"AIC 全闲"的段，就能把它们藏起来。
输出：① 机会总量 ② 天花板 ③ 位置邻近性（决定能否配对）
"""
import csv, sys, statistics
from collections import defaultdict
D=sys.argv[1]
rows=[]
with open(D+"/kernel_details.csv",newline="") as fh:
    for r in csv.DictReader(fh):
        if (r.get("Stream ID") or "").strip()!="109": continue
        try:
            st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
            aic=float(r.get("aicore_time(us)") or 0); aiv=float(r.get("aiv_time(us)") or 0)
        except: continue
        rows.append(dict(st=st,du=du,en=st+du,core=(r.get("Accelerator Core") or "").strip(),
                         nm=(r.get("Name") or ""),aic=aic,aiv=aiv))
rows.sort(key=lambda x:x["st"])
anch=sorted(x["st"] for x in rows if x["nm"].startswith("HcPre")); PER=86
STEP=statistics.median([anch[i+PER]-anch[i] for i in range(len(anch)-PER)])
S,E=anch[PER*5],anch[PER*6]
seq=[x for x in rows if x["st"]<E and x["en"]>S]
def kind(x):
    a,v=x["aic"],x["aiv"]
    if a>1 and v>1: return "MIX"
    if a>1: return "AIC_ONLY"
    if v>1: return "AIV_ONLY"
    return "OTHER"
for x in seq: x["k"]=kind(x)
tot=sum(x["du"] for x in seq)
print("步长(profile) %.1f µs | 主流 %d 算子 | 合计 %.1f µs"%(STEP,len(seq),tot))
agg=defaultdict(lambda:[0.0,0,0.0,0.0])
for x in seq:
    a=agg[x["k"]]; a[0]+=x["du"]; a[1]+=1; a[2]+=x["aic"]; a[3]+=x["aiv"]
print("\n=== 按引擎占用分类 ===")
print("%-10s %6s %10s %9s %9s %8s"%("类别","n","Duration","aicore","aiv","占步长"))
for k in ("MIX","AIC_ONLY","AIV_ONLY","OTHER"):
    if k not in agg: continue
    a=agg[k]
    print("%-10s %6d %10.1f %9.1f %9.1f %7.1f%%"%(k,a[1],a[0],a[2],a[3],a[0]/tot*100))
aic_only=agg["AIC_ONLY"][0]; aiv_only=agg["AIV_ONLY"][0]
mix=agg["MIX"][0]
print("\n=== 机会与天花板 ===")
print("  纯 AIC 段 %.1f µs（此时 AIV 100%% 闲）"%(aic_only))
print("  纯 AIV 段 %.1f µs（此时 AIC 100%% 闲）"%(aiv_only))
print("  ⇒ **互相填坑的天花板 = min(两者) = %.1f µs = 步长的 %.1f%%**"%(min(aic_only,aiv_only),min(aic_only,aiv_only)/tot*100))
print("  ⇒ 理想加速 = %.1f/(%.1f) = %.3fx"%(STEP,STEP-min(aic_only,aiv_only),STEP/(STEP-min(aic_only,aiv_only))))
# 位置邻近性
idx={id(x):i for i,x in enumerate(seq)}
aic_idx=[i for i,x in enumerate(seq) if x["k"]=="AIC_ONLY"]
aiv_idx=[i for i,x in enumerate(seq) if x["k"]=="AIV_ONLY"]
print("\n=== 位置邻近性（纯AIC算子 ±k 位置内有没有纯AIV算子）===")
import bisect
for k in (1,2,3,5,10):
    hit=0; tot_a=0.0; hit_a=0.0
    for i in aic_idx:
        tot_a+=seq[i]["du"]
        lo=bisect.bisect_left(aiv_idx,i-k); hi=bisect.bisect_right(aiv_idx,i+k)
        if hi>lo: hit+=1; hit_a+=seq[i]["du"]
    print("  ±%-3d 位置: %3d/%3d 个纯AIC算子有纯AIV邻居, 覆盖 %.1f/%.1f µs"%(k,hit,len(aic_idx),hit_a,tot_a))
# 连续段结构
print("\n=== 纯 AIV 段的长度分布（可塞入 AIC 工作的容器）===")
segs=[]; cur=None
for x in seq:
    if x["k"]=="AIV_ONLY":
        if cur is None: cur=[x["du"],1]
        else: cur[0]+=x["du"]; cur[1]+=1
    else:
        if cur: segs.append(cur); cur=None
if cur: segs.append(cur)
print("  纯AIV段 %d 个, 时长中位 %.1f µs / 均值 %.1f / 最大 %.1f"%(len(segs),
    statistics.median([s[0] for s in segs]), sum(s[0] for s in segs)/len(segs), max(s[0] for s in segs)))
big=[s for s in segs if s[0]>=20]
print("  ≥20µs 的段: %d 个, 合计 %.1f µs（可容纳较大的 AIC 算子）"%(len(big),sum(s[0] for s in big)))
