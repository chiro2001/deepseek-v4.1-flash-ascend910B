#!/usr/bin/env python3
"""从真实 trace 导出：① 算子类型汇总 ② 一个 step 的完整序列（类型/时长/block/名字）
   ③ 相邻切换对与间隙"""
import csv, sys, statistics
from collections import Counter, defaultdict
D=sys.argv[1]
N_OP=int(sys.argv[2]) if len(sys.argv)>2 else 120
AIC={"AI_CORE","MIX_AIC"}; AIV={"AI_VECTOR_CORE","MIX_AIV"}
rows=[]
with open(D+"/kernel_details.csv",newline="") as fh:
    for r in csv.DictReader(fh):
        try: st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
        except: continue
        if (r.get("Stream ID") or "").strip()!="109": continue
        rows.append(dict(st=st, du=du, en=st+du,
                         core=(r.get("Accelerator Core") or "").strip(),
                         blk=(r.get("Block Num") or ""),
                         nm=(r.get("Name") or "")))
rows.sort(key=lambda x:x["st"])
anch=sorted(x["st"] for x in rows if x["nm"].startswith("HcPre")); PER=86
STEP=statistics.median([anch[i+PER]-anch[i] for i in range(len(anch)-PER)])
k0=PER*5; S,E=anch[k0],anch[k0+PER]
seq=[x for x in rows if x["st"]<E and x["en"]>S]
def T(c):
    if c in AIC: return "AIC"
    if c in AIV: return "AIV"
    return "OTH"
print("步长 %.1f µs | 一 step 主流算子 %d 个"%(STEP,len(seq)))
# ① 类型汇总
print("\n=== ① 按 (类型, block) 汇总 ===")
agg=defaultdict(lambda:[0.0,0])
for x in seq:
    agg[(T(x["core"]),x["blk"])][0]+=x["du"]; agg[(T(x["core"]),x["blk"])][1]+=1
tot=sum(v[0] for v in agg.values())
print("%-8s %-8s %7s %11s %9s %8s"%("类型","block","个数","合计µs","占比","均µs"))
for (t,b),(du,c) in sorted(agg.items(), key=lambda x:-x[1][0]):
    print("%-8s %-8s %7d %11.1f %8.1f%% %8.2f"%(t,b,c,du,du/tot*100,du/c))
# ② 序列
print("\n=== ② 前 %d 个算子的完整顺序（T=类型 切换处标 *）==="%N_OP)
print("%4s %-4s %-18s %-5s %8s %8s  %s"%("idx","T","core","blk","durµs","gapµs","name"))
prev=None
for i,x in enumerate(seq[:N_OP]):
    t=T(x["core"]); gap=(x["st"]-prev["en"]) if prev else 0.0
    mark="*" if prev and T(prev["core"])!=t else " "
    print("%4d %-4s %-18s %-5s %8.1f %8.1f %s %s"%(i,t,x["core"][:18],x["blk"],x["du"],gap,mark,x["nm"][:40]))
    prev=x
# ③ 切换对
print("\n=== ③ 类型切换统计（一个 step 内）===")
sw=0; gaps=[]; pairs=Counter()
for i in range(1,len(seq)):
    p,c=seq[i-1],seq[i]
    if T(p["core"])!=T(c["core"]):
        sw+=1; gaps.append(c["st"]-p["en"]); pairs[(p["nm"][:22],c["nm"][:22])]+=1
print("切换次数 %d 次/步；间隙：中位 %.2f µs 均值 %.2f µs 合计 %.3f ms"%(
    sw, statistics.median(gaps), sum(gaps)/len(gaps), sum(gaps)/1000))
print("\n最常见切换对:")
for (a,b),n in pairs.most_common(10): print("   %3d 次  %-24s → %s"%(n,a,b))
