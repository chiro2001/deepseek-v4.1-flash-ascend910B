#!/usr/bin/env python3
"""统计真实主流的"可应用面"：按 core 类型（AI_CORE / MIX_AIC / AIV）分时长与块结构。"""
import csv, sys, statistics
from collections import Counter, defaultdict
D=sys.argv[1]
rows=[]
with open(D+"/kernel_details.csv",newline="") as fh:
    for r in csv.DictReader(fh):
        try: st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
        except: continue
        if (r.get("Stream ID") or "").strip()!="109": continue
        rows.append(dict(st=st,du=du,en=st+du,core=(r.get("Accelerator Core") or "").strip(),
                         blk=(r.get("Block Num") or ""),nm=(r.get("Name") or "")))
rows.sort(key=lambda x:x["st"])
anch=sorted(x["st"] for x in rows if x["nm"].startswith("HcPre")); PER=86
STEP=statistics.median([anch[i+PER]-anch[i] for i in range(len(anch)-PER)])
S,E=anch[PER*5],anch[PER*6]
seq=[x for x in rows if x["st"]<E and x["en"]>S]
tot=sum(x["du"] for x in seq)
print("步长 %.1f µs | 主流算子 %d 个 | 合计 %.1f µs (%.1f ms)\n"%(STEP,len(seq),tot,tot/1000))
print("=== ① 按 Accelerator Core 分（决定能不能吃到两流收益）===")
agg=defaultdict(lambda:[0.0,0])
for x in seq:
    agg[x["core"]][0]+=x["du"]; agg[x["core"]][1]+=1
print("%-18s %7s %11s %8s %9s"%("core","个数","合计µs","占比","均µs"))
CU=0.0
for k,(du,c) in sorted(agg.items(),key=lambda x:-x[1][0]):
    print("%-18s %7d %11.1f %7.1f%% %9.2f"%(k,c,du,du/tot*100,du/c))
pure_aic=agg.get("AI_CORE",[0,0])[0]; mix=agg.get("MIX_AIC",[0,0])[0]
aiv=agg.get("AI_VECTOR_CORE",[0,0])[0]+agg.get("MIX_AIV",[0,0])[0]
print("\n=== ② 可应用面（S4 已验证：纯 AI_CORE ∥ AIV 有 1.4x）===")
print("  纯 AI_CORE       : %7.1f µs (%.1f%%)   ← 已验证可重叠"%(pure_aic,pure_aic/tot*100))
print("  MIX_AIC          : %7.1f µs (%.1f%%)   ← 待验证（48blk 代理为负）"%(mix,mix/tot*100))
print("  AIV 类           : %7.1f µs (%.1f%%)   ← 被重叠的一方"%(aiv,aiv/tot*100))
print("  其它             : %7.1f µs"%(tot-pure_aic-mix-aiv))
print("\n=== ③ 纯 AI_CORE 算子的构成 ===")
c2=defaultdict(lambda:[0.0,0])
for x in seq:
    if x["core"]=="AI_CORE": c2[x["nm"].replace("aclnn","")[:34]][0]+=x["du"]; c2[x["nm"].replace("aclnn","")[:34]][1]+=1
for k,(du,c) in sorted(c2.items(),key=lambda x:-x[1][0])[:10]:
    print("   %-36s %6d 个 %9.1f µs"%(k,c,du))
print("\n=== ④ MIX_AIC 算子的构成及其 block ===")
c3=defaultdict(lambda:[0.0,0,Counter()])
for x in seq:
    if x["core"]=="MIX_AIC":
        k=x["nm"].replace("aclnn","")[:34]; c3[k][0]+=x["du"]; c3[k][1]+=1; c3[k][2][x["blk"]]+=1
for k,(du,c,bl) in sorted(c3.items(),key=lambda x:-x[1][0])[:10]:
    print("   %-36s %6d 个 %9.1f µs  blk=%s"%(k,c,du,dict(bl)))
