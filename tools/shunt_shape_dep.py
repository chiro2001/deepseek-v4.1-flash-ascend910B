#!/usr/bin/env python3
"""依赖审计（形状法）：用输入/输出形状匹配判定相邻算子对是否有数据依赖。

判定：若 op[i+1] 的任一输入形状 == op[i] 的任一输出形状 ⇒ 疑似依赖（保守）
输出：每层的"独立对"数量与对应时长
"""
import csv, sys, statistics, re
from collections import Counter, defaultdict
D=sys.argv[1]
def shapes(s):
    return [x.strip() for x in (s or "").split(";") if x.strip()]
rows=[]
with open(D+"/kernel_details.csv",newline="") as fh:
    for r in csv.DictReader(fh):
        if (r.get("Stream ID") or "").strip()!="109": continue
        try:
            st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
            aic=float(r.get("aicore_time(us)") or 0); aiv=float(r.get("aiv_time(us)") or 0)
        except: continue
        rows.append(dict(st=st,du=du,en=st+du,aic=aic,aiv=aiv,
                         nm=(r.get("Name") or ""),blk=(r.get("Block Num") or ""),
                         ins=shapes(r.get("Input Shapes")),outs=shapes(r.get("Output Shapes"))))
rows.sort(key=lambda x:x["st"])
anch=sorted(x["st"] for x in rows if x["nm"].startswith("HcPre")); PER=86
S,E=anch[PER*5],anch[PER*6]
seq=[x for x in rows if x["st"]<E and x["en"]>S]
def kind(x):
    if x["aic"]>1 and x["aiv"]>1: return "MIX"
    if x["aic"]>1: return "AIC"
    if x["aiv"]>1: return "AIV"
    return "OTH"
# 相邻对依赖判定
dep=0; indep=0; dep_us=0.0; indep_pairs=[]
for i in range(len(seq)-1):
    a,b=seq[i],seq[i+1]
    linked = any(s in b["ins"] for s in a["outs"] if s) if a["outs"] else False
    if linked: dep+=1; dep_us+=b["du"]
    else:
        indep+=1
        indep_pairs.append((i,kind(a),kind(b),b["du"],a["nm"][:22],b["nm"][:22]))
print("一层 %d 算子 | 相邻对 %d 个"%(len(seq),len(seq)-1))
print("  疑似有依赖: %d 个 (%.0f%%)"%(dep,dep/len(seq)*100))
print("  形状不匹配: %d 个 (%.0f%%)  ← 这些是**潜在可并行对**"%(indep,indep/len(seq)*100))
print("\n=== 形状不匹配的相邻对（按类型组合）===")
c=Counter((a,b) for _,a,b,_,_,_ in indep_pairs)
for k,v in c.most_common(12): print("   %-16s %3d 对"%(str(k),v))
print("\n=== 其中 AIC→AIV 或 AIV→AIC 的（最有价值：可分到两条流）===")
val=[p for p in indep_pairs if {p[1],p[2]} & {"AIC","AIV"} and p[1]!=p[2]]
print("  共 %d 对，涉及时长 %.1f µs"%(len(val),sum(p[3] for p in val)))
for i,a,b,du,x,y in val[:14]:
    print("   [%3d] %-4s→%-4s %7.1fµs  %-22s → %s"%(i,a,b,du,x,y))
print("\n=== 纯 AIC（AI_CORE）算子的邻居依赖 ===")
for i,x in enumerate(seq):
    if kind(x)!="AIC": continue
    nxt=seq[i+1] if i+1<len(seq) else None
    linked = nxt and any(s in nxt["ins"] for s in x["outs"] if s) if x["outs"] else False
    print("   [%3d] %-22s dur=%6.1f → 下一个 %-22s 依赖=%s"%(i,x["nm"][:22],x["du"],
          nxt["nm"][:22] if nxt else "-", "是" if linked else "否"))
