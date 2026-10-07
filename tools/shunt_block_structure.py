#!/usr/bin/env python3
"""① 真实主流的"同类型块"结构  ② 我的微基准排布对照"""
import csv, sys, statistics
from collections import Counter, defaultdict
D=sys.argv[1]
AIC={"AI_CORE","MIX_AIC"}; AIV={"AI_VECTOR_CORE","MIX_AIV"}
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
def T(c): return "AIC" if c in AIC else ("AIV" if c in AIV else "OTH")
# 合并成块
blocks=[]
for x in seq:
    t=T(x["core"])
    if blocks and blocks[-1]["t"]==t:
        blocks[-1]["n"]+=1; blocks[-1]["du"]+=x["du"]; blocks[-1]["ops"].append(x)
    else:
        blocks.append(dict(t=t,n=1,du=x["du"],ops=[x],st=x["st"]))
print("步长 %.1f µs | 算子 %d 个 | **块 %d 个**（块=连续同类型算子）"%(STEP,len(seq),len(blocks)))
for t in ("AIC","AIV"):
    bs=[b for b in blocks if b["t"]==t]
    ns=sorted(b["n"] for b in bs)
    print("  %s 块: %d 个, 每块算子数 中位 %d 均值 %.1f p90 %d 最大 %d"%(t,len(bs),ns[len(ns)//2],sum(ns)/len(ns),ns[int(len(ns)*0.9)],ns[-1]))
print("\n=== 块大小分布（算子数→块数）===")
d=defaultdict(Counter)
for b in blocks: d[b["t"]][b["n"]]+=1
for t in ("AIC","AIV"):
    print("  %s: %s"%(t,dict(sorted(d[t].items())[:12])))
print("\n=== 前 24 个块的构成 ===")
print("%4s %-4s %5s %9s %-6s  %s"%("blk","T","#ops","durµs","blocks","算子"))
for i,b in enumerate(blocks[:24]):
    names=" + ".join("%s(%s,%dµs)"%(o["nm"].replace("aclnn","")[:16],o["blk"],o["du"]) for o in b["ops"][:3])
    print("%4d %-4s %5d %9.1f %-6s  %s"%(i,b["t"],b["n"],b["du"],b["ops"][0]["blk"],names))
# 我的排布
print("\n=== ★ 我的微基准排布（对照）===")
print("每层: AIC 8 个 [mix,mm,mix,mm,mix,mm,mix,mm] + AIV 5 个 [rms,dqs,rms,dqs,rms]")
print("  mix=moe_init_routing(MIX_AIC,48blk,~13µs)  mm=matmul(AI_CORE,20blk,~8µs)")
print("  rms=rms_norm(AIV,48blk,~8µs)               dqs=dynamic_quant(AIV,4blk,~3.6µs)")
print("  interleaved 臂 = 上述 1:1 交替（每层 13 个算子）")
print("  bar32 臂      = AIC 按 32 个算子一格, 同格内 AIV 取累计时长匹配者, 格间加屏障")
print("  free 臂       = AIC 全在一流 / AIV 全在另一流, 仅首尾同步")
print("  块结构: 我的 AIC 块 = 32 算子（远大于真实的 1~2）; AIV 块同理")
print("\n=== ★ 真实 vs 我的排布 关键差异 ===")
real_aic=[b["n"] for b in blocks if b["t"]=="AIC"]; real_aiv=[b["n"] for b in blocks if b["t"]=="AIV"]
print("%-14s %-24s %-24s"%("项","真实主流","我的微基准"))
print("%-14s %-24s %-24s"%("算子数/步","%d"%len(seq),"416  (=32层×13)"))
print("%-14s %-24s %-24s"%("切换次数","%d"%len(blocks),"%d  (=逐算子交替)"%416))
print("%-14s %-24s %-24s"%("AIC 块大小","中位 %d 均值 %.1f"%(sorted(real_aic)[len(real_aic)//2],sum(real_aic)/len(real_aic)),"1 (interleaved) / 32 (bar)"))
print("%-14s %-24s %-24s"%("AIV 块大小","中位 %d 均值 %.1f"%(sorted(real_aiv)[len(real_aiv)//2],sum(real_aiv)/len(real_aiv)),"1 (interleaved) / 6 (bar)"))
print("%-14s %-24s %-24s"%("AIC:AIV 时长比","%.2f"%(sum(b['du'] for b in blocks if b['t']=='AIC')/sum(b['du'] for b in blocks if b['t']=='AIV')),"1.36/0.50 = 2.72"))
print("%-14s %-24s %-24s"%("MIX 代理 block","真实 24（HcPre等）","我的 48（moe_init_routing）"))
