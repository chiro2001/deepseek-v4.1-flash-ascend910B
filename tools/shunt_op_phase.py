import csv, statistics
from collections import defaultdict
D="/home/l00886679/cedpd-repo/results/armF_r6_base/prof/dp0_pp0_tp0_dcp0_ep0_rank0_1434_20261004195431495_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
agg=defaultdict(lambda: defaultdict(list))
def cls(nm,core):
    if nm.startswith("HcPre"): return "HcPre(MIX_AIC)"
    if nm.startswith("HcPost"): return "HcPost(AIV)"
    if nm.startswith("SparseFlashMla"): return "SparseFlashMla(MIX_AIC)"
    if "GroupedMatmul" in nm: return "GroupedMatmul(MIX_AIC)"
    if nm.startswith("RmsNorm") or nm.startswith("RmsNormCast"): return "RmsNorm(AIV)"
    if "MatMulCommon" in nm or "MatMulV3" in nm: return "MatMulV2/V3(AI_CORE)"
    if "QuantMatmul" in nm: return "QuantBatchMatmul(MIX_AIC)"
    if "DynamicQuant" in nm: return "DynamicQuant(AIV)"
    if nm.startswith("MoeInitRouting"): return "MoeInitRouting(MIX_AIV)"
    return None
with open(D,newline="") as fh:
    for r in csv.DictReader(fh):
        if (r.get("Stream ID") or "").strip()!="109": continue
        k=cls(r.get("Name") or "",(r.get("Accelerator Core") or ""))
        if not k: continue
        try:
            dur=float(r["Duration(us)"])
            aic=float(r.get("aicore_time(us)") or 0)
            aiv=float(r.get("aiv_time(us)") or 0)
        except: continue
        agg[k]["dur"].append(dur); agg[k]["aic"].append(aic); agg[k]["aiv"].append(aiv)
        agg[k]["blk"].append(r.get("Block Num") or "")
        agg[k]["mixblk"].append(r.get("Mix Block Num") or "")
print("%-28s %6s %8s %9s %9s %8s %8s %8s"%("算子","n","blk","Duration","aicore_t","aiv_t","AIC/D","AIV/D"))
for k in sorted(agg, key=lambda x:-sum(agg[x]["dur"])):
    d=agg[k]
    med=lambda a: statistics.median(a) if a else 0
    D_=med(d["dur"]); A=med(d["aic"]); V=med(d["aiv"])
    print("%-28s %6d %8s %9.2f %9.2f %8.2f %8s %8s"%(k,len(d["dur"]),d["blk"][0][:6],D_,A,V,
        ("%.0f%%"%(A/D_*100) if D_ else "-"), ("%.0f%%"%(V/D_*100) if D_ else "-")))
print("\n=== 明细：MIX 算子的 aicore_time vs aiv_time 分布 ===")
for k in ("HcPre(MIX_AIC)","SparseFlashMla(MIX_AIC)","GroupedMatmul(MIX_AIC)","QuantBatchMatmul(MIX_AIC)"):
    if k not in agg: continue
    d=agg[k]
    D_=sorted(d["dur"]); A=sorted(d["aic"]); V=sorted(d["aiv"])
    n=len(D_)
    print("### %s (n=%d)"%(k,n))
    for q,lab in ((0,"min"),(n//2,"p50"),(n-1,"max")):
        print("    %-4s dur=%8.2f  aicore=%8.2f (%.0f%%)  aiv=%8.2f (%.0f%%)"%(
            lab,D_[q],A[q],A[q]/D_[q]*100 if D_[q] else 0,V[q],V[q]/D_[q]*100 if D_[q] else 0))
    sA=sum(d["aic"])/sum(d["dur"])*100; sV=sum(d["aiv"])/sum(d["dur"])*100
    print("    合计占比: aicore=%.1f%%  aiv=%.1f%%  (两者之和 %.1f%%)"%(sA,sV,sA+sV))
