import csv, statistics
from collections import defaultdict
D="/home/l00886679/cedpd-repo/results/armF_r6_base/prof/dp0_pp0_tp0_dcp0_ep0_rank0_1434_20261004195431495_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
rows=[]
for r in csv.DictReader(open(D,newline="")):
    if (r.get("Stream ID") or "").strip()!="109": continue
    try:
        st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
        aic=float(r.get("aicore_time(us)") or 0); aiv=float(r.get("aiv_time(us)") or 0)
    except: continue
    rows.append(dict(st=st,du=du,en=st+du,core=(r.get("Accelerator Core") or "").strip(),
                     nm=(r.get("Name") or ""),blk=(r.get("Block Num") or ""),aic=aic,aiv=aiv))
rows.sort(key=lambda x:x["st"])
anch=sorted(x["st"] for x in rows if x["nm"].startswith("HcPre")); PER=86
STEP=statistics.median([anch[i+PER]-anch[i] for i in range(len(anch)-PER)])
S,E=anch[PER*5],anch[PER*6]
seq=[x for x in rows if x["st"]<E and x["en"]>S]
nstep=1
tot_dur=sum(x["du"] for x in seq)
tot_aic=sum(x["aic"] for x in seq); tot_aiv=sum(x["aiv"] for x in seq)
print("步长(profile) %.1f µs | 主流算子 %d 个"%(STEP,len(seq)))
print("ΣDuration = %.1f µs"%(tot_dur))
print("Σaicore_time = %.1f µs  (占步长 %.1f%%)"%(tot_aic,tot_aic/STEP*100))
print("Σaiv_time    = %.1f µs  (占步长 %.1f%%)"%(tot_aiv,tot_aiv/STEP*100))
print("Σ(aic+aiv)   = %.1f µs  (占步长 %.1f%%)  ← 二者可同时忙"%(tot_aic+tot_aiv,(tot_aic+tot_aiv)/STEP*100))
print("\n=== 按算子分组的 AIC/AIV 贡献（每 step, µs）===")
agg=defaultdict(lambda: defaultdict(float))
def cls(nm,core):
    if nm.startswith("HcPre"): return "HcPre MIX_AIC"
    if nm.startswith("SparseFlashMla"): return "SparseFlashMla MIX_AIC"
    if "GroupedMatmul" in nm: return "GroupedMatmul MIX_AIC"
    if "QuantMatmul" in nm: return "QuantBatchMatmul MIX_AIC"
    if "LightningIndexer" in nm: return "QLIndexer MIX_AIC"
    if "MatMulCommon" in nm or "MatMulV3" in nm: return "MatMulV2/V3 AI_CORE"
    if nm.startswith("RmsNorm") or nm.startswith("HcPost") or "DynamicQuant" in nm or nm.startswith("InplacePartialRotary"): return "AIV 常规(RmsNorm/HcPost/DQ/RoPE)"
    if core=="AI_VECTOR_CORE": return "AIV 小算子"
    if core=="MIX_AIV": return "MIX_AIV"
    return "其它"
for x in seq:
    k=cls(x["nm"],x["core"]); a=agg[k]
    a["dur"]+=x["du"]; a["aic"]+=x["aic"]; a["aiv"]+=x["aiv"]; a["n"]+=1
print("%-34s %7s %9s %9s %9s %8s %8s"%("类别","n","Duration","aicore","aiv","AIC/D","AIV/D"))
for k in sorted(agg,key=lambda x:-agg[x]["dur"]):
    a=agg[k]
    print("%-34s %7d %9.1f %9.1f %9.1f %7.0f%% %7.0f%%"%(k,a["n"],a["dur"],a["aic"],a["aiv"],
        a["aic"]/a["dur"]*100 if a["dur"] else 0, a["aiv"]/a["dur"]*100 if a["dur"] else 0))
print("\n=== 理论并行上限 ===")
print("若 AIC 与 AIV 能完全重叠 ⇒ 步长下限 = max(Σaic, Σaiv) = %.1f µs ⇒ 加速 %.2fx"%(
    max(tot_aic,tot_aiv), STEP/max(tot_aic,tot_aiv)))
print("（对比：Σdur = %.1f µs ⇒ 完全串行时步长 %.2fx）"%(tot_dur,STEP/tot_dur))
