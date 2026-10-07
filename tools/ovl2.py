#!/usr/bin/env python3
"""按算子名分组统计：span(A) span(B) 交集 并集，以及串行/并发对比。"""
import csv, os, sys, glob
root=sys.argv[1]
def merge(iv):
    if not iv: return []
    iv.sort(); out=[list(iv[0])]
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
KEY={"MatMulCommon_MatMulV2":"AIC_mm","RmsNorm":"AIV_rms","DynamicQuantV2":"AIV_dq",
     "MoeInitRoutingV3":"MIXV_route"}
res={}
for d in sorted(glob.glob(os.path.join(root,"*"))):
    if not os.path.isdir(d): continue
    f=glob.glob(os.path.join(d,"**","ASCEND_PROFILER_OUTPUT","kernel_details.csv"),recursive=True)
    if not f: continue
    g={}
    with open(f[0],newline="") as fh:
        for r in csv.DictReader(fh):
            try: st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
            except Exception: continue
            nm=r.get("Name") or ""
            for k,tag in KEY.items():
                if k in nm:
                    g.setdefault(tag,[]).append((st,st+du)); break
    g={k:merge(v) for k,v in g.items()}
    res[os.path.basename(d)]=g
print("%-34s %-10s %9s %9s %9s %9s %9s"%("graph","op","span_ms","n","并发交集","并集ms","min占比"))
for name in sorted(res):
    g=res[name]; keys=sorted(g)
    if len(keys)<2: 
        print("%-34s %s"%(name,keys)); continue
    a,b=keys[0],keys[1]
    A,B=g[a],g[b]
    U=merge([tuple(x) for x in A]+[tuple(x) for x in B])
    isec=inter(A,B)
    print("%-34s %-10s %9.3f %9d"%(name,a,span(A)/1000,len(A)))
    print("%-34s %-10s %9.3f %9d %9.3f %9.3f %8.1f%%"%("",b,span(B)/1000,len(B),isec/1000,span(U)/1000,isec/min(span(A),span(B))*100))
print("\n=== 串行 vs 并发（同 pair 对比）===")
print("%-28s %10s %10s %8s"%("pair","串行并集","并发并集","压缩比"))
seen=set()
for name in sorted(res):
    if name.endswith("__serial"):
        base=name[:-8]; conc=base+"__conc"
        if conc not in res: continue
        def uni(g):
            alliv=[]
            for v in g.values(): alliv+= [tuple(x) for x in v]
            return span(merge(alliv))/1000
        us,uc=uni(res[name]),uni(res[conc])
        print("%-28s %10.3f %10.3f %7.2fx"%(base.split("__")[0]+"|"+base.split("__")[1],us,uc,us/uc))
