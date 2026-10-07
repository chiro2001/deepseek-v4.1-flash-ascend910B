#!/usr/bin/env python3
"""决定性：两流并发时，每个算子自己是不是变慢了？（solo vs 2-stream 的逐算子时长）"""
import torch, torch_npu, os, glob, csv, shutil, statistics
from collections import defaultdict
torch.npu.set_device(0); dev="npu:0"
M=48
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
K=int(os.environ.get("K","64"))
G=bf16(5120)
XA=[bf16(M,1024) for _ in range(K)]; XV=[bf16(M,5120) for _ in range(K)]
W=bf16(1024,5120)
aic=lambda i: torch.matmul(XA[i],W)
aiv=lambda i: torch_npu.npu_rms_norm(XV[i],G)[0]

def build(kind, seg=16):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    nseg=(K+seg-1)//seg
    ef=[torch.npu.Event() for _ in range(nseg)]; ej=[torch.npu.Event() for _ in range(nseg)]
    def body():
        if kind=="aic":
            for i in range(K): aic(i)
        elif kind=="aiv":
            for i in range(K): aiv(i)
        elif kind=="serial":
            for i in range(K): aic(i); aiv(i)
        else:
            for s in range(nseg):
                lo,hi=s*seg,min((s+1)*seg,K)
                ef[s].record(root)
                with torch.npu.stream(s2):
                    s2.wait_event(ef[s])
                    for i in range(lo,hi): aiv(i)
                    ej[s].record(s2)
                for i in range(lo,hi): aic(i)
                root.wait_event(ej[s])
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g

OUT="/tmp/shunt_sv"; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
exp=torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False,
    op_attr=False, data_simplification=True, aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
REP=6
for kind in ("aic","aiv","serial","two"):
    sub=os.path.join(OUT,kind)
    g=build(kind)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=exp,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(sub)) as prof:
        for _ in range(REP): g.replay()
        torch.npu.synchronize()
fs={}
for kind in ("aic","aiv","serial","two"):
    f=glob.glob(os.path.join(OUT,kind,"**","ASCEND_PROFILER_OUTPUT","kernel_details.csv"),recursive=True)
    if not f: print(kind,"no csv"); continue
    d=defaultdict(list); span=[]
    for r in csv.DictReader(open(f[0],newline="")):
        try: st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
        except: continue
        nm=(r.get("Name") or "")
        key="AIC" if "MatMul" in nm else ("AIV" if "Rms" in nm else "other")
        d[key].append(du); span.append((st,st+du))
    nrep = REP
    print("### %-8s  每步重放 kernel 数: AIC=%.1f AIV=%.1f other=%d"%(
        kind, len(d["AIC"])/nrep, len(d["AIV"])/nrep, len(d["other"])/nrep))
    for k in ("AIC","AIV"):
        if d[k]:
            print("      %s 中位时长 %.2f µs  (p25 %.2f / p75 %.2f)"%(
                k, statistics.median(d[k]), sorted(d[k])[len(d[k])//4], sorted(d[k])[3*len(d[k])//4]))
    if span:
        print("      窗口 %.3f ms / %d 次重放 = %.3f ms/次"%( (max(e for _,e in span)-min(s for s,_ in span))/1000, nrep,
              (max(e for _,e in span)-min(s for s,_ in span))/1000/nrep))
