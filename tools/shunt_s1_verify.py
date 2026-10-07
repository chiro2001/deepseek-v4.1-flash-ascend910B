#!/usr/bin/env python3
"""核验：每个臂的图里到底有多少 kernel、在哪些流上（防止"图里其实是空的"）。"""
import torch, torch_npu, os, glob, csv, shutil
from collections import Counter
torch.npu.set_device(0); dev="npu:0"
M=48
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
W=bf16(1024,5120); G=bf16(5120)
K=int(os.environ.get("K","64"))
XA=[bf16(M,1024) for _ in range(K)]; XV=[bf16(M,5120) for _ in range(K)]
aic=lambda i: torch.matmul(XA[i],W)
aiv=lambda i: torch_npu.npu_rms_norm(XV[i],G)[0]

def cap(kind,K):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    ea=[torch.npu.Event() for _ in range(K)]; ev=[torch.npu.Event() for _ in range(K)]
    def body():
        for i in range(K):
            aic(i); ea[i].record(root)
            with torch.npu.stream(s2):
                if kind=="wait1way": s2.wait_event(ea[i])
                aiv(i); ev[i].record(s2)
            if kind=="join_full" and i==K-1: root.wait_event(ev[i])
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g

OUT="/tmp/shunt_verify"; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
exp=torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False, op_attr=False, data_simplification=True,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
for kind in ("wait1way","join_full"):
    sub=os.path.join(OUT,kind)
    g=cap(kind,K)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exp,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(sub)) as prof:
        for _ in range(4): g.replay()
        torch.npu.synchronize()
    fs=glob.glob(sub+"/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv",recursive=True)
    if not fs: print(kind,"no csv"); continue
    c=Counter(); streams=Counter()
    tot=0.0
    for r in csv.DictReader(open(fs[0],newline="")):
        try: du=float(r["Duration(us)"])
        except: continue
        nm=(r.get("Name") or "")[:34]
        c[(nm,(r.get("Accelerator Core") or "").strip())]+=1; tot+=du
        streams[(r.get("Stream ID") or "").strip()]+=1
    print("### %s  kernels=%d  总时长=%.3f ms  流分布=%s"%(kind,sum(c.values()),tot/1000,dict(streams.most_common(6))))
    for k,v in c.most_common(6): print("     %-38s %-16s n=%d"%(k[0],k[1],v))
