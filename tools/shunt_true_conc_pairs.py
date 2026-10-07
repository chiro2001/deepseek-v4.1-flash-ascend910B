#!/usr/bin/env python3
"""用"真并发"口径复核各算子对（替换之前的"区间并集相交"口径）。"""
import torch, torch_npu, os, glob, csv, shutil, json
torch.npu.set_device(0); dev="npu:0"
m=24; K=int(os.environ.get("K","64"))
def bf(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s): return torch.randn(*s,dtype=torch.float32,device=dev)
A=[bf(m,1024) for _ in range(K)]; W=bf(1024,5120)
R=[bf(m,5120) for _ in range(K)]; G=bf(5120)
D=[bf(m,1280) for _ in range(K)]
M=[bf(m,5120) for _ in range(K)]
row=torch.arange(m*6,dtype=torch.int32,device=dev).view(m,6)
exp=torch.topk(f32(m,384),6).indices.to(torch.int32)
OPS={"mm":lambda i:torch.matmul(A[i],W),
     "rms":lambda i:torch_npu.npu_rms_norm(R[i],G)[0],
     "dqs":lambda i:torch_npu.npu_dynamic_quant(D[i])[0],
     "mix":lambda i:torch_npu.npu_moe_init_routing(M[i],row,exp,m)}
for f in OPS.values(): f(0)
torch.npu.synchronize()
def cap(opA,opB):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    e1=torch.npu.Event(); e2=torch.npu.Event()
    def body():
        e1.record(root)
        with torch.npu.stream(s2):
            s2.wait_event(e1)
            for i in range(K): opB(i)
            e2.record(s2)
        for i in range(K): opA(i)
        root.wait_event(e2)
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
def cap_ser(opA,opB):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream()
    def body():
        for i in range(K): opA(i)
        for i in range(K): opB(i)
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
OUT="/tmp/shunt_tcp2"; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
exc=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False,op_attr=False,data_simplification=True,aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
def analyze(g,sub):
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exc,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(sub)) as p:
        for _ in range(4): g.replay()
        torch.npu.synchronize()
    f=glob.glob(sub+"/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv",recursive=True)[0]
    iv=[]
    for r in csv.DictReader(open(f,newline="")):
        try: st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
        except: continue
        iv.append((st,st+du))
    ev=[]
    for s,e in iv: ev.append((s,1)); ev.append((e,-1))
    ev.sort(key=lambda x:(x[0],-x[1]))
    cur=0; last=ev[0][0]; dual=0.0; busy=0.0
    for t,d in ev:
        dt=t-last
        if cur>=1: busy+=dt
        if cur>=2: dual+=dt
        cur+=d; last=t
    span=iv[-1][1]-iv[0][0]; sd=sum(e-s for s,e in iv); n=len(iv)/4
    return dict(span=span/4,busy=busy/4,dual=dual/4,sumdur=sd/4,n=n,dual_pct=dual/span*100,median=sorted(e-s for s,e in iv)[len(iv)//2])
print("=== 真并发口径复核（K=%d 每流, m=%d）==="%(K,m))
print("%-22s %9s %9s %9s %9s %8s %9s"%("臂","span ms","Σdur ms","忙%","真并发%","中位µs","并发/Σ"))
for name,a,b in [("mm_vs_rms48",OPS["mm"],OPS["rms"]),("mm_vs_dqs",OPS["mm"],OPS["dqs"]),
                 ("mm_vs_mix",OPS["mm"],OPS["mix"]),("mix_vs_mix",OPS["mix"],OPS["mix"])]:
    r1=analyze(cap_ser(a,b),os.path.join(OUT,"ser_"+name.replace(" ","")))
    r2=analyze(cap(a,b),os.path.join(OUT,"conc_"+name.replace(" ","")))
    print("%-22s %9.3f %9.3f %8.0f%% %8.1f%% %8.2f %8.3fx  [串行 %.3f ms]"%(
        name,r2["span"],r2["sumdur"],r2["busy"]/r2["span"]*100,r2["dual_pct"],r2["median"],
        r2["sumdur"]/r2["span"],r1["span"]))
