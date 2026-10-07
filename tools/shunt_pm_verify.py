#!/usr/bin/env python3
"""核验生产比例下策略 A：真并发率与算子膨胀。"""
import torch, torch_npu, os, glob, csv, shutil
torch.npu.set_device(0); dev="npu:0"
M=48; LAY=int(os.environ.get("LAYERS","16"))
def bf(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s): return torch.randn(*s,dtype=torch.float32,device=dev)
MX=bf(M,5120); row=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6)
exp=torch.topk(f32(M,384),6).indices.to(torch.int32)
RX=bf(M,5120); RG=bf(5120); DX=bf(M,1280); CA=bf(M,5120); CW=bf(5120,1024)
f_mix=lambda: torch_npu.npu_moe_init_routing(MX,row,exp,M)
f_rms=lambda: torch_npu.npu_rms_norm(RX,RG)[0]
f_dq =lambda: torch_npu.npu_dynamic_quant(DX)[0]
f_mm =lambda: torch.matmul(CA,CW)
for f in (f_mix,f_rms,f_dq,f_mm): f()
torch.npu.synchronize()
def e_mix():
    for _ in range(11): f_mix()
def e_other():
    for _ in range(4): f_rms()
    for _ in range(13): f_dq()
    for _ in range(2): f_mm()
def cap(arm):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    def body():
        if arm=="serial":
            for _ in range(LAY): e_mix(); e_other()
        else:
            e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e1)
                for _ in range(LAY): e_other()
                e2.record(s2)
            for _ in range(LAY): e_mix()
            root.wait_event(e2)
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
OUT="/tmp/shunt_pmv"; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
exc=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False,op_attr=False,data_simplification=True,aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
print("%-8s %9s %9s %9s %7s %10s %9s %9s"%("臂","span","Σdur","真并发%","峰值","mix中位","rms中位","mm中位"))
for arm in ("serial","A"):
    g=cap("serial" if arm=="serial" else "A"); sub=os.path.join(OUT,arm)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exc,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(sub)) as p:
        for _ in range(4): g.replay()
        torch.npu.synchronize()
    f=glob.glob(sub+"/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv",recursive=True)[0]
    iv=[]; per={}
    for r in csv.DictReader(open(f,newline="")):
        try: st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
        except: continue
        iv.append((st,st+du)); nm=(r.get("Name") or "")
        k="mix" if "InitRouting" in nm else ("mm" if "MatMul" in nm else ("rms" if "Rms" in nm else "dq"))
        per.setdefault(k,[]).append(du)
    ev=[]
    for s,e in iv: ev.append((s,1)); ev.append((e,-1))
    ev.sort(key=lambda x:(x[0],-x[1]))
    cur=0; last=ev[0][0]; dual=0.0; peak=0
    for t,d in ev:
        if cur>=2: dual+=t-last
        cur+=d; peak=max(peak,cur); last=t
    span=(iv[-1][1]-iv[0][0])/4/1000; sd=sum(e-s for s,e in iv)/4/1000
    med={k:sorted(v)[len(v)//2] for k,v in per.items()}
    print("%-8s %9.3f %9.3f %8.1f%% %7d %10.2f %9.2f %9.2f"%(arm,span,sd,dual/(iv[-1][1]-iv[0][0])*100,peak,
        med.get("mix",0),med.get("rms",0),med.get("mm",0)))
