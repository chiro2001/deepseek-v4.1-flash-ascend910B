#!/usr/bin/env python3
"""核验：真实块结构下 ①交替 / ④逐块屏障 的真并发与算子膨胀。"""
import torch, torch_npu, os, glob, csv, shutil, statistics
torch.npu.set_device(0); dev="npu:0"
M=48; NBLK=int(os.environ.get("NBLK","190"))
def bf(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
A=bf(M,5120); W=bf(5120,1024); X=bf(M,5120); G=bf(5120); D=bf(M,1280)
f_aic=lambda: torch.matmul(A,W)
f_rms=lambda: torch_npu.npu_rms_norm(X,G)[0]
f_dq =lambda: torch_npu.npu_dynamic_quant(D)[0]
for f in (f_aic,f_rms,f_dq): f()
torch.npu.synchronize()
def aiv2(): f_rms(); f_dq()
def cap(arm):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    def body():
        if arm=="interleaved":
            for _ in range(NBLK): f_aic(); aiv2()
        elif arm=="bar":
            for _ in range(NBLK):
                e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
                with torch.npu.stream(s2):
                    s2.wait_event(e1); aiv2(); e2.record(s2)
                f_aic(); root.wait_event(e2)
        elif arm=="free":
            e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e1)
                for _ in range(NBLK): aiv2()
                e2.record(s2)
            for _ in range(NBLK): f_aic()
            root.wait_event(e2)
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
OUT="/tmp/shunt_rv"; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
exc=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False,op_attr=False,data_simplification=True,aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
print("%-12s %9s %9s %9s %7s %9s %9s %9s"%("臂","span","Σdur","真并发%","峰值","matmul中位","rms中位","dq中位"))
for arm in ("interleaved","free","bar"):
    g=cap(arm); sub=os.path.join(OUT,arm)
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
        k="mm" if "MatMul" in nm else ("rms" if "Rms" in nm else "dq")
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
    print("%-12s %9.3f %9.3f %8.1f%% %7d %10.2f %9.2f %9.2f"%(arm,span,sd,dual/(iv[-1][1]-iv[0][0])*100,peak,
        med.get("mm",0),med.get("rms",0),med.get("dq",0)))
print("\n（solo 参考：matmul≈15.3µs  rms≈8.0µs  dq≈3.6µs）")
