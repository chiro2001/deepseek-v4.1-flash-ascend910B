#!/usr/bin/env python3
"""核验 matched-barrier 方案：真并发率 + 每条流的实际速度 + 争用税。"""
import torch, torch_npu, os, glob, csv, shutil, json
from collections import Counter
torch.npu.set_device(0); dev="npu:0"
M=48; LAY=int(os.environ.get("LAYERS","16"))
def bf(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s): return torch.randn(*s,dtype=torch.float32,device=dev)
AIC_KIND=["mix","mm","mix","mm","mix","mm","mix","mm"]; AIV_KIND=["rms","dqs","rms","dqs","rms"]
AIC_SEQ=AIC_KIND*LAY; AIV_SEQ=AIV_KIND*LAY
B=dict(M=M,mm_a=bf(M,1024),mm_w=bf(1024,5120),rms_x=bf(M,5120),rms_g=bf(5120),
       dq_x=bf(M,1280),mix_x=bf(M,5120),
       row=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6),
       exp=torch.topk(f32(M,384),6).indices.to(torch.int32))
F={"mix":lambda: torch_npu.npu_moe_init_routing(B["mix_x"],B["row"],B["exp"],B["M"]),
   "mm": lambda: torch.matmul(B["mm_a"],B["mm_w"]),
   "rms":lambda: torch_npu.npu_rms_norm(B["rms_x"],B["rms_g"])[0],
   "dqs":lambda: torch_npu.npu_dynamic_quant(B["dq_x"])[0]}
for f in F.values(): f()
torch.npu.synchronize()
def solo_ms(kind,n=200):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream()
    def body():
        for _ in range(n): F[kind]()
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    for _ in range(3): g.replay()
    torch.npu.synchronize(); t0=__import__("time").perf_counter()
    for _ in range(10): g.replay()
    torch.npu.synchronize(); return (__import__("time").perf_counter()-t0)/10/n*1e6
DUR={k:solo_ms(k) for k in F}
AIC_SOLO=sum(DUR[k] for k in AIC_SEQ)/1000; AIV_SOLO=sum(DUR[k] for k in AIV_SEQ)/1000
print("单算子 %s"%{k:round(v,2) for k,v in DUR.items()})
print("纯 AIC 串行=%.3f ms | 纯 AIV 串行=%.3f ms | 若完美隐藏 ⇒ 上限=max=%.3f ms"%(AIC_SOLO,AIV_SOLO,max(AIC_SOLO,AIV_SOLO)))

def cap(arm,k=32):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    def body():
        if arm=="interleaved":
            for i in range(max(len(AIC_SEQ),len(AIV_SEQ))):
                if i<len(AIC_SEQ): F[AIC_SEQ[i]]()
                if i<len(AIV_SEQ): F[AIV_SEQ[i]]()
        elif arm=="free":
            e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e1)
                for kk in AIV_SEQ: F[kk]()
                e2.record(s2)
            for kk in AIC_SEQ: F[kk]()
            root.wait_event(e2)
        else:
            i=j=0
            while i<len(AIC_SEQ) or j<len(AIV_SEQ):
                cu=0.0; ai=[]
                while i<len(AIC_SEQ) and len(ai)<k:
                    ai.append(AIC_SEQ[i]); cu+=DUR[AIC_SEQ[i]]; i+=1
                cv=0.0; av=[]
                while j<len(AIV_SEQ) and cv<cu:
                    av.append(AIV_SEQ[j]); cv+=DUR[AIV_SEQ[j]]; j+=1
                e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
                with torch.npu.stream(s2):
                    s2.wait_event(e1)
                    for kk in av: F[kk]()
                    e2.record(s2)
                for kk in ai: F[kk]()
                root.wait_event(e2)
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
OUT="/tmp/shunt_pv"; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
exc=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False,op_attr=False,data_simplification=True,aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
print("\n%-14s %9s %9s %9s %9s %9s %9s %9s"%("臂","span","Σdur","真并发%","峰值","mix中位","mm中位","rms中位"))
for arm in ("interleaved","free","bar32"):
    g=cap("interleaved" if arm=="interleaved" else ("free" if arm=="free" else "bar"),32)
    sub=os.path.join(OUT,arm)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exc,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(sub)) as p:
        for _ in range(4): g.replay()
        torch.npu.synchronize()
    f=glob.glob(sub+"/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv",recursive=True)[0]
    iv=[]; per=Counter(); med={}
    for r in csv.DictReader(open(f,newline="")):
        try: st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
        except: continue
        iv.append((st,st+du)); nm=(r.get("Name") or "")
        key="mix" if "InitRouting" in nm else ("mm" if "MatMul" in nm else ("rms" if "Rms" in nm else "dqs"))
        per.setdefault(key,[]).append(du)
    ev=[]
    for s,e in iv: ev.append((s,1)); ev.append((e,-1))
    ev.sort(key=lambda x:(x[0],-x[1]))
    cur=0; last=ev[0][0]; dual=0.0; peak=0
    for t,d in ev:
        if cur>=2: dual+=t-last
        cur+=d; peak=max(peak,cur); last=t
    span=(iv[-1][1]-iv[0][0])/4/1000; sd=sum(e-s for s,e in iv)/4/1000
    for kk in per: med[kk]=sorted(per[kk])[len(per[kk])//2]
    print("%-14s %9.3f %9.3f %8.1f%% %9d %9.2f %9.2f %9.2f"%(
        arm,span,sd,dual/(iv[-1][1]-iv[0][0])*100,peak,
        med.get("mix",0),med.get("mm",0),med.get("rms",0)))
print("\n* 'mix中位'等为并发时该算子的设备时长；与 solo 对比即争用税")
json.dump({"solo":DUR,"aic_solo":AIC_SOLO,"aiv_solo":AIV_SOLO},open("/tmp/shunt_pv.json","w"),indent=1)
