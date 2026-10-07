#!/usr/bin/env python3
"""真并发度量：区分"区间并集相交"（假象）与"同一时刻多个 kernel 在跑"（真并行）。

对每个时刻，统计同时活跃的 kernel 数 c(t)：
  真并发时间 = ∫ 1[c(t) >= 2] dt
并与 Σ时长 / makespan（设备利用率）交叉验证。
"""
import torch, torch_npu, os, glob, csv, shutil, json
from collections import Counter
torch.npu.set_device(0); dev="npu:0"
m=int(os.environ.get("MBATCH","24"))
TEMPLATE=["mix","rms","dq","mix","rms","dqs","mix","rms","mix","dqs","rms","mm","mm","dqs","rms",
          "mix","rms","mm","dqs","rms","dqs","dqs","dqs","dqs","mix","mix","mix","dqs","dqs","rms","rms","rms"]
OK={"mix":"mix","mm":"mm","rms":"rms","dq":"dqs","dqs":"dqs"}
def mk(M):
    bf=lambda *s: torch.randn(*s,dtype=torch.bfloat16,device=dev)
    f32=lambda *s: torch.randn(*s,dtype=torch.float32,device=dev)
    return dict(M=M,mm_a=bf(M,1024),mm_w=bf(1024,5120),rms_x=bf(M,5120),rms_g=bf(5120),
                dq_x=bf(M,1280),mix_x=bf(M,5120),
                row=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6),
                exp=torch.topk(f32(M,384),6).indices.to(torch.int32))
F={"mix":lambda b: torch_npu.npu_moe_init_routing(b["mix_x"],b["row"],b["exp"],b["M"]),
   "mm":lambda b: torch.matmul(b["mm_a"],b["mm_w"]),
   "rms":lambda b: torch_npu.npu_rms_norm(b["rms_x"],b["rms_g"])[0],
   "dqs":lambda b: torch_npu.npu_dynamic_quant(b["dq_x"])[0]}
L=int(os.environ.get("LAYERS","16"))
seq=[OK[t] for t in TEMPLATE]*L
B1,B2,B3=mk(2*m),mk(m),mk(m)
for b in (B1,B2,B3):
    for k in set(seq): F[k](b)
torch.npu.synchronize()
def emit(b,s):
    for k in s: F[k](b)
def cap(arm):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream()
    def body():
        if arm=="one": emit(B1,seq)
        elif arm=="serial": emit(B2,seq); emit(B3,seq)
        else:
            s2=torch.npu.Stream(); e1=torch.npu.Event(); e2=torch.npu.Event()
            e1.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e1); emit(B3,seq); e2.record(s2)
            emit(B2,seq); root.wait_event(e2)
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g

OUT="/tmp/shunt_tc"; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
exc=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False,op_attr=False,data_simplification=True,aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
def analyze(arm):
    g=cap(arm); sub=os.path.join(OUT,arm)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exc,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(sub)) as p:
        for _ in range(5): g.replay()
        torch.npu.synchronize()
    f=glob.glob(sub+"/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv",recursive=True)[0]
    iv=[]; per_kernel=Counter()
    for r in csv.DictReader(open(f,newline="")):
        try: st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
        except: continue
        iv.append((st,st+du)); per_kernel[round(du,1)]+=1
    iv.sort()
    # 事件扫描求 c(t)
    ev=[]
    for s,e in iv: ev.append((s,1)); ev.append((e,-1))
    ev.sort(key=lambda x:(x[0],-x[1]))
    busy=0.0; dual=0.0; quad=0.0; cur=0; last=ev[0][0]; peak=0
    for t,d in ev:
        dt=t-last
        if cur>=1: busy+=dt
        if cur>=2: dual+=dt
        if cur>=4: quad+=dt
        cur+=d; peak=max(peak,cur); last=t
    span=iv[-1][1]-iv[0][0]
    sumdur=sum(e-s for s,e in iv)
    nrep=5
    print("%-8s span=%7.3fms  Σ时长=%7.3fms  忙=%7.3fms(%.0f%%)  **真并发(≥2)=%7.3fms(%.1f%%)**  ≥4=%6.3fms  峰值并发=%d"%(
        arm, span/1000/nrep, sumdur/1000/nrep, busy/1000/nrep, busy/span*100,
        dual/1000/nrep, dual/span*100, quad/1000/nrep, peak))
    print("         算子数=%d  中位时长%.2fus"%(len(iv)/nrep, sorted(x[1]-x[0] for x in iv)[len(iv)//2]))
    return dict(span=span/nrep,busy=busy/nrep,dual=dual/nrep,sumdur=sumdur/nrep,n=len(iv)/nrep)
r={}
print("=== 真并发度量（L=%d 层, 每流 %d 算子, M=%d/%d）==="%(L,len(seq),2*m,m))
for arm in ("one","serial","two"): r[arm]=analyze(arm)
print("\n读法：'真并发'= 同一时刻 ≥2 个 kernel 在执行（这才是真并行）。")
print("      若两个流的'区间并集相交'很大但'真并发'≈0，说明是交替执行，不是并行。")
json.dump(r,open("/tmp/shunt_tc.json","w"),indent=1)
