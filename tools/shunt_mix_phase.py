#!/usr/bin/env python3
"""① 量 MIX 代理的内部 AIC/AIV 相位占比
   ② 决定性实验：把 MIX 拆成 cube-part + vector-part 并做跨实例流水，能否胜过融合版？"""
import torch, torch_npu, time, os, glob, csv, shutil, statistics, json
torch.npu.set_device(0); dev="npu:0"
M=48; N=int(os.environ.get("N","64"))
def bf(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s): return torch.randn(*s,dtype=torch.float32,device=dev)
MX=[bf(M,5120) for _ in range(4)]
row=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6)
exp=torch.topk(f32(M,384),6).indices.to(torch.int32)
CA=bf(M,5120); CW=bf(5120,1024)     # cube 代理 (AI_CORE 24blk 15.3µs)
RX=bf(M,5120); RG=bf(5120)          # vector 代理 (AIV 48blk 8µs)
f_mix=lambda i: torch_npu.npu_moe_init_routing(MX[i%4],row,exp,M)
f_cube=lambda: torch.matmul(CA,CW)
f_vec=lambda: torch_npu.npu_rms_norm(RX,RG)[0]
for f in (lambda:f_mix(0),f_cube,f_vec): f()
torch.npu.synchronize()
OUT="/tmp/shunt_mp"; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
exc=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False,op_attr=False,data_simplification=True,aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
def prof(g,sub,rep=6):
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exc,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(sub)) as p:
        for _ in range(rep): g.replay()
        torch.npu.synchronize()
    return glob.glob(sub+"/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv",recursive=True)[0]
def cap(fn,n):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream()
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): fn(root,n)
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): fn(root,n)
    return g
def body_solo(root,n):
    for _ in range(n): f_mix(0)
g=cap(body_solo,N)
f=prof(g,os.path.join(OUT,"solo"))
ds=[]; ais=[]; avs=[]
for r in csv.DictReader(open(f,newline="")):
    try: ds.append(float(r["Duration(us)"]))
    except: continue
    ais.append(float(r.get("aicore_time(us)") or 0)); avs.append(float(r.get("aiv_time(us)") or 0))
Dm=statistics.median(ds); Am=statistics.median(ais); Vm=statistics.median(avs)
print("MIX 代理 moe_init_routing: dur=%.2f  aicore=%.2f (%.0f%%)  aiv=%.2f (%.0f%%)  ⇒ 内部重叠≈%.0f%%"%(
    Dm,Am,Am/Dm*100,Vm,Vm/Dm*100,(Am+Vm-Dm)/Dm*100))
# cube / vec solo
for nm,fn in (("cube",lambda r,n:[f_cube() for _ in range(n)]),("vec",lambda r,n:[f_vec() for _ in range(n)])):
    g=cap(fn,N); f=prof(g,os.path.join(OUT,nm))
    ds=[float(r["Duration(us)"]) for r in csv.DictReader(open(f,newline="")) if r.get("Duration(us)")]
    print("  %s 代理: dur=%.2f µs"%(nm,statistics.median(ds)))
REP=25
def meas(g):
    for _ in range(3): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(REP): g.replay()
    torch.npu.synchronize(); return (time.perf_counter()-t0)/REP*1000
# 臂
def body_fused(root,n):
    for _ in range(n): f_mix(0)
def body_split_serial(root,n):          # 拆开但同流串行（依赖强制）
    for _ in range(n): f_cube(); f_vec()
def body_split_pipe(root,n):            # 拆开 + 跨实例流水：cube_{i+1} 与 vec_i 重叠
    s2=torch.npu.Stream()
    ev=[torch.npu.Event() for _ in range(8)]
    for i in range(n):
        f_cube(); ev[i%8].record(root)
        with torch.npu.stream(s2):
            s2.wait_event(ev[i%8]); f_vec()
            ev[(i+4)%8].record(s2)
        root.wait_event(ev[(i+4)%8])    # 保证 vec_i 完成后再 cube_{i+1}（保守）
res={}
for lab,fn in (("① 融合 MIX（现状）",body_fused),
               ("② 拆开·同流串行",body_split_serial),
               ("③ 拆开·跨实例流水",body_split_pipe)):
    try: g=cap(fn,N)
    except Exception as e:
        print("%-22s FAIL %s"%(lab,str(e).replace("\n"," ")[:60])); continue
    res[lab]=meas(g)
b=res.get("① 融合 MIX（现状）")
print("\n%-22s %12s %10s"%("臂","makespan(ms)","vs 融合"))
for k,v in res.items(): print("%-22s %12.3f %9.3fx"%(k,v,b/v))
json.dump(res,open("/tmp/shunt_mp.json","w"),indent=1)
