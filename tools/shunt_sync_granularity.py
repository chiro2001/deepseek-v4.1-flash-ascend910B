#!/usr/bin/env python3
"""同步粒度扫描 + profiler 验证：两流自由漂移能拿多少重叠？
臂：端点同步(=用户设想) / 每 N 算子同步 / 逐算子同步 / 串行 / 单批
"""
import torch, torch_npu, time, os, json, glob, csv, shutil, statistics
from collections import Counter
torch.npu.set_device(0); dev="npu:0"
LAYERS=int(os.environ.get("LAYERS","16")); m=int(os.environ.get("MBATCH","24"))
TEMPLATE=["mix","rms","dq","mix","rms","dqs","mix","rms","mix","dqs","rms","mm","mm","dqs","rms",
          "mix","rms","mm","dqs","rms","dqs","dqs","dqs","dqs","mix","mix","mix","dqs","dqs","rms","rms","rms"]
OK={"mix":"mix","mm":"mm","rms":"rms","dq":"dqs","dqs":"dqs"}
SEQ=[OK[t] for t in TEMPLATE]*LAYERS
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
B1=mk(2*m); B2=mk(m); B3=mk(m)
for b in (B1,B2,B3):
    for k in set(SEQ): F[k](b)
torch.npu.synchronize()
def emit(b,seq):
    for k in seq: F[k](b)

def build(arm, sync_every=None):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream()
    def body():
        if arm=="one_batch": emit(B1,SEQ)
        elif arm=="serial":  emit(B2,SEQ); emit(B3,SEQ)
        elif arm=="endpoint":
            s2=torch.npu.Stream(); ef=torch.npu.Event(); ej=torch.npu.Event()
            ef.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(ef); emit(B3,SEQ); ej.record(s2)
            emit(B2,SEQ); root.wait_event(ej)
        else:  # chunked：每 sync_every 个算子同步一次
            s2=torch.npu.Stream(); n=len(SEQ)
            for lo in range(0,n,sync_every):
                hi=min(lo+sync_every,n)
                e1=torch.npu.Event(); e2=torch.npu.Event()
                e1.record(root)
                with torch.npu.stream(s2):
                    s2.wait_event(e1)
                    for k in SEQ[lo:hi]: F[k](B3)
                    e2.record(s2)
                for k in SEQ[lo:hi]: F[k](B2)
                root.wait_event(e2)
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
REP=int(os.environ.get("REP","25"))
def meas(g):
    for _ in range(3): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(REP): g.replay()
    torch.npu.synchronize(); return (time.perf_counter()-t0)/REP*1000

print("层=%d 每流 %d 算子 | 大 M=%d 小 M=%d×2"%(LAYERS,len(SEQ),2*m,m))
res={}
res["one_batch"]=meas(build("one_batch"))
print("\n%-22s %11s %9s"%("臂","makespan","vs 单批"))
print("%-22s %11.3f %8.3fx"%("A 单批 M=%d"%(2*m),res["one_batch"],1.0))
for arm,lab in (("serial","C 两批串行"),("endpoint","B0 端点同步(你的设想)")):
    res[arm]=meas(build(arm)); print("%-22s %11.3f %8.3fx"%(lab,res[arm],res["one_batch"]/res[arm]))
print()
for se in (32,64,128,256):
    dt=meas(build("chunk",se)); res["chunk%d"%se]=dt
    print("%-22s %11.3f %8.3fx"%("B%d 每 %d 算子同步"%(se,se),dt,res["one_batch"]/dt))
b0=res["endpoint"]; c=res["serial"]
print("\n关键量：")
print("  两流自由漂移 vs 串行 : %.3fx  ← 自由调度的净收益"%(c/b0))
print("  两流自由漂移 vs 单批 : %.3fx  ← 相对现状（分两批的代价）"%(res["one_batch"]/b0))
json.dump(res,open("/tmp/shunt_sync.json","w"),indent=1)

# profiler 验证重叠
OUT="/tmp/shunt_synv"; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
exc=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False,op_attr=False,data_simplification=True,aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
def span_iv(iv):
    if not iv: return 0.0
    iv=sorted(iv); out=[list(iv[0])]
    for s,e in iv[1:]:
        if s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return sum(e-s for s,e in out)
def inter(A,B):
    A=sorted(A); B=sorted(B); i=j=0; t=0.0
    while i<len(A) and j<len(B):
        s=max(A[i][0],B[j][0]); e=min(A[i][1],B[j][1])
        if e>s: t+=e-s
        if A[i][1]<B[j][1]: i+=1
        else: j+=1
    return t
print("\n=== profiler：两流是否真的并行 ===")
for arm in ("endpoint","chunk128"):
    g=build("endpoint" if arm=="endpoint" else "chunk", 128)
    sub=os.path.join(OUT,arm)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exc,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(sub)) as p:
        for _ in range(4): g.replay()
        torch.npu.synchronize()
    f=glob.glob(sub+"/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv",recursive=True)
    per=collections=Counter(); streams={}
    for r in csv.DictReader(open(f[0],newline="")):
        try: st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
        except: continue
        sid=(r.get("Stream ID") or "").strip()
        streams.setdefault(sid,[]).append((st,st+du))
    ss=sorted(streams.items(), key=lambda x:-span_iv(x[1]))[:3]
    if len(ss)>=2:
        (s1,i1),(s2,i2)=ss[0],ss[1]
        sp1,sp2,ov=span_iv(i1)/1000,span_iv(i2)/1000,inter(i1,i2)/1000
        print("  %-10s 流%s=%.3fms 流%s=%.3fms 交集=%.3fms 重叠率=%.1f%%"%(
            arm,s1,sp1,s2,sp2,ov,ov/min(sp1,sp2)*100))
