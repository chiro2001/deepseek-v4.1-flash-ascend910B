#!/usr/bin/env python3
"""正确基线对照：
  ① seq_then_seq —— AIC 全部跑完再跑 AIV（真·零重叠基线）
  ② interleaved  —— 交替顺序（= 现状）
  ③ free         —— 两流，仅首尾同步
  ④ bar32        —— 两流，阶段屏障（用户方案）
判据：③④ 相对 ① 的加速 = 真重叠收益；相对 ② 的加速 = 相对现状的收益。
"""
import torch, torch_npu, time, os, json
torch.npu.set_device(0); dev="npu:0"
M=48; LAY=int(os.environ.get("LAYERS","16"))
def bf(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s): return torch.randn(*s,dtype=torch.float32,device=dev)
AIC_SEQ=["mix","mm","mix","mm","mix","mm","mix","mm"]*LAY
AIV_SEQ=["rms","dqs","rms","dqs","rms"]*LAY
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
def first_solo(kind,n=100):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream()
    def body():
        for _ in range(n): F[kind]()
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    for _ in range(3): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(8): g.replay()
    torch.npu.synchronize(); return (time.perf_counter()-t0)/8/n*1e6
DUR={k:first_solo(k) for k in F}
def cap(arm,k=32):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    def body():
        if arm=="seq_then_seq":
            for x in AIC_SEQ: F[x]()
            for x in AIV_SEQ: F[x]()
        elif arm=="interleaved":
            for i in range(max(len(AIC_SEQ),len(AIV_SEQ))):
                if i<len(AIC_SEQ): F[AIC_SEQ[i]]()
                if i<len(AIV_SEQ): F[AIV_SEQ[i]]()
        elif arm=="free":
            e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e1)
                for x in AIV_SEQ: F[x]()
                e2.record(s2)
            for x in AIC_SEQ: F[x]()
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
                    for x in av: F[x]()
                    e2.record(s2)
                for x in ai: F[x]()
                root.wait_event(e2)
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
REP=30
def meas(g):
    for _ in range(3): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(REP): g.replay()
    torch.npu.synchronize(); return (time.perf_counter()-t0)/REP*1000
r={}
for arm in ("seq_then_seq","interleaved","free","bar32"):
    r[arm]=meas(cap("bar" if arm=="bar32" else arm,32))
s0,s1=r["seq_then_seq"],r["interleaved"]
print("算子时长 %s ; 理论串行和 = %.3f ms"%({k:round(v,2) for k,v in DUR.items()},
      (sum(DUR[x] for x in AIC_SEQ)+sum(DUR[x] for x in AIV_SEQ))/1000))
print("\n%-16s %11s %13s %13s"%("臂","makespan","vs 零重叠基线","vs 现状(交替)"))
for arm,lab in (("seq_then_seq","① 零重叠基线"),("interleaved","② 交替=现状"),("free","③ 两流自由"),("bar32","④ 两流+阶段屏障")):
    print("%-16s %11.3f %12.3fx %12.3fx"%(lab,r[arm],r["seq_then_seq"]/r[arm],r["interleaved"]/r[arm]))
print("\n关键：③④ 相对 ① 的加速 = 真重叠收益；相对 ② = 相对现状的收益")
json.dump(r,open("/tmp/shunt_cb.json","w"),indent=1)
