#!/usr/bin/env python3
"""Shunt 的真实场景判定：MIX 算子 ∥ AIV 算子，是赚还是亏？"""
import torch, torch_npu, time, os, json
torch.npu.set_device(0); dev="npu:0"
M=48; K=int(os.environ.get("K","64"))
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s): return torch.randn(*s,dtype=torch.float32,device=dev)
XM=[bf16(M,5120) for _ in range(K)]
XV=[bf16(M,5120) for _ in range(K)]
XD=[bf16(M,1280) for _ in range(K)]
G=bf16(5120)
LOG=f32(M,384); row=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6)
exp=torch.topk(LOG,6).indices.to(torch.int32)
op_mix=lambda i: torch_npu.npu_moe_init_routing(XM[i],row,exp,M)     # MIX_AIC, 48 blk
op_rms=lambda i: torch_npu.npu_rms_norm(XV[i],G)[0]                  # AI_VECTOR, 48 blk
op_dq =lambda i: torch_npu.npu_dynamic_quant(XD[i])[0]               # AI_VECTOR, blk 4~16
for f in (lambda:op_mix(0), lambda:op_rms(0), lambda:op_dq(0)): f()
torch.npu.synchronize()

def cap(kind, opB, seg=None):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    def body():
        if kind=="serial":
            for i in range(K): op_mix(i)
            for i in range(K): opB(i)
        elif kind=="concurrent":
            ef=torch.npu.Event(); ej=torch.npu.Event()
            if seg is None:
                ef.record(root)
                with torch.npu.stream(s2):
                    s2.wait_event(ef)
                    for i in range(K): opB(i)
                    ej.record(s2)
                for i in range(K): op_mix(i)
                root.wait_event(ej)
            else:
                nseg=(K+seg-1)//seg
                for s in range(nseg):
                    lo,hi=s*seg,min((s+1)*seg,K)
                    e1=torch.npu.Event(); e2=torch.npu.Event()
                    e1.record(root)
                    with torch.npu.stream(s2):
                        s2.wait_event(e1)
                        for i in range(lo,hi): opB(i)
                        e2.record(s2)
                    for i in range(lo,hi): op_mix(i)
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

print("K=%d（每臂总量 = K 个 MIX + K 个 AIV）"%K)
print("%-28s %11s %9s %11s"%("臂","makespan(ms)","每单元µs","vs 串行"))
res={}
for label,opB in (("AIV=rms48",op_rms),("AIV=dq_small",op_dq)):
    ts=meas(cap("serial",opB))
    tc=meas(cap("concurrent",opB))
    res[label]=(ts,tc)
    print("%-28s %11.3f %9.2f %10s"%(label+" [串行]",ts,ts/K*1000,"1.000x"))
    print("%-28s %11.3f %9.2f %10.3fx"%(label+" [并发]",tc,tc/K*1000,ts/tc))
print()
# 分段并发（更接近生产）
for seg in (8,16,32):
    ts=res["AIV=rms48"][0]
    tc=meas(cap("concurrent",op_rms,seg))
    print("rms48 分段并发 seg=%-3d %11.3f %9.2f %10.3fx"%(seg,tc,tc/K*1000,ts/tc))
json.dump({k:list(v) for k,v in res.items()},open("/tmp/shunt_mixaiv.json","w"),indent=1)
