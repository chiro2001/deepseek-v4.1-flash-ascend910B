#!/usr/bin/env python3
"""判定真正的限制资源：AIC 与 AIV 是不是在抢同一批 AI Core？

对照：
  mm23 ∥ rms48   —— 23-block cube  ∥ 48-block vector（占满全部 48 个 vector）
  mm23 ∥ dq4     —— 23-block cube  ∥  4-block vector（只占 4 个 vector）
  mm23 ∥ mm23    —— cube ∥ cube
  rms48 ∥ rms48  —— vector ∥ vector
若"核池争用"成立：dq4 的重叠率应显著高于 rms48。
"""
import torch, torch_npu, time, os, json
torch.npu.set_device(0); dev="npu:0"
M=48
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
K=int(os.environ.get("K","64"))
W=bf16(1024,5120); G=bf16(5120)
XA=[bf16(M,1024) for _ in range(K)]; XV=[bf16(M,5120) for _ in range(K)]; XD=[bf16(M,1280) for _ in range(K)]
op_mm =lambda i: torch.matmul(XA[i],W)
op_rms=lambda i: torch_npu.npu_rms_norm(XV[i],G)[0]
op_dq =lambda i: torch_npu.npu_dynamic_quant(XD[i])[0]
for f,i in ((op_mm,0),(op_rms,0),(op_dq,0)): f(i)
torch.npu.synchronize()

def cap(opA,opB,seg=16):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    nseg=(K+seg-1)//seg
    ef=[torch.npu.Event() for _ in range(nseg)]; ej=[torch.npu.Event() for _ in range(nseg)]
    def body():
        for s in range(nseg):
            lo,hi=s*seg,min((s+1)*seg,K)
            ef[s].record(root)
            with torch.npu.stream(s2):
                s2.wait_event(ef[s])
                for i in range(lo,hi): opB(i)
                ej[s].record(s2)
            for i in range(lo,hi): opA(i)
            root.wait_event(ej[s])
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
def cap_serial(opA,opB):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream()
    def body():
        for i in range(K): opA(i); opB(i)
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
def cap_solo(op):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream()
    def body():
        for i in range(K): op(i)
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
solo={"mm":meas(cap_solo(op_mm)),"rms":meas(cap_solo(op_rms)),"dq":meas(cap_solo(op_dq))}
print("单算子链（K=%d）: mm=%.3f ms  rms=%.3f  dq=%.3f"%(K,solo["mm"],solo["rms"],solo["dq"]))
print("\n%-16s %10s %10s %10s %9s %9s"%("组合","串行(ms)","并发(ms)","理想(ms)","并发/理想","vs串行"))
res={}
for name,a,b in [("mm ∥ rms48",op_mm,op_rms),("mm ∥ dq4",op_mm,op_dq),
                 ("mm ∥ mm",op_mm,op_mm),("rms48 ∥ rms48",op_rms,op_rms)]:
    try:
        ts=meas(cap_serial(a,b)); tc=meas(cap(a,b))
    except Exception as e:
        print("%-16s FAIL %s"%(name,str(e).replace("\n"," ")[:60])); continue
    ka={"mm":"mm","rms":"rms","dq":"dq"}
    ideal=max(solo["mm"] if a is op_mm else (solo["rms"] if a is op_rms else solo["dq"]),
              solo["mm"] if b is op_mm else (solo["rms"] if b is op_rms else solo["dq"]))
    res[name]=(ts,tc,ideal)
    print("%-16s %10.3f %10.3f %10.3f %8.2fx %8.2fx"%(name,ts,tc,ideal,tc/ideal,ts/tc))
json.dump(res,open("/tmp/shunt_coreshare.json","w"),indent=1,default=str)
