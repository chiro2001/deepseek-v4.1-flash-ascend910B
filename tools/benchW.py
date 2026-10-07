#!/usr/bin/env python3
"""图重放壁钟对比 + 检查 MIXV_route 是否真的入图。"""
import torch, torch_npu, time
torch.npu.set_device(0); dev="npu:0"
M=48
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s):  return torch.randn(*s,dtype=torch.float32,device=dev)
L1=bf16(M,1024); L2=bf16(1024,5120); XR=bf16(M,5120); GR=bf16(5120); XD=bf16(M,5120)
LOG=f32(M,384); row_idx=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6)
expert_idx=torch.topk(LOG,6).indices.to(torch.int32)
ops={"AIC_mm":lambda: torch.matmul(L1,L2),
     "AIV_rms":lambda: torch_npu.npu_rms_norm(XR,GR)[0],
     "AIV_dq":lambda: torch_npu.npu_dynamic_quant(XD)[0],
     "MIXV_route":lambda: torch_npu.npu_moe_init_routing(XR,row_idx,expert_idx,M)}
for f in ops.values(): f()
torch.npu.synchronize()
N=24
def b_serial(fa,fb):
    for _ in range(N): fa()
    for _ in range(N): fb()
def b_conc(fa,fb,root,s2,ef,ej):
    ef.record(root)
    with torch.npu.stream(s2):
        s2.wait_event(ef)
        for _ in range(N): fb()
        ej.record(s2)
    for _ in range(N): fa()
    root.wait_event(ej)
def cap(kind,fa,fb):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    ef=torch.npu.Event(); ej=torch.npu.Event()
    ws=torch.npu.Stream()
    with torch.npu.stream(ws):
        (b_serial if kind=="serial" else b_conc)(fa,fb) if kind=="serial" else b_conc(fa,fb,root,s2,ef,ej)
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root):
        if kind=="serial": b_serial(fa,fb)
        else: b_conc(fa,fb,root,s2,ef,ej)
    return g
pairs=[("AIC_mm","AIV_rms"),("AIC_mm","AIV_dq"),("AIV_rms","AIV_dq"),
       ("AIC_mm","MIXV_route"),("AIV_rms","MIXV_route")]
REP=50
print("%-26s %12s %12s %9s %9s"%("pair","串行图(ms)","并发图(ms)","比值","每流op数"))
for a,b in pairs:
    fa,fb=ops[a],ops[b]
    try:
        gs=cap("serial",fa,fb); gc=cap("conc",fa,fb)
    except Exception as e:
        print("%-26s [capture FAIL] %s"%(f"{a}|{b}",str(e).replace(chr(10),' ')[:80])); continue
    def t(g):
        for _ in range(5): g.replay()
        torch.npu.synchronize(); t0=time.perf_counter()
        for _ in range(REP): g.replay()
        torch.npu.synchronize(); return (time.perf_counter()-t0)/REP*1000
    ts=t(gs); tc=t(gc)
    print("%-26s %12.3f %12.3f %8.2fx %6d/%d"%(f"{a}|{b}",ts,tc,ts/tc,N,N))
