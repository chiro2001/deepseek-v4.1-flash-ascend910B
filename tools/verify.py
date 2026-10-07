#!/usr/bin/env python3
"""核验：并发图重放的输出是否与 eager 逐位一致（防止算子被静默丢弃）。"""
import torch, torch_npu, time
torch.npu.set_device(0); dev="npu:0"
M=48
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s):  return torch.randn(*s,dtype=torch.float32,device=dev)
L1=bf16(M,1024); L2=bf16(1024,5120); XR=bf16(M,5120); GR=bf16(5120); XD=bf16(M,5120)
LOG=f32(M,384); row_idx=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6)
expert_idx=torch.topk(LOG,6).indices.to(torch.int32)
OUT={}
def op(name):
    if name=="AIC_mm":   return torch.matmul(L1,L2)
    if name=="AIV_rms":  return torch_npu.npu_rms_norm(XR,GR)[0]
    if name=="AIV_dq":   return torch_npu.npu_dynamic_quant(XD)[0]
    if name=="MIXV_route":
        r=torch_npu.npu_moe_init_routing(XR,row_idx,expert_idx,M); return r[0]
for n in ("AIC_mm","AIV_rms","AIV_dq","MIXV_route"): op(n)
torch.npu.synchronize()
REF={n:op(n).clone() for n in ("AIC_mm","AIV_rms","AIV_dq","MIXV_route")}
torch.npu.synchronize()
for n,v in REF.items(): print("[ref] %-12s shape=%s sum=%.4f"%(n,tuple(v.shape),float(v.float().sum())))
N=8; RES={}
def b_conc(fa,fb,root,s2,ef,ej):
    ef.record(root)
    with torch.npu.stream(s2):
        s2.wait_event(ef)
        for _ in range(N): RES["b"]=fb()
        ej.record(s2)
    for _ in range(N): RES["a"]=fa()
    root.wait_event(ej)
def cap(kind,fa,fb):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    ef=torch.npu.Event(); ej=torch.npu.Event(); ws=torch.npu.Stream()
    with torch.npu.stream(ws):
        if kind=="serial":
            for _ in range(N): RES["a"]=fa()
            for _ in range(N): RES["b"]=fb()
        else: b_conc(fa,fb,root,s2,ef,ej)
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    RES["a"]=None; RES["b"]=None
    with torch.npu.graph(g, stream=root):
        if kind=="serial":
            for _ in range(N): RES["a"]=fa()
            for _ in range(N): RES["b"]=fb()
        else: b_conc(fa,fb,root,s2,ef,ej)
    return g
OPS={"AIC_mm":lambda: torch.matmul(L1,L2),
     "AIV_rms":lambda: torch_npu.npu_rms_norm(XR,GR)[0],
     "AIV_dq":lambda: torch_npu.npu_dynamic_quant(XD)[0],
     "MIXV_route":lambda: torch_npu.npu_moe_init_routing(XR,row_idx,expert_idx,M)[0]}
print("\n=== 并发图重放后的输出 vs eager 参考 ===")
print("%-12s %-12s %-12s %s"%("opA","opB","哪个不对","最大绝对差"))
for a,b in [("AIC_mm","AIV_rms"),("AIC_mm","AIV_dq"),("AIC_mm","MIXV_route"),
            ("AIV_rms","AIV_dq"),("AIV_rms","MIXV_route")]:
    g=cap("conc",OPS[a],OPS[b]); torch.npu.synchronize()
    RES["a"]=None; RES["b"]=None
    g.replay(); torch.npu.synchronize()
    msg=[]
    for slot,opn in (("a",a),("b",b)):
        got=RES[slot]
        if got is None: msg.append(f"{opn}:未产出"); continue
        d=float((got.float()-REF[opn].float()).abs().max())
        if d>1e-3: msg.append(f"{opn}:DIFF")
    print("%-12s %-12s %-12s %s"%(a,b,",".join(msg) if msg else "一致",""))
