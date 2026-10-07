#!/usr/bin/env python3
"""U3 微基准 G：把"双流并发"捕获成一张 NPUGraph 并重放，用 profiler 看真实设备重叠。"""
import torch, torch_npu, time, os, json, shutil
torch.npu.set_device(0); dev="npu:0"
M=48
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s):  return torch.randn(*s,dtype=torch.float32,device=dev)
L1=bf16(M,1024); L2=bf16(1024,5120)
XR=bf16(M,5120); GR=bf16(5120)
XD=bf16(M,5120)
LOG=f32(M,384)
row_idx=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6)
expert_idx=torch.topk(LOG,6).indices.to(torch.int32)

def op_aic():  return torch.matmul(L1,L2)                      # AI_CORE
def op_aiv():  return torch_npu.npu_rms_norm(XR,GR)[0]         # AI_VECTOR blk48
def op_aiv2(): return torch_npu.npu_dynamic_quant(XD)[0]       # AI_VECTOR blk16
def op_mixv(): return torch_npu.npu_moe_init_routing(XR,row_idx,expert_idx,M)  # MIX_AIV blk48
OPS={"AIC_mm":op_aic,"AIV_rms":op_aiv,"AIV_dq":op_aiv2,"MIXV_route":op_mixv}
for n,f in OPS.items(): f()
torch.npu.synchronize()

N=24
def body_serial(fa,fb):
    for _ in range(N): fa()
    for _ in range(N): fb()
def body_conc(fa,fb,root,s2,ef,ej):
    ef.record(root)
    with torch.npu.stream(s2):
        s2.wait_event(ef)
        for _ in range(N): fb()
        ej.record(s2)
    for _ in range(N): fa()
    root.wait_event(ej)

def capture(kind,fa,fb):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    ef=torch.npu.Event(); ej=torch.npu.Event()
    # warmup on side stream
    ws=torch.npu.Stream()
    with torch.npu.stream(ws):
        if kind=="serial": body_serial(fa,fb)
        else: body_conc(fa,fb,root,s2,ef,ej)
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root):
        if kind=="serial": body_serial(fa,fb)
        else: body_conc(fa,fb,root,s2,ef,ej)
    return g

def run_block(fa,fb,conc):
    """eager 版（无图）"""
    root=torch.npu.Stream(); s2=torch.npu.Stream(); ef=torch.npu.Event(); ej=torch.npu.Event()
    cur=torch.npu.current_stream(); root.wait_stream(cur); s2.wait_stream(cur)
    if conc: body_conc(fa,fb,root,s2,ef,ej)
    else: body_serial(fa,fb)
    cur.wait_stream(root)

OUT=os.environ.get("OUT","/tmp/u3prof")
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT, exist_ok=True)
pairs=[("AIC_mm","AIV_rms"),("AIC_mm","AIV_dq"),("AIC_mm","MIXV_route"),
       ("AIV_rms","AIV_dq"),("AIV_rms","MIXV_route")]
graphs={}
print("=== 捕获双流图 ===")
for a,b in pairs:
    fa,fb=OPS[a][0],OPS[b][0]
    for kind in ("serial","conc"):
        try:
            g=capture(kind,fa,fb); graphs[(a,b,kind)]=g
            print("  [ok]  %-22s %s"%(f"{a}|{b}",kind))
        except Exception as e:
            print("  [FAIL]%-22s %s -> %s"%(f"{a}|{b}",kind,str(e).replace(chr(10),' ')[:100]))

exp=torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False,
    op_attr=False, data_simplification=True, aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
REP=12
print("\n=== profiler 采集 ===")
for (a,b,kind),g in graphs.items():
    sub=os.path.join(OUT,f"{a}__{b}__{kind}")
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exp,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(sub)) as prof:
        for _ in range(REP): g.replay()
        torch.npu.synchronize()
    print("  profiled", sub)
json.dump({f"{a}|{b}|{k}":1 for (a,b,k) in graphs}, open(os.path.join(OUT,"index.json"),"w"))
print("done ->",OUT)
