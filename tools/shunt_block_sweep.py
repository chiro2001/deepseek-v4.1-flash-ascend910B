#!/usr/bin/env python3
"""block 数 → 重叠效率：固定 MIX 侧，改变 AIV 侧的宽度（决定 block 数）。
规则若成立：AIV 越窄（block 越少）⇒ 重叠效率越高。"""
import torch, torch_npu, time, os, json, glob, csv, shutil
from collections import Counter
torch.npu.set_device(0); dev="npu:0"
M=48; K=int(os.environ.get("K","64"))
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s): return torch.randn(*s,dtype=torch.float32,device=dev)
XM=[bf16(M,5120) for _ in range(K)]
G={w:bf16(w) for w in (128,512,1280,5120)}
XV={w:[bf16(M,w) for _ in range(K)] for w in G}
LOG=f32(M,384); row=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6)
exp=torch.topk(LOG,6).indices.to(torch.int32)
op_mix=lambda i: torch_npu.npu_moe_init_routing(XM[i],row,exp,M)
def op_rms(w):
    return lambda i: torch_npu.npu_rms_norm(XV[w][i],G[w])[0]
op_mix(0)
for w in G: op_rms(w)(0)
torch.npu.synchronize()

def cap(width, kind):
    opB=op_rms(width)
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    def body():
        if kind=="serial":
            for i in range(K): op_mix(i)
            for i in range(K): opB(i)
        else:
            e1=torch.npu.Event(); e2=torch.npu.Event()
            e1.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e1)
                for i in range(K): opB(i)
                e2.record(s2)
            for i in range(K): op_mix(i)
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

# 先量 block 数
OUT="/tmp/shunt_bs"; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
exc=torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False,
    op_attr=False, data_simplification=True, aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
blocks={}
for w in G:
    g=cap(w,"serial"); sub=os.path.join(OUT,str(w))
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exc,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(sub)) as prof:
        g.replay(); torch.npu.synchronize()
    f=glob.glob(sub+"/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv",recursive=True)
    c=Counter(); 
    for r in csv.DictReader(open(f[0],newline="")):
        nm=(r.get("Name") or "")
        if "Rms" in nm: c[(r.get("Block Num") or "?",(r.get("Accelerator Core") or "").strip())]+=1
    blocks[w]=c.most_common(1)[0][0][0] if c else "?"
print("MIX 侧 = moe_init_routing（48 blocks）；AIV 侧 = rms_norm，宽度可变")
print("%-8s %-10s %11s %11s %11s %9s"%("AIV宽度","AIV blocks","串行(ms)","并发(ms)","理想(ms)","vs串行"))
res={}
for w in (128,512,1280,5120):
    ts=meas(cap(w,"serial")); tc=meas(cap(w,"concurrent"))
    # 理想：max(mix链, rms链)
    gm=cap(w,"serial")  # 仅用于结构说明
    res[w]=(ts,tc,blocks.get(w,"?"))
    print("%-8d %-10s %11.3f %11.3f %11s %9.3fx"%(w,blocks.get(w,"?"),ts,tc,"-",ts/tc))
json.dump({str(k):list(v) for k,v in res.items()},open("/tmp/shunt_bs.json","w"),indent=1)
