#!/usr/bin/env python3
"""核验：MIX 算子两流 vs 单流，kernel 数与设备时长是否可信。"""
import torch, torch_npu, os, glob, csv, shutil, statistics, time
from collections import Counter
torch.npu.set_device(0); dev="npu:0"
M=48; K=int(os.environ.get("K","64"))
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s): return torch.randn(*s,dtype=torch.float32,device=dev)
XA=[bf16(M,5120) for _ in range(K)]; XB=[bf16(M,5120) for _ in range(K)]
LOG=f32(M,384); row=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6)
exp=torch.topk(LOG,6).indices.to(torch.int32)
mixA=lambda i: torch_npu.npu_moe_init_routing(XA[i],row,exp,M)
mixB=lambda i: torch_npu.npu_moe_init_routing(XB[i],row,exp,M)
mixA(0); mixB(0); torch.npu.synchronize()

def build(mode):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    ef=torch.npu.Event(); ej=torch.npu.Event()
    def body():
        if mode=="serial":
            for i in range(K): mixA(i)
            for i in range(K): mixB(i)
        else:
            ef.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(ef)
                for i in range(K): mixB(i)
                ej.record(s2)
            for i in range(K): mixA(i)
            root.wait_event(ej)
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g

OUT="/tmp/shunt_mixv"; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
exc=torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False,
    op_attr=False, data_simplification=True, aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
REP=5
for mode in ("serial","two"):
    g=build(mode); sub=os.path.join(OUT,mode)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exc,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(sub)) as prof:
        for _ in range(REP): g.replay()
        torch.npu.synchronize()
    f=glob.glob(sub+"/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv",recursive=True)
    tot=0.0; cnt=0; durs=[]; streams=Counter()
    for r in csv.DictReader(open(f[0],newline="")):
        try: du=float(r["Duration(us)"])
        except: continue
        tot+=du; cnt+=1; durs.append(du); streams[(r.get("Stream ID") or "").strip()]+=1
    durs.sort()
    print("### %-8s kernel=%d (%.1f/次重放)  设备时长合计 %.3f ms/次重放  中位 %.2f µs  p90 %.2f"%(
        mode, cnt, cnt/REP, tot/REP/1000, statistics.median(durs), durs[int(len(durs)*0.9)]))
    print("      流分布:", dict(streams.most_common(5)))
