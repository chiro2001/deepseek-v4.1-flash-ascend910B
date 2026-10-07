#!/usr/bin/env python3
"""决定性：把真实算子按真实顺序串成"一层"，看每个算子的时长能否复现 trace。
若不能复现 ⇒ 微基准无法代表真实条件，必须上真机。
"""
import torch, torch_npu, os, glob, csv, shutil, json, statistics
from collections import Counter
torch.npu.set_device(0); dev="npu:0"
M=48
bf=lambda *s: torch.randn(*s,dtype=torch.bfloat16,device=dev)
f32=lambda *s: torch.randn(*s,dtype=torch.float32,device=dev)
i8=lambda *s: torch.randint(-8,8,s,dtype=torch.int8,device=dev)
x5120=bf(M,5120); g5120=bf(5120); x1280=bf(M,1280); g1280=bf(1280)
mm1a=bf(M,4096); mm1w=bf(4096,1024)
mm2a=bf(M,1024); mm2w=bf(5120,1024)
logits=f32(M,384); row=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6)
exp=torch.topk(logits,6).indices.to(torch.int32)
# 真实顺序中我们能真实实例化的算子（其余省略，仅测这些的时长）
CHAIN=[
 ("RmsNorm_5120",  48, 18.7, lambda: torch_npu.npu_rms_norm(x5120,g5120)[0]),
 ("DynQuant_5120", 16,  5.6, lambda: torch_npu.npu_dynamic_quant(x5120)[0]),
 ("RmsNorm_1280",  48, 12.4, lambda: torch_npu.npu_rms_norm(x1280,g1280)[0]),
 ("DynQuant_1280",  4,  3.7, lambda: torch_npu.npu_dynamic_quant(x1280)[0]),
 ("MatMulV2_a",    22, 28.0, lambda: torch.matmul(mm1a,mm1w)),
 ("MoeGating",     48, 15.5, lambda: torch_npu.npu_moe_gating_top_k_softmax(logits,None,6)),
 ("MoeInitRouting",48, 16.4, lambda: torch_npu.npu_moe_init_routing(x5120,row,exp,M)),
]
for _,_,_,f in CHAIN: f()
torch.npu.synchronize()
OUT="/tmp/chain"; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
exc=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False,op_attr=False,data_simplification=True,aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
def prof(fn,tag,rep=20):
    sub=os.path.join(OUT,tag)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exc,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(sub)) as p:
        for _ in range(rep): fn()
        torch.npu.synchronize()
    fs=glob.glob(sub+"/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv",recursive=True)[0]
    out=[]
    for r in csv.DictReader(open(fs,newline="")):
        try: out.append((float(r["Duration(us)"]),(r.get("Block Num") or ""),(r.get("Name") or "")[:26]))
        except: pass
    return out
# ① 隔离跑
print("=== ① 隔离跑（每个算子单独 profiler）===")
solo={}
for nm,tb,td,f in CHAIN:
    rows=prof(f,nm,20)
    ds=sorted(x[0] for x in rows)
    solo[nm]=(statistics.median(ds), tb, td)
    print("  %-16s blk=%-3s 隔离 %6.2fµs | trace %6.1fµs | 比 %.2fx"%(nm,rows[0][1],statistics.median(ds),td,statistics.median(ds)/td))
# ② 真实顺序串成链
def chain():
    for _,_,_,f in CHAIN: f()
print("\n=== ② 真实顺序串成一层（NPUGraph 内）===")
rows=prof(chain,"chain",20)
ds=sorted(x[0] for x in rows)
print("  该层总时长 %.1f µs（7 个算子）"%(sum(x[0] for x in rows)/20))
for nm,tb,td,f in CHAIN:
    ms=[x for x in rows if nm.split("_")[0].lower() in x[2].lower()]
print("\n  逐个（按名字匹配，可能重名）:")
for d,b,n in sorted(rows)[:12]:
    print("    %-28s blk=%-3s %7.2fµs"%(n,b,d))
print("\n=== ③ 双流版（A 流 / B 流 各一半）===")
def two():
    s2=torch.npu.Stream(); e1=torch.npu.Event(); e2=torch.npu.Event()
    root=torch.npu.current_stream()
    e1.record(root)
    with torch.npu.stream(s2):
        s2.wait_event(e1)
        for i,(nm,tb,td,f) in enumerate(CHAIN):
            if i%2==1: f()
        e2.record(s2)
    for i,(nm,tb,td,f) in enumerate(CHAIN):
        if i%2==0: f()
    root.wait_event(e2)
rows2=prof(two,"two",20)
print("  双流总时长 %.1f µs  vs 单流 %.1f µs  = %.3fx"%(
    sum(x[0] for x in rows2)/20, sum(x[0] for x in rows)/20,
    (sum(x[0] for x in rows)/20)/(sum(x[0] for x in rows2)/20)))
