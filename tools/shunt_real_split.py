#!/usr/bin/env python3
"""真实算子 + 真实 shape：三种分流设计的对照。
 A) 均匀分半（奇偶交替）      —— 两条流都 AIV 重
 B) 按引擎分（AIC/MIX 一流，AIV 另一流）
 C) 单流基线
"""
import torch, torch_npu, os, glob, csv, shutil, statistics, time, json
torch.npu.set_device(0); dev="npu:0"
M=48
bf=lambda *s: torch.randn(*s,dtype=torch.bfloat16,device=dev)
f32=lambda *s: torch.randn(*s,dtype=torch.float32,device=dev)
x5120=bf(M,5120); g5120=bf(5120); x1280=bf(M,1280); g1280=bf(1280)
mm1a=bf(M,4096); mm1w=bf(4096,1024)
logits=f32(M,384); row=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6)
exp=torch.topk(logits,6).indices.to(torch.int32)
# (标签, 引擎类, trace时长µs, 函数)
OPS=[
 ("RmsNorm5120","V",18.7, lambda: torch_npu.npu_rms_norm(x5120,g5120)[0]),
 ("DynQuant5120","V",5.6, lambda: torch_npu.npu_dynamic_quant(x5120)[0]),
 ("RmsNorm1280","V",12.4, lambda: torch_npu.npu_rms_norm(x1280,g1280)[0]),
 ("DynQuant1280","V",3.7, lambda: torch_npu.npu_dynamic_quant(x1280)[0]),
 ("MatMulV2","C",28.0, lambda: torch.matmul(mm1a,mm1w)),
 ("MoeGating","V",15.5, lambda: torch_npu.npu_moe_gating_top_k_softmax(logits,None,6)),
 ("MoeInitRouting","M",16.4, lambda: torch_npu.npu_moe_init_routing(x5120,row,exp,M)),
]
for *_,f in OPS: f()
torch.npu.synchronize()
REP=int(os.environ.get("REP","30"))
LAY=int(os.environ.get("LAYERS","20"))
print("层数=%d（每层 7 个真实算子，真实 shape）"%LAY)
def meas(g):
    for _ in range(3): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(REP): g.replay()
    torch.npu.synchronize(); return (time.perf_counter()-t0)/REP*1000
def cap(body):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream()
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body(root)
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body(root)
    return g
def b_serial(root):
    for _ in range(LAY):
        for *_,f in OPS: f()
def b_even(root):
    s2=torch.npu.Stream(); e1=torch.npu.Event(); e2=torch.npu.Event()
    e1.record(root)
    with torch.npu.stream(s2):
        s2.wait_event(e1)
        for _ in range(LAY):
            for i,(_l,_k,_d,f) in enumerate(OPS):
                if i%2: f()
        e2.record(s2)
    for _ in range(LAY):
        for i,(_l,_k,_d,f) in enumerate(OPS):
            if i%2==0: f()
    root.wait_event(e2)
def b_engine(root):
    s2=torch.npu.Stream(); e1=torch.npu.Event(); e2=torch.npu.Event()
    e1.record(root)
    with torch.npu.stream(s2):
        s2.wait_event(e1)
        for _ in range(LAY):
            for l,k,d,f in OPS:
                if k=="V": f()
        e2.record(s2)
    for _ in range(LAY):
        for l,k,d,f in OPS:
            if k in ("C","M"): f()
    root.wait_event(e2)
def b_engine_bar(root):     # 按引擎分 + 逐层屏障
    s2=torch.npu.Stream()
    for _ in range(LAY):
        e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
        with torch.npu.stream(s2):
            s2.wait_event(e1)
            for l,k,d,f in OPS:
                if k=="V": f()
            e2.record(s2)
        for l,k,d,f in OPS:
            if k in ("C","M"): f()
        root.wait_event(e2)
res={}
for lab,fn in (("A 单流基线",b_serial),("B 均匀分半",b_even),
               ("C 按引擎分",b_engine),("D 按引擎分+逐算子屏障",b_engine_bar)):
    try: g=cap(fn)
    except Exception as e:
        print("%-24s FAIL %s"%(lab,str(e).replace("\n"," ")[:60])); continue
    res[lab]=meas(g)
b=res.get("A 单流基线")
print("7 个真实算子（真实 shape），trace 时长和 = %.1f µs\n"%sum(d for *_ ,d,_ in [(a,b_,c,None) for a,b_,c,_ in [(o[0],o[1],o[2],0) for o in OPS]]))
print("%-24s %11s %10s"%("臂","makespan","vs 单流"))
for lab in ("A 单流基线","B 均匀分半","C 按引擎分","D 按引擎分+逐算子屏障"):
    if lab in res: print("%-24s %11.3f %9.3fx"%(lab,res[lab],b/res[lab]))
print("\n引擎构成: V(AIV)=5 个/%.1fµs  C(AIC)=1 个/%.1fµs  M(MIX)=1 个/%.1fµs"%(
    sum(d for _,k,d,_ in OPS if k=="V"),sum(d for _,k,d,_ in OPS if k=="C"),sum(d for _,k,d,_ in OPS if k=="M")))
json.dump(res,open("/tmp/shunt_rs.json","w"),indent=1)
