#!/usr/bin/env python3
"""依赖审计（经验法）：延迟注入探针。

对每个候选算子注入 Δ 额外延迟，测步长增量：
  · 步长几乎不涨（<0.2Δ） ⇒ 它**不在关键路径**（已被其它工作掩盖）⇒ **可挪**
  · 步长涨满 Δ（>0.8Δ）    ⇒ 它在关键路径上 **被完全暴露** ⇒ **不可挪**
  · 介于两者之间            ⇒ 部分暴露

注入方式：在算子后插 k 个"纯 AIV 空转算子"（dynamic_quant on 2-token 输入），
每个 ≈ Δ/k。
"""
import torch, torch_npu, time, os, json, statistics
torch.npu.set_device(0); dev="npu:0"
M=48
def bf(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s): return torch.randn(*s,dtype=torch.float32,device=dev)
# 层模板（真实顺序，前 32 个算子）
T=[("HcPre","mix"),("RmsNorm","rms"),("DynamicQuant","dq"),("QuantMatmul","qmm"),
   ("RmsNorm","rms"),("DynamicQuant","dq"),("QuantMatmul","qmm"),("RoPE","rope"),
   ("SparseFlashMla","sfa"),("Neg","tiny"),("RoPE","rope"),
   ("MatMulV2a","mm"),("MatMulV2b","mm"),("TensorMove","tm"),("HcPost","hcpost"),
   ("HcPre","mix"),("RmsNormCast","rms"),("MatMulV3","mmv3"),
   ("Cast","tiny"),("MoeGating","gate"),("Less","tiny"),("GE","tiny"),("Or","tiny"),("Mask","tiny"),
   ("MoeInitRouting","route"),("GroupedSwiglu","gsw"),("GroupedMM","gmm"),
   ("Cast2","tiny"),("Abs","tiny"),("Unpermute","unpermute"),("Add","add"),("HcPost2","hcpost")]
# 张量池
P={}
P["x"]=bf(M,5120); P["g"]=bf(5120); P["d"]=bf(M,1280)
P["qmm_x"]=bf(M,5120); P["qmm_w"]=torch.randint(-8,8,(40,320,16,32),dtype=torch.int8,device=dev)
P["sc"]=bf(1280).abs()+.01; P["ps"]=f32(M).abs()+.01; P["bias"]=f32(M)
P["mm_a"]=bf(M,1024); P["mm_w"]=bf(1024,5120)
P["mmv3_a"]=bf(M,5120); P["mmv3_w"]=bf(5120,1024)
P["rope_x"]=bf(M,1,8,512); P["cos"]=bf(M,1,1,64); P["sin"]=bf(M,1,1,64)
P["logits"]=f32(M,384); P["row"]=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6)
P["exp"]=torch.topk(P["logits"],6).indices.to(torch.int32)
P["gsw_x"]=torch.randint(-8,8,(288,5120),dtype=torch.int8,device=dev)
P["gsw_w"]=torch.randint(-8,8,(48,72,320,16,64),dtype=torch.int8,device=dev)
P["gsw_ws"]=bf(48,4608); P["gsw_xs"]=f32(288).abs()+.01
P["gsw_gl"]=torch.tensor([288]+[0]*47,dtype=torch.int64,device=dev)
P["p2"]=bf(M,5120); P["d2"]=bf(M,1280)
def op(kind):
    if kind=="mix":   return torch_npu.npu_moe_init_routing(P["x"],P["row"],P["exp"],M)
    if kind=="rms":   return torch_npu.npu_rms_norm(P["x"],P["g"])[0]
    if kind=="dq":    return torch_npu.npu_dynamic_quant(P["d"])[0]
    if kind=="qmm":   return torch.matmul(P["mm_a"],P["mm_w"])   # 代理
    if kind=="rope":  return torch_npu.npu_rms_norm(P["p2"],P["g"])[0]   # 代理
    if kind=="mm":    return torch.matmul(P["mm_a"],P["mm_w"])
    if kind=="mmv3":  return torch.matmul(P["mmv3_a"],P["mmv3_w"])
    if kind=="hcpost":return torch_npu.npu_rms_norm(P["p2"],P["g"])[0]
    if kind=="gate":  return torch_npu.npu_moe_gating_top_k_softmax(P["logits"],None,6)
    if kind=="tiny":  return torch_npu.npu_dynamic_quant(P["d2"])[0]
    if kind=="route": return torch_npu.npu_moe_init_routing(P["x"],P["row"],P["exp"],M)
    if kind=="sfa":   return torch_npu.npu_rms_norm(P["p2"],P["g"])[0]
    if kind=="tm":    return P["p2"].add(0)
    if kind=="unpermute": return torch_npu.npu_rms_norm(P["p2"],P["g"])[0]
    if kind=="add":   return P["p2"].add(1)
    if kind=="gsw" or kind=="gmm": return torch_npu.npu_rms_norm(P["p2"],P["g"])[0]
    return P["p2"].add(0)
for _k,_ in T:
    try: op(_k)
    except Exception: pass
torch.npu.synchronize()
def cap(delay_at=None, k_delay=0):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream()
    def body():
        for rep in range(LAY):
            for i,(nm,k) in enumerate(T):
                op(k)
                if delay_at is not None and i==delay_at:
                    for _ in range(k_delay): op("tiny")
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
LAY=int(os.environ.get("LAYERS","10"))
REP=20
def meas(g):
    for _ in range(3): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(REP): g.replay()
    torch.npu.synchronize(); return (time.perf_counter()-t0)/REP*1000
# 先量 Δ（一个 tiny 算子）
g1=cap(); base=meas(g1)
gd=cap(0,8); d8=meas(gd)
DELTA=(d8-base)/8
print("基线 %.3f ms（%d 层 × %d 算子）| 注入单元 Δ=%.3f µs"%(base,LAY,len(T),DELTA*1000))
print("\n%-14s %10s %10s %9s %s"%("注入位置","makespan","Δ步长","暴露度","判定"))
res={}
for i,(nm,k) in enumerate(T):
    if k in ("mix","qmm","sfa","gsw","gmm"): continue     # 先测 AIV/小算子
    try: gd=cap(i,8)
    except Exception as e: continue
    dt=meas(gd); inc=(dt-base)*1000; inj=DELTA*1000*8
    expo=inc/inj
    verdict="可挪" if expo<0.35 else ("不可挪" if expo>0.75 else "部分")
    res[nm]=dict(inc=inc,inj=inj,expo=expo,verdict=verdict)
    print("%-14s %10.3f %9.1fµs %8.0f%%  %s"%(nm,dt,inc,expo*100,verdict))
json.dump(res,open("/tmp/shunt_delay.json","w"),indent=1)
