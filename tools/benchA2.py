#!/usr/bin/env python3
"""U3 微基准 A2：AIC/AIV 跨流重叠可行性矩阵（真实形状 M=48，修复计时与入参）。"""
import torch, torch_npu, time, json
torch.npu.set_device(0); dev="npu:0"
M=48; K=5120; K2=1280
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s):  return torch.randn(*s,dtype=torch.float32,device=dev)
def i8(*s):   return torch.randint(-8,8,s,dtype=torch.int8,device=dev)

x5120=bf16(M,5120); g5120=bf16(5120)
x1024=bf16(M,1024); w1024=bf16(1024,5120)
qx=i8(M,5120); wq_nz=i8(40,320,16,32); sw=bf16(1280).abs()+0.01; ps=f32(M).abs()+0.01; bias=f32(M)
qx2=i8(M,1280); wq_nz2=i8(128,80,16,32); sw2=bf16(1280).abs()+0.01
logits=f32(M,384)
row_idx=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6)
expert_idx=torch.topk(logits,6).indices.to(torch.int32)
sc_var=bf16(4096*128,512); sc_idx=torch.randint(0,4096*128,(M,2),dtype=torch.int32,device=dev); sc_upd=bf16(M,512)
sel_kv=bf16(25893,128,1,512); sel_q=bf16(M,8,512); cos=bf16(M,1,1,64); sin=bf16(M,1,1,64)
rope_x=bf16(M,1,8,512)

OPS={}
def reg(n,fn,tag): OPS[n]=(fn,tag)
reg("AIC_mm23",   lambda: torch.matmul(x1024,w1024), "AI_CORE blk~23")
reg("MIXA_qbmm20",lambda: torch_npu.npu_quant_matmul(qx,wq_nz,sw,pertoken_scale=ps,bias=bias,output_dtype=torch.bfloat16), "MIX_AIC blk20")
reg("MIXA_qbmm16",lambda: torch_npu.npu_quant_matmul(qx2,wq_nz2,sw2,pertoken_scale=ps,bias=bias,output_dtype=torch.bfloat16), "MIX_AIC blk16")
reg("AIV_rms48",  lambda: torch_npu.npu_rms_norm(x5120,g5120)[0], "AIV blk48")
reg("AIV_rope48", lambda: torch_npu.npu_rotary_mul(rope_x,cos,sin), "AIV blk48")
reg("AIV_dq16",   lambda: torch_npu.npu_dynamic_quant(x5120)[0], "AIV blk16")
reg("MIXV_route", lambda: torch_npu.npu_moe_init_routing(x5120,row_idx,expert_idx,M), "MIX_AIV blk48")
reg("MIXV_scat",  lambda: torch_npu.npu_scatter_nd_update(sc_var,sc_idx,sc_upd), "MIX_AIV blk48")

ok={}
for n,(fn,tag) in OPS.items():
    try: fn(); torch.npu.synchronize(); ok[n]=True; print("[ok]  %-14s %s"%(n,tag))
    except Exception as e: print("[FAIL]%-14s %s"%(n,str(e).replace(chr(10),' ')[:110]))
OPS={k:v for k,v in OPS.items() if ok.get(k)}

N=40; REP=15
def timed(fn):
    fn(); torch.npu.synchronize()
    t0=time.perf_counter()
    for _ in range(REP): fn()
    torch.npu.synchronize()
    return (time.perf_counter()-t0)/REP

solo={}
for n,(fn,_) in OPS.items():
    solo[n]=timed(lambda fn=fn:[fn() for _ in range(N)])/N
print("\n=== 单算子（µs）===")
for n,v in sorted(solo.items(),key=lambda x:x[1]): print("  %-14s %8.2f"%(n,v*1e6))

root=torch.npu.Stream(); s2=torch.npu.Stream()
ef=torch.npu.Event(); ej=torch.npu.Event()
def blk_serial(a,b):
    fa=OPS[a][0]; fb=OPS[b][0]
    for _ in range(N): fa()
    for _ in range(N): fb()
def blk_conc(a,b):
    fa=OPS[a][0]; fb=OPS[b][0]; cur=torch.npu.current_stream()
    root.wait_stream(cur); s2.wait_stream(cur)
    with torch.npu.stream(root):
        ef.record(root)
        for _ in range(N): fa()
    with torch.npu.stream(s2):
        s2.wait_event(ef)
        for _ in range(N): fb()
        ej.record(s2)
    with torch.npu.stream(root): root.wait_event(ej)
    cur.wait_stream(root)

pairs=[k for k in [("AIC_mm23","AIV_rms48"),("AIC_mm23","AIV_rope48"),("AIC_mm23","AIV_dq16"),
     ("AIC_mm23","MIXV_route"),("AIC_mm23","MIXA_qbmm20"),
     ("MIXA_qbmm20","AIV_rms48"),("MIXA_qbmm16","AIV_rms48"),("MIXA_qbmm20","AIV_dq16"),
     ("MIXA_qbmm20","MIXV_route"),("MIXA_qbmm20","MIXA_qbmm16"),
     ("AIV_rms48","AIV_rope48"),("AIV_rms48","AIV_dq16"),("AIV_rms48","MIXV_route"),
     ("MIXV_route","MIXV_scat")] if k[0] in OPS and k[1] in OPS]
print("\n=== 双流重叠可行性（N=%d/流）==="%N)
print("%-34s %9s %9s %9s %8s"%("pair","串行","并发","理论串行","加速比"))
out=[]
for a,b in pairs:
    ts=timed(lambda a=a,b=b:blk_serial(a,b)); tc=timed(lambda a=a,b=b:blk_conc(a,b))
    out.append(dict(a=a,b=b,serial=ts,conc=tc,speedup=ts/tc,
                    ov=(ts-tc)/min(solo[a],solo[b])/N))
    print("%-34s %9.1f %9.1f %9.1f %7.2fx  ov=%5.1f%%"%(f"{a}|{b}",ts*1e6,tc*1e6,(solo[a]+solo[b])*N*1e6,ts/tc,(ts-tc)/min(solo[a],solo[b])/N*100))
json.dump(out,open("/tmp/benchA2.json","w"),indent=1)
