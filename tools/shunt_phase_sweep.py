#!/usr/bin/env python3
"""相位扫描：两条流"错开"到底有没有用？

背景：MIX 核内部 cube 相位 → vector 相位是**先后执行**的（MIX_AIC ∩ MIX_AIV = 0.000，
`DECODE-AIC-AIV-PIPELINE` §2 实测）。若两条流上的 MIX 核相位**对齐**，两者的 cube 需求同时到达；
若**错开半拍**，A 流的 cube 可与 B 流的 vector 重叠。

做法：在 B 链头部插入 n 个"移相算子"（每个 ≈ 半个 MIX 核时长），扫描 n。
若相位重要，makespan 应随 n 出现明显谷底；若无关，则近似平坦。
"""
import torch, torch_npu, time, os, json
torch.npu.set_device(0); dev="npu:0"
M=48
def bf16(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s):  return torch.randn(*s,dtype=torch.float32,device=dev)

K=int(os.environ.get("K","64"))
# MIX 算子：moe_init_routing（profile 实测 MIX_AIC，48 blocks，~30µs）
XA=[bf16(M,5120) for _ in range(K)]
XB=[bf16(M,5120) for _ in range(K)]
LOG=f32(M,384)
row_idx=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6)
exp_idx=torch.topk(LOG,6).indices.to(torch.int32)
# 移相算子：dynamic_quant（AI_VECTOR，blk4/16，短）
SHIFT_X=bf16(M,1280)
mixA=lambda i: torch_npu.npu_moe_init_routing(XA[i],row_idx,exp_idx,M)
mixB=lambda i: torch_npu.npu_moe_init_routing(XB[i],row_idx,exp_idx,M)
shift =lambda: torch_npu.npu_dynamic_quant(SHIFT_X)[0]
for f in (lambda: mixA(0), shift): f()
torch.npu.synchronize()

def build(mode, shift_n=0):
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
                for _ in range(shift_n): shift()
                for i in range(K): mixB(i)
                ej.record(s2)
            for i in range(K): mixA(i)
            root.wait_event(ej)
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

print("K=%d 每条流（2K=%d 个 MIX 算子总量）"%(K,2*K))
try:
    ts=meas(build("serial")); print("参照  单流串行 2K 个: %8.3f ms"%ts)
except Exception as e:
    print("serial capture FAIL", str(e).replace("\n"," ")[:80]); ts=None
print("\n%-14s %11s %11s %10s"%("移相算子数","makespan","相对对齐","每算子µs"))
res={}
base=None
for n in range(0,7):
    try: g=build("two",n)
    except Exception as e:
        print("%-14d FAIL %s"%(n,str(e).replace("\n"," ")[:60])); continue
    dt=meas(g)
    if base is None: base=dt
    res[n]=dt
    print("%-14d %11.3f %10.3fx %9.2f"%(n,dt,base/dt,dt/K*1000))
if ts:
    print("\n两流 vs 单流串行: 最好 %.3f× 最差 %.3f×"%(
        ts/min(res.values()) if res else 0, ts/max(res.values()) if res else 0))
    print("（若两流完全无重叠 ⇒ 比值≈1.00×；若完美互补 ⇒ 可达 ~2.00×）")
json.dump(res,open("/tmp/shunt_phase.json","w"),indent=1)
