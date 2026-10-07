#!/usr/bin/env python3
"""融合天花板实测：把纯 AIC 工作藏进"AIC 全闲"的纯 AIV 段里。
比例取自真实 trace：纯AIC 3741 µs / 纯AIV 8300 µs ≈ 1 : 2.2
臂：
  ① serial   —— AIC 段 → AIV 段 交替（= 现状形态）
  ② fused    —— 两流自由（= 内核融合的上界）
  ③ delayed  —— 把 AIC 延后 1 层（跨层流水）
"""
import torch, torch_npu, time, os, json
torch.npu.set_device(0); dev="npu:0"
M=48; LAY=int(os.environ.get("LAYERS","24"))
def bf(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
CA=[bf(M,5120) for _ in range(4)]; CW=bf(5120,1024)     # AI_CORE 24blk ~15.3µs
RX=[bf(M,5120) for _ in range(4)]; RG=bf(5120)          # AIV 48blk ~8µs
DX=[bf(M,1280) for _ in range(4)]                       # AIV 4blk ~3.6µs
f_cube=lambda i: torch.matmul(CA[i%4],CW)
f_aiv =lambda i: (torch_npu.npu_rms_norm(RX[i%4],RG)[0], torch_npu.npu_dynamic_quant(DX[i%4])[0])
for i in range(4): f_cube(i); f_aiv(i)
torch.npu.synchronize()
# 每层: 2 个 cube(30.6µs) + 5 组 aiv(5×11.6=58µs)  → 比例 1:1.9，接近真实的 1:2.2
def emit_cube(k):
    for _ in range(2): f_cube(k)
def emit_aiv(k):
    for _ in range(5): f_aiv(k)
def cap(arm):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    def body():
        if arm=="serial":
            for k in range(LAY): emit_cube(k); emit_aiv(k)
        elif arm=="fused":
            e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e1)
                for k in range(LAY): emit_aiv(k)
                e2.record(s2)
            for k in range(LAY): emit_cube(k)
            root.wait_event(e2)
        elif arm=="delayed":     # AIC 延后一层（跨层流水，避开同层依赖）
            e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e1)
                for k in range(LAY): emit_aiv(k)
                e2.record(s2)
            emit_cube(0)
            for k in range(LAY-1):
                root.wait_event(e1)   # 占位
                emit_cube(k+1)
            root.wait_event(e2)
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
REP=25
def meas(g):
    for _ in range(3): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(REP): g.replay()
    torch.npu.synchronize(); return (time.perf_counter()-t0)/REP*1000
res={}
for a,lab in (("serial","① 串行交替(=现状)"),("fused","② 两流(=融合上界)"),("delayed","③ 跨层延后")):
    try: g=cap(a)
    except Exception as e:
        print("%-22s FAIL %s"%(lab,str(e).replace("\n"," ")[:60])); continue
    res[a]=meas(g)
b=res.get("serial")
print("每层: 2 个纯cube(24blk) + 5 组纯AIV | LAY=%d\n"%LAY)
print("%-22s %12s %10s"%("臂","makespan(ms)","vs 串行"))
for a,lab in (("serial","① 串行交替(=现状)"),("fused","② 两流(=融合上界)"),("delayed","③ 跨层延后")):
    if a in res: print("%-22s %12.3f %9.3fx"%(lab,res[a],b/res[a]))
print("\n★ 理论：纯cube占总时长 30.6/88.6 = 34.5% ⇒ 完美隐藏加速 = 1/(1-0.345) = 1.53x")
json.dump(res,open("/tmp/shunt_fb.json","w"),indent=1)
