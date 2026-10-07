#!/usr/bin/env python3
"""生产比例混合实验：按真实主流的时长比例（MIX 55% / AIV 31% / CUBE 13%）
构造合成序列，测三种流分配策略的净收益。

⚠️ MIX 代理只能用 moe_init_routing（48blk）；生产的 HcPre 是 24blk。
   因此 MIX 相关的结论是【下界】（48blk 更吃资源）。
"""
import torch, torch_npu, time, os, json
torch.npu.set_device(0); dev="npu:0"
M=48; LAY=int(os.environ.get("LAYERS","16"))
def bf(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s): return torch.randn(*s,dtype=torch.float32,device=dev)
# 代理
MX=bf(M,5120); row=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6)
exp=torch.topk(f32(M,384),6).indices.to(torch.int32)
RX=bf(M,5120); RG=bf(5120); DX=bf(M,1280)
CA=bf(M,5120); CW=bf(5120,1024)
f_mix=lambda: torch_npu.npu_moe_init_routing(MX,row,exp,M)   # MIX_AIC 48blk (真实 24blk)
f_rms=lambda: torch_npu.npu_rms_norm(RX,RG)[0]                # AIV 48blk
f_dq =lambda: torch_npu.npu_dynamic_quant(DX)[0]              # AIV 4blk
f_mm =lambda: torch.matmul(CA,CW)                             # AI_CORE 24blk
for f in (f_mix,f_rms,f_dq,f_mm): f()
torch.npu.synchronize()
# 生产比例：MIX 55% / AIV 31% / CUBE 13%（按时长）
# 时长: mix 12.4, rms 7.6, dq 3.6, mm 15.3  → 每层配比
MIX_N, RMS_N, DQ_N, MM_N = 11, 4, 13, 2    # 目标精确比例 55/31/13

DUR_A={"mix":11*12.4,"rms":4*7.6,"dq":13*3.6,"mm":2*15.3}
tot=sum(DUR_A.values())
print("每层时长占比: mix %.0f%%  aiv %.0f%%  cube %.0f%%"%(
    DUR_A["mix"]/tot*100,(DUR_A["rms"]+DUR_A["dq"])/tot*100,DUR_A["mm"]/tot*100))
def emit_mix(): 
    for _ in range(MIX_N): f_mix()
def emit_aiv():
    for _ in range(RMS_N): f_rms()
    for _ in range(DQ_N): f_dq()
def emit_cube():
    for _ in range(MM_N): f_mm()
def cap(arm):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    def body():
        if arm=="serial":                     # 现状：全部交替串行
            for _ in range(LAY):
                emit_mix(); emit_aiv(); emit_cube()
        elif arm=="split_A":                  # 流A=全部MIX, 流B=AIV+cube
            e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e1)
                for _ in range(LAY): emit_aiv(); emit_cube()
                e2.record(s2)
            for _ in range(LAY): emit_mix()
            root.wait_event(e2)
        elif arm=="split_B":                  # 流A=MIX+cube(都用cube), 流B=AIV
            e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e1)
                for _ in range(LAY): emit_aiv()
                e2.record(s2)
            for _ in range(LAY): emit_mix(); emit_cube()
            root.wait_event(e2)
        elif arm=="split_C":                  # 流A=cube, 流B=MIX+AIV  (cube与谁都能重叠)
            e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e1)
                for _ in range(LAY): emit_mix(); emit_aiv()
                e2.record(s2)
            for _ in range(LAY): emit_cube()
            root.wait_event(e2)
        elif arm=="split_D":                  # 逐层屏障：MIX|cube 与 AIV 交替
            for _ in range(LAY):
                e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
                with torch.npu.stream(s2):
                    s2.wait_event(e1); emit_aiv(); e2.record(s2)
                emit_mix(); emit_cube()
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
for a,lab in (("serial","现状(全交替)"),("split_A","A: MIX ∥ (AIV+cube)"),
              ("split_B","B: (MIX+cube) ∥ AIV"),("split_C","C: cube ∥ (MIX+AIV)"),
              ("split_D","D: 逐层屏障 (MIX+cube) ∥ AIV")):
    try: g=cap(a)
    except Exception as e:
        print("%-30s FAIL %s"%(lab,str(e).replace("\n"," ")[:60])); continue
    res[a]=meas(g)
b=res.get("serial")
print("\n%-30s %12s %10s"%("策略","makespan(ms)","vs 现状"))
for a,lab in (("serial","现状(全交替)"),("split_A","A: MIX ∥ (AIV+cube)"),
              ("split_B","B: (MIX+cube) ∥ AIV"),("split_C","C: cube ∥ (MIX+AIV)"),
              ("split_D","D: 逐层屏障 (MIX+cube) ∥ AIV")):
    if a in res: print("%-30s %12.3f %9.3fx"%(lab,res[a],b/res[a]))
json.dump(res,open("/tmp/shunt_pm.json","w"),indent=1)
