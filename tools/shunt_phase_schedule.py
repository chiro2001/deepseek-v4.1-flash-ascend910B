#!/usr/bin/env python3
"""人工相位调度：把 AIC 工作与 AIV 工作按"时长匹配"编成阶段，阶段间加屏障。

臂：
  interleaved   —— 单流，按真实交替顺序（= 现状）
  split_free    —— AIC 全在流 A、AIV 全在流 B，仅首尾同步
  split_bar_k   —— 同上，但每 k 个算子一道屏障
  split_matched —— 同上，但阶段按"累计时长匹配"切分（用户设想）
  split_stagger —— matched + 流 B 延迟半个阶段启动（抗相位漂移）
"""
import torch, torch_npu, time, os, json
torch.npu.set_device(0); dev="npu:0"
M=48; LAY=int(os.environ.get("LAYERS","16"))
def bf(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s): return torch.randn(*s,dtype=torch.float32,device=dev)

# 每层 AIC 8 个 / AIV 5 个 —— 时长比 ≈2.2x，对齐真实 trace 的 AIC:AIV = 17.7:8.25
AIC_KIND=["mix","mm","mix","mm","mix","mm","mix","mm"]
AIV_KIND=["rms","dqs","rms","dqs","rms"]
AIC_SEQ=AIC_KIND*LAY; AIV_SEQ=AIV_KIND*LAY

def mkbatch():
    return dict(M=M, mm_a=bf(M,1024), mm_w=bf(1024,5120),
                rms_x=bf(M,5120), rms_g=bf(5120), dq_x=bf(M,1280),
                mix_x=bf(M,5120),
                row=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6),
                exp=torch.topk(f32(M,384),6).indices.to(torch.int32))
B=mkbatch()
F={"mix":lambda: torch_npu.npu_moe_init_routing(B["mix_x"],B["row"],B["exp"],B["M"]),
   "mm": lambda: torch.matmul(B["mm_a"],B["mm_w"]),
   "rms":lambda: torch_npu.npu_rms_norm(B["rms_x"],B["rms_g"])[0],
   "dqs":lambda: torch_npu.npu_dynamic_quant(B["dq_x"])[0]}
for f in F.values(): f()
torch.npu.synchronize()

# --- 先量各自时长（用于"匹配"切分）---
def solo_ms(kind,n=200):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream()
    def body():
        for _ in range(n): F[kind]()
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    for _ in range(3): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(10): g.replay()
    torch.npu.synchronize(); return (time.perf_counter()-t0)/10/n*1e6   # µs
DUR={k:solo_ms(k) for k in F}
print("单算子时长:", {k:round(v,2) for k,v in DUR.items()})

def phases_matched(aic,aiv,target=60.0):
    """按累计时长配对：每阶段 AIC 与 AIV 的累计时长都尽量接近 target"""
    ph=[]; i=j=0
    while i<len(aic) or j<len(aiv):
        cu=cv=0.0; ai=[]; av=[]
        while i<len(aic) and cu<target:
            ai.append(aic[i]); cu+=DUR[aic[i]]; i+=1
        while j<len(aiv) and cv<cu:
            av.append(aiv[j]); cv+=DUR[aiv[j]]; j+=1
        if not ai and not av: break
        ph.append((ai,av))
    return ph
PH=phases_matched(AIC_SEQ,AIV_SEQ)
print("匹配阶段数=%d（共 %d AIC + %d AIV 算子）"%(len(PH),len(AIC_SEQ),len(AIV_SEQ)))
assert sum(len(a) for a,_ in PH)==len(AIC_SEQ) and sum(len(b) for _,b in PH)==len(AIV_SEQ), "阶段覆盖率错"
mis=[abs(sum(DUR[x] for x in a)-sum(DUR[y] for y in b)) for a,b in PH]
print("阶段时长失配: 中位 %.2f µs / 最大 %.2f µs"%(sorted(mis)[len(mis)//2],max(mis)))

def cap(arm,k=None,stagger=0.0):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    def body():
        if arm=="interleaved":
            for i in range(max(len(AIC_SEQ),len(AIV_SEQ))):
                if i<len(AIC_SEQ): F[AIC_SEQ[i]]()
                if i<len(AIV_SEQ): F[AIV_SEQ[i]]()
        elif arm=="split_free":
            e1=torch.npu.Event(); e2=torch.npu.Event()
            e1.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e1)
                for kk in AIV_SEQ: F[kk]()
                e2.record(s2)
            for kk in AIC_SEQ: F[kk]()
            root.wait_event(e2)
        elif arm=="split_bar_k":
            # 每 k 个 AIC 算子一格；同格内 AIV 取累计时长与之匹配的那些
            i=j=0
            while i<len(AIC_SEQ) or j<len(AIV_SEQ):
                cu=0.0; ai=[]
                while i<len(AIC_SEQ) and len(ai)<k:
                    ai.append(AIC_SEQ[i]); cu+=DUR[AIC_SEQ[i]]; i+=1
                cv=0.0; av=[]
                while j<len(AIV_SEQ) and cv<cu:
                    av.append(AIV_SEQ[j]); cv+=DUR[AIV_SEQ[j]]; j+=1
                e1=torch.npu.Event(); e2=torch.npu.Event()
                e1.record(root)
                with torch.npu.stream(s2):
                    s2.wait_event(e1)
                    for kk in av: F[kk]()
                    e2.record(s2)
                for kk in ai: F[kk]()
                root.wait_event(e2)
        else:  # matched / stagger
            first=True
            for ai,av in PH:
                e1=torch.npu.Event(); e2=torch.npu.Event()
                e1.record(root)
                with torch.npu.stream(s2):
                    s2.wait_event(e1)
                    if first and stagger>0:
                        for _ in range(int(stagger//DUR["dqs"])): F["dqs"]()
                    for kk in av: F[kk]()
                    e2.record(s2)
                for kk in ai: F[kk]()
                root.wait_event(e2)
                first=False
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
def count_ops(g_note=""):
    pass
REP=25
def meas(g):
    for _ in range(3): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(REP): g.replay()
    torch.npu.synchronize(); return (time.perf_counter()-t0)/REP*1000
print("\n%-30s %11s %9s"%("臂","makespan(ms)","vs 现状"))
res={}; base=None
arms=[("interleaved",None),("split_free",None),("split_matched",None)]
for a,k in arms:
    try: g=cap(a,k)
    except Exception as e: print("%-30s FAIL %s"%(a,str(e).replace("\n"," ")[:60])); continue
    dt=meas(g); res[a]=dt
    if base is None: base=dt
    print("%-30s %11.3f %8.3fx"%(a,dt,base/dt))
for kk in (16,32,64):
    try: g=cap("split_bar_k",kk)
    except Exception as e: print("split_bar_k=%d FAIL"%kk); continue
    dt=meas(g); res["bar%d"%kk]=dt
    print("%-30s %11.3f %8.3fx"%("split_barrier_k=%d"%kk,dt,base/dt))
try:
    g=cap("split_stagger",None,stagger=20.0); dt=meas(g); res["stagger"]=dt
    print("%-30s %11.3f %8.3fx"%("split_stagger(20µs)",dt,base/dt))
except Exception as e: print("stagger FAIL",str(e)[:50])
json.dump(res,open("/tmp/shunt_ps.json","w"),indent=1)
