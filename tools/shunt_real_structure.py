#!/usr/bin/env python3
"""按真实主流块结构复刻：AIC 1 算子/块(24blk) ←→ AIV 2 算子/块(48blk+4blk)。
臂：
  1 real_interleaved   —— AIC块/AIV块 交替（= 真实结构）
  2 same_type_batched  —— 同算子集，先全 AIC 再全 AIV（消除切换的上界）
  3 two_stream_free    —— AIC 全在流A / AIV 全在流B，仅首尾同步
  4 two_stream_bar     —— 每对块加屏障（用户的人工方案）
  5 real_wide_AIV      —— 把相邻 2 个 AIV 块并成 1 块（同流内加大同类型块）
"""
import torch, torch_npu, time, os, json
torch.npu.set_device(0); dev="npu:0"
M=48; NBLK=int(os.environ.get("NBLK","190"))
def bf(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s): return torch.randn(*s,dtype=torch.float32,device=dev)
# AIC 代理：AI_CORE 24 blocks / 15.26µs（真实 24-block 档）
AIC_A=bf(M,5120); AIC_W=bf(5120,1024)
# AIV 代理（真实结构：块内 1 个 48blk + 1 个 4blk）
AIV_X=bf(M,5120); AIV_G=bf(5120); AIV_D=bf(M,1280)
f_aic=lambda: torch.matmul(AIC_A,AIC_W)                       # 24 blk
f_rms=lambda: torch_npu.npu_rms_norm(AIV_X,AIV_G)[0]           # 48 blk
f_dq =lambda: torch_npu.npu_dynamic_quant(AIV_D)[0]            #  4 blk
for f in (f_aic,f_rms,f_dq): f()
torch.npu.synchronize()
def aiv_block(): f_rms(); f_dq()          # 真实 AIV 块 = 2 算子
print("NBLK=%d (每块: AIC 1 算子 / AIV 2 算子)  总算子=%d"%(NBLK,NBLK*(1+2)))

def cap(arm):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    def body():
        if arm=="real_interleaved":
            for _ in range(NBLK): f_aic(); aiv_block()
        elif arm=="same_type_batched":
            for _ in range(NBLK): f_aic()
            for _ in range(NBLK): aiv_block()
        elif arm=="real_wide_AIV":
            for i in range(NBLK):
                f_aic()
                aiv_block()
                if i%2==0: aiv_block()      # 每 2 个 AIV 块并成 1 块（同流内）
        elif arm=="two_stream_free":
            e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e1)
                for _ in range(NBLK): aiv_block()
                e2.record(s2)
            for _ in range(NBLK): f_aic()
            root.wait_event(e2)
        elif arm=="two_stream_bar":     # 每对块一道屏障（用户的人工方案）
            for _ in range(NBLK):
                e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
                with torch.npu.stream(s2):
                    s2.wait_event(e1); aiv_block(); e2.record(s2)
                f_aic(); root.wait_event(e2)
    ws=torch.npu.Stream()
    with torch.npu.stream(ws): body()
    torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
    with torch.npu.graph(g, stream=root): body()
    return g
REP=int(os.environ.get("REP","25"))
def meas(g):
    for _ in range(3): g.replay()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(REP): g.replay()
    torch.npu.synchronize(); return (time.perf_counter()-t0)/REP*1000
res={}
print("\n%-22s %12s %12s %12s"%("臂","makespan(ms)","vs 真实交替","vs 同类型合并"))
arms=[("real_interleaved","① 真实交替结构"),("same_type_batched","② 同类型合并(上界)"),
      ("real_wide_AIV","⑤ 同流内加宽AIV块"),("two_stream_free","③ 两流自由漂移"),
      ("two_stream_bar","④ 两流+逐块屏障")]
for a,lab in arms:
    try: g=cap(a)
    except Exception as e:
        print("%-22s FAIL %s"%(lab,str(e).replace("\n"," ")[:60])); continue
    dt=meas(g); res[a]=dt
print()
base=res.get("real_interleaved"); st=res.get("same_type_batched")
for a,lab in arms:
    if a not in res: continue
    print("%-22s %12.3f %11.3fx %14s"%(lab,res[a],
        (base/res[a]) if base else 0, ("%.3fx"%(st/res[a])) if st else "-"))
if base and st:
    print("\n★ 切换消除的上界 = %.3fx ; 真实结构下 ③/④ 相对 ① = %.3fx / %.3fx"%(
        base/st, base/res.get("two_stream_free",base), base/res.get("two_stream_bar",base)))
json.dump(res,open("/tmp/shunt_real.json","w"),indent=1)
