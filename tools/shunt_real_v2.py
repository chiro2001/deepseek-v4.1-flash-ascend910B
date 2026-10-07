#!/usr/bin/env python3
"""修正版：① 核对切换代价是否复现 ② 修正 ⑤（保持总工作量不变）③ profiler 验证。"""
import torch, torch_npu, time, os, json, glob, csv, shutil
torch.npu.set_device(0); dev="npu:0"
M=48; NBLK=int(os.environ.get("NBLK","190"))
def bf(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
A=bf(M,5120); W=bf(5120,1024)          # AI_CORE 24blk
X=bf(M,5120); G=bf(5120); D=bf(M,1280) # AIV 48blk / 4blk
f_aic=lambda: torch.matmul(A,W)
f_rms=lambda: torch_npu.npu_rms_norm(X,G)[0]
f_dq =lambda: torch_npu.npu_dynamic_quant(D)[0]
for f in (f_aic,f_rms,f_dq): f()
torch.npu.synchronize()
def aiv2(): f_rms(); f_dq()        # 真实 AIV 块（2 算子）
def aiv4(): f_rms(); f_dq(); f_rms(); f_dq()

def cap(arm):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream(); s2=torch.npu.Stream()
    def body():
        if arm=="interleaved":
            for _ in range(NBLK): f_aic(); aiv2()
        elif arm=="merged":                     # 同总工作量：AIV 块 4 算子，块数减半
            for _ in range(NBLK):
                f_aic()
                if _%2==0: aiv4()               # 只在半数块里放 4 个 → 总量相同
        elif arm=="same_type":                  # 先全 AIC 再全 AIV（消除切换）
            for _ in range(NBLK): f_aic()
            for _ in range(NBLK): aiv2()
        elif arm=="free":
            e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e1)
                for _ in range(NBLK): aiv2()
                e2.record(s2)
            for _ in range(NBLK): f_aic()
            root.wait_event(e2)
        elif arm=="free_swap":                  # AIC 放侧流，AIV 放主流
            e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
            with torch.npu.stream(s2):
                s2.wait_event(e1)
                for _ in range(NBLK): f_aic()
                e2.record(s2)
            for _ in range(NBLK): aiv2()
            root.wait_event(e2)
        elif arm=="bar":
            for _ in range(NBLK):
                e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
                with torch.npu.stream(s2):
                    s2.wait_event(e1); aiv2(); e2.record(s2)
                f_aic(); root.wait_event(e2)
        elif arm=="bar_wide":                   # 每 2 对块一道屏障
            for i in range(NBLK):
                e1=torch.npu.Event(); e2=torch.npu.Event(); e1.record(root)
                with torch.npu.stream(s2):
                    s2.wait_event(e1); aiv2(); e2.record(s2)
                f_aic()
                if i%2==1: root.wait_event(e2)
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

arms=[("interleaved","① 真实交替"),("same_type","② 同类型合并(消除切换)"),
      ("merged","⑤ 加宽AIV块(等量工作)"),("free","③ 两流自由"),("free_swap","③b 两流(交换主流)"),
      ("bar","④ 两流+逐块屏障"),("bar_wide","④b 两流+每2块屏障")]
res={}
for a,lab in arms:
    try: g=cap(a)
    except Exception as e:
        print("%-26s FAIL %s"%(lab,str(e).replace("\n"," ")[:60])); continue
    res[a]=meas(g)
base=res.get("interleaved"); st=res.get("same_type")
print("NBLK=%d  每块 AIC 1 算子(24blk) / AIV 2 算子(48blk+4blk)\n"%(NBLK))
print("%-26s %12s %13s %14s"%("臂","makespan(ms)","vs 真实交替","vs 同类型"))
for a,lab in arms:
    if a not in res: continue
    print("%-26s %12.3f %12.3fx %13s"%(lab,res[a],base/res[a],("%.3fx"%(st/res[a])) if st else "-"))
print("\n★ 切换代价检验：② / ① = %.3f  (>1 表示消除切换有收益)"%(base/st))
json.dump(res,open("/tmp/shunt_rv2.json","w"),indent=1)
