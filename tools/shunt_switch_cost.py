#!/usr/bin/env python3
"""类型切换成本：同类型连排 vs AIC/AIV 交替，各排多久？
真实结构（trace 实测）：272 个块交替，AIC 块 1.6 个算子 / AIV 块 4.0 个算子。
"""
import torch, torch_npu, time, os, json
torch.npu.set_device(0); dev="npu:0"
M=48
def bf(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
def f32(*s): return torch.randn(*s,dtype=torch.float32,device=dev)
B=dict(M=M,mm_a=bf(M,1024),mm_w=bf(1024,5120),rms_x=bf(M,5120),rms_g=bf(5120),
       dq_x=bf(M,1280),mix_x=bf(M,5120),
       row=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6),
       exp=torch.topk(f32(M,384),6).indices.to(torch.int32))
F={"mix":lambda: torch_npu.npu_moe_init_routing(B["mix_x"],B["row"],B["exp"],B["M"]),
   "mm": lambda: torch.matmul(B["mm_a"],B["mm_w"]),
   "rms":lambda: torch_npu.npu_rms_norm(B["rms_x"],B["rms_g"])[0],
   "dqs":lambda: torch_npu.npu_dynamic_quant(B["dq_x"])[0]}
for f in F.values(): f()
torch.npu.synchronize()

NBLK=int(os.environ.get("NBLK","64"))     # 交替块数
AC=2   # 每 AIC 块算子数（真实 1.6）
VC=4   # 每 AIV 块算子数（真实 4.0）
AIC=["mix","mm"]; AIV=["rms","dqs"]       # 交替取

def cap(kind):
    g=torch.npu.NPUGraph(); root=torch.npu.Stream()
    def body():
        if kind=="all_aic":
            for _ in range(NBLK*AC): F[AIC[0]]()
        elif kind=="all_aiv":
            for _ in range(NBLK*VC): F[AIV[0]]()
        elif kind=="block_alt":          # 真实结构：AIC块 → AIV块 交替
            for b in range(NBLK):
                for i in range(AC): F[AIC[i%2]]()
                for i in range(VC): F[AIV[i%2]]()
        elif kind=="block_alt_wide":     # 块更大（AIC 8 / AIV 16）
            for b in range(max(1,NBLK//4)):
                for i in range(8): F[AIC[i%2]]()
                for i in range(16): F[AIV[i%2]]()
        elif kind=="mixed_noswitch":     # 同样的算子总量，但不切换：先全 AIC 再全 AIV
            for b in range(NBLK):
                for i in range(AC): F[AIC[i%2]]()
            for b in range(NBLK):
                for i in range(VC): F[AIV[i%2]]()
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
print("NBLK=%d  AIC %d 算子/块, AIV %d 算子/块"%(NBLK,AC,VC))
print("%-22s %11s %9s %11s"%("结构","makespan","算子数","每算子µs"))
res={}
for kind,lab,nexp in (("mixed_noswitch","同算子集·不切换",NBLK*(AC+VC)),
                      ("block_alt","AIC/AIV 块交替(真实)",NBLK*(AC+VC)),
                      ("block_alt_wide","大块交替(8/16)",(NBLK//4)*(24)),
                      ("all_aic","纯 AIC",NBLK*AC),("all_aiv","纯 AIV",NBLK*VC)):
    try: g=cap(kind)
    except Exception as e: print("%-22s FAIL %s"%(lab,str(e).replace("\n"," ")[:50])); continue
    dt=meas(g); res[kind]=dt
    print("%-22s %11.3f %9d %10.2f"%(lab,dt,nexp,dt/nexp*1000))
if "mixed_noswitch" in res and "block_alt" in res:
    n=NBLK
    excess=res["block_alt"]-res["mixed_noswitch"]
    print("\n★ 块切换代价：%.3f ms / %d 次切换 = %.2f µs/次"%(excess,n,excess/n*1000))
    print("  外推到生产：272 次切换/步 × %.2f µs = %.3f ms/步（占 24.6ms 的 %.1f%%）"%(
        excess/n*1000, excess/n*1000*272/1000, excess/n*1000*272/24.6/1000*100))
json.dump(res,open("/tmp/shunt_sw.json","w"),indent=1)
