#!/usr/bin/env python3
"""用真实 shape 实例化真实算子，并逐个核对 (block, duration) 是否与 trace 一致。
这是"真实数据"的入口：只有 block 数和时长都对得上的算子，才算真复刻。
"""
import torch, torch_npu, os, glob, csv, shutil, json, statistics
from collections import Counter
torch.npu.set_device(0); dev="npu:0"
M=48
bf =lambda *s: torch.randn(*s,dtype=torch.bfloat16,device=dev)
f32=lambda *s: torch.randn(*s,dtype=torch.float32,device=dev)
f16=lambda *s: torch.randn(*s,dtype=torch.float16,device=dev)
i32=lambda *s: torch.randint(0,4,s,dtype=torch.int32,device=dev)
i64=lambda *s: torch.randint(0,4,s,dtype=torch.int64,device=dev)
i8 =lambda *s: torch.randint(-8,8,s,dtype=torch.int8,device=dev)
b8 =lambda *s: torch.randint(0,2,s,dtype=torch.bool,device=dev)

SPEC=[]   # (label, target_blk, target_dur, callable)
def add(label,tblk,tdur,fn,**kw): SPEC.append(dict(label=label,tblk=tblk,tdur=tdur,fn=fn,**kw))

# ---- 真实形状（取自 trace 一层）----
x5120=bf(M,5120); g5120=bf(5120)
x1280=bf(M,1280); g1280=bf(1280)
q5120=i8(M,5120); q1280=i8(M,1280)
wq_1280=i8(40,320,16,32); wq_4096=i8(128,80,16,32)
sc=bf(1280).abs()+.01
mm1a=bf(M,4096); mm1w=bf(4096,1024)
mm2a=bf(M,1024); mm2w=bf(5120,1024)
rope=bf(M,1,8,512); cs=bf(M,1,1,64); sn=bf(M,1,1,64)
logits=f32(M,384); gbias=f32(384)
routing_x=bf(M,5120)
row=torch.arange(M*6,dtype=torch.int32,device=dev).view(M,6)
mmv3a=f32(M,5120); mmv3w=f32(384,5120)

# ① RmsNorm 48blk / 18.7µs
add("RmsNorm_5120",48,18.7, lambda: torch_npu.npu_rms_norm(x5120,g5120)[0])
# ② RmsNorm 48blk / 12.4µs (1280)
add("RmsNorm_1280",48,12.4, lambda: torch_npu.npu_rms_norm(x1280,g1280)[0])
# ③ DynamicQuant 16blk / 5.6µs
add("DynQuant_5120",16,5.6, lambda: torch_npu.npu_dynamic_quant(x5120)[0])
# ④ DynamicQuant 4blk / 3.7µs
add("DynQuant_1280",4,3.7, lambda: torch_npu.npu_dynamic_quant(x1280)[0])
# ⑤ MatMulV2 22blk / 28.0µs
add("MatMulV2_4096x1024",22,28.0, lambda: torch.matmul(mm1a,mm1w))
# ⑥ MatMulV2 23blk / 16.5µs
add("MatMulV2_1024x1024",23,16.5, lambda: torch.matmul(mm2a,mm2w))
# ⑦ MatMulV3 24blk / 19.3µs (fp32)
add("MatMulV3_5120x384",24,19.3, lambda: torch.matmul(mmv3a,mmv3w))
# ⑧ MoeGatingTopKHash 48blk / 15.5µs
add("MoeGating",48,15.5, lambda: torch_npu.npu_moe_gating_top_k_softmax(logits,None,6))
# ⑨ MoeInitRouting 48blk / 16.4µs
add("MoeInitRouting",48,16.4, lambda: torch_npu.npu_moe_init_routing(routing_x,row,torch.topk(logits,6).indices.to(torch.int32),M))
# ⑩ TensorMove 24blk / 5.1µs
add("TensorMove",24,5.1, lambda: x5120.add(0))
# ⑪ RoPE 48blk / 10.9µs
try:
    add("RoPE_rotary_mul",48,10.9, lambda: torch_npu.npu_rotary_mul(rope,cs,sn))
except Exception: pass
# ⑫ 小算子
add("Neg_tiny",3,1.3, lambda: -cs)
add("Cast_tiny",1,1.2, lambda: mm1a[:48].to(torch.int32) if False else torch.arange(48,device=dev,dtype=torch.int32).to(torch.float32))
# ⑬ HcPre（真实算子，若可用）
try:
    phi=f32(24,20480); alpha=f32(3); hb=f32(24)
    add("HcPre_mhc",24,38.2, lambda: torch.ops.npu.npu_mhc_pre(bf(M,4,5120),phi,alpha,hb))
except Exception as e: print("HcPre spec err",str(e)[:60])

# ---- 逐个实例化 + profiler 核对 ----
OUT="/tmp/realops"; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
exc=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False,op_attr=False,data_simplification=True,aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
print("%-24s %-6s %-6s %9s %9s %s"%("算子","目标blk","实测blk","目标µs","实测µs","判定"))
ok=[]; bad=[]
for s in SPEC:
    try:
        r=s["fn"](); torch.npu.synchronize()
    except Exception as e:
        print("%-24s %-6s %-6s %9.1f %9s 实例化失败: %s"%(s["label"],s["tblk"],"-",s["tdur"],"-",str(e).replace("\n"," ")[:60]))
        bad.append(s["label"]); continue
    sub=os.path.join(OUT,s["label"])
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exc,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(sub)) as p:
        for _ in range(20): s["fn"]()
        torch.npu.synchronize()
    fs=glob.glob(sub+"/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv",recursive=True)
    if not fs: bad.append(s["label"]); continue
    c=Counter(); ds=[]
    for r in csv.DictReader(open(fs[0],newline="")):
        try: ds.append(float(r["Duration(us)"]))
        except: continue
        c[(r.get("Block Num") or "")]+=1
    if not ds: bad.append(s["label"]); continue
    ds.sort(); mblk=c.most_common(1)[0][0]; mdur=ds[len(ds)//2]
    verdict = "✅" if (str(s["tblk"])==str(mblk) and abs(mdur-s["tdur"])/s["tdur"]<0.35) else ("⚠️blk" if str(s["tblk"])!=str(mblk) else "⚠️dur")
    print("%-24s %-6s %-6s %9.1f %9.2f %s"%(s["label"],s["tblk"],mblk,s["tdur"],mdur,verdict))
    (ok if verdict=="✅" else bad).append(s["label"])
print("\n匹配: %d / %d"%(len(ok),len(SPEC)))
