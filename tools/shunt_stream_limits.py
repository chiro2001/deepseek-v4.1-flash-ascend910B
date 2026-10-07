import torch, torch_npu, time
torch.npu.set_device(0); dev="npu:0"
# ① 能创建多少条流？
ok=0
try:
    for i in range(64):
        torch.npu.Stream(); ok+=1
except Exception as e:
    print("创建第 %d 条流失败: %s"%(ok+1,str(e)[:90]))
print("可创建流数（未捕获）: %d"%ok)
# ② 图内能有多少条并发流？
M=48
def bf(*s): return torch.randn(*s,dtype=torch.bfloat16,device=dev)
x=bf(M,5120); g=bf(5120)
for nstream in (2,3,4,6,8):
    try:
        gr=torch.npu.NPUGraph(); root=torch.npu.Stream()
        sides=[torch.npu.Stream() for _ in range(nstream-1)]
        ev_f=torch.npu.Event(); ev_j=[torch.npu.Event() for _ in sides]
        def body():
            ev_f.record(root)
            for k,s in enumerate(sides):
                with torch.npu.stream(s):
                    s.wait_event(ev_f)
                    for _ in range(8): torch_npu.npu_rms_norm(x,g)[0]
                    ev_j[k].record(s)
            for _ in range(8): torch_npu.npu_rms_norm(x,g)[0]
            for e in ev_j: root.wait_event(e)
        ws=torch.npu.Stream()
        with torch.npu.stream(ws): body()
        torch.npu.current_stream().wait_stream(ws); torch.npu.synchronize()
        with torch.npu.graph(gr, stream=root): body()
        for _ in range(3): gr.replay()
        torch.npu.synchronize()
        t0=time.perf_counter()
        for _ in range(20): gr.replay()
        torch.npu.synchronize()
        print("  %d 流: OK  %.3f ms"%(nstream,(time.perf_counter()-t0)/20*1000))
    except Exception as e:
        print("  %d 流: FAIL %s"%(nstream,str(e).replace("\n"," ")[:70]))
