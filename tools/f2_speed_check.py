import vllm_ascend.vllm_ascend_C  # noqa
import torch, torch_npu, time
torch.npu.set_device("npu:0")
C = torch.ops._C_ascend
def split(x, w, eps):
    y = torch_npu.npu_rms_norm(x, w, eps)
    y = y[0] if isinstance(y, (tuple, list)) else y
    return torch_npu.npu_dynamic_quant(y)
def fused(x, w, eps):
    return C.npu_rms_norm_dynamic_quant(x, w, epsilon=eps)
for shape in ((6, 5120), (6, 1280), (8064, 5120)):
    x = torch.randn(*shape, dtype=torch.bfloat16, device="npu:0")
    w = torch.randn(shape[-1], dtype=torch.bfloat16, device="npu:0")
    for name, fn in (("split(norm+quant)", split), ("fused", fused)):
        for _ in range(5): fn(x, w, 1e-6)
        torch.npu.synchronize()
        N = 30 if shape[0] <= 256 else 10
        t0 = time.perf_counter()
        for _ in range(N): fn(x, w, 1e-6)
        t1 = time.perf_counter(); torch.npu.synchronize(); t2 = time.perf_counter()
        print(f"shape={shape} {name:20s} host {(t1-t0)/N*1e6:8.1f} us  incl-device {(t2-t0)/N*1e6:8.1f} us")
