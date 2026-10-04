import vllm_ascend.vllm_ascend_C  # noqa
import torch, torch_npu
torch.npu.set_device("npu:0")
C = torch.ops._C_ascend

def rms(x, w, eps):
    for fn, args, kw in (
        (getattr(torch_npu, "npu_rms_norm", None), (x, w), {"epsilon": eps}),
        (getattr(torch_npu, "npu_rms_norm", None), (x, w), {"eps": eps}),
        (getattr(C, "npu_rms_norm", None), (x, w, eps), {}),
    ):
        if fn is None: continue
        try: return fn(*args, **kw)
        except Exception as e: last = e
    raise RuntimeError(f"npu_rms_norm 调用失败: {last}")

def dq(x):
    return torch_npu.npu_dynamic_quant(x)

def rmsdq(x, w, eps):
    try:
        return C.npu_rms_norm_dynamic_quant(x, w, epsilon=eps)
    except Exception:
        return torch_npu.npu_rms_norm_dynamic_quant(x, w, epsilon=eps)

for shape in ((6, 5120), (6, 1280), (256, 5120)):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16, device="npu:0")
    w = torch.randn(shape[-1], dtype=torch.bfloat16, device="npu:0") * 0.1 + 1.0
    eps = 1e-6
    r = rms(x, w, eps)
    n = r[0] if isinstance(r, (tuple, list)) else r
    q1, s1 = dq(n)
    q2, s2 = rmsdq(x, w, eps)
    q1 = q1.view(torch.int8); q2 = q2.view(torch.int8)
    same_q = bool((q1 == q2).all().item())
    ndiff = int((q1 != q2).sum().item())
    d1 = (s1.float() - s2.float()).abs()
    print(f"shape={shape}  int8 逐位相同={same_q} ndiff={ndiff}/{q1.numel()}  "
          f"scale max|Δ|={float(d1.max()):.3e}  scale ndiff={int((d1>0).sum().item())}/{s1.numel()}")
