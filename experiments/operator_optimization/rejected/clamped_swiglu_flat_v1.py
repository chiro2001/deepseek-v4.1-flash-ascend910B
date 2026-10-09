"""Vector fusion of the measured clamped SwiGLU paths on DAV_2201."""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def clamped_swiglu_kernel(
    X, Y, M: tl.constexpr, D: tl.constexpr,
    STRIDE_ROW: tl.constexpr, STRIDE_COL: tl.constexpr,
    LIMIT: tl.constexpr, ALPHA: tl.constexpr, BETA: tl.constexpr,
    MUTATE_INPUT: tl.constexpr, STAGED_ROUNDING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < M * D
    row = i // D
    col = i % D
    offset = row * STRIDE_ROW + col * STRIDE_COL
    gate = tl.load(X + offset, mask, 0).to(tl.float32)
    up = tl.load(X + offset + D * STRIDE_COL, mask, 0).to(tl.float32)
    # aclnnClamp materializes bounds in the input dtype for these paths.
    bound = tl.full((), LIMIT, tl.float32).to(X.dtype.element_ty).to(tl.float32)
    lower = tl.full((), -LIMIT, tl.float32).to(X.dtype.element_ty).to(tl.float32)
    # Comparisons leave NaN and signed zero intact, like the reference clamp.
    gate = tl.where(gate > bound, bound, gate)
    up = tl.where(up < lower, lower, up)
    up = tl.where(up > bound, bound, up)
    if MUTATE_INPUT:
        tl.store(X + offset, gate, mask)
        tl.store(X + offset + D * STRIDE_COL, up, mask)
    if STAGED_ROUNDING:
        scaled = (gate * ALPHA).to(X.dtype.element_ty).to(tl.float32)
        sigmoid = (1.0 / (1.0 + tl.exp(-scaled))).to(X.dtype.element_ty).to(tl.float32)
        gated = (gate * sigmoid).to(X.dtype.element_ty).to(tl.float32)
        biased = (up + BETA).to(X.dtype.element_ty).to(tl.float32)
        result = gated * biased
    else:
        # Routed native npu_swiglu uses its FP32 internal activation path.
        result = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
    tl.store(Y + i, result, mask)


def clamped_swiglu(x, limit, *, alpha=1.0, beta=0.0,
                   mutate_input=False, staged_rounding=False, block=1024):
    if x.ndim != 2 or x.shape[-1] % 2:
        raise ValueError("Expected a two-dimensional input with an even width")
    if x.dtype != torch.bfloat16 or x.device.type != "npu":
        raise ValueError("The measured candidate supports NPU BF16 input")
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("limit must be finite and positive")
    if not staged_rounding and (alpha != 1.0 or beta != 0.0):
        raise ValueError("The routed contract has alpha=1 and beta=0")
    m, width = x.shape
    d = width // 2
    out = torch.empty((m, d), dtype=x.dtype, device=x.device)
    if m:
        clamped_swiglu_kernel[(triton.cdiv(m * d, block),)](
            x, out, m, d, *x.stride(), float(limit), float(alpha), float(beta),
            mutate_input, staged_rounding, block, enable_fp_fusion=False,
        )
    return out


class ModelNew(torch.nn.Module):
    def __init__(self, limit=7.0, alpha=1.0, beta=0.0,
                 mutate_input=True, staged_rounding=False):
        super().__init__()
        self.limit = limit
        self.alpha = alpha
        self.beta = beta
        self.mutate_input = mutate_input
        self.staged_rounding = staged_rounding

    def forward(self, x):
        return clamped_swiglu(
            x, self.limit, alpha=self.alpha, beta=self.beta,
            mutate_input=self.mutate_input, staged_rounding=self.staged_rounding,
        )
