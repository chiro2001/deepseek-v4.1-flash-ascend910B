"""Experimental HC collapse with the vendor's row0+row1+row2+row3 order.

Not enabled in formal TP8 until the original precision gates pass.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def collapse_sequential(X, P, Y, D: tl.constexpr, B: tl.constexpr):
    col = tl.program_id(0) * B + tl.arange(0, B)
    valid = col < D
    value = tl.load(X + col, valid, 0).to(tl.float32) * tl.load(P)
    for row in tl.static_range(1, 4):
        product = tl.load(X + row * D + col, valid, 0).to(tl.float32) * tl.load(P + row)
        value = value + product
    tl.store(Y + col, value, valid)


def replace_collapse(x, outputs, pre_mix=None):
    assert x.shape == (1, 4, 5120) and x.dtype == torch.bfloat16 and x.is_contiguous()
    pre = outputs[3] if pre_mix is None else pre_mix
    y = torch.empty((1, 5120), dtype=x.dtype, device=x.device)
    collapse_sequential[(10,)](x, pre, y, 5120, 512, enable_fp_fusion=False)
    return (y, *outputs[1:])
