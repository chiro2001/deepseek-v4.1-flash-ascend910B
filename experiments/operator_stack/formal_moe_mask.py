"""Exact local-expert probability mask, optionally with the consumer BF16 cast."""
import torch
import triton
import triton.language as tl


@triton.jit
def _mask(ids, weights, output, FIRST:tl.constexpr, LAST:tl.constexpr,
          SIZE:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    expert=tl.load(ids+i,i<SIZE,other=-1)
    prob=tl.load(weights+i,i<SIZE,other=0)
    prob=tl.where((expert>=FIRST)&(expert<LAST),prob,0.)
    tl.store(output+i,prob,i<SIZE)


def local_probs(ids,weights,first,last,*,consumer_bf16=False):
    assert ids.shape==weights.shape and ids.ndim==2 and ids.shape[1]==6
    assert ids.dtype==torch.int32 and weights.dtype==torch.float32
    assert ids.is_contiguous() and weights.is_contiguous() and ids.device==weights.device
    assert 0<=first<last<=384 and last-first==48
    result=torch.empty_like(weights,dtype=torch.bfloat16 if consumer_bf16 else torch.float32)
    _mask[(triton.cdiv(ids.numel(),128),)](ids,weights,result,first,last,ids.numel(),128,
                                      enable_fp_fusion=False)
    return result
