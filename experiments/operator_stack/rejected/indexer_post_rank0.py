"""Experimental one-coefficient Div-RN per row; original numerical gates."""
import torch
import triton
import triton.language as tl

from indexer_post import COEFFICIENT_NUMERATORS


def initialize_coefficients(device):
    if device not in COEFFICIENT_NUMERATORS:
        COEFFICIENT_NUMERATORS[device]=torch.full((128,),127.0,dtype=torch.float32,device=device)
    return COEFFICIENT_NUMERATORS[device]


@triton.jit
def post_kernel_scalar(X,G,C,S,COORD,K_CACHE,S_CACHE,NORM,ROT,QUANT,SCALE,NUMERATOR,
                XS0:tl.constexpr,XS1:tl.constexpr,CS0:tl.constexpr,CS1:tl.constexpr,
                SS0:tl.constexpr,SS1:tl.constexpr,IS0:tl.constexpr,IS1:tl.constexpr,
                KS0:tl.constexpr,KS1:tl.constexpr,KS3:tl.constexpr,
                FS0:tl.constexpr,FS1:tl.constexpr,
                EPS:tl.constexpr,DEBUG:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,128)
    x=tl.load(X+row*XS0+col*XS1).to(tl.float32)
    gamma=tl.load(G+col).to(tl.float32)
    mean=tl.sum(x*x,axis=0)*(1.0/128.0)
    normal=(x*tl.rsqrt(mean+EPS)*gamma).to(tl.bfloat16).to(tl.float32)
    partner=tl.gather(normal,col ^ 1,axis=0)
    rotated=col>=64
    # Keep masked-off addresses in bounds during Ascend strided-load lowering.
    rope_col=col & 63
    cosine=tl.load(C+row*CS0+rope_col*CS1,rotated,0).to(tl.float32)
    sine=tl.load(S+row*SS0+rope_col*SS1,rotated,0).to(tl.float32)
    sign=tl.where((col & 1)==0,-1.0,1.0)
    value=tl.where(rotated,normal*cosine+partner*sine*sign,normal)
    value=value.to(tl.bfloat16).to(tl.float32)
    maximum=tl.max(tl.abs(value),axis=0)
    scale=maximum*(1.0/127.0)
    # The installed CANN single/multi-row kernels Div(127,max) then Mul(x,
    # coefficient). Use rounded division for that coefficient, rather than
    # Triton's approximate div or division by the dequantization scale.
    # Rank-zero Div-RN aborts this Ascend MLIR backend. Keep a one-element
    # tensor coefficient and broadcast it to the 128 values.
    numerator=tl.load(NUMERATOR)
    coefficient=tl.div_rn(numerator,maximum)
    scaled=tl.where(maximum>0,value*coefficient,0.0)
    # Exact nearest-even at positive/negative ties without GPU-only libdevice.
    lower=tl.floor(scaled)
    fraction=scaled-lower
    odd=(lower.to(tl.int32) & 1)!=0
    rounded=lower+tl.where((fraction>0.5)|((fraction==0.5)&odd),1.0,0.0)
    quant=tl.minimum(tl.maximum(rounded,-127.0),127.0).to(tl.int8)
    block=tl.load(COORD+row*IS0).to(tl.int64)
    position=tl.load(COORD+row*IS0+IS1).to(tl.int64)
    valid=(block>=0)&(position>=0)
    tl.store(K_CACHE+block*KS0+position*KS1+col*KS3,quant,valid)
    tl.store(S_CACHE+block*FS0+position*FS1,scale,valid)
    if DEBUG:
        tl.store(NORM+row*128+col,normal)
        tl.store(ROT+row*128+col,value)
        tl.store(QUANT+row*128+col,quant)
        tl.store(SCALE+row,scale)


def indexer_post_scalar(projected,gamma,cos,sin,coords,k_cache,scale_cache,epsilon,*,debug=False):
    assert projected.ndim==2 and projected.shape[-1]==128 and projected.dtype==torch.bfloat16
    assert gamma.shape==(128,) and gamma.is_contiguous()
    assert cos.ndim==sin.ndim==2 and cos.shape[1]==sin.shape[1]==64
    assert coords.ndim==2 and coords.shape[1]==2
    assert k_cache.ndim==scale_cache.ndim==4 and k_cache.shape[2:]==(1,128)
    assert scale_cache.shape[2:]==(1,1) and scale_cache.dtype==torch.float16
    assert k_cache.dtype==torch.int8
    rows=projected.shape[0]
    initialize_coefficients(projected.device)
    if debug:
        # Debug stores are row-major even when the projected input is a view.
        norm=torch.empty(projected.shape,dtype=projected.dtype,device=projected.device)
        rot=torch.empty_like(norm)
        quant=torch.empty_like(norm,dtype=torch.int8)
        scale=torch.empty(rows,dtype=torch.float32,device=projected.device)
    else:
        norm,rot,quant,scale=projected,projected,k_cache,scale_cache
    if rows:
        post_kernel_scalar[(rows,)](projected,gamma,cos,sin,coords,k_cache,scale_cache,norm,rot,quant,scale,
                            COEFFICIENT_NUMERATORS[projected.device],
                            *projected.stride(),*cos.stride(),*sin.stride(),*coords.stride(),
                            k_cache.stride(0),k_cache.stride(1),k_cache.stride(3),
                            scale_cache.stride(0),scale_cache.stride(1),
                            float(epsilon),debug,enable_fp_fusion=False)
    return (norm,rot,quant,scale) if debug else None
