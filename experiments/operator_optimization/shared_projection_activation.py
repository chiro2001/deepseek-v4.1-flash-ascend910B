"""Shared expert N-major BF16 projection and staged clamped activation."""
import torch
import triton
import triton.language as tl


@triton.jit
def shared_projection_activation_kernel(x,w,raw,y,K:tl.constexpr,D:tl.constexpr,
        LIMIT:tl.constexpr,ALPHA:tl.constexpr,BETA:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    gate=tl.zeros((BN,),tl.float32);up=tl.zeros((BN,),tl.float32)
    for chunk in range(tl.cdiv(K,BK)):
        kk=chunk*BK+k
        value=tl.load(x+kk,kk<K,0).to(tl.float32)
        wg=tl.load(w+n[:,None]*K+kk[None,:],(n[:,None]<D)&(kk[None,:]<K),0).to(tl.float32)
        wu=tl.load(w+(n[:,None]+D)*K+kk[None,:],(n[:,None]<D)&(kk[None,:]<K),0).to(tl.float32)
        gate+=tl.sum(wg*value[None,:],1);up+=tl.sum(wu*value[None,:],1)
    gate=gate.to(tl.bfloat16).to(tl.float32);up=up.to(tl.bfloat16).to(tl.float32)
    # Shared clamp does not mutate its projection output.
    tl.store(raw+n,gate,n<D);tl.store(raw+D+n,up,n<D)
    upper=tl.full((),LIMIT,tl.float32).to(tl.bfloat16).to(tl.float32)
    lower=tl.full((),-LIMIT,tl.float32).to(tl.bfloat16).to(tl.float32)
    gate=tl.where(gate>upper,upper,gate);up=tl.where(up<lower,lower,up);up=tl.where(up>upper,upper,up)
    scaled=(gate*ALPHA).to(tl.bfloat16).to(tl.float32)
    sigmoid=(1/(1+tl.exp(-scaled))).to(tl.bfloat16).to(tl.float32)
    gated=(gate*sigmoid).to(tl.bfloat16).to(tl.float32)
    biased=(up+BETA).to(tl.bfloat16).to(tl.float32)
    tl.store(y+n,gated*biased,n<D)


def shared_projection_activation(x,w,limit,alpha=1.,beta=0.,*,bn=16,bk=512):
    assert x.shape==(1,5120) and w.shape==(512,5120)
    assert x.dtype==w.dtype==torch.bfloat16 and x.is_contiguous() and w.is_contiguous()
    raw=torch.empty((1,512),device=x.device,dtype=x.dtype)
    y=torch.empty((1,256),device=x.device,dtype=x.dtype)
    shared_projection_activation_kernel[(triton.cdiv(256,bn),)](x,w,raw,y,5120,256,
                     float(limit),float(alpha),float(beta),bn,bk,enable_fp_fusion=False)
    return raw,y
