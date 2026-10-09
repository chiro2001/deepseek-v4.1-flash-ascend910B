"""Two selected-expert GEMVs with BF16 output boundary and routed epilogue."""
import torch
import triton
import triton.language as tl


@triton.jit
def gmm1_activation_kernel(x,w,counts,raw,y,K:tl.constexpr,D:tl.constexpr,
                           COUNT_TYPE:tl.constexpr,LIMIT:tl.constexpr,
                           BN:tl.constexpr,BK:tl.constexpr):
    slot=tl.program_id(1)
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    e=tl.arange(0,8)
    ends=tl.load(counts+e)
    if COUNT_TYPE==1:ends=tl.cumsum(ends,0)
    expert=tl.sum((ends<=slot).to(tl.int32),0)
    k=tl.arange(0,BK)
    gate=tl.zeros((BN,),tl.float32);up=tl.zeros((BN,),tl.float32)
    for chunk in range(tl.cdiv(K,BK)):
        kk=chunk*BK+k
        value=tl.load(x+slot*K+kk,kk<K,0).to(tl.float32)
        wg=tl.load(w+expert*2*D*K+n[:,None]*K+kk[None,:],
                   (n[:,None]<D)&(kk[None,:]<K),0).to(tl.float32)
        wu=tl.load(w+expert*2*D*K+(n[:,None]+D)*K+kk[None,:],
                   (n[:,None]<D)&(kk[None,:]<K),0).to(tl.float32)
        gate+=tl.sum(wg*value[None,:],1)
        up+=tl.sum(wu*value[None,:],1)
    # GMM1 materializes BF16 before its activation, even in a fused epilogue.
    gate=gate.to(tl.bfloat16).to(tl.float32)
    up=up.to(tl.bfloat16).to(tl.float32)
    upper=tl.full((),LIMIT,tl.float32).to(tl.bfloat16).to(tl.float32)
    lower=tl.full((),-LIMIT,tl.float32).to(tl.bfloat16).to(tl.float32)
    gate=tl.where(gate>upper,upper,gate)
    up=tl.where(up<lower,lower,up);up=tl.where(up>upper,upper,up)
    tl.store(raw+slot*2*D+n,gate,n<D)
    tl.store(raw+slot*2*D+D+n,up,n<D)
    output=gate*(1/(1+tl.exp(-gate)))*up
    tl.store(y+slot*D+n,output,n<D)


def gmm1_activation(x,nk_weight,counts,limit,*,count_type=1,bn=8,bk=512):
    assert x.shape==(2,5120) and nk_weight.shape==(8,512,5120) and counts.shape==(8,)
    assert x.dtype==nk_weight.dtype==torch.bfloat16
    assert x.is_contiguous() and nk_weight.is_contiguous() and count_type in [0,1]
    raw=torch.empty((2,512),device=x.device,dtype=x.dtype)
    y=torch.empty((2,256),device=x.device,dtype=x.dtype)
    gmm1_activation_kernel[(triton.cdiv(256,bn),2)](x,nk_weight,counts,raw,y,
                     5120,256,count_type,float(limit),bn,bk,enable_fp_fusion=False)
    return raw,y
