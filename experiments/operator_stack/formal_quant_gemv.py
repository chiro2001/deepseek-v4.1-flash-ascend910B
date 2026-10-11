"""INT8 single-token projection candidates with explicit dequantization order."""
import torch
import triton
import triton.language as tl


@triton.jit
def _cube(x,w,ws,xs,y,K:tl.constexpr,N:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
          NZ:tl.constexpr,ORDER:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    m=tl.arange(0,16)
    k=tl.arange(0,BK)
    acc=tl.zeros((16,BN),tl.int32)
    for start in range(tl.cdiv(K,BK)):
        kk=start*BK+k
        a=tl.load(x+m[:,None]*K+kk[None,:],(m[:,None]==0)&(kk[None,:]<K),other=0)
        if NZ:
            offset=((n[None,:]//32)*(K//16)+kk[:,None]//16)*512+(kk[:,None]%16)*32+n[None,:]%32
        else:
            offset=kk[:,None]*N+n[None,:]
        b=tl.load(w+offset,(kk[:,None]<K)&(n[None,:]<N),other=0)
        acc=tl.dot(a,b,acc)
    weights=tl.load(ws+n,n<N,other=0).to(tl.float32)
    token=tl.load(xs).to(tl.float32)
    value=acc.to(tl.float32)
    if ORDER==0:
        value=(value*weights[None,:])*token
    elif ORDER==1:
        value=value*(weights[None,:]*token)
    else:
        value=(value*token)*weights[None,:]
    tl.store(y+m[:,None]*N+n[None,:],value,(m[:,None]==0)&(n[None,:]<N))


def quant_gemv(x,w,weight_scale,token_scale,*,nz=False,bn=64,bk=256,order=0):
    assert x.ndim==2 and x.shape[0]==1 and x.dtype==w.dtype==torch.int8
    assert x.is_contiguous() and weight_scale.is_contiguous() and token_scale.is_contiguous()
    k=x.shape[1];assert w.shape[0]==k and w.ndim==2
    n=w.shape[1]
    assert weight_scale.shape==(n,) and weight_scale.dtype==torch.bfloat16
    assert token_scale.shape==(1,) and token_scale.dtype==torch.float32
    assert order in (0,1,2) and k%16==0 and n%32==0
    y=torch.empty((1,n),device=x.device,dtype=torch.bfloat16)
    _cube[(triton.cdiv(n,bn),)](x,w,weight_scale,token_scale,y,k,n,bn,bk,nz,order,
                              enable_fp_fusion=False)
    return y
