"""Decode-only HcPre alternative: SIMD projection then the unchanged Sinkhorn."""
import torch
import triton
import triton.language as tl


@triton.jit
def project_hf32(x,fn,dots,squares,K:tl.constexpr,PARTS:tl.constexpr,BK:tl.constexpr):
    n=tl.program_id(0)
    part=tl.program_id(1)
    k=tl.arange(0,BK)
    size=tl.cdiv(K,PARTS)
    dot=tl.full((),0,tl.float32)
    sq=tl.full((),0,tl.float32)
    for start in range(tl.cdiv(size,BK)):
        offset=part*size+start*BK+k
        valid=(offset<K)&(start*BK+k<size)
        xv=tl.load(x+offset,valid,0).to(tl.float32)
        fv=tl.load(fn+n*K+offset,valid,0)
        # Calibrated on 910_9382: 11 explicit fraction bits, halfway away
        # from zero. A TF32-style 10-bit truncation does not match native.
        xf=((xv.to(tl.int32,bitcast=True)+2048)&-4096).to(tl.float32,bitcast=True)
        ff=((fv.to(tl.int32,bitcast=True)+2048)&-4096).to(tl.float32,bitcast=True)
        dot+=tl.sum(xf*ff,0)
        if n==0:
            sq+=tl.sum(xv*xv,0)
    tl.store(dots+n*PARTS+part,dot)
    if n==0:
        tl.store(squares+part,sq)


@triton.jit
def finish_hc(x,dots,squares,scale,base,pmix,y,post,comb,pre,
              D:tl.constexpr,PARTS:tl.constexpr,BY:tl.constexpr,ITERS:tl.constexpr,
              HAS_MIX:tl.constexpr,NORM_EPS:tl.constexpr,HC_EPS:tl.constexpr):
    block=tl.program_id(0)
    rows=tl.arange(0,4)
    parts=tl.arange(0,PARTS)
    inv=tl.rsqrt(tl.sum(tl.load(squares+parts),0)/(4*D)+NORM_EPS)
    s0=tl.load(scale);s1=tl.load(scale+1);s2=tl.load(scale+2)
    dpre=tl.sum(tl.load(dots+rows[:,None]*PARTS+parts[None,:]),1)*inv
    p=1/(1+tl.exp(-(dpre*s0+tl.load(base+rows))))+HC_EPS
    if block==0:
        tl.store(pre+rows,p)
        dpost=tl.sum(tl.load(dots+(rows[:,None]+4)*PARTS+parts[None,:]),1)*inv
        q=2/(1+tl.exp(-(dpost*s1+tl.load(base+rows+4))))
        tl.store(post+rows,q)
        indices=rows[:,None]*4+rows[None,:]
        flat=tl.arange(0,16)
        mixes=tl.sum(tl.load(dots+(flat[:,None]+8)*PARTS+parts[None,:]),1)*inv
        c=tl.reshape(mixes*s2+tl.load(base+flat+8),(4,4))
        e=tl.exp(c-tl.max(c,1)[:,None])
        c=e/tl.sum(e,1)[:,None]+HC_EPS
        c=c/(tl.sum(c,0)[None,:]+HC_EPS)
        for _ in range(ITERS-1):
            c=c/(tl.sum(c,1)[:,None]+HC_EPS)
            c=c/(tl.sum(c,0)[None,:]+HC_EPS)
        tl.store(comb+indices,c)
    if HAS_MIX:
        p=tl.load(pmix+rows)
    d=block*BY+tl.arange(0,BY)
    xx=tl.load(x+rows[:,None]*D+d[None,:],d[None,:]<D,0).to(tl.float32)
    yy=tl.sum(xx*p[:,None],0)
    tl.store(y+d,yy,d<D)


def hc_vector(x,fn,scale,base,pre_mix=None,*,parts=1,bk=4096,by=1024,
              hc_sinkhorn_iters=20,norm_eps=1e-20,hc_eps=1e-6):
    assert x.shape==(1,4,5120) and fn.shape==(24,20480)
    assert x.dtype==torch.bfloat16 and fn.dtype==torch.float32
    assert all(a.is_contiguous() for a in [x,fn,scale,base])
    dots=torch.empty((24,parts),dtype=torch.float32,device=x.device)
    squares=torch.empty((parts,),dtype=torch.float32,device=x.device)
    y=torch.empty((1,5120),dtype=x.dtype,device=x.device)
    post=torch.empty((1,4),dtype=torch.float32,device=x.device)
    comb=torch.empty((1,4,4),dtype=torch.float32,device=x.device)
    pre=torch.empty((1,4),dtype=torch.float32,device=x.device)
    project_hf32[(24,parts)](x,fn,dots,squares,20480,parts,bk,enable_fp_fusion=False)
    finish_hc[(triton.cdiv(5120,by),)](x,dots,squares,scale,base,
             pre_mix if pre_mix is not None else pre,y,post,comb,pre,
             5120,parts,by,hc_sinkhorn_iters,pre_mix is not None,norm_eps,hc_eps,
             enable_fp_fusion=False)
    return y,post,comb,pre
