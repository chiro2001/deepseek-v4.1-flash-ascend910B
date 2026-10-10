"""Experimental native-Div sigmoid; formal deployment remains gated.

The installed HC SigmoidPerf uses Exp/Adds/Div, rather than approximate
reciprocal division. All other equations follow the frozen HC baseline.
"""
import torch
import triton
import triton.language as tl
from hc_static import project_static

@triton.jit
def finish_hc_divrn(x,dots,squares,scale,base,pmix,y,post,comb,pre,
              D:tl.constexpr,PARTS:tl.constexpr,BY:tl.constexpr,ITERS:tl.constexpr,
              HAS_MIX:tl.constexpr,NORM_EPS:tl.constexpr,HC_EPS:tl.constexpr):
    block=tl.program_id(0)
    rows=tl.arange(0,4)
    parts=tl.arange(0,PARTS)
    inv=tl.rsqrt(tl.sum(tl.load(squares+parts),0)/(4*D)+NORM_EPS)
    s0=tl.load(scale);s1=tl.load(scale+1);s2=tl.load(scale+2)
    dpre=tl.sum(tl.load(dots+rows[:,None]*PARTS+parts[None,:]),1)*inv
    p=tl.div_rn(1.0,1.0+tl.exp(-(dpre*s0+tl.load(base+rows))))+HC_EPS
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


def hc_divrn(x, rounded_fn, scale, base, pre_mix=None, *, hc_sinkhorn_iters=20,
             norm_eps=1e-20, hc_eps=1e-6):
    assert x.shape == (1,4,5120) and x.dtype == torch.bfloat16
    dots=torch.empty((24,1),dtype=torch.float32,device=x.device)
    squares=torch.empty((1,),dtype=torch.float32,device=x.device)
    y=torch.empty((1,5120),dtype=x.dtype,device=x.device)
    post=torch.empty((1,4),dtype=torch.float32,device=x.device)
    comb=torch.empty((1,4,4),dtype=torch.float32,device=x.device)
    pre=torch.empty((1,4),dtype=torch.float32,device=x.device)
    project_static[(24,1)](x,rounded_fn,dots,squares,20480,1,4096,enable_fp_fusion=False)
    finish_hc_divrn[(5,)](x,dots,squares,scale,base,pre_mix if pre_mix is not None else pre,
                         y,post,comb,pre,5120,1,1024,hc_sinkhorn_iters,pre_mix is not None,
                         norm_eps,hc_eps,enable_fp_fusion=False)
    return y,post,comb,pre
