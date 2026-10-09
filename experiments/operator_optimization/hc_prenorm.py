"""Candidate HC finish + whole-vector RMS, retaining the BF16 HC boundary."""
import torch
import triton
import triton.language as tl

from hc_static import project_static


@triton.jit
def finish_prenorm(x,dots,squares,scale,base,pmix,weight,raw,post,comb,pre,
                   normalized,fp32,D:tl.constexpr,BY:tl.constexpr,
                   HAS_MIX:tl.constexpr,TREE:tl.constexpr,EMIT_FP32:tl.constexpr,
                   ITERS:tl.constexpr,NORM_EPS:tl.constexpr,HC_EPS:tl.constexpr,
                   RMS_EPS:tl.constexpr):
    rows=tl.arange(0,4)
    inv=tl.rsqrt(tl.load(squares)/(4*D)+NORM_EPS)
    s0=tl.load(scale);s1=tl.load(scale+1);s2=tl.load(scale+2)
    p=1/(1+tl.exp(-(tl.load(dots+rows)*inv*s0+tl.load(base+rows))))+HC_EPS
    tl.store(pre+rows,p)
    q=2/(1+tl.exp(-(tl.load(dots+rows+4)*inv*s1+tl.load(base+rows+4))))
    tl.store(post+rows,q)
    flat=tl.arange(0,16)
    c=tl.reshape(tl.load(dots+flat+8)*inv*s2+tl.load(base+flat+8),(4,4))
    e=tl.exp(c-tl.max(c,1)[:,None])
    c=e/tl.sum(e,1)[:,None]+HC_EPS
    c=c/(tl.sum(c,0)[None,:]+HC_EPS)
    for _ in range(ITERS-1):
        c=c/(tl.sum(c,1)[:,None]+HC_EPS)
        c=c/(tl.sum(c,0)[None,:]+HC_EPS)
    tl.store(comb+rows[:,None]*4+rows[None,:],c)
    if HAS_MIX:
        p=tl.load(pmix+rows)
    p0=tl.sum(tl.where(rows==0,p,0),0)
    p1=tl.sum(tl.where(rows==1,p,0),0)
    p2=tl.sum(tl.where(rows==2,p,0),0)
    p3=tl.sum(tl.where(rows==3,p,0),0)
    d=tl.arange(0,BY)
    mask=d<D
    # Load rows separately to avoid a 4 x BY FP32 reduction scratch allocation.
    a=tl.load(x+d,mask,0).to(tl.float32)*p0
    b=tl.load(x+D+d,mask,0).to(tl.float32)*p1
    first=a+b
    u=tl.load(x+2*D+d,mask,0).to(tl.float32)*p2
    v=tl.load(x+3*D+d,mask,0).to(tl.float32)*p3
    if TREE:
        yy=first+(u+v)
    else:
        yy=(first+u)+v
    rounded=yy.to(tl.bfloat16)
    tl.store(raw+d,rounded,mask)
    xx=rounded.to(tl.float32)
    xx=tl.where(mask,xx,0)
    if EMIT_FP32:
        # Match the current vendor RmsNormCast: scale every squared element
        # before reduction, then sqrt and reciprocal.
        scaled=(xx*xx)*(1.0/D)
        # Vendor reduction first folds repeated 64-lane vectors, then does
        # WholeReduceSum over those lanes. Test this layout without relaxing
        # the fixed FP32 routing tolerance.
        lanes=tl.reshape(scaled,(BY//64,64))
        variance=tl.sum(tl.sum(lanes,0),0)
        rms=1.0/tl.sqrt(variance+RMS_EPS)
    else:
        rms=tl.rsqrt(tl.sum(xx*xx,0)/D+RMS_EPS)
    gamma=tl.load(weight+d,mask,0).to(tl.float32)
    result=xx*rms*gamma
    result_bf16=result.to(tl.bfloat16)
    tl.store(normalized+d,result_bf16,mask)
    if EMIT_FP32:
        # Source and on-board verification show intentional widening of the
        # final rounded BF16 values for HashTopK, not the unrounded affine.
        tl.store(fp32+d,result_bf16.to(tl.float32),mask)


def hc_prenorm(x,rounded_fn,scale,base,weight,pre_mix=None,*,by=5120,tree=True,
               emit_fp32=True,rms_eps=1e-6,hc_sinkhorn_iters=20,norm_eps=1e-20,hc_eps=1e-6):
    assert x.shape==(1,4,5120) and x.dtype==torch.bfloat16
    assert rounded_fn.shape==(24,20480) and weight.shape==(5120,)
    assert by>=5120 and all(v.is_contiguous() for v in [x,rounded_fn,weight])
    dots=torch.empty((24,),device=x.device,dtype=torch.float32)
    squares=torch.empty((1,),device=x.device,dtype=torch.float32)
    raw=torch.empty((1,5120),device=x.device,dtype=x.dtype)
    post=torch.empty((1,4),device=x.device,dtype=torch.float32)
    comb=torch.empty((1,4,4),device=x.device,dtype=torch.float32)
    pre=torch.empty((1,4),device=x.device,dtype=torch.float32)
    normalized=torch.empty_like(raw)
    fp32=torch.empty((1,5120) if emit_fp32 else (0,),device=x.device,dtype=torch.float32)
    project_static[(24,1)](x,rounded_fn,dots,squares,20480,1,4096,enable_fp_fusion=False)
    finish_prenorm[(1,)](x,dots,squares,scale,base,pre_mix if pre_mix is not None else pre,
                        weight,raw,post,comb,pre,normalized,fp32,5120,by,pre_mix is not None,
                        tree,emit_fp32,hc_sinkhorn_iters,norm_eps,hc_eps,rms_eps,
                        enable_fp_fusion=False,multibuffer=False)
    return raw,post,comb,pre,normalized,fp32
