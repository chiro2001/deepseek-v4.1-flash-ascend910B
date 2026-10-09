"""Decode q_b packed weight panels and joint native QA/KV projection."""
import torch
import torch_npu
import triton
import triton.language as tl

PACKED={}
JOINT={}


def nz_pack(weight):
    previous=torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT')
    if isinstance(previous,bytes):previous=previous.decode()
    previous=previous or 'disable'
    assert previous in ['enable','disable'],previous
    try:
        torch.npu.config.allow_internal_format=True
        result=torch_npu.npu_format_cast(weight,29)
    finally:
        torch_npu._C._npu_setOption({'ALLOW_INTERNAL_FORMAT':previous})
    assert torch_npu.get_npu_format(result)==29
    return result


def matmul_validation(actual,expected):
    a=actual.float();b=expected.float();difference=(a-b).abs()
    peak=float(b.abs().max().item())
    rms=float(b.square().mean().sqrt().item())
    error=float(difference.square().mean().sqrt().item())
    finite=bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item())
    close=finite and bool(torch.allclose(a,b,rtol=1/128,atol=max(peak*1e-6,1e-30)))
    relative_rms=error/max(rms,1e-30)
    return {'equal':torch.equal(actual,expected),'different':int((actual!=expected).sum().item()),
            'max_abs':float(difference.max().item()),'rms_error':error,
            'relative_rms_error':relative_rms,'bf16_close':close and relative_rms<=1e-3}


def key(weight):
    try:version=weight._version
    except RuntimeError:version=None  # inference weights require explicit invalidation
    return (weight.data_ptr(),tuple(weight.shape),weight.dtype,weight.device,version)


def invalidate():
    PACKED.clear();JOINT.clear()


def pack_weight(weight):
    assert weight.shape==(32768,512) and weight.dtype==torch.bfloat16
    cache_key=key(weight)
    if cache_key not in PACKED:
        nd=weight if torch_npu.get_npu_format(weight)==2 else torch_npu.npu_format_cast(weight,2)
        PACKED[cache_key]=nd.t().reshape(4,128,256,128).permute(2,0,1,3).contiguous()
    return PACKED[cache_key]


def joint_weight(qa,kv):
    assert qa.shape==kv.shape==(512,5120) and qa.dtype==kv.dtype==torch.bfloat16
    cache_key=(key(qa),key(kv))
    if cache_key not in JOINT:
        JOINT[cache_key]=torch.cat((qa,kv),dim=0).contiguous()
    return JOINT[cache_key]


@triton.jit
def panel_kernel(X,P,Y,T:tl.constexpr,XS0:tl.constexpr,XS1:tl.constexpr):
    tile=tl.program_id(0)
    rows=tl.arange(0,16)
    cols=tl.arange(0,128)
    reduction=tl.arange(0,128)
    accumulator=tl.zeros((16,128),tl.float32)
    for part in range(4):
        a=tl.load(X+rows[:,None]*XS0+(part*128+reduction[None,:])*XS1,
                  rows[:,None]<T,0)
        b=tl.load(P+tile*65536+part*16384+reduction[:,None]*128+cols[None,:])
        accumulator=tl.dot(a,b,accumulator)
    tl.store(Y+rows[:,None]*32768+tile*128+cols[None,:],
             accumulator.to(tl.bfloat16),rows[:,None]<T)


@triton.jit
def prefetch_kernel(X,P,Y,T:tl.constexpr,XS0:tl.constexpr,XS1:tl.constexpr):
    tile=tl.program_id(0)
    rows=tl.arange(0,16)
    cols=tl.arange(0,128)
    reduction=tl.arange(0,128)
    a=tl.load(X+rows[:,None]*XS0+reduction[None,:]*XS1,rows[:,None]<T,0)
    b=tl.load(P+tile*65536+reduction[:,None]*128+cols[None,:])
    accumulator=tl.zeros((16,128),tl.float32)
    # Unroll the three handoffs: this Ascend toolchain cannot carry a loaded
    # NZ CBUF tensor through scf.for's ND iter_arg layout.
    for part in tl.static_range(3):
        # Keep the next A/B panel live before current Cube consumption.
        # K order and the final BF16 store remain identical to panel_kernel.
        next_a=tl.load(X+rows[:,None]*XS0+((part+1)*128+reduction[None,:])*XS1,
                       rows[:,None]<T,0)
        next_b=tl.load(P+tile*65536+(part+1)*16384+reduction[:,None]*128+cols[None,:])
        accumulator=tl.dot(a,b,accumulator)
        a=next_a;b=next_b
    accumulator=tl.dot(a,b,accumulator)
    tl.store(Y+rows[:,None]*32768+tile*128+cols[None,:],
             accumulator.to(tl.bfloat16),rows[:,None]<T)


def q_b_panel(query,packed,*,prefetch=False):
    assert query.ndim==2 and query.shape[1]==512 and query.dtype==torch.bfloat16
    assert 0<query.shape[0]<=4 and packed.shape==(256,4,128,128) and packed.is_contiguous()
    if torch_npu.get_npu_format(query)!=2:
        query=torch_npu.npu_format_cast(query,2)
    output=torch.empty((query.shape[0],32768),dtype=query.dtype,device=query.device)
    if prefetch:
        prefetch_kernel[(256,)](query,packed,output,query.shape[0],*query.stride(),
                                num_stages=2,enable_preload=True)
    else:
        panel_kernel[(256,)](query,packed,output,query.shape[0],*query.stride(),num_stages=2)
    return output
