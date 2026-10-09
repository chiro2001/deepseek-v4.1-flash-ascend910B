"""BF16 inverse RoPE into the eight-group BMM input layout."""
import torch
import triton
import triton.language as tl


@triton.jit
def pack_kernel(X,C,S,Y,T:tl.constexpr,XS0:tl.constexpr,XS1:tl.constexpr,XS2:tl.constexpr,
                CS0:tl.constexpr,CS1:tl.constexpr,SS0:tl.constexpr,SS1:tl.constexpr):
    head=tl.program_id(0)
    row=tl.program_id(1)
    col=tl.arange(0,512)
    value=tl.load(X+row*XS0+head*XS1+col*XS2).to(tl.float32)
    partner=tl.load(X+row*XS0+head*XS1+(col ^ 1)*XS2).to(tl.float32)
    rotated=col>=448
    rope_col=col & 63
    cosine=tl.load(C+row*CS0+rope_col*CS1,rotated,0).to(tl.float32)
    sine=tl.load(S+row*SS0+rope_col*SS1,rotated,0).to(tl.float32)
    sign=tl.where((col & 1)==0,1.0,-1.0)
    result=tl.where(rotated,value*cosine+partner*sine*sign,value).to(tl.bfloat16)
    offset=((head//8)*T+row)*4096+(head%8)*512+col
    tl.store(Y+offset,result)


def inverse_rope_pack(raw,cos,sin):
    assert raw.ndim==3 and raw.shape[1:]==(64,512) and raw.dtype==torch.bfloat16
    rows=raw.shape[0]
    c=cos.reshape(rows,64);s=sin.reshape(rows,64)
    result=torch.empty((8,rows,4096),device=raw.device,dtype=raw.dtype)
    if rows:
        pack_kernel[(64,rows)](raw,c,s,result,rows,*raw.stride(),*c.stride(),*s.stride(),
                              enable_fp_fusion=False)
    return result


def grouped_epilogue(raw,cos,sin,weight):
    assert weight.shape==(8,4096,512) and weight.dtype==torch.bfloat16
    prepared=inverse_rope_pack(raw,cos,sin)
    projected=torch.bmm(prepared,weight)
    return projected.transpose(0,1).reshape(raw.shape[0],4096)


@triton.jit
def inverse_kernel(X,C,S,XS0:tl.constexpr,XS1:tl.constexpr,XS2:tl.constexpr,
                   CS0:tl.constexpr,CS1:tl.constexpr,SS0:tl.constexpr,SS1:tl.constexpr):
    row=tl.program_id(1)
    head=tl.program_id(0)*4+tl.arange(0,4)
    col=tl.arange(0,64)
    offset=row*XS0+head[:,None]*XS1+(448+col[None,:])*XS2
    value=tl.load(X+offset).to(tl.float32)
    other=row*XS0+head[:,None]*XS1+(448+(col[None,:]^1))*XS2
    partner=tl.load(X+other).to(tl.float32)
    cosine=tl.load(C+row*CS0+col*CS1).to(tl.float32)
    sine=tl.load(S+row*SS0+col*SS1).to(tl.float32)
    sign=tl.where((col&1)==0,1.0,-1.0)
    result=(value*cosine[None,:]+partner*sine[None,:]*sign[None,:]).to(tl.bfloat16)
    tl.store(X+offset,result)


def inverse_rope_inplace(raw,cos,sin):
    assert raw.ndim==3 and raw.shape[1:]==(64,512) and raw.dtype==torch.bfloat16
    rows=raw.shape[0]
    c=cos.reshape(rows,64);s=sin.reshape(rows,64)
    if rows:
        inverse_kernel[(16,rows)](raw,c,s,*raw.stride(),*c.stride(),*s.stride(),enable_fp_fusion=False)
    return raw
