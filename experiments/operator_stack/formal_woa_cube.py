"""Formal single-group wo_a Cube candidate with capture/consumer receipts."""
import hashlib
import inspect
import os
from pathlib import Path

import torch
from selected_gemv import gemv

ARM='tp8base'
INSTALLED=False
ORIGINAL=torch.matmul
WEIGHTS={}
REFS={}
SOURCE={}


def set_arm(name):
    global ARM
    ARM=name


@torch.library.custom_op('formal_woa::matmul',mutates_args=())
def matmul_op(x:torch.Tensor,w:torch.Tensor)->torch.Tensor:
    selected=ARM=='tp8woa'
    result=gemv(x,w,kind='cube',bn=64,bk=256) if selected else ORIGINAL(x,w)
    from vllm.forward_context import get_forward_context
    if get_forward_context().capturing:
        audit=os.getenv('STACK_REAL_AUDIT')=='1'
        layer=WEIGHTS[w.data_ptr()][1]
        REFS.setdefault(ARM,[]).append((x.clone() if audit else x,w,result.clone() if audit else result,selected,layer))
    return result


@matmul_op.register_fake
def matmul_fake(x,w):
    return torch.empty((x.shape[0],w.shape[1]),device=x.device,dtype=x.dtype)


def install(model):
    global INSTALLED
    if INSTALLED or os.getenv('STACK_WOA_CUBE_ENABLED')!='1':return
    assert os.getenv('STACK_REAL_WEIGHTS')=='1'
    for name,module in model.named_modules():
        if name.endswith('.self_attn.wo_a'):
            w=module.weight
            assert w.shape==(1,4096,1024) and w.dtype==torch.bfloat16 and w.is_contiguous(),(name,w.shape,w.stride())
            WEIGHTS[w.data_ptr()]=(w,name,w._version)
    assert len(WEIGHTS)==40,len(WEIGHTS)
    SOURCE['torch_matmul_module']=ORIGINAL.__module__
    SOURCE['candidate_sha256']=hashlib.sha256(inspect.getsource(gemv).encode()).hexdigest()
    SOURCE['kernel_file_sha256']=hashlib.sha256(Path(inspect.getfile(gemv)).read_bytes()).hexdigest()

    def dispatch(x,w,*args,**kwargs):
        entry=WEIGHTS.get(w.data_ptr()) if isinstance(w,torch.Tensor) else None
        eligible=(entry is not None and not args and not kwargs and x.shape==(1,4096) and
                  w.shape==(4096,1024) and x.dtype==w.dtype==torch.bfloat16 and
                  x.is_contiguous() and w.is_contiguous() and entry[0]._version==entry[2])
        if eligible:return matmul_op(x,w)
        return ORIGINAL(x,w,*args,**kwargs)

    torch.matmul=dispatch;INSTALLED=True


def reset(name):REFS[name]=[]


@torch.inference_mode()
def warm():
    for w,name,version in WEIGHTS.values():
        x=torch.zeros((1,4096),device=w.device,dtype=w.dtype)
        gemv(x,w.squeeze(0),kind='cube',bn=64,bk=256)
    torch.npu.synchronize()


def status(name):
    rows=REFS.get(name,[])
    return {'enabled':INSTALLED,'arm':name,'registered_layers':len(WEIGHTS),
            'captured_calls':len(rows),'selected_calls':sum(r[3] for r in rows),
            'captured_unique_layers':len({r[4] for r in rows}),'source':SOURCE}


@torch.inference_mode()
def audit(name):
    rows=REFS.get(name,[])
    assert len(rows)==40 and all(r[3] for r in rows) and len({r[4] for r in rows})==40,('Missing wo_a coverage',status(name))
    errors=[]
    for x,w,actual,selected,layer in rows:
        expected=ORIGINAL(x,w)
        bits_a=actual.view(torch.int16).to(torch.int32)&65535
        bits_b=expected.view(torch.int16).to(torch.int32)&65535
        order_a=torch.where((bits_a&32768)!=0,32768-(bits_a&32767),32768+bits_a)
        order_b=torch.where((bits_b&32768)!=0,32768-(bits_b&32767),32768+bits_b)
        ulp=(order_a-order_b).abs()
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        assert int(ulp.max())<=1,('Real consumer exceeds original BF16 gate',layer,int(ulp.max()))
        errors.append(int(ulp.max()))
    return {'passed':True,'consumer_calls':len(rows),'max_bf16_ulp':max(errors),
            'scope':'Captured actual40-layer wo_a inputs; full routes/token/logprob gates remain separate'}
