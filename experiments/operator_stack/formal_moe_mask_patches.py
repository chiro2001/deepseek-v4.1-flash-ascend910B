"""Exact MoE mask/cast bank; bind each captured consumer to its real layer."""
import ast
import hashlib
import inspect
import os
from pathlib import Path
import textwrap

import torch
from formal_moe_mask import local_probs

ARM='tp8base'
INSTALLED=False
REFS={}
SOURCE={}
PENDING=None
LAYERS={}


def set_arm(name):
    global ARM
    ARM=name


def native(ids,weights,first,last):
    from vllm_ascend.ops.fused_moe.token_dispatcher import _i32_scalar
    return weights.masked_fill((ids<_i32_scalar(first,ids.device))|
                               (ids>=_i32_scalar(last,ids.device)),0.)


@torch.library.custom_op('formal_mask::probs',mutates_args=())
def probs(ids:torch.Tensor,weights:torch.Tensor,first:int,last:int,selected:bool)->torch.Tensor:
    global PENDING
    result=local_probs(ids,weights,first,last,consumer_bf16=True) if selected else native(ids,weights,first,last)
    from vllm.forward_context import get_forward_context
    if get_forward_context().capturing:
        audit=os.getenv('STACK_REAL_AUDIT')=='1'
        row={'ids':ids.clone() if audit else ids,'weights':weights.clone() if audit else weights,
             'result':result.clone() if audit else result,'first':first,'last':last,
             'selected':selected,'layer':None}
        REFS.setdefault(ARM,[]).append(row)
        PENDING=row
    return result


@probs.register_fake
def probs_fake(ids,weights,first,last,selected):
    return torch.empty_like(weights,dtype=torch.bfloat16 if selected else torch.float32)


def dispatch_mask(ids,weights,first,last):
    eligible=(ids.shape==weights.shape==(1,6) and ids.dtype==torch.int32 and
              weights.dtype==torch.float32 and ids.is_contiguous() and weights.is_contiguous())
    if eligible:return probs(ids,weights,first,last,ARM=='tp8mask')
    return native(ids,weights,first,last)


def install(model):
    global INSTALLED,LAYERS
    if INSTALLED or os.getenv('STACK_MOE_MASK_ENABLED')!='1':return
    assert os.getenv('STACK_REAL_WEIGHTS')=='1'
    from vllm_ascend.ops.fused_moe import token_dispatcher as td
    from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod as W4A8
    cls=td.TokenDispatcherWithAllGather
    original=cls.token_dispatch
    raw=inspect.getsource(original)
    combine=inspect.getsource(cls.token_combine)
    assert hashlib.sha256(raw.encode()).hexdigest()==os.environ['STACK_MASK_DISPATCH_SHA256']
    assert hashlib.sha256(combine.encode()).hexdigest()==os.environ['STACK_MASK_COMBINE_SHA256']
    tree=textwrap.dedent(raw)
    expressions=[ast.get_source_segment(tree,node) for node in ast.walk(ast.parse(tree))
                 if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and
                 node.func.attr=='masked_fill' and isinstance(node.func.value,ast.Name) and
                 node.func.value.id=='topk_weights']
    assert len(expressions)==1 and '_i32_scalar(first_expert_idx' in expressions[0],'Unexpected native mask expression'
    updated=tree.replace(expressions[0],'formal_dispatch_mask(topk_ids, topk_weights, first_expert_idx, last_expert_idx)')
    namespace={**original.__globals__,'formal_dispatch_mask':dispatch_mask}
    exec(compile(updated,'<formal_mask_dispatch>','exec'),namespace)
    cls.token_dispatch=namespace['token_dispatch']
    LAYERS={id(module):name for name,module in model.named_modules()}
    original_gmm=W4A8.apply_gmm1_act_quant
    def bind_layer(self,mlp_compute_input,*args,**kwargs):
        global PENDING
        if PENDING is not None:
            PENDING['layer']=LAYERS[id(mlp_compute_input.layer)]
            PENDING=None
        return original_gmm(self,mlp_compute_input,*args,**kwargs)
    W4A8.apply_gmm1_act_quant=bind_layer
    SOURCE.update(dispatch_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                  combine_sha256=hashlib.sha256(combine.encode()).hexdigest(),
                  kernel_sha256=hashlib.sha256(Path(inspect.getfile(local_probs)).read_bytes()).hexdigest())
    inspect.signature(cls.token_dispatch).bind(object(),token_dispatch_input=object())
    inspect.signature(W4A8.apply_gmm1_act_quant).bind(object(),mlp_compute_input=object())
    INSTALLED=True


def reset(name):
    global PENDING
    REFS[name]=[];PENDING=None


@torch.inference_mode()
def warm():
    from vllm.distributed import get_tensor_model_parallel_rank
    first=get_tensor_model_parallel_rank()*48
    ids=torch.zeros((1,6),dtype=torch.int32,device='npu')
    weights=torch.ones((1,6),dtype=torch.float32,device='npu')
    local_probs(ids,weights,first,first+48,consumer_bf16=True)
    torch.npu.synchronize()


def status(name):
    rows=REFS.get(name,[])
    return {'enabled':INSTALLED,'arm':name,'captured_calls':len(rows),
            'selected_calls':sum(r['selected'] for r in rows),
            'unique_layers':len({r['layer'] for r in rows if r['layer'] is not None}),'source':SOURCE}


@torch.inference_mode()
def audit(name):
    rows=REFS.get(name,[])
    assert len(rows)==40 and all(r['selected'] for r in rows) and status(name)['unique_layers']==40,status(name)
    for row in rows:
        expected=native(row['ids'],row['weights'],row['first'],row['last']).to(torch.bfloat16)
        assert torch.equal(row['result'].view(torch.int16),expected.view(torch.int16)),row['layer']
    return {'passed':True,'consumer_calls':40,'unique_layers':40,'all_bitwise_equal':True,
            'scope':'Actual probabilities consumed by native BF16 token-unpermute; full routing/token/logprob gates separate'}
