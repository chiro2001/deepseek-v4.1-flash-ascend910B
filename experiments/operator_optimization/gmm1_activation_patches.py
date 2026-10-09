"""Hook the routed MLP boundary while keeping general/prefill paths intact."""
import sys
import os

import torch
import torch_npu

import activation_patches as activation
import goal20_patches as goal
from gmm1_activation import gmm1_activation

FUSED_REFS={}
BN=int(os.getenv('GMM_ACT_BN','16'))
BK=int(os.getenv('GMM_ACT_BK','512'))


@torch.library.custom_op('goal20::gmm1_activation',mutates_args=())
def fused_op(x:torch.Tensor,w:torch.Tensor,counts:torch.Tensor,count_type:int,limit:float)->tuple[torch.Tensor,torch.Tensor]:
    selected=(goal.enabled('gmm1act') and x.shape==(2,5120) and x.is_contiguous()
              and torch_npu.get_npu_format(x)==2)
    if selected:
        raw,output=gmm1_activation(x,goal.cached_weight(w,'nk'),counts,limit,
                                  count_type=count_type,bn=BN,bk=BK)
        goal.remember('gmm1',(x,w,counts,count_type),raw,True)
        activation.remember('routed',raw,(limit,),output,True)
        from vllm.forward_context import get_forward_context
        if get_forward_context().capturing:
            FUSED_REFS.setdefault(goal.ARM,[]).append((x,w,counts,count_type,limit,raw,output))
    else:
        # Opaque batch dispatch prevents a prefill example freezing the branch.
        raw=goal.gmm_op(x,w,counts,count_type) if x.shape==(2,5120) else goal.ORIGINAL_GMM(
            [x],[w],group_list=counts,group_type=0,group_list_type=count_type,split_item=2)[0]
        output=activation.routed_op(raw,limit)
    return raw,output


@fused_op.register_fake
def fused_fake(x,w,counts,count_type,limit):
    return (torch.empty((x.shape[0],512),device=x.device,dtype=x.dtype),
            torch.empty((x.shape[0],256),device=x.device,dtype=x.dtype))


def install(model):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_ascend.ops.fused_moe import moe_mlp
    from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
    original=moe_mlp.apply_moe_mlp
    def dispatch(inp,quant_method):
        layer=inp.layer
        if (isinstance(quant_method,AscendUnquantizedFusedMoEMethod)
                and not inp.quant.is_quant and inp.activation==MoEActivation.SILU
                and inp.swiglu_limit>0 and inp.hidden_states.dtype==torch.bfloat16
                and inp.hidden_states.shape[-1]==5120 and inp.group_list.shape==(8,)
                and inp.group_list_type in [0,1] and layer is not None
                and not quant_method.moe.has_bias and inp.lora_context is None
                and not inp.dynamic_eplb):
            w,_=quant_method.get_mlp_weights(layer)
            w=quant_method._maybe_transpose(w,inp.need_trans)
            if isinstance(w,torch.Tensor) and w.shape==(8,5120,512):
                quant_method._lora_routing=None
                _,hidden=fused_op(inp.hidden_states,w,inp.group_list,inp.group_list_type,float(inp.swiglu_limit))
                hidden,scale=quant_method.apply_act_quant(inp,hidden)
                event=torch.npu.current_stream().record_event()
                hidden=quant_method.apply_gmm2(inp,hidden,scale)
                return hidden,event
        return original(inp,quant_method)
    patched=[]
    for name,module in list(sys.modules.items()):
        if name.startswith('vllm_ascend.ops.fused_moe') and module is not None:
            if getattr(module,'apply_moe_mlp',None) is original:
                module.apply_moe_mlp=dispatch;patched.append(name)
    assert 'vllm_ascend.ops.fused_moe.moe_mlp' in patched,patched
    goal.CONFIGS['gmmact']={'hcstatic','hcpost','route','gmm1','gmm1act'}
    old_warm=goal.warm
    def warm(worker):
        result=old_warm(worker)
        counts=torch.tensor([1,1,0,0,0,0,0,0],device='npu',dtype=torch.int64)
        for args,_,_ in goal.REFS['baseline']['gmm1']:
            gmm1_activation(args[0],goal.cached_weight(args[1],'nk'),counts,7.,bn=BN,bk=BK)
        torch.npu.synchronize();return result
    goal.warm=warm
    old_save=goal.save_bank
    def save(worker,name):
        result=old_save(worker,name)
        fused=len(FUSED_REFS.get(name,[]))
        assert fused==(40 if goal.enabled('gmm1act') else 0),(name,fused)
        result['fused_gmm1_activation_calls']=fused
        return result
    goal.save_bank=save
    old_audit=goal.audit
    def audit(worker,name):
        result=old_audit(worker,name)
        if name in FUSED_REFS:
            from verify_activation import compare,native_routed
            errors=[]
            native_errors=[]
            for x,w,counts,count_type,limit,_,output in FUSED_REFS[name]:
                from selected_gemv import gemv
                from clamped_swiglu import clamped_swiglu
                raw=goal.ORIGINAL_GMM([x],[w],group_list=counts,group_type=0,
                                     group_list_type=count_type,split_item=2)[0]
                expected=native_routed(raw,limit)
                raw_proven=gemv(x,goal.cached_weight(w,'nk'),counts=counts,count_type=count_type,
                               kind='vector',bn=8,bk=512,nk_layout=True)
                proven=clamped_swiglu(raw_proven,limit,mutate_input=True)
                # This fusion replaces the already validated Vector path.
                # Existing GMM-vs-native and same-input activation gates above
                # remain unchanged; the composed fusion gate stays <=1 ULP.
                metric=compare(output,proven)
                assert metric['passed'],metric
                errors.append(metric)
                native_errors.append({'fused':compare(output,expected),'proven':compare(proven,expected)})
            result['fused_gmm1_activation']={'calls':len(errors),'reference':'proven Vector GEMV + clamped SwiGLU',
                'max_bf16_ulp':max(x['max_bf16_ulp'] for x in errors),
                'fused_native_max_ulp':max(x['fused']['max_bf16_ulp'] for x in native_errors),
                'proven_native_max_ulp':max(x['proven']['max_bf16_ulp'] for x in native_errors)}
        return result
    goal.audit=audit
    print('FUSED_GMM_ACT_INSTALL',patched,flush=True)
