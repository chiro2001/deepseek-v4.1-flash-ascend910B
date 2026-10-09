"""Reversible process-local candidates and independent decode graph banks."""
import torch
import torch_npu
import os
import inspect
from router_gemv import vector
from hc_vector import hc_vector

ORIGINAL_LINEAR = torch.nn.functional.linear
ARM = 'native'
ND_WEIGHTS = {}
ROUTER_REFS = {}
HC_REFS = {}
GRAPH_BANKS = {}
WRAPPERS = None
ROUTER_INSTALLED = False
HC_INSTALLED = False


def nd_weight(weight):
    key = weight.data_ptr()
    if key not in ND_WEIGHTS:
        # torch.is_contiguous does not identify Ascend FRACTAL_NZ storage.
        ND_WEIGHTS[key] = weight if torch_npu.get_npu_format(weight) == 2 else torch_npu.npu_format_cast(weight,2)
    return ND_WEIGHTS[key]


@torch.library.custom_op('tinyperf::router',mutates_args=())
def router_op(x:torch.Tensor,weight:torch.Tensor)->torch.Tensor:
    if x.shape == (1,5120) and ARM in ['router','both']:
        result = vector(x,nd_weight(weight))
    else:
        result = ORIGINAL_LINEAR(x,weight)
    from vllm.forward_context import get_forward_context
    ctx = get_forward_context()
    if ctx.capturing and x.shape == (1,5120):
        ROUTER_REFS.setdefault(ARM,[]).append((x,weight,result))
    return result


@router_op.register_fake
def router_fake(x,weight):
    return torch.empty((*x.shape[:-1],weight.shape[0]),dtype=weight.dtype,device=x.device)


def install_router():
    global ROUTER_INSTALLED
    if ROUTER_INSTALLED:return
    def dispatch(x,weight,bias=None):
        if (bias is None and weight.shape == (8,5120)
                and x.dtype == weight.dtype == torch.float32 and x.shape[-1] == 5120):
            return router_op(x,weight)
        return ORIGINAL_LINEAR(x,weight,bias)
    torch.nn.functional.linear = dispatch
    ROUTER_INSTALLED=True


@torch.library.custom_op('tinyperf::hc_pre',mutates_args=())
def hc_op(x:torch.Tensor,fn:torch.Tensor,scale:torch.Tensor,base:torch.Tensor,
          pmix:torch.Tensor|None,iters:int,norm_eps:float,hc_eps:float
          )->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    if ARM in ['hc','both'] and x.numel()==20480:
        output=compute_hc_candidate(x,fn,scale,base,pmix,iters,norm_eps,hc_eps)
    else:
        output=torch.ops._C_ascend.npu_hc_pre_v2(x,fn,scale,base,pmix,
                  hc_mult=4,hc_sinkhorn_iters=iters,norm_eps=norm_eps,hc_eps=hc_eps)
    from vllm.forward_context import get_forward_context
    if get_forward_context().capturing and x.numel()==20480:
        HC_REFS.setdefault(ARM,[]).append((x,fn,scale,base,pmix,output))
    return output


@hc_op.register_fake
def hc_fake(x,fn,scale,base,pmix,iters,norm_eps,hc_eps):
    batch=x.shape[:-2];device=x.device
    return (torch.empty((*batch,5120),dtype=x.dtype,device=device),
            torch.empty((*batch,4),dtype=torch.float32,device=device),
            torch.empty((*batch,4,4),dtype=torch.float32,device=device),
            torch.empty((*batch,4),dtype=torch.float32,device=device))


def compute_hc_candidate(x,fn,scale,base,pmix,iters=20,norm_eps=1e-20,hc_eps=1e-6):
    batch=x.shape[:-2]
    output=hc_vector(x.reshape(1,4,5120),nd_weight(fn),scale,base,
                     pmix.reshape(1,4) if pmix is not None else None,
                     hc_sinkhorn_iters=iters,norm_eps=norm_eps,hc_eps=hc_eps)
    return (output[0].reshape(*batch,5120),output[1].reshape(*batch,4),
            output[2].reshape(*batch,4,4),output[3].reshape(*batch,4))


def install_hc(model):
    global HC_INSTALLED
    if HC_INSTALLED:return
    classes={type(m) for m in model.modules() if hasattr(m,'hc_pre') and hasattr(m,'hc_mult')}
    assert classes,'No decoder class with hc_pre found'
    for cls in classes:
        original=cls.hc_pre
        has_mix='pre_mix' in inspect.signature(original).parameters
        print('HC_PATCH_CLASS',cls.__module__,cls.__name__,str(inspect.signature(original)),flush=True)
        def make_dispatch(original,has_mix):
            def dispatch(self,x,fn,scale,base,pre_mix=None):
                if self.hc_mult==4 and x.shape[-2:]==(4,5120) and fn.shape==(24,20480):
                    output=hc_op(x,fn,scale,base,pre_mix,self.hc_sinkhorn_iters,self.norm_eps,self.hc_eps)
                    return output if has_mix else output[:3]
                return original(self,x,fn,scale,base,pre_mix) if has_mix else original(self,x,fn,scale,base)
            return dispatch
        cls.hc_pre=make_dispatch(original,has_mix)
    HC_INSTALLED=True


def actual(worker):
    return getattr(worker,'worker',worker)


def save_graph_bank(worker,name):
    global WRAPPERS
    from vllm_ascend.compilation import acl_graph as ag
    if WRAPPERS is None:
        WRAPPERS = list(ag._acl_graph_wrappers)
    entries = [(w,w.concrete_aclgraph_entries) for w in WRAPPERS]
    count = sum(len(e) for _,e in entries)
    if count != 1:
        raise RuntimeError(f'Expected exactly one full decode graph, got {count}')
    GRAPH_BANKS[name] = (entries,ag._graph_params)
    assert len(ROUTER_REFS.get(name,[]))==40
    if os.getenv('TINY_PERF_HC_ENABLE')=='1':assert len(HC_REFS.get(name,[]))==80
    return {'bank':name,'graphs':count,'router_refs':len(ROUTER_REFS.get(name,[])),
            'hc_refs':len(HC_REFS.get(name,[])),
            'hc_shapes':sorted(set(str(tuple(v[0].shape)) for v in HC_REFS.get(name,[])))}


def create_bank(worker,name):
    global ARM
    from vllm_ascend.compilation import acl_graph as ag
    worker=actual(worker)
    torch.npu.synchronize()
    ARM=name
    # Prepare NDA weights and Triton compilation before capture.
    if name in ['router','both']:
        for x,weight,_ in ROUTER_REFS['native']:
            vector(x,nd_weight(weight))
    if name in ['hc','both']:
        for x,fn,scale,base,pmix,_ in HC_REFS['native']:
            compute_hc_candidate(x,fn,scale,base,pmix)
    torch.npu.synchronize()
    for wrapper in WRAPPERS:
        wrapper.concrete_aclgraph_entries={}
    ag._graph_params=None
    ag.set_graph_params([1])
    worker.model_runner.capture_model()
    return save_graph_bank(worker,name)


def switch_bank(worker,name):
    global ARM
    from vllm_ascend.compilation import acl_graph as ag
    torch.npu.synchronize()
    entries,params=GRAPH_BANKS[name]
    for wrapper,entries_for_wrapper in entries:
        wrapper.concrete_aclgraph_entries=entries_for_wrapper
    ag._graph_params=params
    ARM=name
    return {'bank':name}


def audit_router(worker,name):
    torch.npu.synchronize()
    refs=ROUTER_REFS[name]
    max_abs=0.0
    matches=0
    max_graph_abs=0.0
    weight_stds=[]
    for x,weight,stored in refs:
        weight_stds.append(weight.std().item())
        expected=ORIGINAL_LINEAR(x,weight)
        candidate=vector(x,nd_weight(weight))
        torch.testing.assert_close(candidate,expected,rtol=1e-4,atol=2e-5)
        max_abs=max(max_abs,(candidate-expected).abs().max().item())
        matches += int(torch.equal(candidate.topk(2).indices,expected.topk(2).indices))
        chosen=candidate if name in ['router','both'] else expected
        torch.testing.assert_close(stored,chosen,rtol=1e-4,atol=2e-5)
        max_graph_abs=max(max_graph_abs,(stored-chosen).abs().max().item())
    assert len(refs)==40,(name,len(refs))
    assert matches==40,(name,matches)
    if os.getenv('TINY_PERF_RANDOM_VALIDATION')=='1':assert min(weight_stds)>1e-3,weight_stds
    return {'layers':len(refs),'raw_top2_agreement':matches,'max_abs_score_error':max_abs,
            'max_graph_score_error':max_graph_abs,'min_weight_std':min(weight_stds),
            'max_weight_std':max(weight_stds)}


@torch.inference_mode()
def randomize_validation_weights(worker):
    torch.manual_seed(20261009)
    counts={'gate':0,'hc_fn':0,'hc_scale':0,'hc_base':0}
    for name,param in actual(worker).model_runner.model.named_parameters():
        local=name.rsplit('.',1)[-1]
        if name.endswith('.gate.weight') and param.shape==(8,5120):
            param.copy_(torch.randn_like(param)/5120**.5);counts['gate']+=1
        elif 'hc' in local and local.endswith('_fn') and param.shape==(24,20480):
            param.copy_(torch.randn_like(param)*.1/20480**.5);counts['hc_fn']+=1
        elif 'hc' in local and local.endswith('_scale') and param.shape==(3,):
            param.fill_(1.);counts['hc_scale']+=1
        elif 'hc' in local and local.endswith('_base') and param.shape==(24,):
            param.copy_(torch.randn_like(param)*.1);counts['hc_base']+=1
    assert counts=={'gate':40,'hc_fn':80,'hc_scale':80,'hc_base':80},counts
    ND_WEIGHTS.clear()
    for module in actual(worker).model_runner.model.modules():
        if getattr(module,'precast_fp32_weight',False) and hasattr(module,'weight_fp32'):
            module.weight_fp32.copy_(module.weight.float())
    torch.npu.synchronize()
    return counts


def audit_hc(worker,name):
    torch.npu.synchronize()
    refs=HC_REFS[name]
    assert len(refs)==80,(name,len(refs))
    worst=[0.,0.,0.,0.]
    fn_stds=[]
    for x,fn,scale,base,pmix,_ in refs:
        fn_stds.append(fn.std().item())
        native=torch.ops._C_ascend.npu_hc_pre_v2(x,fn,scale,base,pmix,
                    hc_mult=4,hc_sinkhorn_iters=20,norm_eps=1e-20,hc_eps=1e-6)
        candidate=compute_hc_candidate(x,fn,scale,base,pmix)
        for i,(a,b) in enumerate(zip(candidate,native)):
            torch.testing.assert_close(a.float(),b.float(),rtol=4e-3 if i==0 else 1e-4,
                                       atol=4e-3 if i==0 else 1e-4)
            worst[i]=max(worst[i],(a.float()-b.float()).abs().max().item())
    if os.getenv('TINY_PERF_RANDOM_VALIDATION')=='1':assert min(fn_stds)>1e-4,fn_stds
    return {'calls':len(refs),'max_abs_errors':worst,'min_fn_std':min(fn_stds),'max_fn_std':max(fn_stds)}
