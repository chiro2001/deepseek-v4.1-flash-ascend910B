"""Diagnostic native HC/router repeats on real decode inputs; no new math."""
import inspect
import torch

RECORDS = []
NAMES = {}
SEEN = set()
ORIGINALS = {}
ENABLED = False
ENGRAM_GRAPHS = []
ROUTER_WEIGHTS = {}
ROUTER_OWNER_STACK = []
ROUTER_OWNER_HANDLES = []


def disable_engram_subgraph(worker):
    worker = getattr(worker,'worker',worker)
    assert worker.vllm_config.model_config.enforce_eager
    torch.npu.synchronize()
    layers = []
    for name,module in worker.model_runner.model.named_modules():
        if getattr(module,'_engram_dev_ready',False) and getattr(module,'_engram_dev_graph',None) is not None:
            # Retain the old graph until the diagnosis ends; direct lookup
            # remains the production implementation using the same tables.
            ENGRAM_GRAPHS.append(module._engram_dev_graph)
            module._engram_dev_graph = None
            layers.append(name)
    assert len(layers)==1,layers
    return {'disabled_modules':layers,'engram_enabled':True,'tables_reused':True,
            'note':'Only Engram subgraph replay disabled; hash/gather/dequant stay enabled'}


def eligible(x):
    if not ENABLED or not isinstance(x,torch.Tensor) or x.ndim == 0 or x.shape[0] != 1:
        return False
    from vllm.forward_context import get_forward_context
    ctx = get_forward_context()
    return not ctx.capturing and ctx.attn_metadata is not None


def tensors(output):
    return tuple(v.clone() for v in (output if isinstance(output,(tuple,list)) else (output,))
                 if isinstance(v,torch.Tensor))


def install(worker):
    global ENABLED
    worker = getattr(worker,'worker',worker)
    assert worker.vllm_config.model_config.enforce_eager
    model = worker.model_runner.model
    for name,module in model.named_modules():
        NAMES[id(module)] = name
        if getattr(module,'is_internal_router',False):
            def bind_owner(name):
                def before(module,inputs):
                    if ENABLED: ROUTER_OWNER_STACK.append((name,module))
                def after(module,inputs,output):
                    if ROUTER_OWNER_STACK and ROUTER_OWNER_STACK[-1][1] is module:
                        ROUTER_OWNER_STACK.pop()
                return before,after
            before,after=bind_owner(name)
            ROUTER_OWNER_HANDLES.extend([module.register_forward_pre_hook(before),
                module.register_forward_hook(after,always_call=True)])
        if hasattr(module,'hc_pre') and getattr(module,'hc_mult',None) == 4:
            cls = type(module)
            if cls not in ORIGINALS:
                original = cls.hc_pre
                has_mix = 'pre_mix' in inspect.signature(original).parameters
                ORIGINALS[cls] = original
                def bind(original,has_mix):
                    def wrapped(self,x,fn,scale,base,pre_mix=None):
                        key = ('hc',id(self),fn.data_ptr())
                        selected = eligible(x) and key not in SEEN
                        saved = x.clone() if selected else None
                        mix = pre_mix.clone() if selected and pre_mix is not None else None
                        out = original(self,x,fn,scale,base,pre_mix) if has_mix else original(self,x,fn,scale,base)
                        if selected:
                            SEEN.add(key)
                            RECORDS.append(('hc',NAMES[id(self)],self,original,has_mix,
                                            (saved,fn,scale,base,mix),tensors(out)))
                        return out
                    return wrapped
                cls.hc_pre = bind(original,has_mix)
        weight = getattr(module,'weight',None)
        if isinstance(weight,torch.Tensor) and tuple(weight.shape) == (384,5120):
            ROUTER_WEIGHTS[weight.data_ptr()] = (name,module)
    # Production MoE calls F.linear(input, self.gate.weight), bypassing the
    # gate module's forward hooks. Observe that exact call and retain its
    # native callable for replay rather than introducing a different path.
    import torch.nn.functional as functional
    original_linear = functional.linear
    def linear(x,weight,bias=None):
        owner = ROUTER_WEIGHTS.get(weight.data_ptr())
        if owner is None and tuple(weight.shape)==(384,5120) and ROUTER_OWNER_STACK:
            # Internal routing may use weight_fp32 or a fresh .to(float32)
            # tensor; the expert module establishes its real owner.
            owner=ROUTER_OWNER_STACK[-1]
        key = ('router',id(owner[1])) if owner is not None else None
        selected = owner is not None and eligible(x) and key not in SEEN
        saved = x.clone() if selected else None
        output = original_linear(x,weight,bias)
        if selected:
            SEEN.add(key)
            name,module = owner
            RECORDS.append(('router',name,module,original_linear,None,
                            (saved,weight,bias),tensors(output)))
        return output
    functional.linear = linear
    ENABLED = True
    return {'native_hc_classes':len(ORIGINALS),'router_weights':len(ROUTER_WEIGHTS),
            'internal_router_owners':len(ROUTER_OWNER_HANDLES)//2,
            'scope':'eager first decode; retain actual inputs, no D2H during model forward'}


@torch.inference_mode()
def repeat(worker,repeats=5):
    global ENABLED
    worker = getattr(worker,'worker',worker)
    from vllm.forward_context import set_forward_context
    ENABLED = False
    torch.npu.synchronize()
    assert repeats >= 2 and RECORDS
    result = []
    for kind,name,module,original,has_mix,args,observed in RECORDS:
        x = args[0].clone()
        mix = args[-1].clone() if kind=='hc' and args[-1] is not None else None
        reference = None; equal = True; observed_equal = True; max_abs = 0.0
        for index in range(repeats):
            x.copy_(args[0])
            with set_forward_context(None,worker.vllm_config,num_tokens=1):
                if kind=='hc':
                    if mix is not None: mix.copy_(args[-1])
                    output = (original(module,x,*args[1:4],mix) if has_mix
                              else original(module,x,*args[1:4]))
                else:
                    output = original(x,*args[1:])
            output = tensors(output)
            if reference is None: reference = output
            assert len(output)==len(reference)==len(observed)
            for a,b,c in zip(output,reference,observed):
                equal = equal and torch.equal(a,b)
                observed_equal = observed_equal and torch.equal(a,c)
                max_abs = max(max_abs,float((a.float()-b.float()).abs().max().item()))
        result.append({'kind':kind,'module':name,'input_shape':list(args[0].shape),
                       'input_dtype':str(args[0].dtype),'repeats':repeats,
                       'identical_input_repeats_equal':equal,
                       'repeats_equal_observed_forward':observed_equal,'max_repeat_abs':max_abs})
    from vllm.distributed import get_tensor_model_parallel_rank
    counts = {kind:sum(r['kind']==kind for r in result) for kind in ('hc','router')}
    return {'rank':get_tensor_model_parallel_rank(),'records':result,'coverage':counts,
            'expected_coverage_reached':counts=={'hc':80,'router':40},
            'scope':'native HC/router only; cannot prove whole-model stability or timing'}
