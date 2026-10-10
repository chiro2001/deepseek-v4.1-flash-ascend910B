"""Diagnostic native HC/router repeats on real decode inputs; no new math."""
import inspect
import torch

RECORDS = []
NAMES = {}
SEEN = set()
HANDLES = []
ORIGINALS = {}
ENABLED = False
ENGRAM_GRAPHS = []


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
            pending = {}
            def bind_hooks(name,pending):
                def before(module,inputs):
                    key = ('router',id(module))
                    if inputs and eligible(inputs[0]) and key not in SEEN:
                        pending['input'] = inputs[0].clone()
                        SEEN.add(key)
                def after(module,inputs,out):
                    x = pending.pop('input',None)
                    if x is not None:
                        RECORDS.append(('router',name,module,None,None,(x,),tensors(out)))
                return before,after
            before,after = bind_hooks(name,pending)
            HANDLES.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    ENABLED = True
    return {'native_hc_classes':len(ORIGINALS),'router_hooks':len(HANDLES)//2,
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
                    output = module(x)
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
    return {'rank':get_tensor_model_parallel_rank(),'records':result,
            'scope':'native HC/router only; cannot prove whole-model stability or timing'}
