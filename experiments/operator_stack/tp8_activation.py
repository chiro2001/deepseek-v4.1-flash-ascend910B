"""Adapt the proven row-wise clamp/SwiGLU to actual TP8 decode shapes."""
import torch
import torch_npu
import goal20_patches as goal
from clamped_swiglu import clamped_swiglu

REFS = {}


def remember(kind, x, params, output, selected):
    from vllm.forward_context import get_forward_context
    if get_forward_context().capturing:
        REFS.setdefault(goal.ARM, {'routed': [], 'shared': []})[kind].append((x, params, output, selected))


@torch.library.custom_op('stack8::routed_act', mutates_args={'x'})
def routed(x: torch.Tensor, limit: float) -> torch.Tensor:
    selected = goal.enabled('activation_generic') and x.shape[0] <= 16 and x.shape[0] > 0
    if selected:
        output = clamped_swiglu(x, limit, mutate_input=True)
    else:
        gate, up = x.chunk(2, -1); gate.clamp_(max=limit); up.clamp_(min=-limit, max=limit)
        output = torch_npu.npu_swiglu(x)
    remember('routed', x, (limit,), output, selected)
    return output


@routed.register_fake
def routed_fake(x, limit):
    return torch.empty((*x.shape[:-1], x.shape[-1]//2), device=x.device, dtype=x.dtype)


@torch.library.custom_op('stack8::shared_act', mutates_args=())
def shared(x: torch.Tensor, limit: float, alpha: float, beta: float) -> torch.Tensor:
    selected = goal.enabled('activation_generic') and 0 < x.shape[0] <= 16
    if selected:
        output = clamped_swiglu(x, limit, alpha=alpha, beta=beta, staged_rounding=True)
    else:
        gate, up = x.chunk(2, -1)
        gate = gate.clamp(max=limit); up = up.clamp(min=-limit, max=limit)
        output = gate * torch.sigmoid(alpha * gate) * (up + beta)
    remember('shared', x, (limit, alpha, beta), output, selected)
    return output


@shared.register_fake
def shared_fake(x, limit, alpha, beta):
    return torch.empty((*x.shape[:-1], x.shape[-1]//2), device=x.device, dtype=x.dtype)


def install(model):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm_ascend.ops.fused_moe import moe_mlp
    original = moe_mlp._unified_apply_activation
    def dispatch(inp, hidden, quant_method):
        if (inp.activation == MoEActivation.SILU and inp.swiglu_limit > 0 and
                hidden.dtype == torch.bfloat16 and hidden.ndim == 2 and hidden.shape[-1] == 512 and
                hidden.is_contiguous() and torch_npu.get_npu_format(hidden) == 2):
            return routed(hidden, float(inp.swiglu_limit))
        return original(inp, hidden, quant_method)
    moe_mlp._unified_apply_activation = dispatch
    for module in model.modules():
        if not isinstance(module, SiluAndMulWithClamp): continue
        original_forward = module._forward_method
        def bind(layer, fallback):
            def forward(x):
                if (x.ndim == 2 and x.shape[-1] in [64, 512] and x.dtype == torch.bfloat16
                        and x.is_contiguous() and torch_npu.get_npu_format(x) == 2):
                    return shared(x, layer.swiglu_limit, layer.alpha, layer.beta)
                return fallback(x)
            return forward
        module._forward_method = bind(module, original_forward)


@torch.inference_mode()
def audit(name):
    from verify_activation import compare, native_routed, native_shared
    result = {}
    for kind, rows in REFS.get(name, {}).items():
        metrics = []
        for x, params, output, selected in rows:
            if not x.numel(): continue
            reference = native_routed(x.clone(), params[0]) if kind == 'routed' else native_shared(x, *params)
            metric = compare(output, reference); assert metric['passed'], (name, kind, metric)
            metrics.append(metric)
        result[kind] = {'calls': len(rows), 'selected': sum(r[3] for r in rows),
                        'shapes': sorted({tuple(r[0].shape) for r in rows}),
                        'max_bf16_ulp': max((m['max_bf16_ulp'] for m in metrics), default=0)}
    return result
