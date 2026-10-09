"""Process-local activation dispatch and complete graph banks for matched A/B."""
import math
import os

import torch
import torch_npu

import runtime_patches as baseline
from clamped_swiglu import clamped_swiglu

ARM = os.getenv("OPT_ACT_ARM", "baseline")
REFS = {}
BANKS = {}
INSTALLED = False


def remember(kind, x, params, output, selected):
    from vllm.forward_context import get_forward_context
    if get_forward_context().capturing:
        # Metadata and tensor references only; no device reads during capture.
        REFS.setdefault(ARM, {"routed": [], "shared": []})[kind].append(
            (x, params, output, selected)
        )


@torch.library.custom_op("tinyopt::routed_activation", mutates_args={"x"})
def routed_op(x: torch.Tensor, limit: float) -> torch.Tensor:
    eligible = x.shape == (2, 512) and x.is_contiguous() and torch_npu.get_npu_format(x) == 2
    selected = eligible and ARM in ["routed", "fused", "overlap"]
    if selected:
        output = clamped_swiglu(x, limit, mutate_input=True)
    else:
        gate, up = x.chunk(2, dim=-1)
        gate.clamp_(max=limit)
        up.clamp_(min=-limit, max=limit)
        output = torch_npu.npu_swiglu(x)
    if x.shape == (2, 512):
        remember("routed", x, (limit,), output, selected)
    return output


@routed_op.register_fake
def routed_fake(x, limit):
    return torch.empty((*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device)


@torch.library.custom_op("tinyopt::shared_activation", mutates_args=())
def shared_op(x: torch.Tensor, limit: float, alpha: float, beta: float) -> torch.Tensor:
    eligible = x.shape == (1, 512) and x.is_contiguous() and torch_npu.get_npu_format(x) == 2
    selected = eligible and ARM in ["shared", "fused", "overlap"]
    if selected:
        output = clamped_swiglu(x, limit, alpha=alpha, beta=beta, staged_rounding=True)
    else:
        gate, up = x.chunk(2, dim=-1)
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        output = gate * torch.sigmoid(alpha * gate) * (up + beta)
    if x.shape == (1, 512):
        remember("shared", x, (limit, alpha, beta), output, selected)
    return output


@shared_op.register_fake
def shared_fake(x, limit, alpha, beta):
    return torch.empty((*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device)


def install(model):
    global INSTALLED
    if INSTALLED:
        return
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm_ascend.ops.fused_moe import moe_mlp

    original = moe_mlp._unified_apply_activation

    def routed_dispatch(mlp_compute_input, hidden_states, quant_method):
        if (mlp_compute_input.activation == MoEActivation.SILU
                and mlp_compute_input.swiglu_limit > 0
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.ndim == 2 and hidden_states.shape[-1] == 512):
            return routed_op(hidden_states, float(mlp_compute_input.swiglu_limit))
        return original(mlp_compute_input, hidden_states, quant_method)

    moe_mlp._unified_apply_activation = routed_dispatch
    count = 0
    for module in model.modules():
        if isinstance(module, SiluAndMulWithClamp):
            original_method = module._forward_method

            def bind(layer, fallback):
                def dispatch(x):
                    if x.ndim == 2 and x.shape[-1] == 512 and x.dtype == torch.bfloat16:
                        return shared_op(x, layer.swiglu_limit, layer.alpha, layer.beta)
                    return fallback(x)
                return dispatch

            module._forward_method = bind(module, original_method)
            count += 1
    assert count == 40, ("Expected 40 tiny shared activations", count)
    INSTALLED = True


def randomize_expert_validation(worker):
    """Randomize actual MLP parameters before capture, in addition to gate/HC."""
    torch.manual_seed(20261010)
    counts = {"routed": 0, "shared": 0}
    for name, param in worker.model_runner.model.named_parameters():
        if ".mlp." not in name or param.dtype != torch.bfloat16:
            continue
        routed = name.endswith(("w13_weight", "w2_weight"))
        shared = ".shared_experts." in name and name.endswith("weight") and param.ndim == 2
        if routed or shared:
            with torch.no_grad():
                param.copy_(torch.randn_like(param) / math.sqrt(param.shape[-1]))
            counts["routed" if routed else "shared"] += 1
    assert counts["routed"] == 80 and counts["shared"] == 80, counts
    baseline.ND_WEIGHTS.clear()
    torch.npu.synchronize()
    return counts


def save_bank(worker, name):
    from vllm_ascend.compilation import acl_graph as ag
    wrappers = list(ag._acl_graph_wrappers)
    entries = [(wrapper, wrapper.concrete_aclgraph_entries) for wrapper in wrappers]
    assert sum(len(value) for _, value in entries) == 1
    refs = REFS[name]
    assert len(refs["routed"]) == len(refs["shared"]) == 40
    selected = {kind: sum(item[3] for item in values) for kind, values in refs.items()}
    assert selected["routed"] == (40 if name in ["routed", "fused", "overlap"] else 0), selected
    assert selected["shared"] == (40 if name in ["shared", "fused", "overlap"] else 0), selected
    assert len(baseline.ROUTER_REFS["both"]) == 40
    assert len(baseline.HC_REFS["both"]) == 80
    BANKS[name] = (entries, ag._graph_params, baseline.ROUTER_REFS["both"], baseline.HC_REFS["both"])
    return {"arm": name, "selected": selected, "routed_calls": 40, "shared_calls": 40,
            "hc_calls": 80, "router_calls": 40}


def create_bank(worker, name):
    global ARM
    from vllm_ascend.compilation import acl_graph as ag
    worker = baseline.actual(worker)
    torch.npu.synchronize()
    first = next(iter(BANKS))
    for kind, refs in REFS[first].items():
        if kind == "routed" and name in ["routed", "fused", "overlap"]:
            for x, params, _, _ in refs:
                clamped_swiglu(x, params[0], mutate_input=True)
        if kind == "shared" and name in ["shared", "fused", "overlap"]:
            for x, params, _, _ in refs:
                clamped_swiglu(x, params[0], alpha=params[1], beta=params[2], staged_rounding=True)
    torch.npu.synchronize()
    if os.getenv("OPT_TEST_OVERLAP") == "1":
        import overlap_patches
        print("OVERLAP_CAPTURE", overlap_patches.set_enabled(name == "overlap"), flush=True)
    ARM = name
    REFS[name] = {"routed": [], "shared": []}
    baseline.ROUTER_REFS["both"] = []
    baseline.HC_REFS["both"] = []
    for wrapper in ag._acl_graph_wrappers:
        wrapper.concrete_aclgraph_entries = {}
    ag._graph_params = None
    ag.set_graph_params([1])
    worker.model_runner.capture_model()
    return save_bank(worker, name)


def switch_bank(worker, name):
    global ARM
    from vllm_ascend.compilation import acl_graph as ag
    torch.npu.synchronize()
    entries, params, router_refs, hc_refs = BANKS[name]
    for wrapper, value in entries:
        wrapper.concrete_aclgraph_entries = value
    ag._graph_params = params
    baseline.ROUTER_REFS["both"] = router_refs
    baseline.HC_REFS["both"] = hc_refs
    if os.getenv("OPT_TEST_OVERLAP") == "1":
        import overlap_patches
        overlap_patches.set_enabled(name == "overlap")
    ARM = name
    return {"arm": name}


def audit(worker, name):
    from verify_activation import compare, native_routed, native_shared
    torch.npu.synchronize()
    result = {}
    for kind, refs in REFS[name].items():
        errors = []
        stds = []
        for x, params, stored, selected in refs:
            expected = native_routed(x.clone(), params[0]) if kind == "routed" else native_shared(x, *params)
            metric = compare(stored, expected)
            assert metric["passed"], (name, kind, metric)
            errors.append(metric)
            stds.append(float(x.float().std().item()))
        if os.getenv("TINY_PERF_RANDOM_VALIDATION") == "1":
            assert min(stds) > 1e-4, (kind, stds)
        result[kind] = {"calls": len(refs), "min_input_std": min(stds),
                        "max_bf16_ulp": max(item["max_bf16_ulp"] for item in errors)}
    return result
