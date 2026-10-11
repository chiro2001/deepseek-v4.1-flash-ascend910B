"""Private TP1 ID-hoist graph banks and capture-only consumer witnesses."""
from collections import Counter
import hashlib
import os

import torch

ARM = "off"
REFS = {}
BANKS = {}
TOPOLOGY = None


def actual(worker):
    return getattr(worker, "worker", worker)


def _refs():
    from vllm.forward_context import get_forward_context
    if not get_forward_context().capturing:
        return None
    return REFS.setdefault(ARM, {"model": [], "router": []})


def record_model_ids(original, sanitized):
    refs = _refs()
    if refs is not None:
        refs["model"].append((original, sanitized))


def record_router_before(router, ids):
    refs = _refs()
    if refs is not None:
        refs["router"].append({"router": router, "before": ids})


def record_router_after_cast(ids):
    refs = _refs()
    if refs is not None:
        refs["router"][-1]["after_cast"] = ids


def record_router_consumer(ids):
    refs = _refs()
    if refs is not None:
        refs["router"][-1]["consumer"] = ids


def record_router_result(weights, ids):
    refs = _refs()
    if refs is not None and refs["router"] and "consumer" in refs["router"][-1]:
        refs["router"][-1]["weights"] = weights
        refs["router"][-1]["topk_ids"] = ids


def set_arm(name):
    global ARM
    from vllm_ascend.models.deepseek_v41 import model
    assert name.startswith("off") or name.startswith("on"), name
    ARM = name
    model._IDS64_HOIST = name.startswith("on")


@torch.inference_mode()
def initialize(worker):
    global TOPOLOGY
    from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model
    from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
    worker = actual(worker)
    model = worker.model_runner.model
    # The existing flag is a Python global. VLLM's no-guard compiled callable
    # would otherwise retain its first value while graph banks are re-captured.
    # Both arms use the identical native forward and FULL NPU graph replay.
    # Only the backbone's Dynamo wrapper is bypassed, before first warm-up.
    backbones = [m for m in model.modules() if isinstance(m, DeepseekV41Model)]
    assert len(backbones) == 1, len(backbones)
    for backbone in backbones:
        backbone.do_not_compile = True
    counts = {"gate": 0, "hc_fn": 0, "hc_scale": 0, "hc_base": 0}
    torch.manual_seed(20261011)
    for name, param in model.named_parameters():
        local = name.rsplit(".", 1)[-1]
        if name.endswith(".gate.weight") and param.shape == (8, 5120):
            param.copy_(torch.randn_like(param) / 5120 ** .5)
            counts["gate"] += 1
        elif "hc" in local and local.endswith("_fn") and param.shape == (24, 20480):
            param.copy_(torch.randn_like(param) * .1 / 20480 ** .5)
            counts["hc_fn"] += 1
        elif "hc" in local and local.endswith("_scale") and param.shape == (3,):
            param.fill_(1.)
            counts["hc_scale"] += 1
        elif "hc" in local and local.endswith("_base") and param.shape == (24,):
            param.copy_(torch.randn_like(param) * .1)
            counts["hc_base"] += 1
    assert counts == {"gate": 40, "hc_fn": 80, "hc_scale": 80, "hc_base": 80}, counts
    for module in model.modules():
        if getattr(module, "precast_fp32_weight", False) and hasattr(module, "weight_fp32"):
            module.weight_fp32.copy_(module.weight.float())
    routers = [m for m in model.modules() if isinstance(m, AscendFusedTopKRouter)]
    # Routers may be plain objects rather than registered nn.Modules.
    if not routers:
        routers = []
        for layer in backbones[0].layers:
            experts = layer.mlp.experts
            for candidate in [getattr(experts, "router", None), getattr(getattr(experts, "runner", None), "router", None)]:
                if isinstance(candidate, AscendFusedTopKRouter) and candidate not in routers:
                    routers.append(candidate)
    config = worker.model_runner.model_config.hf_config
    TOPOLOGY = {"layers": int(config.num_hidden_layers), "hidden": int(config.hidden_size),
                "experts": int(config.n_routed_experts), "topk": int(config.num_experts_per_tok),
                "moe_intermediate": int(config.moe_intermediate_size), "tp": 1, "ep": 1,
                "dtype": "bfloat16", "quantization": "unquantized BF16 dummy",
                "num_hash_layers": int(config.num_hash_layers), "vision_n_layers": int(config.vision_n_layers),
                "needs_moe_input_ids": bool(backbones[0].needs_moe_input_ids),
                "backbone_dynamo_bypassed": True, "native_reductions": True,
                "randomized_validation_weights": counts, "randomization_seed": 20261011,
                "physical_chip": 4, "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
                "hccl_deterministic": os.getenv("HCCL_DETERMINISTIC"),
                "model_class": type(model).__module__ + "." + type(model).__name__,
                "router_objects_found_before_capture": len(routers),
                "scope": "40-layer TP1 diagnostic; 8 experts/top2; not formal TP8 384/top6"}
    assert TOPOLOGY["needs_moe_input_ids"] and TOPOLOGY["vision_n_layers"] == 32
    torch.npu.synchronize()
    return TOPOLOGY


def save(worker, name):
    from vllm_ascend.compilation import acl_graph as ag
    entries = [(w, w.concrete_aclgraph_entries, w.graph_pool) for w in ag._acl_graph_wrappers]
    count = sum(len(v) for _, v, _ in entries)
    assert count == 1, (name, count)
    refs = REFS[name]
    assert len(refs["model"]) == 1 and len(refs["router"]) == 40, (name, len(refs["model"]), len(refs["router"]))
    BANKS[name] = (entries, ag._graph_params, refs)
    return coverage(name)


def create(worker, name):
    from vllm_ascend.compilation import acl_graph as ag
    worker = actual(worker)
    torch.npu.synchronize()
    set_arm(name)
    REFS[name] = {"model": [], "router": []}
    pool = torch.npu.graph_pool_handle()
    for wrapper in ag._acl_graph_wrappers:
        wrapper.concrete_aclgraph_entries = {}
        wrapper.graph_pool = pool
    ag._graph_params = None
    ag.set_graph_params([1])
    worker.model_runner.capture_model()
    return save(worker, name)


def switch(worker, name):
    from vllm_ascend.compilation import acl_graph as ag
    torch.npu.synchronize()
    set_arm(name)
    entries, params, refs = BANKS[name]
    for wrapper, values, pool in entries:
        wrapper.concrete_aclgraph_entries = values
        wrapper.graph_pool = pool
    ag._graph_params = params
    REFS[name] = refs
    return {"arm": name}


def _description(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "stride": list(tensor.stride()), "contiguous": tensor.is_contiguous()}


def coverage(name):
    refs = REFS[name]
    original, prepared = refs["model"][0]
    rows = refs["router"]
    return {"arm": name, "model_prepare_calls": len(refs["model"]), "eligible_router_calls": len(rows),
            "model_input": _description(original), "prepared_input": _description(prepared),
            "router_input_dtypes": dict(Counter(str(r["before"].dtype) for r in rows)),
            "router_material_casts": sum(r["before"].dtype != torch.int64 for r in rows),
            "router_noop_casts": sum(r["before"].dtype == torch.int64 for r in rows),
            "model_hoist_material_casts": int(name.startswith("on") and original.dtype != torch.int64),
            "router_cast_storage_count": len({r["after_cast"].data_ptr() for r in rows}),
            "router_input_storage_count": len({r["before"].data_ptr() for r in rows}),
            "consumer_dtypes": dict(Counter(str(r["consumer"].dtype) for r in rows)),
            "consumer_shapes": sorted({str(tuple(r["consumer"].shape)) for r in rows}),
            "consumer_bias_vl_nonnull": sum(r["router"].bias_vl is not None for r in rows),
            "consumer_tid2eid_nonnull": sum(r["router"].tid2eid is not None for r in rows),
            "graph_isolated_pool": name != "off"}


@torch.inference_mode()
def audit(worker, name):
    switch(worker, name)
    torch.npu.synchronize()
    refs = REFS[name]
    original, prepared = refs["model"][0]
    expected = torch.where(original == -1, 0, original).to(torch.int64)
    assert torch.equal(prepared.to(torch.int64), expected)
    seen = []
    for row in refs["router"]:
        before, after, consumer = row["before"], row["after_cast"], row["consumer"]
        assert torch.equal(before.to(torch.int64), after)
        assert torch.equal(after, consumer), (after.shape, consumer.shape)
        assert torch.equal(expected, consumer)
        assert consumer.dtype == torch.int64
        if name.startswith("on"):
            assert before.data_ptr() == after.data_ptr()
        ids = row["topk_ids"]
        assert ids.shape == (1, 2) and int(ids.min()) >= 0 and int(ids.max()) < 8
        assert int(ids[0, 0]) != int(ids[0, 1])
        seen.append(ids.cpu().tolist()[0])
    result = coverage(name)
    result.update({"all_integer_values_equal": True, "all_40_consumers_equal": True,
                   "consumer_input_sha256": hashlib.sha256(expected.cpu().numpy().tobytes()).hexdigest(),
                   "captured_top2_unique_experts": sorted({v for row in seen for v in row})})
    return result


def disable_route_export(worker):
    worker = actual(worker)
    torch.npu.synchronize()
    worker.model_runner.model_config.enable_return_routed_experts = False
    worker.model_runner.vllm_config.model_config.enable_return_routed_experts = False
    # The sync D2H bookkeeping is guarded by this initialized-state latch,
    # independently of ModelConfig; merely changing the config is insufficient.
    worker.model_runner.routed_experts_initialized = False
    routers = {id(row["router"]): row["router"] for refs in REFS.values() for row in refs["router"]}
    for router in routers.values():
        router.set_capture_fn(None)
    assert len(routers) == 40, len(routers)
    return {"route_export_enabled": False, "routed_experts_initialized": False,
            "disabled_router_capture_callbacks": len(routers)}


def topology(worker):
    return TOPOLOGY
