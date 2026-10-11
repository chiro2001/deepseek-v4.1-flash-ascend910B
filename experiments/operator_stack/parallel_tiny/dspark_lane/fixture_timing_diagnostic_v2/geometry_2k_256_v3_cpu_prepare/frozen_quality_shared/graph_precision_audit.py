"""Private actual router/attention outputs and logical draft-cache comparisons.

Capture hooks retain native output references only. Device reads happen after
real engine steps, with audit enabled, never inside graph capture or performance.
"""
import hashlib
import os
from pathlib import Path
from types import MethodType

STATE = {"arm": "on", "enabled": False, "router_roles": {}, "route_capture": {},
         "route_eager": {}, "attention_capture": {}, "attention_eager": {},
         "requests": {}, "current": None, "parameter_seal": None,
         "router_layers": {}, "diagnostic_weight_seal": {}, "diagnostic_sequence": 0}


def _diagnostic_inspect(tensor):
    """Copy one actual tensor outside capture; never replace or sanitize it."""
    import torch
    assert not _capturing(), "Diagnostic device reads are forbidden during capture"
    value = tensor.detach().to("cpu", copy=True).contiguous()
    finite = torch.isfinite(value)
    invalid = (~finite).nonzero()
    view = {"ptr": tensor.data_ptr(), "storage_ptr": tensor.untyped_storage().data_ptr(),
            "storage_offset": tensor.storage_offset(), "stride": list(tensor.stride()),
            "version": None if tensor.is_inference() else tensor._version, "device": str(tensor.device)}
    if tensor.device.type == "npu":
        import torch_npu
        view["npu_format"] = torch_npu.get_npu_format(tensor)
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    info = {"shape": list(value.shape), "dtype": str(value.dtype), "view": view,
            "elements": value.numel(), "finite": bool(finite.all()),
            "invalid_count": invalid.shape[0], "invalid_positions_first64": invalid[:64].tolist(),
            "value_sha256": hashlib.sha256(raw).hexdigest(), "sanitized": False}
    if info["finite"] and value.numel():
        info["min"] = float(value.min())
        info["max"] = float(value.max())
    return value, info


def _diagnostic_write(label, payload):
    import json
    output = Path(os.environ["DSPARK_AUDIT_OUTPUT_PATH"])
    output.mkdir(parents=True, exist_ok=True)
    name = label.replace(".", "_").replace("/", "_")
    (output / (name + ".json")).write_text(json.dumps(payload, indent=2) + "\n")
    return output / name


def _diagnostic_reject(values, summaries, label, context):
    """Reject the first observed bad input after retaining all sibling evidence."""
    import torch
    for name, info in summaries.items():
        if info["finite"]:
            continue
        payload = {"label": label + "." + name, **info, "context": context,
                   "all_observed_inputs": summaries, "quality_gate_rejected": True,
                   "diagnostic_not_consumer_proof": True, "first_observed_nonfinite": True}
        path = _diagnostic_write(label + "." + name + "_nonfinite", payload)
        torch.save(values[name], path.with_suffix(".pt"))
        raise AssertionError("Nonfinite actual diagnostic input: " + label + "." + name)


def _router_diagnostic_tensors(layer, router):
    gate = layer.mlp.gate
    tensors = {"gate.weight": gate.weight}
    for name in ("bias", "e_score_correction_bias", "bias_vl", "tid2eid"):
        for role, obj in (("gate", gate), ("router", router)):
            tensor = getattr(obj, name, None)
            if tensor is not None:
                tensors[role + "." + name] = tensor
    return tensors


def _diagnostic_router_inputs(runner, layer, router, label, hidden, logits):
    assert not _capturing()
    tensors = {"hidden": hidden, "logits": logits, **_router_diagnostic_tensors(layer, router)}
    assert hidden is not None and logits is not None
    observed = {name: _diagnostic_inspect(tensor) for name, tensor in tensors.items()}
    values = {name: row[0] for name, row in observed.items()}
    summaries = {name: row[1] for name, row in observed.items()}
    sealed = STATE["diagnostic_weight_seal"][label]
    for name, seal in sealed.items():
        assert summaries[name]["view"] == seal["view"], (label, name, "Router parameter view changed")
        assert summaries[name]["value_sha256"] == seal["value_sha256"], (label, name, "Router parameter value changed")
    STATE["diagnostic_sequence"] += 1
    context = {"arm": STATE["arm"], "capture": False,
               "point": "before native router arithmetic and MoE consumers",
               "actual_scheduled_round": runner._tiny_dspark_pending,
               "sequence": STATE["diagnostic_sequence"], "parameter_value_and_view_seal_passed": True,
               "diagnostic_not_consumer_proof": True}
    _diagnostic_write(label + ".router_inputs", {"context": context, "inputs": summaries})
    _diagnostic_reject(values, summaries, label, context)
    return context | {"inputs": summaries}


def _capturing():
    from vllm.forward_context import get_forward_context
    try:
        return bool(getattr(get_forward_context(), "capturing", False))
    except AssertionError:
        return False


def _router(layer):
    from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
    experts = layer.mlp.experts
    candidates = [getattr(experts, "router", None),
                  getattr(getattr(experts, "runner", None), "router", None)]
    found = [r for r in candidates if isinstance(r, AscendFusedTopKRouter)]
    assert found
    assert all(r is found[0] for r in found)
    return found[0]


def install(worker):
    from vllm_ascend.spec_decode import llm_base_proposer
    from vllm_ascend.ops.fused_moe.router import fused_topk_router
    from vllm_ascend.attention import dsa_v1
    from vllm_ascend.attention.dsa_v1 import AscendDSAImpl
    runner = worker.model_runner
    for module, env in ((fused_topk_router, "DSPARK_EXPECTED_ROUTER_SHA256"),
                        (dsa_v1, "DSPARK_EXPECTED_DSA_SHA256")):
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == os.environ[env]
    original_flag = llm_base_proposer._dspark_rt_flag

    def runtime_flag(name, default="0"):
        if name == "DRAFT_FORCE_EAGER":
            return "1" if STATE["arm"] == "off" else "0"
        return original_flag(name, default)

    # Use the complete helper's existing callable+runtime-context hot switch;
    # keep the real constructor flag and already-created graph wrapper intact.
    llm_base_proposer._dspark_rt_flag = runtime_flag
    main = runner._tiny_dspark_native_backbone
    draft = runner._tiny_dspark_native_draft
    groups = [(f"target.{index}", layer) for index, layer in enumerate(main.layers)]
    groups += [(f"draft.{index}", layer) for index, layer in enumerate(draft.layers.values())]
    assert len(groups) == 43
    for name, layer in groups:
        router = _router(layer)
        assert id(router) not in STATE["router_roles"]
        STATE["router_roles"][id(router)] = name
        STATE["router_layers"][name] = (layer, router)
        original = router._compute_routing

        def make_observer(original, label, layer):
            def observed(this, *args, **kwargs):
                capturing = _capturing()
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                logits = kwargs.get("router_logits", args[1] if len(args) > 1 else None)
                diagnostic = (not capturing and STATE["enabled"]
                              and os.environ.get("DSPARK_ROUTE_IMMEDIATE_DIAGNOSTIC") == "1")
                context = (_diagnostic_router_inputs(runner, layer, this, label, hidden, logits)
                           if diagnostic else None)
                result = original(*args, **kwargs)
                if capturing or STATE["enabled"]:
                    bank = STATE["route_capture"] if capturing else STATE["route_eager"]
                    ref = {"output": result, "hidden": hidden, "logits": logits,
                           "capture_reference": capturing}
                    if diagnostic:
                        context.update({"point": "native router return before later consumers",
                                   "actual_scheduled_round": runner._tiny_dspark_pending,
                                   "hidden_shape": list(hidden.shape) if hidden is not None else None,
                                   "logits_shape": list(logits.shape) if logits is not None else None,
                                   "tuple_shapes": [list(t.shape) for t in result],
                                   "tuple_dtypes": [str(t.dtype) for t in result],
                                   "capture": False, "diagnostic_not_consumer_proof": True})
                        immediate = [_cpu(t, label + f".immediate.{i}", context) for i, t in enumerate(result)]
                        ref["immediate_diagnostic"] = immediate
                    bank[label] = ref
                return result
            return observed

        router._compute_routing = MethodType(make_observer(original, name, layer), router)
    original_attention = AscendDSAImpl._forward_attention

    def attention(this, layer_name, hidden_states, kv_cache, layer_metadata, *args, **kwargs):
        result = original_attention(this, layer_name, hidden_states, kv_cache, layer_metadata, *args, **kwargs)
        capturing = _capturing()
        if str(layer_name).startswith("mtp.") and (capturing or STATE["enabled"]):
            bank = STATE["attention_capture"] if capturing else STATE["attention_eager"]
            bank[str(layer_name)] = {"output": result, "kv_cache": kv_cache, "metadata": layer_metadata,
                                     "compress_ratio": this.compress_ratio}
        return result

    AscendDSAImpl._forward_attention = attention
    return {"registered_native_routers": 43, "capture_only_references": True,
            "cache_reader": "AscendDSAImpl._forward_attention", "capture_d2h": False,
            "native_arithmetic_unchanged": True, "constructor_bool_flipped": False}


def _parameter_metadata(worker):
    import torch_npu
    roots = [worker.model_runner._tiny_dspark_native_target, worker.model_runner.drafter.model]
    result = {}
    for index, root in enumerate(roots):
        for name, tensor in root.named_parameters():
            result[f"{index}:{name}"] = {"ptr": tensor.data_ptr(),
                "storage_ptr": tensor.untyped_storage().data_ptr(), "storage_offset": tensor.storage_offset(),
                "shape": tuple(tensor.shape), "stride": tuple(tensor.stride()), "dtype": str(tensor.dtype),
                "device": str(tensor.device), "format": torch_npu.get_npu_format(tensor),
                "version": tensor._version}
    return result


def freeze_parameters(worker):
    STATE["parameter_seal"] = _parameter_metadata(worker)
    if os.environ.get("DSPARK_ROUTE_IMMEDIATE_DIAGNOSTIC") == "1":
        assert not _capturing()
        seals = {}
        for label, (layer, router) in STATE["router_layers"].items():
            observed = {name: _diagnostic_inspect(tensor)
                        for name, tensor in _router_diagnostic_tensors(layer, router).items()}
            values = {name: row[0] for name, row in observed.items()}
            summaries = {name: row[1] for name, row in observed.items()}
            _diagnostic_reject(values, summaries, label + ".weight_seal",
                               {"point": "after all capture before first real request", "capture": False})
            seals[label] = summaries
        STATE["diagnostic_weight_seal"] = seals
        _diagnostic_write("router_weight_value_seal", {"layers": seals, "capture": False,
            "native_parameter_values_unchanged": True, "full_weight_or_consumer_precision_proof": False})
    return {"parameter_views": len(STATE["parameter_seal"]), "sealed_after_all_capture": True,
            "seal_is_metadata_not_weight_value_or_deviceproof": True,
            "diagnostic_router_weight_value_seals": len(STATE["diagnostic_weight_seal"])}


def switch(worker, arm, audit, tag):
    assert arm in ("off", "on") and STATE["current"] is None
    assert _parameter_metadata(worker) == STATE["parameter_seal"], "Captured parameters were replaced/mutated"
    STATE["arm"], STATE["enabled"] = arm, bool(audit)
    if audit:
        assert tag not in STATE["requests"]
        STATE["current"] = {"tag": tag, "arm": arm, "steps": [], "cache": None}
    return {"arm": arm, "audit": audit, "actual_constructor_graph_configured": worker.model_runner.drafter.use_cuda_graph,
            "hot_switch": "native helper DRAFT_FORCE_EAGER gate (callable and context)", "bool_flip": False}


def _cpu(tensor, label="unlabeled", context=None):
    import torch
    result = tensor.detach().to("cpu", copy=True).contiguous()
    finite = torch.isfinite(result)
    if not bool(finite.all()):
        import json
        invalid = (~finite).nonzero().tolist()
        output = Path(os.environ["DSPARK_AUDIT_OUTPUT_PATH"])
        output.mkdir(parents=True, exist_ok=True)
        name = label.replace(".", "_").replace("/", "_")
        info = {"label": label, "shape": list(result.shape), "dtype": str(result.dtype),
                "source_ptr": tensor.data_ptr(), "source_stride": list(tensor.stride()),
                "invalid_count": len(invalid), "invalid_positions_first64": invalid[:64],
                "context": context, "sanitized": False, "quality_gate_rejected": True}
        (output / (name + "_nonfinite.json")).write_text(json.dumps(info,indent=2)+"\n")
        torch.save(result, output / (name + "_nonfinite.pt"))
        raise AssertionError("Nonfinite actual consumer values: " + label)
    return result


def _bitwise(left, right):
    import torch
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.view(torch.uint8), right.view(torch.uint8)))


def snapshot_step(worker, is_decode, draft_replayed):
    import torch
    runner = worker.model_runner
    assert STATE["enabled"] and STATE["current"] is not None and not _capturing()
    torch.npu.synchronize()
    row = {"is_decode": bool(is_decode), "draft_replayed": bool(draft_replayed), "routes": {}, "attention": {}}
    for name in STATE["router_roles"].values():
        capture = bool(is_decode) if name.startswith("target.") else bool(draft_replayed)
        bank = STATE["route_capture"] if capture else STATE["route_eager"]
        assert name in bank, (name, "No actual router output observed")
        reference = bank[name]
        weights, ids = reference["output"]
        count = 6 if name.startswith("target.") and is_decode else None
        if name.startswith("draft."):
            count = runner.drafter.num_query_per_req
        if count is not None:
            assert weights.shape[0] >= count and ids.shape[0] >= count
            weights, ids = weights[:count], ids[:count]
        actual_round = runner._tiny_dspark_receipts[-1]
        context = {"is_decode": bool(is_decode), "arm": STATE["arm"], "capture_reference": capture,
                   "scheduled_target_tokens": actual_round["scheduled_target_tokens"],
                   "prompt_tokens": actual_round["prompt_tokens"], "computed_before": actual_round["computed_before"],
                   "verified_draft_tokens": actual_round["verified_draft_tokens"],
                   "raw_generated_tokens": actual_round["raw_generated_tokens"],
                   "hidden_shape": list(reference["hidden"].shape) if reference["hidden"] is not None else None,
                   "logits_shape": list(reference["logits"].shape) if reference["logits"] is not None else None,
                   "native_tuple_dtypes": [str(t.dtype) for t in reference["output"]],
                   "native_tuple_shapes": [list(t.shape) for t in reference["output"]]}
        row["routes"][name] = {"weights": _cpu(weights, name + ".weights", context),
                              "ids": _cpu(ids, name + ".ids", context)}
        if "immediate_diagnostic" in reference:
            immediate = reference["immediate_diagnostic"]
            assert _bitwise(row["routes"][name]["weights"], immediate[0]), (name, "router output changed after native return")
            assert _bitwise(row["routes"][name]["ids"], immediate[1]), (name, "router indices changed after native return")
    assert len(row["routes"]) == 43
    bank = STATE["attention_capture"] if draft_replayed else STATE["attention_eager"]
    assert len(bank) == 3, "Expected three actual draft attention consumers"
    for name, refs in sorted(bank.items()):
        from vllm_ascend.attention.dsa_v1 import DeviceOperator, _require_req_metadata
        index = int(name.split(".")[1])
        layer = list(runner._tiny_dspark_native_draft.layers.values())[index]
        cache = layer.self_attn.dsa_attn.swa_cache_layer.kv_cache
        while isinstance(cache, (tuple, list)) and len(cache) == 1:
            cache = cache[0]
        actual_cache = DeviceOperator.unpack_dsa_forward_kv_cache(refs["kv_cache"], refs["compress_ratio"])[1]
        assert actual_cache.data_ptr() == cache.data_ptr(), "Observed reader does not bind its actual draft cache"
        metadata = _require_req_metadata(refs["metadata"].swa)
        group = runner.drafter._layer_group_idx[index]
        table = runner.drafter._per_group_block_tables[group]
        assert metadata.block_table.data_ptr() == table.data_ptr(), "Reader block table binding changed"
        output = refs["output"]
        count = runner.drafter.num_query_per_req
        assert output.shape[0] >= count
        row["attention"][name] = _cpu(output[:count])
    draft_ids = runner._draft_token_ids
    assert isinstance(draft_ids, torch.Tensor)
    row["proposed_ids"] = _cpu(draft_ids)
    STATE["current"]["steps"].append(row)
    return {"observed_native_route_layers": 43, "observed_actual_draft_attention_layers": 3,
            "draft_replayed": bool(draft_replayed), "is_decode": bool(is_decode), "audit_device_sync": 1}


def finish(worker):
    import torch
    runner = worker.model_runner
    assert STATE["enabled"] and STATE["current"] is not None and not _capturing()
    torch.npu.synchronize()
    drafter = runner.drafter
    positions = _cpu(drafter._get_positions(drafter.num_query_per_req)).reshape(-1).tolist()
    assert positions and min(positions) >= 0
    valid_rows = max(positions) + 1
    last_round = runner._tiny_dspark_receipts[-1]
    assert min(positions) >= last_round["prompt_tokens"], "Draft positions are stale capture/profile values"
    block_size = 128
    blocks = (valid_rows + block_size - 1) // block_size
    caches = {}
    for index, layer in enumerate(runner._tiny_dspark_native_draft.layers.values()):
        cache = layer.self_attn.dsa_attn.swa_cache_layer.kv_cache
        while isinstance(cache, (tuple, list)) and len(cache) == 1:
            cache = cache[0]
        assert tuple(cache.shape[1:]) == (128, 1, 512) and cache.dtype == torch.bfloat16
        group = drafter._layer_group_idx[index]
        table = drafter._per_group_block_tables[group]
        ids = table[0, :blocks].to(torch.long)
        cpu_ids = _cpu(ids).tolist()
        assert len(cpu_ids) == blocks and min(cpu_ids) >= 0 and max(cpu_ids) < cache.shape[0]
        values = _cpu(cache.index_select(0, ids).reshape(-1, 1, 512)[:valid_rows])
        caches[f"draft.{index}"] = {"physical_blocks": cpu_ids, "logical_rows": valid_rows,
                                  "cache_ptr": cache.data_ptr(), "table_ptr": table.data_ptr(), "values": values}
    current = STATE["current"]
    current["cache"] = caches
    current["query_positions"] = positions
    assert _parameter_metadata(worker) == STATE["parameter_seal"]
    current["parameter_seal_passed"] = True
    STATE["requests"][current["tag"]] = current
    STATE["current"] = None
    STATE["enabled"] = False
    return {"tag": current["tag"], "actual_route_layers": 43, "actual_draft_attention_layers": 3,
            "actual_step_snapshots": len(current["steps"]), "logical_cache_rows": valid_rows,
            "draft_cache_layers": 3, "query_positions": positions, "parameter_seal_passed": True}


def compare(worker, left_tag, right_tag):
    import torch
    left, right = STATE["requests"][left_tag], STATE["requests"][right_tag]
    assert STATE["parameter_seal"] is not None and left["parameter_seal_passed"] and right["parameter_seal_passed"]
    assert left["query_positions"] == right["query_positions"]
    assert len(left["steps"]) == len(right["steps"])
    route_positions = attention_values = 0
    for step, (a, b) in enumerate(zip(left["steps"], right["steps"])):
        assert a["is_decode"] == b["is_decode"]
        assert _bitwise(a["proposed_ids"], b["proposed_ids"]), (step, "draft proposals")
        assert set(a["routes"]) == set(b["routes"]) and len(a["routes"]) == 43
        for name in a["routes"]:
            for role in ("weights", "ids"):
                x, y = a["routes"][name][role], b["routes"][name][role]
                assert x.shape == y.shape and x.dtype == y.dtype
                assert _bitwise(x, y), (step, name, role, "Native consumer values differ")
            route_positions += a["routes"][name]["ids"].numel()
        assert set(a["attention"]) == set(b["attention"]) and len(a["attention"]) == 3
        for name in a["attention"]:
            x, y = a["attention"][name], b["attention"][name]
            assert _bitwise(x, y), (step, name, "attention output")
            attention_values += x.numel()
    assert set(left["cache"]) == set(right["cache"]) and len(left["cache"]) == 3
    cache_values = 0
    for name in left["cache"]:
        a, b = left["cache"][name], right["cache"][name]
        assert a["logical_rows"] == b["logical_rows"] and _bitwise(a["values"], b["values"]), (name, "logical cache")
        cache_values += a["values"].numel()
    return {"passed": True, "left": left_tag, "right": right_tag,
            "native_proposals_bitwise": True, "ordered_route_ids_weights_bitwise": True,
            "draft_attention_consumers_bitwise": True, "draft_logical_cache_bitwise": True,
            "route_values_checked": route_positions, "attention_values_checked": attention_values,
            "cache_values_checked": cache_values, "parameter_seal_passed": True,
            "formal_quality_claim": False}
