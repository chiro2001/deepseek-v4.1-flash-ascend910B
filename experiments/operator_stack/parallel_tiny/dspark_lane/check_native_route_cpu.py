"""CPU tensor controls for first-bad-input inspection; no NPU initialization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["DSPARK_AUDIT_OUTPUT_PATH"] = str(args.output / "audit")
    import torch
    context = types.SimpleNamespace(capturing=False)
    module = types.ModuleType("vllm.forward_context")
    module.get_forward_context = lambda: context
    sys.modules["vllm.forward_context"] = module
    import graph_precision_audit as audit
    checks = []
    tensor = torch.tensor([[1., -2.], [3., 4.]], dtype=torch.bfloat16)
    value, info = audit._diagnostic_inspect(tensor)
    assert value.device.type == "cpu" and info["finite"] and info["min"] == -2 and info["max"] == 4
    assert info["value_sha256"] == hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    checks.append("finite BF16 input is copied and bitwise sealed")
    with torch.inference_mode():
        inference_tensor = torch.ones(2, 2)
    _, inference_info = audit._diagnostic_inspect(inference_tensor)
    assert inference_info["finite"] and inference_info["view"]["version"] is None
    checks.append("inference activations keep value evidence without a version counter")
    _, scalar_info = audit._diagnostic_inspect(torch.tensor(1.))
    assert scalar_info["finite"] and scalar_info["shape"] == []
    checks.append("scalar tensor value seals preserve the scalar shape")
    _, positive = audit._diagnostic_inspect(torch.tensor([0.]))
    _, negative = audit._diagnostic_inspect(torch.tensor([-0.]))
    assert positive["value_sha256"] != negative["value_sha256"]
    checks.append("negative zero has a different value seal")
    bad = torch.tensor([[float("nan"), 1.], [2., float("inf")]])
    bad_value, bad_info = audit._diagnostic_inspect(bad)
    assert bad_info["invalid_count"] == 2 and bad_info["invalid_positions_first64"] == [[0, 0], [1, 1]]
    try:
        audit._diagnostic_reject({"logits": bad_value}, {"logits": bad_info}, "cpu.router", {"capture": False})
    except AssertionError:
        data = json.loads((args.output / "audit/cpu_router_logits_nonfinite.json").read_text())
        assert data["quality_gate_rejected"] and not data["sanitized"]
        assert torch.isnan(torch.load(args.output / "audit/cpu_router_logits_nonfinite.pt", weights_only=True)[0, 0])
        checks.append("NaN and infinity are retained and rejected")
    else:
        raise AssertionError("Nonfinite input was admitted")
    context.capturing = True

    class CaptureReadBomb:
        def detach(self):
            raise RuntimeError("Device read inside capture")

    try:
        audit._diagnostic_inspect(CaptureReadBomb())
    except AssertionError:
        checks.append("capture rejects before any device read")
    else:
        raise AssertionError("Capture read was admitted")
    context.capturing = False
    router = types.SimpleNamespace(e_score_correction_bias=None, bias_vl=None, tid2eid=None)
    gate = types.SimpleNamespace(weight=torch.tensor([[.1, .2], [.3, .4]]))
    layer = types.SimpleNamespace(mlp=types.SimpleNamespace(gate=gate))
    runner = types.SimpleNamespace(_tiny_dspark_pending={"is_decode": True, "scheduled_target_tokens": 6})
    label = "draft.0"
    audit.STATE["diagnostic_weight_seal"][label] = {
        name: audit._diagnostic_inspect(item)[1]
        for name, item in audit._router_diagnostic_tensors(layer, router).items()}
    audit._diagnostic_router_inputs(runner, layer, router, label, tensor, torch.ones(2, 2))
    checks.append("unchanged native router weight value and view pass")
    gate.weight.data[0, 0] = .7
    try:
        audit._diagnostic_router_inputs(runner, layer, router, label, tensor, torch.ones(2, 2))
    except AssertionError:
        checks.append("weight mutation through data is rejected by value seal")
    else:
        raise AssertionError("Changed weight was admitted")
    import native_first_value_trace as trace
    graph = object()
    active = {"eligible": True, "graph": graph, "event": {"role": "target", "capturing": False}}
    assert trace.target_replay_matches(active, graph)
    for variant, candidate in ((None, graph), ({**active, "eligible": False}, graph),
            ({**active, "event": {"role": "draft", "capturing": False}}, graph),
            ({**active, "event": {"role": "target", "capturing": True}}, graph), (active, object())):
        assert not trace.target_replay_matches(variant, candidate)
    checks.append("target cache boundaries reject foreign, draft, capture and ineligible graph submissions")
    source_weight = torch.arange(12, dtype=torch.bfloat16).view(4, 3)
    cache_weight = source_weight.view(2, 2, -1).transpose(2, 1).contiguous()
    assert audit._bitwise(cache_weight, source_weight.view(2, 2, -1).transpose(2, 1).contiguous())
    changed_weight = cache_weight.clone()
    changed_weight[0, 0, 0] = 99
    assert not audit._bitwise(changed_weight, source_weight.view(2, 2, -1).transpose(2, 1).contiguous())
    checks.append("native dummy wo_a loader permutation preserves bytes and rejects changed cached weights")
    cache = torch.arange(256, dtype=torch.float32).reshape(2, 128, 1, 1).expand(-1, -1, 1, 512).to(torch.bfloat16).contiguous()
    indices = torch.tensor([[[0, 2, -1]], [[127, 128, 129]]], dtype=torch.int32)
    selected, selected_info = trace.select_visible_cache(cache, indices, 2)
    assert selected[:, 0, 0].tolist() == [0., 2., 127., 128., 129.]
    assert selected_info["pad_entries_excluded_by_native_minus_one_abi"] == 1 and not selected_info["finite_filter_used"]
    checks.append("actual sparse indices bind physical cache slots across block boundaries")
    cache.view(256, 1, 512)[31] = float("nan")
    selected, _ = trace.select_visible_cache(cache, indices, 2)
    assert torch.isfinite(selected).all()
    bad_indices = torch.tensor([[[31, -1, -1]]], dtype=torch.int32)
    selected, _ = trace.select_visible_cache(cache, bad_indices, 1)
    _, selected_info = audit._diagnostic_inspect(selected)
    assert not selected_info["finite"] and selected_info["invalid_count"] == 512
    checks.append("unreferenced cache padding is excluded, referenced nonfinite values are retained")
    for name, bad in (("unknown pad sentinel", -2), ("out of range slot", 256)):
        try:
            trace.select_visible_cache(cache, torch.tensor([[[bad]]], dtype=torch.int32), 1)
        except AssertionError:
            checks.append(name + " rejected")
        else:
            raise AssertionError(name + " was admitted")
    operator = torch.ops.aten.add.Tensor
    bound = trace.bind_operator_arguments(operator._schema, (tensor, torch.ones(2, 2, dtype=torch.bfloat16)), {"alpha": 2})
    assert bound["self"] is tensor and bound["alpha"] == 2
    try:
        trace.bind_operator_arguments(operator._schema, (tensor,), {"self": tensor})
    except AssertionError:
        checks.append("native operator argument binding rejects duplicate ABI names")
    else:
        raise AssertionError("Duplicate operator argument was admitted")
    before = tensor.clone()
    test_trace = trace.FirstValueTrace(types.SimpleNamespace(model_runner=runner))
    audit.STATE["enabled"] = True
    result = test_trace.observe_operator(torch._ops.OpOverload.__call__, operator,
          (tensor, torch.ones(2, 2, dtype=torch.bfloat16)), {"alpha": 2}, "cpu.operator")
    assert torch.equal(result, tensor + 2) and torch.equal(tensor, before)
    checks.append("operator boundary delegates original arguments and preserves input values")
    context.capturing = True
    sequence = test_trace.sequence
    result = test_trace.observe_operator(torch._ops.OpOverload.__call__, operator,
          (tensor, torch.ones(2, 2, dtype=torch.bfloat16)), {}, "cpu.capture")
    assert torch.equal(result, tensor + 1) and test_trace.sequence == sequence
    context.capturing = False
    audit.STATE["enabled"] = False
    checks.append("operator boundary performs no observation device reads during capture")
    import o_proj_cache_fixture as fixture
    owned = source_weight.view(2, 2, -1).transpose(2, 1).contiguous()
    fixture_owner = types.SimpleNamespace(_buffers={fixture.BUFFER_NAME: owned})
    setattr(fixture_owner, fixture.BUFFER_NAME, owned)
    fixture_impl = types.SimpleNamespace(_dummy_wo_a_3d=owned,
                                         wo_a=types.SimpleNamespace(weight=source_weight))
    fixture.STATE["entries"] = {"draft.0": {"owner": fixture_owner, "impl": fixture_impl,
          "owner_id": id(fixture_owner), "cache": audit._diagnostic_inspect(owned)[1],
          "source": audit._diagnostic_inspect(source_weight)[1]}}
    assert fixture.check_entry("draft.0", True, "cpu_owned_fixture")["cache_view_and_value_seal_equal"]
    checks.append("explicit fixture owner, consumer pointer and source/value seal match")
    owned.data[0, 0, 0] = 99
    try:
        fixture.check_entry("draft.0", False, "cpu_changed_fixture")
    except AssertionError:
        checks.append("finite cached-weight mutation is rejected by frozen fixture value seal")
    else:
        raise AssertionError("Changed fixture cache was admitted")
    replacement = owned.clone()
    setattr(fixture_owner, fixture.BUFFER_NAME, replacement)
    fixture_owner._buffers[fixture.BUFFER_NAME] = replacement
    fixture_impl._dummy_wo_a_3d = replacement
    try:
        fixture.check_entry("draft.0", False, "cpu_replaced_fixture")
    except AssertionError:
        checks.append("cache replacement is rejected even when owner and consumer agree")
    else:
        raise AssertionError("Replaced fixture cache was admitted")
    history_trace = types.SimpleNamespace(helper_history=[
        {"capturing": True, "native_helper_returned": True},
        {"capturing": False, "native_helper_returned": True},
        {"capturing": False, "native_helper_returned": False}], helper_marker=1)
    helper_counts = trace.read_helper_receipt(types.SimpleNamespace(_tiny_first_value_trace=history_trace))
    assert helper_counts["native_helper_calls_this_request"] == 2
    assert helper_counts["native_helper_successful_returns_this_request"] == 1
    assert helper_counts["cold_capture_helper_calls_global"] == 1
    checks.append("helper receipts separate cold capture, successful returns and exceptions")
    result = {"passed": True, "cpu_tensors_only_no_npu_init": True, "checks": checks,
              "actual_device_precision_proof": False,
              "audit_source_sha256": hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest(),
              "first_value_trace_source_sha256": hashlib.sha256(Path(trace.__file__).read_bytes()).hexdigest(),
              "o_proj_cache_fixture_source_sha256": hashlib.sha256(Path(fixture.__file__).read_bytes()).hexdigest()}
    (args.output / "cpu_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
