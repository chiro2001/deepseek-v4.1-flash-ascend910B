"""Meaningful CPU controls: forbid timed diagnostic reads and reject bad counters."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

from timing_controls import WindowGuard, period2_after_prefill, pooled_triplet, triplet, validate_geometry
from launch_fixture_timing import RUNTIME_FILES, remote_program

BASE = Path(__file__).resolve().parent
ARCHIVE = BASE.parent / "evidence/graph_fixture_v1_raw/results"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expect_reject(function):
    try:
        function()
    except AssertionError:
        return
    raise AssertionError("Negative control was not rejected")


def prepared_raw(index):
    old = json.loads((ARCHIVE / f"raw_{index:03d}.json").read_text())
    raw = copy.deepcopy(old)
    old_helper = raw["helper"]
    raw["helper"] = {key: old_helper[key] for key in (
        "native_helper_calls_this_request", "native_helper_successful_returns_this_request",
        "cold_capture_helper_calls_global", "cold_capture_helper_successful_returns_global")}
    raw["helper"]["history"] = [{**row, "returned": row["native_helper_returned"]}
                                   for row in old_helper["helper_history_this_request"]]
    raw["window_guard"] = {"diagnostic_device_reads_or_hash_calls_in_window": 0,
                            "explicit_terminal_device_fences": 1,
                            "route_logprob_audit_and_profiler": False}
    return raw


def extracted_function(path, name, globals_dict):
    tree = ast.parse(path.read_text())
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    module = ast.fix_missing_locations(ast.Module(body=[matches[0]], type_ignores=[]))
    exec(compile(module, str(path), "exec"), globals_dict)
    return globals_dict[name]


def main():
    controls = []

    guard = WindowGuard()
    called = []
    protected = guard.protect("cpu.fake_d2h", lambda value: called.append(value) or value)
    assert protected(3) == 3 and called == [3]
    controls.append("outside_window_diagnostic_forwarding")
    guard.begin()
    expect_reject(lambda: protected(4))
    assert called == [3] and guard.rejected_diagnostic_calls == 1
    controls.append("timed_diagnostic_read_rejected_before_original")
    synchronize = guard.observe_synchronize(lambda: called.append("fence"))
    synchronize()
    guard.terminal_fence_expected = True
    synchronize()
    guard.terminal_fence_expected = False
    guard.end()
    receipt = guard.receipt()
    assert receipt["explicit_terminal_device_fences"] == receipt["other_native_explicit_fences"] == 1
    controls.append("terminal_fence_count_separated_from_native_fences")
    expect_reject(guard.end)
    controls.append("invalid_window_transition_rejected")

    fixture = BASE / "frozen_shared/o_proj_cache_fixture.py"
    calls = []
    impl = object()
    consumer_guard = WindowGuard()
    consumer_guard.begin()
    namespace = {"roles_by_impl": {id(impl): "draft.0"},
                 "runner": SimpleNamespace(_tiny_dspark_recording=True),
                 "audit": SimpleNamespace(_capturing=lambda: False),
                 "os": os, "STATE": {"consumer_counts": {"draft.0": 0}},
                 "check_entry": consumer_guard.protect("frozen_consumer_d2h", lambda *a, **k: calls.append("read")),
                 "original": lambda this, *args, **kwargs: calls.append((this, args, kwargs)) or args[0]}
    consumer = extracted_function(fixture, "consumer", namespace)
    old_env = os.environ.get("DSPARK_FIXTURE_VALIDATE_CONSUMERS")
    try:
        os.environ["DSPARK_FIXTURE_VALIDATE_CONSUMERS"] = "0"
        opaque = object()
        assert consumer(impl, opaque, native_keyword=5) is opaque
        assert calls == [(impl, (opaque,), {"native_keyword": 5})]
        assert consumer_guard.rejected_diagnostic_calls == 0
        controls.append("same_frozen_fixture_consumer_flag0_preserves_native_call_without_d2h")
        os.environ["DSPARK_FIXTURE_VALIDATE_CONSUMERS"] = "1"
        expect_reject(lambda: consumer(impl, opaque))
        assert len(calls) == 1
        controls.append("accidental_consumer_d2h_enable_rejected")
    finally:
        if old_env is None:
            os.environ.pop("DSPARK_FIXTURE_VALIDATE_CONSUMERS", None)
        else:
            os.environ["DSPARK_FIXTURE_VALIDATE_CONSUMERS"] = old_env
        consumer_guard.end()

    context = SimpleNamespace(capturing=True, cudagraph_runtime_mode="FULL", batch_descriptor="cold6")
    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_context = ModuleType("vllm.forward_context")
    fake_context.get_forward_context = lambda: context
    original_modules = {name: sys.modules.get(name) for name in ("vllm", "vllm.forward_context")}
    state = {"helper_history": []}
    original_args = []
    native_result = object()
    def original_helper(*args, **kwargs):
        original_args.append((args, kwargs))
        return native_result
    try:
        sys.modules.update({"vllm": fake_vllm, "vllm.forward_context": fake_context})
        helper = extracted_function(BASE / "fixture_timing_worker.py", "helper", {
            "runner": SimpleNamespace(_tiny_dspark_pending=None), "state": state,
            "time": time, "original_helper": original_helper})
        operand = object()
        assert helper(None, operand, native_keyword=7) is native_result
        assert original_args == [((operand,), {"native_keyword": 7})]
        assert len(state["helper_history"]) == 1 and state["helper_history"][0]["capturing"] and state["helper_history"][0]["returned"]
        controls.append("host_helper_counter_preserves_native_arguments_and_return")
        controls.append("cold_capture_counter_reads_context_without_tensor_values")
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    refs = json.loads((BASE / "reference_token_ids.json").read_text())
    for index, arm in ((3, "off"), (10, "on")):
        raw = prepared_raw(index)
        result = validate_geometry(raw, raw["token_ids"], arm, raw["core_id"], refs["0"])
        assert result["actual_draft_propose_calls"] == 17 and result["initial_actual_draft_calls"] == result["terminal_actual_draft_calls"] == 1
        controls.append(arm + "_geometry_includes_initial_and_terminal_draft")
    base = prepared_raw(10)
    changes = [
        ("missing_initial_draft", lambda x: x["counter"]["rounds"][0].update(draft_calls=0)),
        ("missing_terminal_draft", lambda x: x["counter"]["rounds"][-1].update(draft_calls=0)),
        ("wrong_decode_scheduled_geometry", lambda x: x["counter"]["rounds"][1].update(scheduled_target_tokens=7)),
        ("wrong_verified_geometry", lambda x: x["counter"]["rounds"][1].update(verified_draft_tokens=4)),
        ("missing_terminal_fence", lambda x: x["counter"].update(terminal_sync_count=0)),
        ("missing_hot_draft_replay", lambda x: next(e for e in x["graph"]["events"] if e["role"] == "draft" and e["native_replay_returned"]).update(native_replay_returned=False)),
        ("foreign_replay_address", lambda x: next(e for e in x["graph"]["events"] if e["native_replay_returned"]).update(positional_address_match=False)),
        ("wrong_on_native_helper_count", lambda x: x["helper"].update(native_helper_calls_this_request=17)),
        ("timed_diagnostic_read_receipt", lambda x: x["window_guard"].update(diagnostic_device_reads_or_hash_calls_in_window=1)),
        ("missing_counted_terminal_completion", lambda x: x["window_guard"].update(explicit_terminal_device_fences=0)),
    ]
    for name, mutate in changes:
        altered = copy.deepcopy(base)
        mutate(altered)
        expect_reject(lambda: validate_geometry(altered, altered["token_ids"], "on", altered["core_id"], refs["0"]))
        controls.append(name + "_rejected")
    assert period2_after_prefill(refs["0"]) and period2_after_prefill(refs["2"])
    assert not period2_after_prefill([1] * 32) and not period2_after_prefill(list(range(32)))
    controls.append("period2_classifier_distinguishes_constant_and_nonrepeating_tokens")
    t = triplet(.5, 16, 31)
    assert t["A"] == 1.9375 and t["ms_per_step"] == 31.25 and t["single_stream_tok_s"] == 62
    p = pooled_triplet([{"decode_s": .5, "geometry": {"target_decode_rounds": 16, "decode_tokens": 31}},
                        {"decode_s": 1., "geometry": {"target_decode_rounds": 16, "decode_tokens": 31}}])
    assert p["steps"] == 32 and p["decode_tokens"] == 62 and p["decode_s"] == 1.5
    controls.append("pooled_triplet_uses_matching_steps_tokens_and_time")
    expect_reject(lambda: triplet(0, 16, 31))
    controls.append("invalid_timing_duration_rejected")
    for path in RUNTIME_FILES.values():
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
    remote = remote_program({"job": "cpu", "pairs": 8, "known": {}, "files": {},
                             "freeze_manifest_sha256": "cpu"})
    script_line = next(line for line in remote.splitlines() if line.startswith("script="))
    namespace = {"job": "cpu", "payload": {"pairs": 8}}
    exec(script_line, namespace)
    assert namespace["script"].startswith("set -e\npython ") and namespace["script"].endswith("--pairs 8\n")
    assert namespace["script"].count("\n") == 2
    controls.append("rendered_remote_shell_has_real_newlines_and_exact_benchmark_arguments")
    assert not any(name == "torch" or name.startswith("torch.") or name == "torch_npu" for name in sys.modules)
    result = {"passed": True, "controls_passed": len(controls), "controls": controls,
              "cpu_only_no_torch_or_npu_import": True, "npu_proof": False,
              "runtime_source_sha256": {name: sha(path) for name, path in RUNTIME_FILES.items()},
              "launcher_sha256": sha(BASE / "launch_fixture_timing.py"),
              "fixture_source_sha256": sha(fixture), "checker_sha256": sha(Path(__file__))}
    (BASE / "evidence/cpu_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("controls", "runtime_source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
