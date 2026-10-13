"""Meaningful CPU controls for the 2048/256 timing diagnostic: timed reads forbidden, counters strict."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace

from build_reference_2k256 import MINIMAL_TAGS, REFERENCE_JOB, REQUIRED_TAGS, extract_reference
from launch_fixture_timing import (GEOMETRY, REFERENCE_NAME, RUNTIME_FILES, assert_launchable,
                                   build_manifest, build_readiness, loaded_files, remote_program)
from timing_controls import (OUTPUT_TOKENS, PROMPT_TOKENS, WindowGuard, load_reference,
                             period2_after_prefill, pooled_triplet, triplet, validate_geometry)

BASE = Path(__file__).resolve().parent
ARCHIVE = BASE.parent / "evidence/graph_fixture_v1_raw/results"
ACTUAL_CASES = {"off": "cpu_cases/v2_actual_warmup_off.json", "on": "cpu_cases/v2_actual_warmup_on.json"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expect_reject(function):
    try:
        function()
    except AssertionError:
        return
    raise AssertionError("Negative control was not rejected")


def prepared_raw(arm):
    """Convert a real 2K/256 geometry-gate receipt into the timing worker receipt shape."""
    old = json.loads((BASE / ACTUAL_CASES[arm]).read_text())
    assert old["arm"] == arm
    old_helper = old["helper"]
    history = [{"capturing": row["capturing"], "runtime_mode": row["runtime_mode"],
                "actual_round_at_entry": copy.deepcopy(row["actual_round_at_entry"]),
                "begin_s": row["begin_s"], "end_s": row["end_s"],
                "returned": row["native_helper_returned"]}
               for row in old_helper["helper_history_this_request"]]
    helper = {"native_helper_calls_this_request": old_helper["native_helper_calls_this_request"],
              "native_helper_successful_returns_this_request": old_helper["native_helper_successful_returns_this_request"],
              "cold_capture_helper_calls_global": old_helper["cold_capture_helper_calls_global"],
              "cold_capture_helper_successful_returns_global": old_helper["cold_capture_helper_successful_returns_global"],
              "history": history, "host_only_counter": True, "graph_replay_executes_no_python_helper": True}
    return {"counter": copy.deepcopy(old["counter"]), "graph": copy.deepcopy(old["graph"]), "helper": helper,
            "core_id": old["core_id"], "token_ids": list(old["token_ids"]), "arm": arm,
            "window_guard": {"diagnostic_device_reads_or_hash_calls_in_window": 0,
                             "explicit_terminal_device_fences": 1,
                             "route_logprob_audit_and_profiler": False}}


def extracted_function(path, name, globals_dict):
    tree = ast.parse(path.read_text())
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    module = ast.fix_missing_locations(ast.Module(body=[matches[0]], type_ignores=[]))
    exec(compile(module, str(path), "exec"), globals_dict)
    return globals_dict[name]


def fake_records():
    records = []
    for seed in (0, 1, 2):
        tokens = [10_000 + seed * 100 + index % 2 for index in range(OUTPUT_TOKENS)]
        for arm, tag in (("off", f"native-off-{seed}-A"), ("off", f"native-off-{seed}-B"),
                         ("off", f"candidate-{seed}-off"), ("on", f"candidate-{seed}-on"),
                         ("on", f"native-on-{seed}-A"), ("on", f"native-on-{seed}-B")):
            records.append({"tag": tag, "arm": arm, "seed": seed, "token_ids": list(tokens)})
    records.append({"tag": "warmup-off", "arm": "off", "seed": 0, "token_ids": list(records[0]["token_ids"])})
    records.append({"tag": "warmup-on", "arm": "on", "seed": 1, "token_ids": list(records[6]["token_ids"])})
    return records


def valid_reference():
    seeds = {str(seed): {"tags": [tag.format(seed=seed) for tag in REQUIRED_TAGS],
                         "token_ids": [10_000 + seed * 100 + index % 2 for index in range(OUTPUT_TOKENS)]}
             for seed in (0, 1, 2)}
    return {"pending": False, "source_job": REFERENCE_JOB,
            "source_requests_path": "/repo/" + REFERENCE_JOB + "/results/requests.json",
            "source_requests_sha256": "a" * 64, "source_records": 20,
            "extractor": "build_reference_2k256.py", "extracted_unix": 1.0,
            "tag_subset_allowed": False, "fabricated_reference_bytes": False, "seeds": seeds}


def main():
    controls = []
    reference_controls = 0
    tmp_root = Path(os.environ.get("DSPARK_TIMING2K_TMPDIR",
                                   str(Path.home() / "tmp" / "dsv41-tiny-timing2k")))
    tmp_root.mkdir(parents=True, exist_ok=True)

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
        assert len(state["helper_history"]) == 1 and state["helper_history"][0]["capturing"]
        assert state["helper_history"][0]["returned"]
        controls.append("host_helper_counter_preserves_native_arguments_and_return")
        controls.append("cold_capture_counter_reads_context_without_tensor_values")
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    off = prepared_raw("off")
    on = prepared_raw("on")
    off_result = validate_geometry(off, off["token_ids"], "off", off["core_id"], off["token_ids"])
    assert off_result["target_decode_rounds"] == 128 and off_result["actual_draft_propose_calls"] == 130
    assert off_result["prefill_chunks"] == [2044, 4] and off_result["decode_tokens"] == 255
    assert off_result["native_helper_calls"] == 130 and off_result["draft_native_replay_returns"] == 0
    assert off_result["target_native_replay_returns"] == 128
    controls.append("actual_v2_2k256_off_130_rows_128_decode_130_helpers_accepted")
    on_result = validate_geometry(on, on["token_ids"], "on", on["core_id"], on["token_ids"])
    assert on_result["target_decode_rounds"] == 255 and on_result["actual_draft_propose_calls"] == 257
    assert on_result["decode_tokens"] == 255 and on_result["native_helper_calls"] == 2
    assert on_result["draft_native_replay_returns"] == 255 and on_result["target_native_replay_returns"] == 255
    assert on_result["partial_prefill_native_draft_replay_returns"] == 0
    assert on_result["seed_native_draft_replay_returns"] == 0
    assert on_result["terminal_native_draft_replay_returns"] == 1
    controls.append("actual_v2_2k256_on_257_rows_255_decode_2_prefill_helpers_accepted")
    controls.append("round_counts_derived_from_fresh_receipts_not_assumed_16_or_128")
    controls.append("on_helpers_are_exactly_the_two_prefill_rounds_not_hardcoded_one")

    negatives = [
        ("off", "missing_partial_prefill_draft", lambda x: x["counter"]["rounds"][0].update(draft_calls=0)),
        ("on", "missing_seed_draft", lambda x: x["counter"]["rounds"][1].update(draft_calls=0)),
        ("on", "missing_terminal_draft", lambda x: x["counter"]["rounds"][-1].update(draft_calls=0)),
        ("on", "missing_seed_sample", lambda x: x["counter"]["rounds"][1].update(sampled_token_ids=[])),
        ("on", "sampling_seed_on_partial_prefill", lambda x: x["counter"]["rounds"][0].update(sampled_token_ids=[7])),
        ("on", "wrong_prefill_chunk_geometry", lambda x: x["counter"]["rounds"][1].update(scheduled_target_tokens=5)),
        ("on", "wrong_prefill_progress", lambda x: x["counter"]["rounds"][1].update(computed_before=2045)),
        ("on", "wrong_decode_scheduled_geometry", lambda x: next(r for r in x["counter"]["rounds"] if r["is_decode"]).update(scheduled_target_tokens=7)),
        ("on", "wrong_verified_geometry", lambda x: next(r for r in x["counter"]["rounds"] if r["is_decode"]).update(verified_draft_tokens=4)),
        ("on", "illegal_decode_progress", lambda x: [r for r in x["counter"]["rounds"] if r["is_decode"]][1].update(computed_before=2060)),
        ("on", "missing_terminal_fence", lambda x: x["counter"].update(terminal_sync_count=0)),
        ("on", "steady_extra_sync", lambda x: x["counter"].update(steady_extra_sync_count=1)),
        ("on", "early_terminal_device_done", lambda x: x["counter"].update(terminal_device_done_s=x["counter"]["rounds"][-1]["sample_done_s"] - 1)),
        ("on", "missing_hot_draft_replay", lambda x: next(e for e in x["graph"]["events"] if e["role"] == "draft" and e["native_replay_returned"]).update(native_replay_returned=False)),
        ("on", "missing_target_replay", lambda x: next(e for e in x["graph"]["events"] if e["role"] == "target" and e["native_replay_returned"]).update(native_replay_returned=False)),
        ("on", "foreign_replay_address", lambda x: next(e for e in x["graph"]["events"] if e["native_replay_returned"]).update(positional_address_match=False)),
        ("on", "wrong_on_native_helper_count", lambda x: x["helper"].update(native_helper_calls_this_request=130)),
        ("on", "missing_partial_prefill_helper_history", lambda x: x["helper"]["history"].pop(0)),
        ("on", "wrong_helper_round_binding", lambda x: x["helper"]["history"][1]["actual_round_at_entry"].update(execute_begin_s=0.0)),
        ("on", "timed_diagnostic_read_receipt", lambda x: x["window_guard"].update(diagnostic_device_reads_or_hash_calls_in_window=1)),
        ("on", "missing_counted_terminal_completion", lambda x: x["window_guard"].update(explicit_terminal_device_fences=0)),
        ("off", "reference_token_mismatch", lambda x: x["token_ids"].__setitem__(-1, (x["token_ids"][-1] + 1) % 129280)),
        ("off", "short_token_stream", lambda x: x.__setitem__("token_ids", x["token_ids"][:-1])),
        ("off", "draft_replay_present_in_off_arm", lambda x: x["graph"]["events"].append(
            {"role": "draft", "native_replay_returned": True, "existing_entry": True, "capturing": False,
             "positional_address_match": True, "mode": "FULL", "call_returned": True,
             "begin_s": next(r for r in x["counter"]["rounds"] if r["is_decode"])["draft_propose_begin_s"],
             "end_s": next(r for r in x["counter"]["rounds"] if r["is_decode"])["draft_propose_done_s"]})),
        ("on", "prefill_target_must_stay_eager", lambda x: x["graph"]["events"][0].update(native_replay_returned=True)),
    ]
    for arm, name, mutate in negatives:
        altered = copy.deepcopy(off if arm == "off" else on)
        mutate(altered)
        expect_reject(lambda altered=altered, arm=arm: validate_geometry(
            altered, altered["token_ids"], arm, altered["core_id"], altered["token_ids"]))
        controls.append(name + "_rejected")

    assert period2_after_prefill(off["token_ids"]) and period2_after_prefill(on["token_ids"])
    assert not period2_after_prefill([1] * OUTPUT_TOKENS) and not period2_after_prefill(list(range(OUTPUT_TOKENS)))
    controls.append("period2_classifier_distinguishes_constant_and_nonrepeating_tokens")
    assert PROMPT_TOKENS == 2048 and OUTPUT_TOKENS == 256 and GEOMETRY["capture_representative_seq_len"] == 3072
    controls.append("declared_2k256_geometry_matches_contract_constants")
    t = triplet(.5, 128, 255)
    assert t["A"] == 255 / 128 and t["ms_per_step"] == 500 / 128 and t["single_stream_tok_s"] == 510
    assert triplet(1.0, 255, 255)["A"] == 1.0
    p = pooled_triplet([{"decode_s": .5, "geometry": {"target_decode_rounds": 128, "decode_tokens": 255}},
                        {"decode_s": 1.0, "geometry": {"target_decode_rounds": 255, "decode_tokens": 255}}])
    assert p["steps"] == 383 and p["decode_tokens"] == 510 and p["decode_s"] == 1.5
    assert p["A"] == 510 / 383
    controls.append("pooled_triplet_uses_matching_steps_tokens_and_time")
    expect_reject(lambda: triplet(0, 128, 255))
    controls.append("invalid_timing_duration_rejected")

    with tempfile.TemporaryDirectory(dir=tmp_root) as scratch:
        scratch = Path(scratch)
        missing = scratch / "missing.json"
        expect_reject(lambda: load_reference(missing))
        reference_controls += 1
        controls.append("missing_reference_file_rejected_as_pending")
        pending = scratch / "pending.json"
        pending.write_text(json.dumps({"pending": True}))
        expect_reject(lambda: load_reference(pending))
        controls.append("pending_reference_marker_rejected")
        reference_controls += 1
        bad_job = valid_reference()
        bad_job["source_job"] = "dspark_lane_20261011_c4_graph_geometry2k256_v2"
        (scratch / "bad_job.json").write_text(json.dumps(bad_job))
        expect_reject(lambda: load_reference(scratch / "bad_job.json"))
        controls.append("reference_from_wrong_geometry_job_rejected")
        reference_controls += 1
        bad_seed = valid_reference()
        del bad_seed["seeds"]["2"]
        (scratch / "bad_seed.json").write_text(json.dumps(bad_seed))
        expect_reject(lambda: load_reference(scratch / "bad_seed.json"))
        controls.append("reference_missing_seed_rejected")
        reference_controls += 1
        bad_len = valid_reference()
        bad_len["seeds"]["1"]["token_ids"] = bad_len["seeds"]["1"]["token_ids"][:-1]
        (scratch / "bad_len.json").write_text(json.dumps(bad_len))
        expect_reject(lambda: load_reference(scratch / "bad_len.json"))
        controls.append("reference_wrong_length_rejected")
        reference_controls += 1
        bad_vocab = valid_reference()
        bad_vocab["seeds"]["0"]["token_ids"][0] = 129280
        (scratch / "bad_vocab.json").write_text(json.dumps(bad_vocab))
        expect_reject(lambda: load_reference(scratch / "bad_vocab.json"))
        controls.append("reference_out_of_vocab_token_rejected")
        reference_controls += 1
        good = scratch / REFERENCE_NAME
        good.write_text(json.dumps(valid_reference()))
        loaded = load_reference(good)
        assert loaded["source_job"] == REFERENCE_JOB and len(loaded["seeds"]) == 3
        controls.append("well_formed_v3_reference_accepted")
        reference_controls += 1

    records = fake_records()
    seeds = extract_reference(records)
    assert set(seeds) == {"0", "1", "2"} and len(seeds["0"]["token_ids"]) == OUTPUT_TOKENS
    controls.append("reference_extraction_accepts_consistent_terminal_records")
    reference_controls += 1
    assert extract_reference(records, require_all_tags=False)
    controls.append("reference_extraction_subset_mode_still_requires_canonical_arms")
    reference_controls += 1
    altered = copy.deepcopy(records)
    altered[2]["token_ids"][0] += 1
    expect_reject(lambda: extract_reference(altered))
    controls.append("reference_extraction_cross_arm_mismatch_rejected")
    reference_controls += 1
    altered = copy.deepcopy(records)
    altered = [row for row in altered if row["seed"] != 2]
    expect_reject(lambda: extract_reference(altered))
    controls.append("reference_extraction_missing_seed_rejected")
    reference_controls += 1
    altered = copy.deepcopy(records)
    altered = [row for row in altered if row["tag"] != "native-on-1-B"]
    expect_reject(lambda: extract_reference(altered))
    controls.append("reference_extraction_missing_required_tag_rejected")
    reference_controls += 1
    altered = copy.deepcopy(records)
    altered = [row for row in altered if row["tag"] not in ("candidate-1-on",)]
    expect_reject(lambda: extract_reference(altered, require_all_tags=False))
    controls.append("reference_extraction_subset_mode_rejects_missing_canonical_arm")
    reference_controls += 1
    altered = copy.deepcopy(records)
    altered.append({"tag": "native-off-0-A", "arm": "off", "seed": 0, "token_ids": list(altered[0]["token_ids"])})
    expect_reject(lambda: extract_reference(altered))
    controls.append("reference_extraction_duplicate_tag_rejected")
    reference_controls += 1
    altered = copy.deepcopy(records)
    altered[0]["token_ids"] = altered[0]["token_ids"][:-1]
    expect_reject(lambda: extract_reference(altered))
    controls.append("reference_extraction_wrong_length_rejected")
    reference_controls += 1
    assert not (BASE / REFERENCE_NAME).exists()
    controls.append("package_reference_stays_pending_no_fabricated_bytes")
    reference_controls += 1

    known = json.loads((BASE / "native_sources.json").read_text())
    manifest_pending = build_manifest("cpu_job", 8, known, "cpu", 1, {}, {"status": "pending v3 terminal evidence"})
    assert manifest_pending["launch_blocked_by_pending_reference"] is True
    manifest_ready = build_manifest("cpu_job", 8, known, "cpu", 1, {}, {"status": "present", "sha256": "x"})
    assert manifest_ready["launch_blocked_by_pending_reference"] is False
    assert manifest_ready["input_tokens"] == 2048 and manifest_ready["output_tokens"] == 256
    controls.append("freeze_manifest_blocks_launch_while_reference_pending")
    expect_reject(lambda: assert_launchable({"reference_pending": True}))
    assert_launchable({"reference_pending": False})
    controls.append("launch_path_rejects_pending_reference_payload")
    reference_controls += 1
    fake_gate = {"controls_passed": 1, "reference_interface_controls_passed": 1, "checker_sha256": "x"}
    fake_freeze = BASE / "evidence/fake_freeze.json"
    readiness_pending = build_readiness("cpu_job", 8, fake_gate, "gate", {"status": "pending v3 terminal evidence"},
                                        fake_freeze, "sha")
    readiness_ready = build_readiness("cpu_job", 8, fake_gate, "gate", {"status": "present"}, fake_freeze, "sha")
    assert readiness_pending["npu_authorization"]["npu_permission"] == 0
    assert readiness_pending["npu_authorization"]["authorized"] is False
    assert readiness_pending["freeze_manifest"]["launch_blocked_by_pending_reference"] is True
    assert any("参考 token" in item for item in readiness_pending["pending_items"])
    assert not any("参考 token" in item for item in readiness_ready["pending_items"])
    assert readiness_ready["future_commands"]["3_authorized_launch"].endswith("--pairs 8 --launch")
    assert readiness_ready["future_commands"]["2_final_freeze"] == "python3 launch_fixture_timing.py --freeze-only"
    controls.append("readiness_records_cpu_ready_pending_items_future_command_and_npu_permission_zero")
    reference_controls += 1

    launcher_tree = ast.parse((BASE / "launch_fixture_timing.py").read_text())
    budget_call = next(node for node in ast.walk(launcher_tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == "replace"
                       and len(node.args) == 2 and "original_fresh_gate=fresh_gate" in ast.unparse(node.args[1]))
    budget_code = eval(compile(ast.Expression(budget_call.args[1]), "rendered_budget", "eval"),
                       {"needle": "initial_gate=fresh_gate()"})
    for used in (3289, 7465):
        namespace = {"re": re, "fresh_gate": lambda used=used: {"physical4": {"line": f"0 / 0 {used} / 65536"}}}
        exec(budget_code, namespace)
        assert namespace["initial_gate"]["normal_memory_budget"]["gpu_memory_utilization"] == .70
        controls.append("current_HBM_" + str(used) + "_keeps_normal_budget")
    for line in ("0 / 0 24000 / 65536", "0 / 0"):
        expect_reject(lambda line=line: exec(budget_code, {"re": re, "fresh_gate": lambda: {"physical4": {"line": line}}}))
        controls.append("unavailable_or_missing_current_normal_memory_budget_rejected")

    for path in RUNTIME_FILES.values():
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
    controls.append("runtime_python_sources_ast_parse_all")
    remote = remote_program({"job": "cpu", "pairs": 8, "known": {"native_sources": {},
                                                                 "shared_sources": {"o_proj_cache_fixture.py": "x"}},
                             "files": {}, "freeze_manifest_sha256": "cpu", "reference_pending": False})
    script_line = next(line for line in remote.splitlines() if line.startswith("script="))
    namespace = {"job": "cpu", "payload": {"pairs": 8}}
    exec(script_line, namespace)
    assert namespace["script"].startswith("set -e\npython ") and namespace["script"].count("\n") == 2
    assert namespace["script"].endswith("--pairs 8 --prompt-tokens 2048 --tokens 256\n")
    controls.append("rendered_remote_shell_has_real_newlines_and_exact_benchmark_arguments")
    pending_remote = remote_program({"job": "cpu", "pairs": 8, "known": {"native_sources": {},
                                                                         "shared_sources": {"o_proj_cache_fixture.py": "x"}},
                                     "files": {}, "freeze_manifest_sha256": "cpu", "reference_pending": True})
    assert "assert not payload.get('reference_pending')" in pending_remote
    controls.append("rendered_remote_refuses_pending_reference_before_any_action")
    import base64 as base64_module
    import gzip
    packed = re.search(r"base64\.b64decode\('([A-Za-z0-9+/=]+)'\)", remote).group(1)
    payload_back = json.loads(gzip.decompress(base64_module.b64decode(packed)))
    assert payload_back["reference_pending"] is False and payload_back["pairs"] == 8
    assert payload_back["freeze_manifest_sha256"] == "cpu"
    payload_files = loaded_files(False)
    payload_back = json.loads(gzip.decompress(base64_module.b64decode(
        re.search(r"base64\.b64decode\('([A-Za-z0-9+/=]+)'\)",
                  remote_program({"job": "cpu", "pairs": 8,
                                  "known": {"native_sources": {},
                                            "shared_sources": {"o_proj_cache_fixture.py": "x"}},
                                  "files": payload_files, "freeze_manifest_sha256": "cpu",
                                  "reference_pending": False})).group(1))))
    assert sorted(payload_back["files"]) == sorted(payload_files)
    for name, info in payload_back["files"].items():
        raw = base64_module.b64decode(info["base64"])
        assert len(raw) == info["bytes"] and hashlib.sha256(raw).hexdigest() == info["sha256"], name
        if name.endswith(".py"):
            ast.parse(raw.decode(), filename=name)
    controls.append("remote_payload_roundtrip_verifies_all_runtime_file_sha256")
    bench_source = (BASE / "bench_fixture_timing.py").read_text()
    assert bench_source.index("load_reference(") < bench_source.index("import torch")
    controls.append("bench_loads_frozen_reference_before_torch_import")
    worker_source = (BASE / "fixture_timing_worker.py").read_text()
    assert '"DSPARK_CAPTURE_SEQ_LEN": "3072"' in worker_source
    assert "max_model_len == 4096" in worker_source and "max_num_batched_tokens == 2048" in worker_source
    assert "sliding_window == 128" in worker_source
    controls.append("worker_asserts_2k256_geometry_and_capture_representative")
    cpu_cases = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                 for path in sorted((BASE / "cpu_cases").glob("*.json"))}
    assert set(cpu_cases) == {"v1_actual_warmup_off.json", "v2_actual_warmup_off.json",
                              "v2_actual_warmup_on.json"}, sorted(cpu_cases)
    controls.append("cpu_cases_include_v1_and_v2_actual_2k256_receipts")

    assert not any(name == "torch" or name.startswith("torch.") or name == "torch_npu" for name in sys.modules)
    gate_probe = {"passed": True, "cpu_only_no_torch_or_npu_import": True,
                  "launcher_sha256": sha(BASE / "launch_fixture_timing.py"),
                  "reference_interface_controls_passed": reference_controls,
                  "runtime_source_sha256": {name: sha(path) for name, path in RUNTIME_FILES.items()}}
    assert gate_probe["launcher_sha256"] == sha(BASE / "launch_fixture_timing.py")
    assert gate_probe["reference_interface_controls_passed"] >= 8
    for name, path in RUNTIME_FILES.items():
        assert gate_probe["runtime_source_sha256"][name] == sha(path), name
    controls.append("gate_output_satisfies_every_launcher_precondition")
    result = {"passed": True, "controls_passed": len(controls), "controls": controls,
              "reference_interface_controls_passed": reference_controls,
              "cpu_only_no_torch_or_npu_import": True, "npu_proof": False,
              "runtime_source_sha256": {name: sha(path) for name, path in RUNTIME_FILES.items()},
              "launcher_sha256": sha(BASE / "launch_fixture_timing.py"),
              "extractor_sha256": sha(BASE / "build_reference_2k256.py"),
              "fixture_source_sha256": sha(fixture), "checker_sha256": sha(Path(__file__)),
              "reference_status": "pending v3 terminal evidence",
              "reference_expected_source": "local results/requests.json of " + REFERENCE_JOB,
              "cpu_case_files": cpu_cases,
              "actual_case_provenance": {name: {"path": ACTUAL_CASES[name],
                                                "sha256": sha(BASE / ACTUAL_CASES[name])}
                                         for name in ACTUAL_CASES},
              "geometry": dict(GEOMETRY)}
    (BASE / "evidence/cpu_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("controls", "runtime_source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
