"""Verify all fresh counters/seals, then report bounded 2048/256 mechanism timing."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

from launch_fixture_timing import RUNTIME_FILES
from timing_controls import pooled_triplet, triplet, validate_geometry

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal_passed(fixture):
    assert fixture["passed"] and fixture["owner_buffers"] == 43 and len(fixture["checks"]) == 43
    assert not fixture["source_or_cache_values_mutated"]
    assert fixture["fixture_source_sha256"] == "7dfa11336f7be4c8c55d0f09eb302673ef31ac51d582686d31cffc432af6fe78"
    for check in fixture["checks"]:
        assert all(check[key] for key in ("finite", "owner_and_consumer_binding_equal",
                                          "cache_view_and_value_seal_equal", "source_checked"))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--job", default="dspark_lane_20261011_c4_fixture_timing_diagnostic_v3_2k_256")
    args = parser.parse_args()
    job = args.job
    directory = BASE / "runtime" / job
    results = directory / "results"
    freeze_path = BASE / "evidence" / (job + "_freeze.json")
    launch_path = BASE / "evidence" / (job + "_launch.json")
    fetch_path = BASE / "evidence" / (job + "_fetch.json")
    freeze, launch, fetch = read(freeze_path), read(launch_path), read(fetch_path)
    assert fetch["terminal"] and fetch["exit"] == "0" and (directory / "run.exit").read_text().strip() == "0"
    assert launch["freeze_manifest_sha256"] == sha(freeze_path)
    assert all(sha(RUNTIME_FILES[name]) == info["sha256"] for name, info in freeze["runtime_source_files"].items())
    assert all(sha(directory / info["relative_path"]) == info["sha256"] for info in fetch["files"])
    requests, pairs, summary, startup = (read(results / name) for name in (
        "requests.json", "pairs.json", "summary.json", "startup.json"))
    reference = read(BASE / "reference_token_ids_2k256.json")
    assert reference["pending"] is False and reference["source_job"] == "dspark_lane_20261011_c4_graph_geometry2k256_v3"
    assert len(requests) == freeze["pairs"] * 2 + 2 and len(pairs) == freeze["pairs"] >= 8
    assert startup["source_guard"]["all_actual_source_sha256_match"]
    assert startup["capture"]["actual_entries"] == 1 and startup["capture"]["constructor_use_cuda_graph"]
    assert not startup["capture"]["bool_flip"]
    assert len(startup["cold_helper_history"]) == 1 and startup["cold_helper_history"][0]["returned"]
    assert startup["cold_helper_history"][0]["capturing"]
    assert startup["parameter_metadata_views"] == 1032
    initializer = read(results / "cache_gate/fixed_o_proj_cache_fixture.json")
    assert len(initializer["entries"]) == 43 and initializer["initialized_before_compile_profile_capture"]
    assert initializer["native_loader_permutation_bitwise_equal"] and initializer["original_parameters_unchanged"]
    assert not initializer["capture_or_warmup_weight_mutation"]
    after_capture = read(results / "cache_gate/fixture_validation_timing_after_capture.json")
    seal_passed(after_capture)
    per_request = []
    for index, record in enumerate(requests, 1):
        raw = read(results / f"raw_{index:03d}.json")
        before = read(results / f"fixture_before_{index:03d}.json")
        after = read(results / f"fixture_after_{index:03d}.json")
        assert before["point"] == record["tag"] + ".before" and after["point"] == record["tag"] + ".after"
        seal_passed(before)
        seal_passed(after)
        assert raw["fixture_after"] == after
        assert raw["core_id"] == record["core_id"] and raw["token_ids"] == record["token_ids"]
        geometry = validate_geometry(raw, record["token_ids"], record["arm"], record["core_id"],
                                     reference["seeds"][str(record["seed"])]["token_ids"])
        assert geometry == record["geometry"]
        first, last = raw["counter"]["rounds"][0], raw["counter"]["rounds"][-1]
        seed_row = raw["counter"]["rounds"][1]
        assert record["decode_window_begin_s"] == first["bookkeeping_done_s"]
        assert record["decode_window_end_s"] == raw["counter"]["terminal_device_done_s"]
        assert record["decode_window_begin_s"] <= first["draft_propose_begin_s"] < first["draft_propose_done_s"]
        assert record["decode_window_begin_s"] <= seed_row["bookkeeping_done_s"]
        assert seed_row["draft_propose_begin_s"] < seed_row["draft_propose_done_s"] <= record["decode_window_end_s"]
        assert last["draft_propose_done_s"] <= record["decode_window_end_s"]
        assert math.isclose(record["decode_s"], record["decode_window_end_s"] - record["decode_window_begin_s"], rel_tol=1e-12)
        recomputed = triplet(record["decode_s"], geometry["target_decode_rounds"], geometry["decode_tokens"])
        assert all(math.isclose(record[key], value, rel_tol=1e-12) for key, value in recomputed.items())
        assert record["initial_and_terminal_draft_included_in_window"]
        assert record["partial_prefill_seed_and_terminal_draft_included_in_window"]
        assert record["before_and_after_full43_fixture_seals_passed"]
        assert record["diagnostic_device_reads_or_hash_calls_in_window"] == 0
        assert not record["route_logprob_audit"] and not record["profiler"]
        assert not record["tiny_candidate_admitted"] and not record["formal_candidate_admitted"] and not record["formal_tp8_claim"]
        per_request.append({"tag": record["tag"], "arm": record["arm"], "measured": record["measured"],
                            **recomputed, "geometry": geometry,
                            "native_explicit_fences_beyond_terminal": record["other_native_explicit_fences"],
                            "before_after_source_cache_seals": 86,
                            "diagnostic_device_reads_or_hash_calls_in_window": 0})
    by_tag = {row["tag"]: row for row in requests}
    for index, pair in enumerate(pairs):
        assert pair["order"] == (["off", "on"] if index % 2 == 0 else ["on", "off"])
        left, right = by_tag[pair["off_tag"]], by_tag[pair["on_tag"]]
        assert left["token_ids"] == right["token_ids"] and left["seed"] == right["seed"]
        assert pair["token_ids_equal"]
    measured = [row for row in requests if row["measured"]]
    assert collections.Counter(row["arm"] for row in measured) == {"off": freeze["pairs"], "on": freeze["pairs"]}
    pooled = {arm: pooled_triplet([row for row in measured if row["arm"] == arm]) for arm in ("off", "on")}
    assert pooled["off"] == summary["off"] and pooled["on"] == summary["on"]
    assert summary["dummy_clean_rate"] == 0 and summary["period2_repetition_requests"] == len(requests)
    assert not summary["tiny_candidate_admitted"] and not summary["formal_candidate_admitted"] and not summary["formal_tp8_claim"]
    assert not fetch["physical4_process_rows"] and any("OK" in line for line in fetch["physical4_smi_lines"])
    remaining = fetch["container_processes"].splitlines()[1:]
    assert len(remaining) == 1 and "sleep infinity" in remaining[0]
    frozen_v7 = read(BASE.parent / "delivery_v7_files.json")
    repo = BASE.parents[4]
    assert repo.name == "operator-stack-tp8"
    assert all(sha(repo / info["path"]) == info["sha256"] for info in frozen_v7["files"])
    output = {"job": job, "exit": 0, "scope": "Fresh physical4 TP1 tiny same-instance 2048/256 mechanism timing only",
              "cpu_controls": freeze["cpu_controls"], "cpu_gate_is_not_npu_proof": True,
              "freeze_manifest_sha256": sha(freeze_path), "actual_source_guard": startup["source_guard"],
              "fixture_source_sha256": freeze["fixture_source_sha256"],
              "model_config_sha256": freeze["model_config_sha256"],
              "vendor_source_py_tree_sha256": freeze["vendor_source_py_tree_sha256"],
              "fetched_files_sha_verified": len(fetch["files"]), "ssh_packed_bytes": fetch["ssh_packed_bytes"],
              "reference": {"source_job": reference["source_job"],
                            "source_requests_sha256": reference["source_requests_sha256"],
                            "extracted_unix": reference["extracted_unix"],
                            "seed_tags": {seed: reference["seeds"][seed]["tags"] for seed in reference["seeds"]}},
              "geometry": {"prompt_tokens": 2048, "completion_tokens": 256, "decode_tokens": 255,
                           "K": 5, "verify_scheduled_tokens": 6, "prefill_chunks": [2044, 4],
                           "target_decode_rounds_by_arm": {arm: sorted({row["geometry"]["target_decode_rounds"]
                                                                        for row in measured if row["arm"] == arm})
                                                           for arm in ("off", "on")},
                           "actual_draft_propose_calls_by_arm": {arm: sorted({row["geometry"]["actual_draft_propose_calls"]
                                                                              for row in measured if row["arm"] == arm})
                                                                 for arm in ("off", "on")},
                           "fresh_geometry_verified": True, "round_counts_derived_not_assumed": True,
                           "128_32_geometry_admission": False},
              "requests": len(requests), "warmup_requests": 2, "measured_requests": len(measured),
              "measured_requests_per_arm": freeze["pairs"], "interleaved_pairs": len(pairs),
              "mechanism_timing_pooled_triplets": pooled, "per_request": per_request,
              "initial_and_terminal_submitted_draft_included_in_every_window": True,
              "terminal_fences_per_request": 1, "cold_draft_capture": startup["capture"],
              "cold_native_helper_calls_and_successful_returns": [1, 1],
              "steady_per_request": {arm: {
                  "helper_calls": sorted({row["geometry"]["native_helper_calls"] for row in measured if row["arm"] == arm}),
                  "helper_successful_returns": sorted({row["geometry"]["native_helper_successful_returns"] for row in measured if row["arm"] == arm}),
                  "prefill_helper_mode": "NONE/eager" if arm == "on" else "NONE/eager",
                  "target_replay": sorted({row["geometry"]["target_native_replay_returns"] for row in measured if row["arm"] == arm}),
                  "draft_replay": sorted({row["geometry"]["draft_native_replay_returns"] for row in measured if row["arm"] == arm}),
                  "partial_prefill_draft_replay": sorted({row["geometry"]["partial_prefill_native_draft_replay_returns"] for row in measured if row["arm"] == arm}),
                  "seed_draft_replay": sorted({row["geometry"]["seed_native_draft_replay_returns"] for row in measured if row["arm"] == arm}),
                  "terminal_draft_replay": sorted({row["geometry"]["terminal_native_draft_replay_returns"] for row in measured if row["arm"] == arm})}
                  for arm in ("off", "on")},
              "fixed_fixture": {"before_compile_profile_capture_owner_buffers": 43,
                                "original_source_parameters_unchanged": True,
                                "after_capture_full43_seals_passed": True,
                                "before_and_after_every_request_full43_seals_passed": True,
                                "before_after_source_cache_seal_checks": 86 * len(requests),
                                "before_after_measured_request_seal_checks": 86 * len(measured),
                                "all_seal_reads_outside_timing_window": True},
              "timing_instrumentation": {"diagnostic_device_reads_or_hash_calls_in_window": 0,
                                         "route_logprob_audit": False, "first_value_trace": False, "profiler": False,
                                         "native_required_cpu_bookkeeping_unchanged": True,
                                         "native_explicit_fences_beyond_terminal_observed": sorted({row["other_native_explicit_fences"] for row in requests}),
                                         "host_counter_and_observer_overhead_included": True,
                                         "device_only_step_latency_claim": False,
                                         "full_consumer_numeric_gate_reexecuted_in_this_timing_job": False},
              "quality": {"period2_requests": len(requests), "period2_measured_requests": len(measured),
                          "all_pair_token_ids_equal": True, "frozen_2k256_reference_token_ids_equal": True,
                          "dummy_clean_rate": 0.0, "dummy_acceptance_must_not_be_used_as_benefit": True},
              "admission": {"tiny": False, "formal": False, "formal_tp8_claim": False,
                            "formal_full_main_mtp_verify_scheduler_cache_20ms_goal_achieved": False},
              "device_terminal": {"health": "OK", "physical_device": 4,
                                  "process_rows": fetch["physical4_process_rows"],
                                  "owned_model_processes_released": True, "time_unix": fetch["time_unix"],
                                  "smi_lines": fetch["physical4_smi_lines"],
                                  "container_processes": fetch["container_processes"]},
              "frozen_v7_files_unchanged": len(frozen_v7["files"]),
              "no_reset_no_dropcache_no_unknown_process_stop": True,
              "no_allocator_or_shared_pool_alias_mechanism_claim": True}
    path = BASE / "evidence" / (job + "_analysis.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"analysis": str(path), "sha256": sha(path), "exit": 0,
                      "mechanism_timing_pooled_triplets": pooled, "period2_requests": len(requests),
                      "dummy_clean_rate": 0.0, "admitted": False, "physical4_released": True}, indent=2))


if __name__ == "__main__":
    main()
