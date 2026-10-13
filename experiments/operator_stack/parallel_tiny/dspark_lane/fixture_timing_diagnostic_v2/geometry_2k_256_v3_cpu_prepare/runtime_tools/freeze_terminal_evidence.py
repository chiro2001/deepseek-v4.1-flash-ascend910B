"""Verify terminal originals, freeze compact copies, and write an exact delivery.

Run only after the bounded terminal fetch, remote CPU geometry aggregation, and
a final status query have finished. No remote commands or accelerator access.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

BASE = Path(__file__).resolve().parents[1]
REPO = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=BASE, text=True).strip())
JOB = "dspark_lane_20261011_c4_graph_geometry2k256_v3"
SOURCE_FREEZE_SHA256 = "827393390eca642c001808644201c83b1856425c4a5b080b7316fd82cfef43f7"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_rows(rows):
    for row in rows:
        path = REPO / row["path"]
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], str(path)


def put_original(target, raw):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == raw, str(target)
    else:
        target.write_bytes(raw)


def main():
    evidence = BASE / "runtime_evidence"
    fetch_path = evidence / (JOB + "_fetch.json")
    status_path = evidence / (JOB + "_status.json")
    aggregate_path = evidence / (JOB + "_geometry_terminal_aggregate.json")
    aggregate_receipt_path = evidence / (JOB + "_geometry_terminal_aggregate_receipt.json")
    collector_cpu_path = evidence / "collector_cpu_validation.json"
    fetch, status, aggregate = map(read, (fetch_path, status_path, aggregate_path))
    assert fetch["terminal"] and status["terminal"] and aggregate["terminal"]
    assert int(fetch["exit"]) == int(status["exit"]) == aggregate["exit"]
    assert not status["physical4_process_rows"] and "953926" not in status["container_processes"]
    assert status["ssh_packed_bytes"] < 900000
    assert fetch["original_compact_bytes_verified_after_assembly"] and fetch["largest_ssh_response_packed_bytes"] < 900000
    assert aggregate["cpu_json_only_no_accelerator_access"] and aggregate["all_runtime_source_files_match_preflight"]
    assert aggregate["remote_step_files_transferred"] is False
    for row in read(collector_cpu_path)["source_scripts"]:
        assert digest(BASE / row["path"]) == row["sha256"]
    source_freeze_path = BASE / "source_freeze_geometry2k256_v3.json"
    assert digest(source_freeze_path) == SOURCE_FREEZE_SHA256
    freeze = read(source_freeze_path)
    verify_rows(freeze["files"])
    cpu_delivery_path = BASE / "delivery_geometry2k256_v3_cpu_files.json"
    cpu_delivery = read(cpu_delivery_path)
    assert len(freeze["files"]) == 42 and len(cpu_delivery["files"]) == 43
    verify_rows(cpu_delivery["files"])
    old_deliveries = []
    for old in freeze["old_delivery_manifests_verified_unchanged"]:
        path = REPO / old["path"]
        assert digest(path) == old["sha256"], str(path)
        manifest = read(path)
        verify_rows(manifest["files"])
        old_deliveries.append({"path": old["path"], "sha256": old["sha256"],
                               "verified_files": len(manifest["files"]), "unchanged": True})
    original = BASE / "runtime" / JOB
    frozen = BASE / "frozen_runtime_terminal" / JOB
    delivered = []
    for row in fetch["files"]:
        path = original / row["relative_path"]
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], str(path)
        target = frozen / row["relative_path"]
        put_original(target, path.read_bytes())
        delivered.append(target)
    for path in (fetch_path, status_path, aggregate_path, aggregate_receipt_path, collector_cpu_path):
        target = frozen / "collector_receipts" / path.name
        put_original(target, path.read_bytes())
        delivered.append(target)
    results = frozen / "results"
    requests_transferred = (results / "requests.json").exists()
    if requests_transferred:
        records = read(results / "requests.json")
        assert digest(results / "requests.json") == aggregate["requests_source_sha256"]
    else:
        skipped = next(row for row in fetch["skipped_large_compact_files"] if row["relative_path"] == "results/requests.json")
        assert skipped["bytes"] == aggregate["requests_source_bytes"] >= 1000000
        assert skipped["sha256"] == aggregate["requests_source_sha256"]
        records = aggregate["request_records_without_tokens_and_logprobs"]
    checks = read(results / "checks.json")
    assert len(records) == aggregate["requests_completed"] == status["requests_completed"]
    assert len(checks) == status["numeric_checks_completed"]
    assert aggregate["verified_geometry_steps"] == sum(r["all_step_cache_geometry_receipts"] for r in records)
    assert all(r["performance_evaluated"] is False and r["formal_tp8_claim"] is False for r in records)
    aa_counts = Counter(c["kind"] for c in checks)
    full_gate = int(status["exit"]) == 0 and len(records) == 20 and aa_counts == {"native_off_AA": 3, "off_on": 3, "native_on_AA": 3}
    if full_gate:
        assert all(c["passed"] and c["token_equal"] and c["top5_keys_equal"] and c["max_logprob_delta"] < 1e-3 for c in checks)
        for check in checks:
            values = check["actual_consumers"]
            assert all(values[key] for key in ("passed", "native_proposals_bitwise", "ordered_route_ids_weights_bitwise",
                                               "draft_attention_consumers_bitwise", "draft_logical_cache_bitwise",
                                               "parameter_seal_passed", "all_step_actual_visible_cache_bitwise",
                                               "unretained_prefix_not_read", "independent_2048_256_gate"))
        accuracy, completion = map(read, (results / "accuracy_gate.json", results / "diagnostic_completion.json"))
        assert accuracy["passed"] and completion["consumer_comparator_passed_in_diagnostic_mode"]
        assert completion["performance_evaluation_disabled"] and completion["tiny_mechanism_candidate_admitted"] is False
        assert completion["formal_candidate_admitted"] is False
        assert aggregate["verified_geometry_steps"] == aggregate["geometry_step_files"] and not aggregate["unassigned_terminal_step_tags"]
        assert status["failure_lines"] == [] and status["nonfinite_files"] == []
    seals = []
    points = set()
    for path in sorted((results / "audit").glob("fixture_validation_*.json")):
        value = read(path)
        assert value["point"] not in points
        points.add(value["point"])
        assert value["owner_buffers"] == 43 and len(value["checks"]) == 43
        assert value["passed"] and value["source_or_cache_values_mutated"] is False
        assert all(c["finite"] and c["owner_and_consumer_binding_equal"] and c["cache_view_and_value_seal_equal"] and c["source_checked"] for c in value["checks"])
        seals.append({"point": value["point"], "owner_buffers": 43, "passed": True, "sha256": digest(path)})
    if full_gate:
        assert points == {"after_capture"} | {r["tag"] for r in records} | {r["tag"] + ".before" for r in records}
    guard = read(results / "audit/geometry2k_snapshot_entry_host_guard.json")
    assert guard["worker_id"] == guard["installed_worker_id"] and guard["worker_identity_equal"] and guard["audit_enabled"]
    assert guard["capture"] is False and guard["device_reads"] is False and guard["host_scalars_only"]
    preflight = read(frozen / "preflight.json")
    fresh = {name: preflight[name] for name in ("initial_gate", "before_sdk_gate", "before_model_gate")}
    for gate in fresh.values():
        memory = gate["normal_memory_budget"]
        assert gate["physical4"]["health"] == "OK" and gate["physical4_processes"] == []
        assert memory["gpu_memory_utilization"] == .70 and memory["budget_lowered"] is False
        assert memory["free_mb"] >= memory["required_free_mb"]
    log = (frozen / "run.log").read_text(errors="replace")
    analysis = {
        "job": JOB, "exit": int(status["exit"]), "complete_9_diagnostic_gate_passed": full_gate,
        "rank_observed": [0], "tensor_parallel_size": 1, "dummy_weights": True,
        "source_freeze_sha256": SOURCE_FREEZE_SHA256, "all_42_frozen_files_unchanged": True,
        "all_runtime_source_files_match_preflight": True, "cpu_controls": 43,
        "cpu_gate_sha256": freeze["cpu_gate_sha256"], "readiness_sha256": freeze["readiness_sha256"],
        "guard_source_trace_analysis_sha256": freeze["guard_source_trace_analysis_sha256"],
        "collector_CPU_validation_sha256": digest(collector_cpu_path),
        "request_records_completed": len(records), "audited_request_records_completed": sum(r["audit"] for r in records),
        "requests_source": {"bytes": aggregate["requests_source_bytes"], "sha256": aggregate["requests_source_sha256"],
                            "original_transferred": requests_transferred,
                            "compact_metadata_used_if_large": not requests_transferred},
        "numeric_checks_completed": len(checks), "checks_by_kind": dict(aa_counts),
        "checks_original_sha256": digest(results / "checks.json"), "checks": checks,
        "geometry": {key: value for key, value in aggregate.items()
                     if key not in ("requests", "runtime_source_files", "request_records_without_tokens_and_logprobs")},
        "geometry_aggregate_sha256": digest(aggregate_path), "per_request_geometry_and_helpers": aggregate["requests"],
        "fixed_fixture": {"distinct_source_cache_seal_points": len(seals), "owner_buffers": 43,
                          "source_cache_seal_checks": 43 * len(seals), "mirror_copies_not_double_counted": True, "seals": seals},
        "snapshot_entry_host_guard": guard, "entry_guard_is_not_precision_proof": True,
        "normal_init": {"fresh_gates": fresh, "sdk_exit": preflight["sdk"]["exit"],
                        "normal_gpu_fraction": .70, "KV_bytes": 2 * 1024**3,
                        "manual_KV_branch_skips_KV_memory_profiling": True,
                        "model_config_sha256": preflight["model_config_sha256"],
                        "vendor_source_py_tree_sha256": preflight["vendor_source"]["source_py_tree_sha256"]},
        "compact_transfer": {"fetched_files_verified": len(fetch["files"]), "fetch_receipt_sha256": digest(fetch_path),
                             "largest_ssh_response_packed_bytes": fetch["largest_ssh_response_packed_bytes"],
                             "all_responses_below_1MB": True, "skipped_large_compact_files": fetch["skipped_large_compact_files"]},
        "terminal_physical4": {"time_unix": status["time_unix"], "smi_lines": status["physical4_smi_lines"],
                               "process_rows": status["physical4_process_rows"], "container_processes": status["container_processes"],
                               "owned_model_processes_released": True},
        "failure_lines": status["failure_lines"], "nonfinite_files": status["nonfinite_files"],
        "traceback_tail_if_failed": log[log.rfind("Traceback"):][-14000:] if "Traceback" in log and int(status["exit"]) else None,
        "old_delivery_manifests_verified_unchanged": old_deliveries,
        "diagnostic_scope": {"invasive_diagnostic_reads_add_sync": True, "performance_evaluated": False,
                             "quoted_performance_triplet": None, "dummy_clean_rate": 0.0, "tiny_admitted": False,
                             "formal_admitted": False, "formal_tp8_claim": False, "formal_20ms_goal_achieved": False},
        "only_one_authorized_NPU_window_used": True, "no_restart_no_reset_no_dropcache_no_tenant_stop": True,
        "current_NPU_permission": 0,
    }
    analysis_path = frozen / "terminal_analysis.json"
    put_original(analysis_path, (json.dumps(analysis, indent=2) + "\n").encode())
    delivered.append(analysis_path)
    delivered += [BASE / "runtime_tools" / name for name in ("fetch_geometry_gate_bounded.py", "fetch_checks_snapshot.py", "aggregate_geometry_terminal.py", "freeze_terminal_evidence.py")]
    delivered.append(cpu_delivery_path)
    paths = {REPO / row["path"] for row in cpu_delivery["files"]} | set(delivered)
    files = [{"path": str(path.relative_to(REPO)), "sha256": digest(path), "bytes": path.stat().st_size} for path in sorted(paths)]
    delivery = {"scope": "Exact v3 frozen CPU preparation and terminal dummy diagnostic evidence; no timing or qualification claim",
                "job": JOB, "source_freeze_sha256": SOURCE_FREEZE_SHA256, "terminal_analysis_sha256": digest(analysis_path),
                "complete_9_diagnostic_gate_passed": full_gate, "performance_evaluated": False,
                "tiny_admitted": False, "formal_admitted": False, "files": files,
                "total_files": len(files), "total_bytes": sum(row["bytes"] for row in files),
                "safe_to_stage_exact_paths_only": True, "no_commit_created": True}
    delivery_path = BASE / "delivery_geometry2k256_v3_terminal_files.json"
    put_original(delivery_path, (json.dumps(delivery, indent=2) + "\n").encode())
    print(json.dumps({"delivery_path": str(delivery_path.relative_to(REPO)), "delivery_sha256": digest(delivery_path),
                      "total_files": len(files), "total_bytes": delivery["total_bytes"],
                      "terminal_analysis_sha256": digest(analysis_path), "complete_9_diagnostic_gate_passed": full_gate}, indent=2))


if __name__ == "__main__":
    main()
