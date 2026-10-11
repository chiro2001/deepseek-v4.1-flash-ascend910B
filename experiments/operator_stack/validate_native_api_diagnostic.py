"""Offline verification of compact, real-weight native DSpark HTTP evidence.

The original full-round logs remain on the remote host. This verifier checks
their recorded hashes, the local original HTTP bytes and all retained rank
boundaries. It never promotes a partial consumer audit to formal acceptance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

from parallel_tiny.dspark_api_counter_bridge_cpu.quality_cpu import quality
from parallel_tiny.dspark_api_counter_bridge_cpu.runtime_guards import token_hash


COUNTER = "da845a5891557f21e80eac06d96aec1955263d1ae93afbc4d84e86dd9585dbec"
CONFIG = "40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424"
ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "evidence/formal_a321/native_api_v2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, name):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), name)


def ranks(items, name):
    require(len(items) == 8 and {item["rank"] for item in items} == set(range(8)), name)
    return {item["rank"]: item for item in items}


def no_admission(record, name, keys=("actual_device_consumer_proof", "performance_receipt_granted", "formal_receipt_granted")):
    for key in keys:
        require(record.get(key) is False, name + ": unsupported " + key)


def verify_window(row, http, selection):
    finish, bridge = row["finish"], http["bridge"]
    metrics, mapping = finish["metrics"], finish["mapping"]
    clock, core = metrics["clock_scope"], mapping["core_request_id"]
    require(clock == selection["identity"]["clock_scope"] and clock["clock"] == "CLOCK_MONOTONIC",
            "Client/worker boot or time namespace changed")
    require(clock["boot_id"] and clock["time_namespace"], "Clock has no actual namespace")
    require(finish["service_id"] == bridge["service_id"] == selection["service_id"], "Foreign service")
    require(finish["window_id"] == bridge["window_id"] and
            finish["phase"] == bridge["phase"] == row["phase"] and
            finish["index"] == bridge["index"] == row["index"], "Window sequence mismatch")
    require(bridge["clock_scope"] == clock and bridge["cpu_mock"] is False
            and finish["cpu_mock"] is False and metrics["cpu_mock"] is False, "Mock/foreign HTTP window")
    no_admission(finish, "finish"); no_admission(metrics, "metrics"); no_admission(bridge, "HTTP")
    require(mapping["completion_response_id"] == "cmpl-" + mapping["http_request_id"]
            and mapping["engine_external_request_id"] == mapping["completion_response_id"] + "-0"
            and core.startswith(mapping["engine_external_request_id"] + "-"), "External/core mapping")
    require(metrics["core_request_id"] == bridge["core_request_id"] == core and
            bridge["engine_external_request_id"] == mapping["engine_external_request_id"] and
            metrics["exact_core_request_id_bound"] is True, "Wrong core request")
    completion = http["completion"]
    require(completion["id"] == mapping["completion_response_id"]
            and completion["model"] == selection["served_model"] and len(completion["choices"]) == 1,
            "Completion identity")
    choice = completion["choices"][0]
    tokens = choice["token_ids"]
    require(len(tokens) == 256 and all(type(t) is int and t >= 0 for t in tokens), "HTTP token budget")
    require(choice["index"] == 0 and choice["finish_reason"] == "length"
            and len(choice["prompt_token_ids"]) == 2048
            and token_hash(choice["prompt_token_ids"]) == row["prompt_sha256"], "HTTP prompt/output coverage")
    require(completion["usage"] == {"prompt_tokens": 2048, "completion_tokens": 256, "total_tokens": 2304},
            "HTTP usage mismatch")
    require(quality(choice["text"], tokens) == row["quality"], "Clean/repetition classification changed")
    require(metrics["decoded_tokens"] == 255 and metrics["prefill_seed_tokens"] == 1
            and metrics["emitted_tokens_including_prefill_seed"] == 256, "Prefill seed counted as decode")
    require(type(metrics["rounds"]) is int and metrics["rounds"] > 0, "Actual verifier rounds unavailable")
    ended = row["client_end_ns"]
    require(metrics["client_end_ns"] == ended, "Client read-end timestamp changed")
    counters = ranks(metrics["rank_counters"], "Counter ranks incomplete")
    receipts = ranks(finish["rank_receipts"], "Receipt ranks incomplete")
    begins = ranks(finish["begin_acks"], "Begin ranks incomplete")
    binds = ranks(finish["bind_acks"], "Bind ranks incomplete")
    descriptions = ranks(selection["worker_descriptions"], "Worker descriptions incomplete")
    events = {event["event"]: event for event in finish["events"]}
    ordered = ["begin_rpc_call", "begin_all_tp8_acked", "stock_generate_external_id_observed",
               "mapping_tap_before_core_submit", "bind_all_tp8_acked", "native_core_submit_call",
               "finished_native_request_output_observed", "stock_completion_response_ready",
               "http_output_send_begin", "http_body_sent", "finish_rpc_call_after_client_end",
               "finish_all_tp8_retrieved"]
    require(len(events) == len(ordered) and set(events) == set(ordered), "Missing/duplicated bridge events")
    times = [events[name]["observed_ns"] for name in ordered]
    require(times == sorted(times) and events["http_body_sent"]["observed_ns"] <= ended
            < events["finish_rpc_call_after_client_end"]["observed_ns"], "Finish RPC included in clock")
    require(events["finish_rpc_call_after_client_end"]["client_end_ns"] == ended
            and events["native_core_submit_call"]["core_request_id"] == core
            and events["stock_generate_external_id_observed"]["engine_external_request_id"] == mapping["engine_external_request_id"],
            "Event request identity")
    tap = events["mapping_tap_before_core_submit"]
    require(tap["prompt_sha256"] == row["prompt_sha256"] and tap["window_id"] == finish["window_id"]
            and all(tap[key] == mapping[key] for key in ("http_request_id", "engine_external_request_id", "core_request_id")),
            "Before-submit tap mismatch")
    starts, joins, end_samples, rank_spans, common_lengths, common_boundary = [], [], [], [], None, None
    for rank, receipt in sorted(receipts.items()):
        for ack in (begins[rank], binds[rank]):
            require(ack["clock_scope"] == clock and ack["counter_sha256"] == COUNTER, "ACK source/clock")
        require(begins[rank]["tag"] == finish["window_id"] and begins[rank]["audit"] is False
                and begins[rank]["phase"] == row["phase"] and begins[rank]["index"] == row["index"], "Begin ACK")
        require(binds[rank]["window_id"] == finish["window_id"] and binds[rank]["before_any_native_execution"] is True
                and all(binds[rank][key] == mapping[key] for key in ("http_request_id", "engine_external_request_id", "core_request_id")),
                "Binding ACK")
        require(begins[rank]["ack_ns"] <= events["begin_all_tp8_acked"]["observed_ns"]
                and binds[rank]["ack_ns"] <= events["bind_all_tp8_acked"]["observed_ns"], "ACK after native submission")
        require(receipt["counter_sha256"] == COUNTER and receipt["clock_scope"] == clock and
                receipt["tag"] == finish["window_id"] and receipt["sources"] == descriptions[rank]["native_sources"], "Rank source/clock")
        require(receipt["audit"] is False and receipt["extra_device_copies"] == 0
                and receipt["capture_profile_counted"] is False and receipt["errors"] == [], "Capture/audit included")
        require(receipt["api_client_end_ns"] == ended and
                events["finish_rpc_call_after_client_end"]["observed_ns"] <= receipt["api_finish_rpc_ns"]
                <= events["finish_all_tp8_retrieved"]["observed_ns"], "Worker finish before read end")
        binding = receipt["api_bridge_binding"]
        require(binding["client_clock"] == clock and binding["window_id"] == finish["window_id"]
                and binding["phase"] == row["phase"] and binding["index"] == row["index"]
                and binding["prompt_sha256"] == row["prompt_sha256"]
                and all(binding[key] == mapping[key] for key in ("http_request_id", "engine_external_request_id", "core_request_id")),
                "Worker bound to another request")
        require(set(receipt["requests"]) == {core}, "Additional/foreign worker requests")
        request = receipt["requests"][core]
        require(request["prompt_tokens"] == 2048 and request["seed_token_ids"] == tokens[:1]
                and request["prefill_calls"] == 2 and request["actual_round_count"] == metrics["rounds"], "Actual prefill/rounds")
        lengths = request["round_token_lengths"]
        require(len(lengths) == metrics["rounds"] and all(type(n) is int and 1 <= n <= 6 for n in lengths), "Round lengths")
        if common_lengths is None: common_lengths = lengths
        require(lengths == common_lengths, "Ranks disagree on per-round emitted lengths")
        initial = request["initial_draft"]
        require(len(initial) == 1, "Initial DSpark draft unavailable")
        first, terminal = request["first_round"], request["terminal_round"]
        require(first["round"] == 1 and terminal["round"] == metrics["rounds"]
                and first["computed_before"] == 2048
                and terminal["computed_before"] == 2048 + sum(lengths[:-1]), "Round/cache offset boundary")
        require(request["decode_start_ns"] <= request["prefill_sample_end_ns"] <= first["execute_start_ns"], "Initial draft excluded")
        require(binds[rank]["ack_ns"] <= request["decode_start_ns"], "Execution before binding")
        for boundary, expected_length in ((first, lengths[0]), (terminal, lengths[-1])):
            require(boundary["phase"] == "decode" and boundary["scheduled_tokens"] == 6
                    and boundary["verified_draft_tokens"] == 5 and len(boundary["scheduled_draft_token_ids"]) == 5
                    and boundary["sampler_called"] is True and boundary["rejection_sampler_called"] is True
                    and len(boundary["token_ids"]) == expected_length
                    and boundary["accepted_draft_tokens_raw"] == expected_length - 1, "Verifier/draft boundary")
            require(boundary["execute_start_ns"] < boundary["bookkeeping_end_ns"] <= boundary["sample_end_ns"], "Boundary execution timing")
        boundary_identity = [(b["scheduled_draft_token_ids"], b["token_ids"], b["draft_calls"])
                             for b in (first, terminal)]
        if common_boundary is None: common_boundary = boundary_identity
        require(boundary_identity == common_boundary, "Ranks disagree on first/terminal token/proposal boundary")
        for draft in initial + first["draft_calls"] + terminal["draft_calls"]:
            require(draft["method"] == "dspark" and draft["k"] == 5 and draft["output_shape"] == [1, 5]
                    and draft["runtime_graph"] is False, "Native DSpark K5 eager execution")
            require(draft["actual_execution_evidence"]["actual_graph_submission"] is False
                    and draft["actual_execution_evidence"]["device_consumer_proof"] is False, "Unsupported graph/consumer claim")
        require(len(first["draft_calls"]) == len(terminal["draft_calls"]) == 1, "Terminal submitted draft omitted")
        require(terminal["terminal_device_synced"] is True, "No terminal device completion fence")
        join = terminal["terminal_tp_join"]
        require(join["backend"] == "gloo" and join["world_size"] == 8 and join["rank"] == rank
                and join["device_fence_preceded_entry"] is True and join["sample_timestamp_preceded_entry"] is True
                and terminal["sample_end_ns"] <= join["entered_ns"] <= join["returned_ns"]
                <= events["finished_native_request_output_observed"]["observed_ns"] <= ended, "Terminal rank join outside client clock")
        starts.append(request["decode_start_ns"]); joins.append(join["returned_ns"]); end_samples.append(terminal["sample_end_ns"])
        rank_spans.append({"rank": rank, "initial_draft_and_prefill_return_ms": (request["prefill_sample_end_ns"] - request["decode_start_ns"]) / 1e6,
                           "first_round_to_bookkeeping_ms": (first["bookkeeping_end_ns"] - first["execute_start_ns"]) / 1e6,
                           "first_bookkeeping_to_sample_end_ms": (first["sample_end_ns"] - first["bookkeeping_end_ns"]) / 1e6,
                           "terminal_round_to_bookkeeping_ms": (terminal["bookkeeping_end_ns"] - terminal["execute_start_ns"]) / 1e6,
                           "terminal_bookkeeping_to_sample_end_ms": (terminal["sample_end_ns"] - terminal["bookkeeping_end_ns"]) / 1e6})
        counter = counters[rank]
        raw = sum(lengths); trimmed = raw - 255
        require(0 <= trimmed < lengths[-1], "Trim crosses final verifier round")
        remaining, accepted = 255, 0
        for n in lengths:
            used = min(remaining, n); accepted += min(used, n - 1); remaining -= used
        expected = {"rank": rank, "rounds": len(lengths), "draft_calls": len(lengths) + 1,
                    "proposed_tokens": 5 * (len(lengths) + 1), "verified_draft_tokens": 5 * len(lengths),
                    "accepted_draft_tokens_raw": raw - len(lengths), "accepted_draft_tokens_used": accepted,
                    "raw_decode_tokens": raw, "terminal_tokens_trimmed": trimmed}
        require(counter == expected, "Rank counters inconsistent with actual per-round lengths")
        require(all(z["total_num_scheduled_tokens"] == 0 and z["counted_as_decode_round"] is False
                    and z["execute_start_ns"] <= z["execute_end_ns"] for z in receipt["zero_token_executions"]), "Cleanup counted as verifier round")
    require(metrics["decode_start_ns"] == min(starts), "Decode start omits an initial rank draft")
    elapsed = (ended - min(starts)) / 1e9
    require(elapsed > 0, "Invalid decode window")
    for name, expected in (("decode_elapsed_s", elapsed), ("A", 255 / metrics["rounds"]),
                           ("ms_per_step", 1000 * elapsed / metrics["rounds"]), ("tokens_per_second", 255 / elapsed)):
        close(metrics[name], expected, "Inconsistent " + name)
    close(metrics["ms_per_step"], 1000 * metrics["A"] / metrics["tokens_per_second"], "Inconsistent triple")
    require(metrics["target_met"] == (metrics["ms_per_step"] <= 20), "Unsupported target claim")
    return {"phase": row["phase"], "index": row["index"], "core_request_id": core,
            "rounds": metrics["rounds"], "decode_elapsed_s": elapsed, "decoded_tokens": 255,
            "ms_per_step": metrics["ms_per_step"], "A": metrics["A"], "tokens_per_second": metrics["tokens_per_second"],
            "clean": row["quality"]["clean"], "repeat_loop": row["quality"]["repeat_loop"],
            "repeated_token_window": row["quality"]["repeated_token_window"], "rank_boundaries_verified": 8,
            "latest_terminal_sample_to_client_read_ms": (ended - max(end_samples)) / 1e6,
            "latest_terminal_join_to_client_read_ms": (ended - max(joins)) / 1e6,
            "rank_boundary_spans": rank_spans,
            "finish_rpc_excluded": True, "initial_and_terminal_submitted_drafts_included": True}


def verify(directory):
    directory = directory.resolve()
    manifest = json.loads((directory / "fetch_manifest.json").read_text())
    for rel, spec in manifest.items():
        require(not Path(rel).is_absolute() and ".." not in Path(rel).parts, "Unsafe evidence path")
        path = directory / rel
        require(path.stat().st_size == spec["bytes"] < 1_000_000 and digest(path) == spec["sha256"], "Original evidence SHA: " + rel)
    selection = json.loads((directory / "service/service_selection.json").read_text())
    launched = json.loads((directory / "launched.json").read_text())
    no_admission(selection, "selection")
    no_admission(launched, "launch", ("full_consumer_audit_passed", "performance_receipt_granted", "formal_receipt_granted"))
    require(launched["chips"] == list(range(8, 16)) and launched["dspark"] is True
            and launched["audit"] is False and launched["profiler"] is False and launched["draft_graph"] is False, "Launch scope")
    require(selection["served_model"] == launched["served_model"] and selection["port"] == launched["port"]
            and selection["frozen_counter_sha256"] == COUNTER, "Launch/selection identity")
    require(selection["checkpoint"]["config_sha256"] == CONFIG
            and selection["checkpoint"]["topology"] == {"hidden_size": 5120, "num_hidden_layers": 40,
                "n_routed_experts": 384, "num_experts_per_tok": 6, "moe_intermediate_size": 2304}
            and selection["checkpoint"]["quantization"]["model_quant_type"] == "W4A8_DYNAMIC", "Actual checkpoint config")
    kwargs = selection["model_kwargs"]
    require(kwargs["load_format"] == "auto" and kwargs["tensor_parallel_size"] == 8
            and kwargs["speculative_config"] == {"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True}
            and kwargs["compilation_config"] == {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [6]}
            and kwargs["additional_config"]["engram_storage"] == "int8" and kwargs["limit_mm_per_prompt"] == {"image": 4}, "Model load/graph scope")
    env = selection["environment"]
    require(env["HCCL_DETERMINISTIC"] == launched["env"]["HCCL_DETERMINISTIC"] == "strict"
            and env["STACK_REAL_WEIGHTS"] == "1" and env["STACK_REAL_AUDIT"] == env["STACK_TINY_PROFILE"] == "0", "Environment scope")
    source_matches = {}
    for rel, expected in launched["source_hashes"].items():
        path = ROOT / rel.removeprefix("stack/")
        if rel in ("stack/bench_formal_dspark.py", "stack/formal_dspark_counters.py"):
            path = ROOT / "parallel_tiny/dspark_formal_audit/stock_compat_v4" / path.name
        require(path.is_file() and digest(path) == expected, "Deployed source differs: " + rel)
        source_matches[rel] = str(path.relative_to(ROOT))
    require((directory / "client.exit").read_text().strip() == (directory / "run.exit").read_text().strip() == "0", "Client/service did not exit normally")
    index = json.loads((directory / "diagnostic_compact/index.json").read_text())
    no_admission(index["flags"], "index")
    require(index["raw_acceptance"]["bytes"] > 1_000_000 and len(index["raw_acceptance"]["sha256"]) == 64, "No full raw evidence binding")
    rows, http_outputs, validated = [], [], []
    for phase, count in (("warmup", 2), ("measured", 8)):
        for i in range(count):
            name = f"{phase}_{i}_compact.json"; path = directory / "diagnostic_compact" / name
            require(digest(path) == index["files"][name]["sha256"], "Compact index SHA")
            row = json.loads(path.read_text()); raw = directory / f"client/{phase}_{i}_http.json"
            require(row["phase"] == phase and row["index"] == i and row["prompt_name"] == f"{phase}_{i}", "Request sequence")
            require(digest(raw) == row["raw_http_sha256"], "HTTP byte SHA")
            http = json.loads(raw.read_text()); validated.append(verify_window(row, http, selection))
            rows.append(row); http_outputs.append(http)
    require(len({r["finish"]["window_id"] for r in rows}) == len(rows)
            and len({r["finish"]["mapping"]["core_request_id"] for r in rows}) == len(rows)
            and len({r["prompt_sha256"] for r in rows}) == len(rows), "Reused request/prompt")
    for previous, current in zip(rows, rows[1:]):
        require(previous["finish"]["events"][-1]["observed_ns"] <= current["finish"]["events"][0]["observed_ns"], "Concurrent measured requests")
    measured = validated[2:]; total = sum(r["rounds"] for r in measured)
    elapsed = sum(r["decode_elapsed_s"] for r in measured)
    aggregate = {"samples": 8, "concurrency": 1, "prompt_tokens": 2048,
                 "total_actual_rounds": total, "total_decoded_tokens": 2040, "decode_elapsed_s_sum": elapsed,
                 "ms_per_step": 1000 * elapsed / total, "A": 2040 / total, "tokens_per_second": 2040 / elapsed,
                 "ms_per_step_median": statistics.median(r["ms_per_step"] for r in measured),
                 "per_stream_tokens_per_second_median": statistics.median(r["tokens_per_second"] for r in measured),
                 "clean_count": sum(r["clean"] for r in measured),
                 "repetition_count": sum(r["repeat_loop"] or r["repeated_token_window"] for r in measured)}
    aggregate["clean_rate"] = aggregate["clean_count"] / 8
    for key, expected in aggregate.items(): close(index["aggregate"][key], expected, "Aggregate " + key)
    archive = directory.parent / "native_api_v2_compact.tgz"
    archive_receipt = json.loads((directory / "archive_receipt.json").read_text())
    require(archive.stat().st_size == archive_receipt["archive_bytes"] < 1_000_000
            and digest(archive) == archive_receipt["archive_sha256"], "Transport archive SHA")
    return {"job": "formal_dspark_native_api_v2", "scope": "Actual native HTTP integration diagnostic; full43 consumer audit still required",
            "offline_validation_passed": True, "runtime_initialized_by_validator": False,
            "local_original_files_verified": len(manifest), "deployed_source_files_verified": len(source_matches),
            "source_paths": source_matches, "raw_acceptance_retained_remotely": index["raw_acceptance"],
            "retained_raw_sources": index["retained_raw_sources"], "archive": archive_receipt,
            "counter_sha256": COUNTER, "checkpoint_config_sha256": CONFIG,
            "service_id": selection["service_id"], "clock_scope": selection["identity"]["clock_scope"],
            "service_started_flag_alone_used_as_http_proof": False,
            "actual_http_proof": "10 original completed HTTP response bodies, matched core IDs and all 8 rank receipts",
            "warmup": validated[:2], "measured": measured, "aggregate": aggregate,
            "actual_api_performance_observed": True, "actual_api_quality_observed": True,
            "actual_client_quality_passed": aggregate["clean_rate"] == 1.0,
            "performance_target_observed": aggregate["ms_per_step"] <= 20,
            "full_consumer_audit_passed": False, "actual_device_consumer_proof": False,
            "performance_receipt_granted": False, "formal_receipt_granted": False,
            "semantic_accuracy_review_completed": False,
            "limitations": ["Intermediate complete round payloads remain remote; compact retains all round lengths and first/terminal payloads",
                            "Timestamp spans are host wall-clock boundaries, not isolated device kernel durations",
                            "Cleanliness heuristics do not certify GSM100/100 or Vision23/23",
                            "SDK initialized-worker gate and checkpoint inventory do not certify full43 consumer seals"]}, rows, http_outputs, selection


def negative_checks(rows, outputs, selection):
    """Evidence corruption checks: each changes a distinct acceptance boundary."""
    def request(r): return next(iter(r["finish"]["rank_receipts"][0]["requests"].values()))
    changes = [
        ("foreign_worker_clock", lambda r: r["finish"]["rank_receipts"][0]["clock_scope"].update(boot_id="foreign")),
        ("external_id_instead_of_core", lambda r: r["finish"]["mapping"].update(core_request_id=r["finish"]["mapping"]["engine_external_request_id"])),
        ("missing_initial_draft", lambda r: request(r).update(initial_draft=[])),
        ("terminal_fence_absent", lambda r: request(r)["terminal_round"].update(terminal_device_synced=False)),
        ("terminal_join_after_read", lambda r: request(r)["terminal_round"]["terminal_tp_join"].update(returned_ns=r["client_end_ns"] + 1)),
        ("missing_terminal_draft", lambda r: request(r)["terminal_round"].update(draft_calls=[])),
        ("invented_round_count", lambda r: request(r).update(actual_round_count=1)),
        ("inconsistent_round_length", lambda r: request(r)["round_token_lengths"].__setitem__(0, 6)),
        ("performance_claim_without_gate", lambda r: r["finish"]["metrics"].update(performance_receipt_granted=True)),
        ("prefill_seed_in_decode", lambda r: r["finish"]["metrics"].update(decoded_tokens=256)),
        ("incorrect_triple", lambda r: r["finish"]["metrics"].update(ms_per_step=20)),
        ("finish_rpc_included", lambda r: r.update(client_end_ns=r["finish"]["events"][-1]["observed_ns"])),
        ("wrong_clean_label", lambda r: r["quality"].update(clean=False)),
    ]
    receipts = []
    for name, change in changes:
        row = copy.deepcopy(rows[2]); change(row)
        try: verify_window(row, outputs[2], selection)
        except (ValueError, KeyError) as exc: receipts.append({"name": name, "rejected": True, "reason": str(exc)})
        else: raise ValueError("Corrupted evidence accepted: " + name)
    return receipts


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--evidence", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--negative-checks", action="store_true")
    args = parser.parse_args()
    report, rows, outputs, selection = verify(args.evidence)
    if args.negative_checks: report["negative_checks"] = negative_checks(rows, outputs, selection)
    report["validator_sha256"] = digest(Path(__file__).resolve())
    if args.output:
        require(not args.output.exists(), "Do not overwrite a validation receipt")
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"job": report["job"], "validated": True, "aggregate": report["aggregate"],
                      "rank_request_cases": len(rows) * 8,
                      "negative_checks": len(report.get("negative_checks", [])), "formal_receipt_granted": False}, ensure_ascii=False))


if __name__ == "__main__": main()
