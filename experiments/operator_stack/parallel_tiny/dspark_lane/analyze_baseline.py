"""Validate actual DSpark rounds and make a compact, explicitly diagnostic receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repeat_period(tokens):
    for width in range(1, min(16, len(tokens) // 3) + 1):
        if tokens[-3 * width:] == tokens[-width:] * 3:
            return width
    return None


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    assert args.job.startswith("dspark_lane_20261011_c4_")
    base = Path(__file__).resolve().parent
    raw = base.parents[1] / "runtime/evidence/dspark_lane" / args.job
    assert (raw / "run.exit").read_text().strip() == "0"
    data = json.loads((raw / "results/requests.json").read_text())
    topology = json.loads((raw / "results/topology.json").read_text())
    aa = json.loads((raw / "results/native_AA.json").read_text())
    assert topology["main_layers"] == 40 and topology["draft_layers"] == 3
    assert topology["proposer"] == "AscendDSparkProposer"
    assert topology["target_layer_ids"] == [37, 38, 39]
    assert topology["num_speculative_tokens"] == 5
    assert aa["passed"] and not aa["complete_consumer_gate"]
    requests = []
    for record in data:
        rows = record["receipts"]
        seed = [r for r in rows if not r["is_decode"]]
        decode = [r for r in rows if r["is_decode"]]
        assert len(seed) == 1 and len(decode) == record["target_decode_rounds"]
        assert all(r["req_id"] == record["core_request_id"] for r in rows)
        assert all(0 <= r["accepted_draft_tokens"] <= r["verified_draft_tokens"] for r in decode)
        assert record["frontend_step_calls"] == len(rows)
        assert sum(r["draft_calls"] for r in rows) == record["draft_calls"]
        assert record["draft_calls"] > 0 and record["verifier_rounds"] > 0
        assert record["completion_tokens"] - record["prefill_seed_tokens"] == record["decode_tokens"]
        assert record["terminal_device_sync_count"] == 1 and record["steady_extra_sync_count"] == 0
        ms, a, tps = record["ms_per_step"], record["A"], record["single_stream_tok_s"]
        assert abs(ms - 1000 * a / tps) < 1e-8
        assert abs(a - record["decode_tokens"] / len(decode)) < 1e-12
        ids = record["token_ids"]
        raw_ids = [token for row in rows for token in row["sampled_token_ids"]]
        assert raw_ids[:len(ids)] == ids, "Native raw token stream must bind the frontend output"
        requests.append({k: record[k] for k in (
            "request_id", "core_request_id", "seed", "warmup", "audit", "completion_tokens",
            "prefill_seed_tokens", "decode_tokens", "target_decode_rounds", "verifier_rounds",
            "draft_calls", "proposed_tokens_verified", "raw_accepted_draft_tokens", "final_budget_clipped_tokens",
            "decode_s", "ms_per_step", "A", "single_stream_tok_s", "draft_ineffective")}
            | {"token_unique_count": len(set(ids)), "empty_output": not ids,
               "token_suffix_repetition_period": repeat_period(ids),
               "target_full_graph": True, "target_capture_sizes": [6],
               "draft_enforce_eager": True, "draft_graph": False,
               "native_token_prefix_matches_output": True})
    perf = [r for r, source in zip(requests, data)
            if not r["warmup"] and not r["audit"] and not source.get("profiler_enabled", False)]
    assert len(perf) >= 8
    wall = sum(r["decode_s"] for r in perf)
    tokens = sum(r["decode_tokens"] for r in perf)
    rounds = sum(r["target_decode_rounds"] for r in perf)
    output = {"job": args.job, "latest_formal_target_ms": 20, "formal_tp8_claim": False,
              "target_full_graph": True, "target_capture_sizes": [6],
              "draft_enforce_eager": True, "draft_graph": False,
              "private_legacy_helper_scope": True, "production_direct_graph_bool_claim": False,
              "dummy_weights": True, "engram_enabled": False, "same_model_instance": True,
              "true_dspark_engine_path": True, "actual_main_layers": 40, "draft_module_layers": 3,
              "draft_layer_device_consumers_audited": False,
              "native_AA": aa, "full_accuracy_gate": False,
              "requests": requests, "serial_performance_requests": len(perf),
              "profiler_enabled": False, "per_request_logprob_audit_during_performance": False,
              "serial_pooled": {"decode_s": wall, "decode_tokens": tokens, "actual_target_rounds": rounds,
                                "ms_per_step": 1000 * wall / rounds, "A": tokens / rounds,
                                "single_stream_tok_s": tokens / wall},
              "per_request_medians": {"ms_per_step": statistics.median(r["ms_per_step"] for r in perf),
                                      "A": statistics.median(r["A"] for r in perf),
                                      "single_stream_tok_s": statistics.median(r["single_stream_tok_s"] for r in perf)},
              "quality_clean_rate": None,
              "quality_reason": "Dummy weights cannot establish text quality or formal DSpark acceptance benefit",
              "token_suffix_repetition_requests": sum(r["token_suffix_repetition_period"] is not None for r in perf),
              "empty_output_requests": sum(r["empty_output"] for r in perf),
              "ineffective_draft_requests": sum(r["draft_ineffective"] for r in perf),
              "candidate_admitted": False,
              "artifact_sha256": {name: digest(raw / name) for name in (
                  "results/requests.json", "results/topology.json", "results/native_AA.json", "preflight.json", "run.log")}}
    target = base / "evidence" / (args.job + "_analysis.json")
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "requests"}, indent=2))


if __name__ == "__main__":
    main()
