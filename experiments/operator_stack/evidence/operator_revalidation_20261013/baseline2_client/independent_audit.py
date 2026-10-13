"""Offline primary-agent re-score of split raw receipts; never calls a service."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[5]
CLIENT = REPO / "experiments/delivery_matrix"
sys.path.insert(0, str(CLIENT))
import benchmark
from frozen_dihuo_inputs import load_frozen_dihuo_inputs

SID = "3cbd844d9f684c43867b4b85c9c78ebe"
FP = "79513a0f20932cf696374573323b715aab0d04df774721fbee92f2f780da4319"
MODEL = "dsv41-operator-revalidation-tp8-20261013"
FORM = "a2-tp8-a3sim"
NEW_PIN = "87c66ad1d78bc6eabc25af32faa34af2d86db10a3aa4fd1589fbe454fb30b39c"
OLD_PIN = "afb141d345f61c84e9ab7a0b5f0682f4bb2ea918dee029636e2fbbc65cbb94ec"
SOURCES = {
    "benchmark.py": "b39d9c3ddc44701d7463f0ee2b2a36b55e01fd6fe69fab85b11db20e8651f666",
    "workloads.py": "62db3507dd74ea69406a940963f93414ae2e9f87aa77424f8204f5c9b7ce4d35",
    "frozen_dihuo_inputs.py": "605cecfb4853e3657de80b95296ae48e862ea91067fe63b3c3ae12793ddeaa2c",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(left, right):
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isclose(
        left, right, rel_tol=1e-12, abs_tol=1e-10)


def audit(args):
    source_hashes = {name: sha((CLIENT / name).read_bytes()) for name in SOURCES}
    require(source_hashes == SOURCES, "auditor must use the exact measured client source version")
    new_inputs = load_frozen_dihuo_inputs(args.manifest, NEW_PIN, 2048, 256)
    old_inputs = load_frozen_dihuo_inputs(args.reference_manifest, OLD_PIN, 2048, 256)
    manifest = json.loads(args.manifest.read_bytes())
    old_manifest = json.loads(args.reference_manifest.read_bytes())
    source = manifest["source"]
    require(source["service_instance_id"] == SID and source["runtime_fingerprint"] == FP
            and source["model"] == MODEL and source["form"] == FORM, "split source identity mismatch")
    require(source["workload_manifest"] == old_manifest["source"]["workload_manifest"],
            "workload differs from historical frozen reference")
    compact = json.loads(args.compact.read_bytes())
    require(compact["config"]["serial"] is True and compact["config"]["rounds"] == 1
            and compact["config"]["warmups"] == 1, "serial/window protocol mismatch")
    require(compact["service"]["service_instance_id"] == SID
            and compact["service"]["runtime_fingerprint"] == FP
            and compact["client_source_sha256"] == SOURCES, "compact identity/source mismatch")
    results, tags = [], set()
    totals = {str(rank): {"decode_rounds": 0, "draft_graph_calls": 0, "draft_eager_calls": 0}
              for rank in range(8)}
    for index, artifact in enumerate(manifest["artifacts"]):
        compressed = (args.manifest.parent / "requests" / artifact["path"]).read_bytes()
        raw = gzip.decompress(compressed)
        require(len(compressed) == artifact["size_bytes"] and sha(compressed) == artifact["sha256"]
                and len(raw) == artifact["uncompressed_size_bytes"]
                and sha(raw) == artifact["uncompressed_sha256"], "raw artifact byte/hash mismatch")
        request = json.loads(raw)["request"]
        usage = request["usage_tokens"]
        require([usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")] == [2048, 256, 2304]
                and request["http_status"] == 200 and request["finish_reason"] == "length"
                and request["error"] is None and request["sse_error_event"] is None,
                "current exact HTTP/SSE output gate failed")
        cleanliness = benchmark.output_cleanliness(request["output_text"])
        require(cleanliness == request["clean"] and cleanliness["clean"] is True,
                "independent current text cleanliness gate failed")
        output_sha = sha(request["output_text"].encode("utf-8"))
        require(output_sha == request["output_sha256"] == artifact["output_sha256"], "output hash mismatch")
        require(new_inputs["samples"][index]["messages"] == old_inputs["samples"][index]["messages"],
                "actual messages differ from frozen reference")
        tag = request["request_tag"]
        require(tag == request["request_id"] and tag not in tags, "request tag reused/mismatched")
        tags.add(tag)
        before, after = request["delivery_metrics_before"], request["delivery_metrics_after"]
        delta = benchmark.counter_metrics_delta(before, after, MODEL, 2048, usage, SID,
                                                tag, 256, request["delivery_arm"], FORM)
        require(delta["available"] is True and delta == request["delivery_counter_delta"],
                "recomputed actual 8-rank counter delta differs")
        require(delta["decode_tokens"] == 255 and delta["request_id"] == request["engine_request_id"],
                "actual seed/token/request mapping mismatch")
        require(len(delta["request_rows_by_rank"]) == 8 and all(
            row["terminal_fence_observed"] is True and row["terminal_tp_join_observed"] is True
            and row["profile_excluded"] is True for row in delta["request_rows_by_rank"].values()),
            "terminal/profile/TP join gate failed")
        steps = delta["steps"]
        http_ms = 1000 * request["decode_s"] / steps
        A = 255 / steps
        rate = 255 / request["decode_s"]
        require(near(http_ms, request["client_decode_ms_per_step"]) and near(A, delta["A"])
                and near(rate, request["decode_tok_s"]) and near(rate, 1000 * A / http_ms),
                "request-level HTTP triplet is not self-consistent")
        for rank in range(8):
            left = before["data"]["ranks"][rank]
            right = after["data"]["ranks"][rank]
            require(left["rank"] == right["rank"] == rank, "raw metric rank order changed")
            changes = {key: right["cumulative"][key] - left["cumulative"][key] for key in totals[str(rank)]}
            require(changes["decode_rounds"] == changes["draft_graph_calls"] == steps
                    and changes["draft_eager_calls"] == 1, "actual rank round/proposal-selector disagreement")
            for key, value in changes.items():
                totals[str(rank)][key] += value
        observed = compact["per_request"][index]
        require(observed["actual_decode_rounds"] == steps and observed["usage_tokens"] == {
            key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
                and near(observed["server_prefill_s"], delta["server_prefill_s"]), "compact request mismatch")
        results.append({"request_index": index, "actual_rounds": steps, "A": A,
                        "http_ms_per_step": http_ms, "http_decode_tok_s": rate,
                        "server_fenced_ms_per_step": delta["server_fenced_ms_per_step"],
                        "decode_s": request["decode_s"], "server_decode_s": delta["decode_seconds_s"],
                        "ttft_s": request["ttft_s"], "server_prefill_s": delta["server_prefill_s"],
                        "prompt_sha256": request["prompt_sha256"],
                        "messages_sha256": request["prompt_messages_sha256"], "output_sha256": output_sha,
                        "output_sha_matches_history": output_sha == old_manifest["artifacts"][index]["output_sha256"]})
    require(len(results) == 8 and totals == compact["per_rank_counter_totals"], "window count mismatch")
    steps = sum(row["actual_rounds"] for row in results)
    decode_s = sum(row["decode_s"] for row in results)
    server_s = sum(row["server_decode_s"] for row in results)
    prefill_s = sum(row["server_prefill_s"] for row in results)
    values = {"steps": steps, "A": 2040 / steps, "client_decode_ms_per_step": 1000 * decode_s / steps,
              "decode_tok_s": 2040 / decode_s, "server_fenced_ms_per_step": 1000 * server_s / steps,
              "server_prefill_s": prefill_s}
    require(all(near(value, compact["summary"][key]) for key, value in values.items()), "aggregate mismatch")
    require(near(1000 * values["A"] / values["client_decode_ms_per_step"], values["decode_tok_s"]),
            "aggregate triplet mismatch")
    return {"schema": "baseline2_primary_raw_audit_v1", "passed": True,
            "service_instance_id": SID, "runtime_configuration_fingerprint": FP,
            "client_source_sha256": source_hashes, "requests": 8, "clean": 8,
            "same_prompt_and_messages_as_history": 8,
            "same_output_text_sha_as_history": sum(row["output_sha_matches_history"] for row in results),
            "current_exact_usage_and_length": 8, "terminal_and_tp_join_all_ranks": True,
            "aggregate": values, "ttft_mean_s": statistics.mean(row["ttft_s"] for row in results),
            "ttft_median_s": statistics.median(row["ttft_s"] for row in results),
            "prefill_mean_s": prefill_s / 8, "prefill_worker_wall_tok_s": 8 * 2048 / prefill_s,
            "per_rank_counter_totals": totals, "per_request": results,
            "scope": "offline raw SHA/usage/text and production counter re-score; no new HTTP or NPU calls",
            "draft_count_scope": "actual proposal runtime selector, not separately instrumented low-level replay events",
            "historical_comparison_scope": "cross-instance same-input trend, not single-variable A/B"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args), ensure_ascii=False, indent=2))
