"""Verify all original HTTP rounds on the host, without transferring large logs.

Use python -S. Only the frozen stdlib-only counter module is loaded; no worker,
model, framework runtime, HTTP request or device initialization occurs.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


COUNTER_SHA = "da845a5891557f21e80eac06d96aec1955263d1ae93afbc4d84e86dd9585dbec"


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(data): return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--counter-file", type=Path, required=True)
    parser.add_argument("--acceptance-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Do not overwrite an original verification receipt")
    raw_path = args.job_root / "client/acceptance.json"; raw = raw_path.read_bytes()
    require(sha(raw) == args.acceptance_sha256, "Large original acceptance SHA changed")
    require(sha(args.counter_file.read_bytes()) == COUNTER_SHA, "Frozen counter SHA changed")
    spec = importlib.util.spec_from_file_location("_dspark_frozen_counter_offline", args.counter_file)
    counter = importlib.util.module_from_spec(spec); spec.loader.exec_module(counter)
    report = json.loads(raw); require(report["cpu_mock"] is False and report["serial_requests"] is True
            and report.get("failed", False) is False, "Not a completed actual serial API report")
    summaries, rank_rounds, hashes = [], 0, {}
    for phase, minimum in (("warmup", 2), ("measured", 8)):
        require(len(report[phase]) == minimum, "Unexpected warmup/measured request count")
        for index, row in enumerate(report[phase]):
            require(row["phase"] == phase and row["index"] == index, "Request sequence")
            finish = row["finish"]; actual = finish["metrics"]; mapping = finish["mapping"]
            path = args.job_root / f"client/{phase}_{index}_http.json"; body = path.read_bytes()
            require(sha(body) == row["raw_http_sha256"], "Original HTTP byte SHA")
            http = json.loads(body); tokens = http["completion"]["choices"][0]["token_ids"]
            require(http["bridge"]["service_id"] == report["service_id"] == finish["service_id"], "Foreign HTTP service")
            result = counter.summarize(finish["rank_receipts"], tokens, row["client_end_ns"],
                report["clock_scope"], draft_graph=False, expected_core_request_id=mapping["core_request_id"])
            for name in ("rounds", "decoded_tokens", "decode_elapsed_s", "A", "ms_per_step", "tokens_per_second"):
                require(math.isclose(actual[name], result[name], rel_tol=1e-12, abs_tol=1e-12), "All-round " + name)
            require(result["rank_counters"] == actual["rank_counters"], "All-round raw/used accepted counters changed")
            for receipt in finish["rank_receipts"]:
                request = receipt["requests"][mapping["core_request_id"]]
                require(len(request["rounds"]) == result["rounds"], "Missing intermediate original round payloads")
                rank_rounds += len(request["rounds"])
            summaries.append({"phase": phase, "index": index, "core_request_id": mapping["core_request_id"],
                              "rounds": result["rounds"], "decode_elapsed_s": result["decode_elapsed_s"],
                              "decoded_tokens": result["decoded_tokens"], "ms_per_step": result["ms_per_step"],
                              "A": result["A"], "tokens_per_second": result["tokens_per_second"],
                              "full_rank_round_payloads_verified": True, "rank_count": 8,
                              "raw_http_sha256": sha(body)})
            hashes[str(path)] = {"bytes": len(body), "sha256": sha(body)}
    measured = [row for row in summaries if row["phase"] == "measured"]
    rounds = sum(row["rounds"] for row in measured); elapsed = sum(row["decode_elapsed_s"] for row in measured)
    for name, expected in (("total_actual_rounds", rounds), ("total_decoded_tokens", 2040),
                           ("decode_elapsed_s_sum", elapsed), ("ms_per_step", 1000 * elapsed / rounds),
                           ("A", 2040 / rounds), ("tokens_per_second", 2040 / elapsed)):
        require(math.isclose(report["aggregate"][name], expected, rel_tol=1e-12, abs_tol=1e-12), "Full raw aggregate " + name)
    backends = [name for name in sys.modules if name.split(".")[0] in ("torch", "torch_npu", "torchair", "vllm", "vllm_ascend", "acl")]
    require(not backends, "Offline verifier imported a framework runtime")
    result = {"scope": "Offline full original native API round verification; no device consumer or formal admission",
              "job": args.job_root.name, "passed": True, "runtime_backend_imports": backends, "NPU_calls": 0,
              "HTTP_requests": 0, "counter_sha256": COUNTER_SHA,
              "raw_acceptance": {"path": str(raw_path), "bytes": len(raw), "sha256": sha(raw)},
              "clock_scope": report["clock_scope"], "service_id": report["service_id"],
              "rank_request_cases": len(summaries) * 8, "all_rank_round_payloads_verified": rank_rounds,
              "summaries": summaries, "original_http_files": hashes, "aggregate": report["aggregate"],
              "full_consumer_audit_passed": False, "formal_receipt_granted": False,
              "performance_receipt_granted": False, "verifier_sha256": sha(Path(__file__).read_bytes())}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__": main()
