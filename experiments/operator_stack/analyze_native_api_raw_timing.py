"""Exact host-boundary accounting for a source-verified native API raw report.

Intervals include asynchronous device work as observed at the host boundaries;
they are not kernel durations or an isolated target/draft cost measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


NAMES = ("initial_draft_and_first_schedule", "execute_to_bookkeeping",
         "bookkeeping_to_sample_return", "between_rounds", "terminal_sample_to_client_read")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--acceptance-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Do not overwrite original timing evidence")
    path = args.job_root / "client/acceptance.json"; raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.acceptance_sha256: raise ValueError("Raw acceptance SHA changed")
    verified_path = args.job_root / "raw_offline_verify_v1.json"
    verified = json.loads(verified_path.read_text())
    if not verified["passed"] or verified["raw_acceptance"]["sha256"] != args.acceptance_sha256:
        raise ValueError("Full raw verification must precede timing accounting")
    data = json.loads(raw); requests = []; totals = {key: 0 for key in NAMES}; body_spans = []; return_spans = []
    for row in data["measured"]:
        finish = row["finish"]; core = finish["mapping"]["core_request_id"]
        rank = min(finish["rank_receipts"], key=lambda r: r["requests"][core]["decode_start_ns"])
        request = rank["requests"][core]; rounds = request["rounds"]
        spans = {"initial_draft_and_first_schedule": rounds[0]["execute_start_ns"] - request["decode_start_ns"],
                 "execute_to_bookkeeping": sum(r["bookkeeping_end_ns"] - r["execute_start_ns"] for r in rounds),
                 "bookkeeping_to_sample_return": sum(r["sample_end_ns"] - r["bookkeeping_end_ns"] for r in rounds),
                 "between_rounds": sum(b["execute_start_ns"] - a["sample_end_ns"] for a,b in zip(rounds,rounds[1:])),
                 "terminal_sample_to_client_read": row["client_end_ns"] - rounds[-1]["sample_end_ns"]}
        elapsed = row["client_end_ns"] - request["decode_start_ns"]
        if min(spans.values()) < 0 or sum(spans.values()) != elapsed:
            raise ValueError("Host intervals overlap or do not reconstruct the exact client decode window")
        for key in NAMES: totals[key] += spans[key]
        body_spans += [(r["bookkeeping_end_ns"] - r["execute_start_ns"]) / 1e6 for r in rounds]
        return_spans += [(r["sample_end_ns"] - r["bookkeeping_end_ns"]) / 1e6 for r in rounds]
        requests.append({"index": row["index"], "rank_with_earliest_initial_boundary": rank["rank"],
                         "actual_rounds": len(rounds), "decode_ns": elapsed,
                         "host_interval_ns": spans, "sum_exactly_matches_decode_ns": True})
    count = sum(row["actual_rounds"] for row in requests); total = sum(totals.values())
    result = {"job": args.job_root.name, "scope": "All measured original rounds; exact host wall-clock accounting only",
              "raw_acceptance_sha256": args.acceptance_sha256,
              "full_raw_verification_sha256": hashlib.sha256(verified_path.read_bytes()).hexdigest(),
              "NPU_calls": 0, "HTTP_requests": 0, "actual_rounds": count, "decode_ns_sum": total,
              "intervals": {key: {"total_s": value / 1e9, "ms_per_actual_round": value / 1e6 / count,
                                    "fraction_of_decode": value / total} for key,value in totals.items()},
              "execute_to_bookkeeping_median_ms": statistics.median(body_spans),
              "bookkeeping_to_sample_return_median_ms": statistics.median(return_spans),
              "requests": requests, "formal_receipt_granted": False,
              "host_intervals_are_isolated_kernel_or_draft_durations": False,
              "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result))


if __name__ == "__main__": main()
