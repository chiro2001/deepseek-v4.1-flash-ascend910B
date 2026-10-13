"""Audit quality wire artifacts independently of semantic evaluator scores."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def artifact(root, name):
    if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name:
        raise ValueError("Wire artifact path must be a single local filename")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked wire artifact")
    return path.read_bytes()


def parse_response(raw, is_sse):
    usage, finishes, done = None, [], False
    if is_sse:
        events = []
        for line in raw.splitlines():
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                continue
            if done:
                raise ValueError("SSE data continues after DONE")
            events.append(json.loads(data))
        if not done:
            raise ValueError("Actual raw SSE has no DONE")
    else:
        events = [json.loads(raw)]
    for event in events:
        if not isinstance(event, dict) or "error" in event:
            raise ValueError("Actual raw response contains an API/SSE error event")
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices") or []:
            if choice.get("finish_reason") is not None:
                finishes.append({"index": choice.get("index", 0), "finish_reason": choice["finish_reason"]})
    return usage, finishes, done


def audit(root, *, dp_count, min_requests, min_unique, allowed_finish=("stop",), expected_hashes=None):
    root = Path(root).resolve(strict=True)
    rows = [json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines() if line]
    generation = [row for row in rows if row.get("generation") is True]
    if len(generation) < min_requests:
        raise ValueError("Too few actual quality generation attempts")
    ranks, sticky = set(), {}
    for row in generation:
        rank = row.get("selected_dp_rank")
        if (type(rank) is not int or not 0 <= rank < dp_count
                or row.get("route_matches_response") is not True
                or row.get("response_dp_rank") != rank
                or row.get("http_status") != 200 or row.get("error") is not None
                or row.get("api_error_event") is not None
                or row.get("response_closed") is not True
                or type(row.get("response_is_sse")) is not bool):
            raise ValueError("Actual quality HTTP/rank/closure evidence failed")
        if (row.get("response_is_sse") is True and row.get("done_consumed") is not True
                or row.get("unparsed_sse_data_lines", 0) != 0 or row.get("unparsed_nonstream")):
            raise ValueError("Quality response was not completely parsed/consumed")
        usage = row.get("usage")
        if (not isinstance(usage, dict) or type(usage.get("completion_tokens")) is not int
                or usage["completion_tokens"] < 1 or type(usage.get("prompt_tokens")) is not int
                or usage["prompt_tokens"] < 1):
            raise ValueError("Actual quality response usage is missing")
        finishes = row.get("finish_reasons")
        if (not isinstance(finishes, list) or not finishes
                or any(item.get("finish_reason") not in allowed_finish for item in finishes)):
            raise ValueError("Quality response finish reason differs from suite contract")
        request_gzip = artifact(root, row.get("request_body_file"))
        request = gzip.decompress(request_gzip)
        response = artifact(root, row.get("response_body_file"))
        if (digest(request_gzip) != row.get("request_gzip_sha256")
                or digest(request) != row.get("request_body_sha256")
                or len(request) != row.get("request_body_bytes")
                or digest(response) != row.get("response_body_sha256")
                or len(response) != row.get("response_body_bytes") or not response):
            raise ValueError("Quality raw wire artifact hash/size differs")
        raw_usage, raw_finishes, raw_done = parse_response(response, row["response_is_sse"])
        if (raw_usage != usage or raw_finishes != finishes
                or row["response_is_sse"] and raw_done != row.get("done_consumed")):
            raise ValueError("Quality sidecar metadata differs from independently parsed raw bytes")
        body_hash = row["request_body_sha256"]
        if body_hash in sticky and sticky[body_hash] != rank:
            raise ValueError("A repeated request body changed native DP rank")
        sticky[body_hash] = rank
        ranks.add(rank)
    if len(sticky) < min_unique or ranks != set(range(dp_count)):
        raise ValueError("Quality requests do not cover every native DP replica/case")
    if expected_hashes is not None and set(sticky) != set(expected_hashes):
        raise ValueError("Quality journal body hashes differ from the evaluator requests")
    return {"wire_audit_passed": True, "generation_attempts": len(generation),
            "unique_request_bodies": len(sticky), "native_dp_ranks": sorted(ranks),
            "semantic_quality_passed": False, "scope": "wire/routing/artifact evidence only"}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dp-count", type=int, choices=(2, 4), required=True)
    parser.add_argument("--min-requests", type=int, required=True)
    parser.add_argument("--min-unique", type=int, required=True)
    parser.add_argument("--allowed-finish", nargs="+", default=["stop"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, dp_count=args.dp_count, min_requests=args.min_requests,
                   min_unique=args.min_unique, allowed_finish=tuple(args.allowed_finish))
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
