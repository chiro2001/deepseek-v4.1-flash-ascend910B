"""Independent native-DP client: one audited request per active DP replica.

Frozen historical prompt bytes are inputs only. No historical measurements or
eight-rank frontend contract is reused. With SEQ1, replica waves do not prove
maximum server saturation; larger per-replica batching needs another runtime.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from .measurement import request_metrics, wave_metrics
from .topology import topology_from_args, validate_identity, DEFAULT_CHIP_LIST
from .code8_replay import load_inputs, MANIFEST_SHA256 as CODE8_MANIFEST_SHA256

if __package__.startswith("experiments."):
    from experiments.delivery_matrix.frozen_dihuo_inputs import load_frozen_dihuo_inputs
    from experiments.delivery_matrix.workloads import output_cleanliness
else:
    from delivery_matrix.frozen_dihuo_inputs import load_frozen_dihuo_inputs
    from delivery_matrix.workloads import output_cleanliness


def http_json(base, path, *, payload=None, timeout=120):
    request = urllib.request.Request(base.rstrip("/") + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def stream(base, model, messages, *, dp_rank, tag, max_tokens, seed,
           ignore_eos, timeout, barrier=None):
    payload = {"model": model, "messages": messages, "temperature": 0,
               "seed": seed, "request_id": tag, "max_tokens": max_tokens,
               "ignore_eos": ignore_eos, "stream": True,
               "stream_options": {"include_usage": True}}
    request = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(), method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                 "X-data-parallel-rank": str(dp_rank), "X-dp-matrix-tag": tag})
    if barrier is not None:
        barrier.wait(timeout=30)
    start = time.monotonic()
    row = {"request_tag": tag, "payload": payload, "http_status": None,
           "dp_rank": None, "done": False, "error": None, "usage": None,
           "finish_reason": None, "response_ids": [], "sse_events": []}
    first = last = None
    text = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            row["http_status"] = response.status
            raw_rank = response.headers.get("X-dp-matrix-rank")
            row["dp_rank"] = int(raw_rank) if raw_rank is not None else None
            for raw in response:
                if not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    row["done"] = True
                    # Drain to EOF so middleware sees the successful final body.
                    continue
                event = json.loads(data)
                row["sse_events"].append(event)
                if "error" in event:
                    raise ValueError("SSE error: " + json.dumps(event["error"]))
                event_id = event.get("id")
                if event_id and event_id not in row["response_ids"]:
                    row["response_ids"].append(event_id)
                if event.get("usage") is not None:
                    row["usage"] = event["usage"]
                choices = event.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    row["finish_reason"] = choice["finish_reason"]
                content = choice.get("delta", {}).get("content")
                if content:
                    if not isinstance(content, str):
                        raise ValueError("Expected actual text content in frozen text workload")
                    last = time.monotonic()
                    if first is None:
                        first = last
                    text.append(content)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, urllib.error.HTTPError):
            row["http_status"] = exc.code
            row["error_body"] = exc.read().decode(errors="replace")
    row.update(start_monotonic=start, end_monotonic=time.monotonic(),
               first_content_monotonic=first, last_content_monotonic=last,
               ttft_s=None if first is None else first-start,
               decode_s=None if first is None or last is None else last-first,
               output_text="".join(text))
    row["output_sha256"] = hashlib.sha256(row["output_text"].encode()).hexdigest()
    row["clean"] = output_cleanliness(row["output_text"])
    return row


def metrics_after(base, rank, tag, timeout):
    path = "/dp/metrics?" + urllib.parse.urlencode({"dp_rank": rank, "tag": tag})
    deadline = time.monotonic() + 10
    while True:
        try:
            return http_json(base, path, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 409 or time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def run(args):
    topology = topology_from_args(args)
    # Historical measurements and answers are never inherited from these inputs.
    workload = getattr(args, "workload", "dihuo")
    if workload == "code":
        if (args.input_tokens != 2048 or args.output_tokens != 1024
                or args.manifest_sha256 != CODE8_MANIFEST_SHA256 or args.requests is None):
            raise ValueError("Code performance requires the exact frozen 2048/1024 Code8 split inputs")
        loaded = load_inputs(args.manifest, args.requests)
        frozen = {"samples": loaded["samples"], "proof": {
            "workload": "code", "manifest_sha256": loaded["manifest_sha256"],
            "input_artifact_sha256": [row["artifact_sha256"] for row in loaded["samples"]]}}
    else:
        frozen = load_frozen_dihuo_inputs(args.manifest, args.manifest_sha256,
                                          args.input_tokens, args.output_tokens)
    ignore_eos = workload == "dihuo"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "input_proof.json").write_text(json.dumps(frozen["proof"], indent=2) + "\n")
    identity = http_json(args.base_url, "/dp/identity")
    validate_identity(identity, topology, served_model=args.model,
                      sid=args.sid, fingerprint=args.fingerprint)
    models = http_json(args.base_url, "/v1/models")
    if [r.get("id") for r in models.get("data", [])].count(args.model) != 1:
        raise ValueError("Actual model ownership differs")
    (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    (output / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    results = []
    width = 1 if args.mode == "serial" else topology.dp
    for round_index in range(args.rounds):
        if workload == "code":
            (output / f"code-outputs-r{round_index:02d}").mkdir()
        for start in range(0, 8, width):
            jobs = []
            for index in range(start, min(start+width, 8)):
                rank = index % topology.dp
                tag = f"dpm-{args.mode}-r{round_index}-i{index}-{uuid.uuid4().hex}"
                arm = http_json(args.base_url, "/dp/arm", payload={
                    "dp_rank": rank, "tag": tag, "max_tokens": args.output_tokens})
                jobs.append((index, rank, tag, arm))
            barrier = threading.Barrier(len(jobs)) if len(jobs) > 1 else None
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                futures = [pool.submit(stream, args.base_url, args.model,
                    frozen["samples"][index]["messages"], dp_rank=rank, tag=tag,
                    max_tokens=args.output_tokens, seed=index, ignore_eos=ignore_eos,
                    timeout=args.timeout, barrier=barrier) for index, rank, tag, arm in jobs]
                streams = [future.result() for future in futures]
            wave = []
            for (index, rank, tag, arm), row in zip(jobs, streams):
                receipt = {"round_index": round_index, "input_index": index,
                           "dp_rank": rank, "arm": arm, "stream": row,
                           "metrics": None, "reduction": None, "error": None}
                try:
                    if row["error"] is not None:
                        raise ValueError(row["error"])
                    actual = metrics_after(args.base_url, rank, tag, args.timeout)
                    receipt["metrics"] = actual
                    receipt["reduction"] = request_metrics(arm, actual, row, topology,
                        dp_rank=rank, tag=tag, sid=args.sid, fingerprint=args.fingerprint,
                        prompt_tokens=args.input_tokens, max_tokens=args.output_tokens,
                        ignore_eos=ignore_eos)
                except Exception as exc:
                    receipt["error"] = f"{type(exc).__name__}: {exc}"
                path = output / f"r{round_index:02d}-i{index:02d}-dp{rank}.json"
                path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
                if workload == "code":
                    sample = frozen["samples"][index]
                    code_row = {"schema": "native_dp_code8_response_v1",
                                "request_scope": "armed_performance",
                                "service_instance_id": args.sid, "runtime_fingerprint": args.fingerprint,
                                "served_model": args.model, "layout": args.layout,
                                "physical_chips": list(topology.chips),
                                "sample_id": sample["sample_id"], "task_id": sample["task_id"],
                                "split_artifact_sha256": sample["artifact_sha256"],
                                "prompt_sha256": sample["prompt_sha256"],
                                "unit_cases_sha256": sample["unit_cases_sha256"],
                                "http_status": row["http_status"], "finish_reason": row["finish_reason"],
                                "error": row["error"], "usage": row["usage"],
                                "output_text": row["output_text"], "output_sha256": row["output_sha256"],
                                "scope": "separate armed performance request; not ordinary Code8 quality gate"}
                    (output / f"code-outputs-r{round_index:02d}" / f"request-{index:02d}.json").write_text(
                        json.dumps(code_row, ensure_ascii=False, indent=2) + "\n")
                wave.append(receipt)
            results.extend(wave)
            if any(row["error"] is not None for row in wave):
                raise ValueError("Actual DP measurement failed; raw receipts preserved, no automatic request retry")
            summary = wave_metrics(streams)
            summary.update(round_index=round_index, input_start=start,
                           clean_requests=sum(r["clean"]["clean"] for r in streams))
            (output / f"wave-r{round_index:02d}-i{start:02d}.json").write_text(
                json.dumps(summary, indent=2) + "\n")
    after = http_json(args.base_url, "/dp/identity")
    validate_identity(after, topology, served_model=args.model,
                      sid=args.sid, fingerprint=args.fingerprint)
    (output / "identity_after.json").write_text(json.dumps(after, indent=2) + "\n")
    final = {"service_instance_id": args.sid, "runtime_fingerprint": args.fingerprint,
             "layout": args.layout, "mode": args.mode, "requests": len(results),
             "physical_chips": list(topology.chips),
             "workload": workload, "ignore_eos": ignore_eos,
             "rounds": args.rounds, "all_metrics_verified": True,
             "clean_requests": sum(r["stream"]["clean"]["clean"] for r in results),
             "maximum_saturation_proven": False,
             "client_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "result.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("base-url", "model", "layout", "sid", "fingerprint", "manifest",
                 "manifest-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=("serial", "replicas"), required=True)
    parser.add_argument("--workload", choices=("dihuo", "code"), default="dihuo")
    parser.add_argument("--requests", help="Code8 split request directory; required only for code")
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--physical-chips", default=DEFAULT_CHIP_LIST)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10:
        parser.error("Explicit bounded rounds 1..10 required")
    run(args)


if __name__ == "__main__":
    main()
