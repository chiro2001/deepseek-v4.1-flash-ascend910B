"""Replay frozen Code8 prompts over ordinary native-DP HTTP chat requests."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import uuid

from .topology import topology_from_args, validate_identity, DEFAULT_CHIP_LIST


MANIFEST_SHA256 = "6ec9d9267695a4b6e7d5e27a1beaf0823df8498c9762b4c7204d2892474a8704"
BENCHMARK_SHA256 = "33c4580e2e458de466a2fd48f20259d97c2d856e29a5fa35c25bbff750d23816"
WORKLOADS_SHA256 = "62db3507dd74ea69406a940963f93414ae2e9f87aa77424f8204f5c9b7ce4d35"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_sha(value) -> str:
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8"))


def load_inputs(manifest_path, request_root, expected_manifest_sha=MANIFEST_SHA256):
    manifest_path = Path(manifest_path)
    request_root = Path(request_root)
    if manifest_path.is_symlink() or request_root.is_symlink():
        raise ValueError("Code8 manifest/request root must not be symlinks")
    manifest_path = manifest_path.resolve(strict=True)
    request_root = request_root.resolve(strict=True)
    raw = manifest_path.read_bytes()
    if sha(raw) != expected_manifest_sha:
        raise ValueError("Frozen Code8 manifest SHA-256 mismatch")
    manifest = json.loads(raw)
    if (manifest.get("schema_version") != "delivery_matrix_split_manifest_v1"
            or manifest.get("artifact_count") != 8 or len(manifest.get("artifacts", [])) != 8
            or manifest.get("artifact_compression") is not None):
        raise ValueError("Unsupported Code8 split manifest")
    source = manifest.get("source")
    workload = source.get("workload_manifest") if isinstance(source, dict) else None
    if (not isinstance(workload, dict) or workload.get("tasks") != ["code"]
            or workload.get("input_tokens_target") != 2048
            or workload.get("code_output_tokens") != 1024
            or len(workload.get("prompt_sha256_by_task", {}).get("code", [])) != 8):
        raise ValueError("Frozen Code8 workload contract differs")
    if request_root.is_symlink() or manifest_path.parent != request_root.parent:
        raise ValueError("Code8 manifest and request directory must have one bounded parent")
    samples = []
    for index, record in enumerate(manifest["artifacts"]):
        expected_name = f"code_round00_request{index:02d}.json"
        if record.get("path") != expected_name or record.get("compression") is not None:
            raise ValueError("Code8 request order or encoding differs")
        path = request_root / expected_name
        if path.is_symlink() or path.parent != request_root or not path.is_file():
            raise ValueError("Code8 request path is not a regular in-directory file")
        payload = path.read_bytes()
        if len(payload) != record.get("size_bytes") or sha(payload) != record.get("sha256"):
            raise ValueError("Code8 split request size/SHA mismatch: " + expected_name)
        row = json.loads(payload)
        request = row.get("request") if isinstance(row, dict) else None
        if (row.get("schema_version") != "delivery_matrix_request_receipt_v1"
                or row.get("task") != "code" or row.get("request_index") != index
                or row.get("round_index") != 0 or request.get("task") != "code"
                or row.get("source") != source
                or request.get("request_index") != index or request.get("round_index") != 0
                or request.get("requested_max_tokens") != 1024
                or request.get("ignore_eos") is not False):
            raise ValueError("Code8 request contract mismatch: " + expected_name)
        prompt = request.get("prompt_text")
        messages = request.get("prompt_messages")
        cases = request.get("unit_cases")
        if (not isinstance(prompt, str) or sha(prompt.encode()) != request.get("prompt_sha256")
                or request.get("prompt_sha256") != record.get("prompt_sha256")
                or messages != [{"role": "user", "content": prompt}]
                or request.get("prompt_messages_sha256") != canonical_sha(messages)
                or request.get("unit_cases_sha256") != canonical_sha(cases)
                or request.get("unit_cases_sha256") != workload.get("code_unit_cases_sha256", {}).get(request.get("code_task_id"))):
            raise ValueError("Code8 prompt/messages/test-case provenance mismatch: " + expected_name)
        samples.append({
            "index": index, "artifact": expected_name,
            "artifact_sha256": record["sha256"],
            "task_id": request["code_task_id"], "sample_id": request["sample_id"],
            "seed": request["seed"], "prompt_text": prompt,
            "prompt_sha256": request["prompt_sha256"],
            "messages": messages, "unit_cases": cases,
            "unit_cases_sha256": request["unit_cases_sha256"],
        })
    if ([row["prompt_sha256"] for row in samples]
            != workload["prompt_sha256_by_task"]["code"]):
        raise ValueError("Code8 prompts differ from the frozen workload manifest")
    return {"manifest": manifest, "manifest_sha256": sha(raw), "samples": samples}


def http_json(base_url, path, timeout):
    request = urllib.request.Request(base_url.rstrip("/") + path)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"GET {path} returned HTTP {response.status}")
        return json.load(response)


def load_frozen_client(source_root):
    source_root = Path(source_root).resolve(strict=True)
    benchmark_path = source_root / "benchmark.py"
    workloads_path = source_root / "workloads.py"
    if sha(benchmark_path.read_bytes()) != BENCHMARK_SHA256:
        raise ValueError("Frozen Code8 benchmark.py SHA-256 mismatch")
    if sha(workloads_path.read_bytes()) != WORKLOADS_SHA256:
        raise ValueError("Frozen Code8 workloads.py SHA-256 mismatch")
    existing = sys.modules.get("workloads")
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() != workloads_path:
        raise ValueError("A different workloads module was already imported")
    sys.path.insert(0, str(source_root))
    spec = importlib.util.spec_from_file_location("_dp_matrix_frozen_code_benchmark", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replay(args):
    topology = topology_from_args(args)
    inputs = load_inputs(args.manifest, args.requests)
    benchmark = load_frozen_client(args.source_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    identity_before = http_json(args.base_url, "/dp/identity", args.timeout)
    validate_identity(identity_before, topology, served_model=args.model,
                      sid=args.sid, fingerprint=args.fingerprint)
    models = http_json(args.base_url, "/v1/models", args.timeout)
    if [item.get("id") for item in models.get("data", [])].count(args.model) != 1:
        raise ValueError("The expected native-DP model is not uniquely served")
    (output / "identity_before.json").write_text(json.dumps(identity_before, indent=2) + "\n")
    (output / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    results = []
    for sample in inputs["samples"]:
        request_id = "dp-code8-" + uuid.uuid4().hex
        started = time.monotonic()
        response = benchmark.stream_chat_completion(
            args.base_url, args.model, sample["messages"], 1024, args.timeout,
            None, sample["seed"], request_id, False)
        row = {"schema": "native_dp_code8_response_v1", "sample_id": sample["sample_id"],
               "request_scope": "ordinary_quality",
               "service_instance_id": args.sid, "runtime_fingerprint": args.fingerprint,
               "served_model": args.model, "layout": args.layout,
               "physical_chips": list(topology.chips),
               "request_id": request_id, "task_id": sample["task_id"],
               "split_artifact": sample["artifact"],
               "split_artifact_sha256": sample["artifact_sha256"],
               "prompt_sha256": sample["prompt_sha256"],
               "unit_cases_sha256": sample["unit_cases_sha256"],
               "request_body_sha256": sha(json.dumps({
                   "model": args.model, "messages": sample["messages"], "temperature": 0,
                   "seed": sample["seed"], "request_id": request_id, "max_tokens": 1024,
                   "stream": True, "stream_options": {"include_usage": True}},
                   ensure_ascii=False).encode("utf-8")),
               "http_status": response["http_status"], "error": response["error"],
               "finish_reason": response["finish_reason"], "usage": response["usage_tokens"],
               "output_text": response["output_text"], "output_sha256": response["output_sha256"],
               "ttft_s": response["ttft_s"], "decode_s": response["decode_s"],
               "request_wall_s": response["request_wall_s"],
               "started_monotonic_s": started, "ended_monotonic_s": time.monotonic(),
               "sse_error_event": response["sse_error_event"],
               "sse_json_events": response["sse_json_events"],
               "response_complete_by_frozen_client": response["error"] is None
                   and response["finish_reason"] == "stop"
                   and response["usage_tokens"].get("completion_tokens") is not None}
        (output / f"request-{sample['index']:02d}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n")
        results.append(row)
    identity_after = http_json(args.base_url, "/dp/identity", args.timeout)
    validate_identity(identity_after, topology, served_model=args.model,
                      sid=args.sid, fingerprint=args.fingerprint)
    if identity_before != identity_after:
        raise ValueError("Service identity changed during the Code8 suite")
    (output / "identity_after.json").write_text(json.dumps(identity_after, indent=2) + "\n")
    result = {"schema": "native_dp_code8_replay_v1", "layout": args.layout,
              "physical_chips": list(topology.chips),
              "served_model": args.model, "service_instance_id": args.sid,
              "runtime_fingerprint": args.fingerprint,
              "split_manifest_sha256": inputs["manifest_sha256"],
              "benchmark_source_sha256": BENCHMARK_SHA256,
              "workloads_source_sha256": WORKLOADS_SHA256,
              "requests": len(results), "successful_http": sum(r["http_status"] == 200 for r in results),
              "complete_responses": sum(r["response_complete_by_frozen_client"] for r in results),
              "maximum_saturation_proven": False}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("base-url", "model", "layout", "sid", "fingerprint", "manifest",
                 "requests", "source-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--physical-chips", default=DEFAULT_CHIP_LIST)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("A positive request timeout is required")
    replay(args)


if __name__ == "__main__":
    main()
