"""Score new Code8 outputs using the byte-pinned isolated workloads scorer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from .code8_replay import (MANIFEST_SHA256, WORKLOADS_SHA256, canonical_sha, load_inputs, sha)
from .quality_audit import audit as audit_wire, artifact
from .topology import topology_from_args, validate_identity, DEFAULT_CHIP_LIST
import gzip


def streamed_content(raw):
    parts = []
    for line in raw.splitlines():
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            continue
        event = json.loads(data)
        for choice in event.get("choices", []):
            content = choice.get("delta", {}).get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(item["text"] for item in content
                             if isinstance(item, dict) and isinstance(item.get("text"), str))
    return "".join(parts)


def load_frozen_workloads(path):
    path = Path(path).resolve(strict=True)
    if sha(path.read_bytes()) != WORKLOADS_SHA256:
        raise ValueError("Frozen Code8 workloads.py SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("_dp_matrix_frozen_code_workloads", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def score(args):
    topology = topology_from_args(args)
    if (args.manifest_sha256 != MANIFEST_SHA256
            or (args.timeout, args.memory_mb, args.cpu_seconds) != (3.0, 512, 2)):
        raise ValueError("Strict Code8 scoring requires the frozen inputs and 3s/512MiB/2CPU limits")
    inputs = load_inputs(args.manifest, args.requests, args.manifest_sha256)
    workloads = load_frozen_workloads(args.workloads)
    output_root = Path(args.responses).resolve(strict=True)
    wire_result = None
    journal = []
    if args.scope == "ordinary_quality":
        if args.proxy_root is None:
            raise ValueError("Ordinary Code8 quality scoring requires the raw proxy journal")
        args.proxy_root = Path(args.proxy_root).resolve(strict=True)
        journal = [json.loads(line) for line in (args.proxy_root / "attempts.jsonl").read_text().splitlines()]
        for phase in ("before", "after"):
            identity = json.loads((output_root / f"identity_{phase}.json").read_text())
            validate_identity(identity, topology, served_model=args.model,
                              sid=args.sid, fingerprint=args.fingerprint)
        expected_hashes = [json.loads((output_root / f"request-{index:02d}.json").read_text())["request_body_sha256"]
                           for index in range(8)]
        wire_result = audit_wire(args.proxy_root, dp_count=topology.dp,
                                 min_requests=8, min_unique=8, expected_hashes=expected_hashes)
    results = []
    for sample in inputs["samples"]:
        response_path = output_root / f"request-{sample['index']:02d}.json"
        if response_path.is_symlink() or not response_path.is_file():
            raise ValueError("Missing or symlinked Code8 response: " + response_path.name)
        response = json.loads(response_path.read_text())
        if response.get("schema") != "native_dp_code8_response_v1":
            raise ValueError("Code8 response schema differs")
        if response.get("request_scope") != args.scope:
            raise ValueError("Quality and armed performance outputs must be scored under their own scope")
        if ("physical_chips" in response and response["physical_chips"] != list(topology.chips)):
            raise ValueError("Code8 response physical-chip placement differs from the explicit run")
        if response.get("error") is not None or response.get("sse_error_event") is not None:
            raise ValueError("Code8 response contains an HTTP/SSE error event")
        for key, value in (("sample_id", sample["sample_id"]),
                            ("service_instance_id", args.sid), ("runtime_fingerprint", args.fingerprint),
                            ("served_model", args.model), ("layout", args.layout),
                            ("task_id", sample["task_id"]),
                            ("split_artifact_sha256", sample["artifact_sha256"]),
                            ("prompt_sha256", sample["prompt_sha256"]),
                            ("unit_cases_sha256", sample["unit_cases_sha256"])):
            if response.get(key) != value:
                raise ValueError("Code8 response provenance mismatch: " + key)
        text = response.get("output_text")
        if canonical_sha(workloads.fixed_unit_cases(sample["task_id"])) != sample["unit_cases_sha256"]:
            raise ValueError("Executed frozen unit cases differ from the recorded input case SHA")
        if (not isinstance(text, str)
                or response.get("output_sha256") != sha(text.encode("utf-8"))):
            raise ValueError("Code8 output text/hash mismatch")
        usage = response.get("usage")
        if (not isinstance(usage, dict) or type(usage.get("prompt_tokens")) is not int
                or usage["prompt_tokens"] != 2048
                or type(usage.get("completion_tokens")) is not int
                or not 1 <= usage["completion_tokens"] <= 1024
                or usage.get("total_tokens") != usage["prompt_tokens"] + usage["completion_tokens"]):
            raise ValueError("Actual Code8 usage differs from the frozen 2048/1024 token contract")
        if args.scope == "ordinary_quality":
            matched = [row for row in journal if row.get("generation") is True
                       and row.get("request_body_sha256") == response.get("request_body_sha256")]
            if len(matched) != 1 or matched[0].get("response_is_sse") is not True:
                raise ValueError("Code8 replay must bind one actual SSE proxy attempt per case")
            wire = matched[0]
            if any(wire["usage"].get(key) != usage[key] for key in
                   ("prompt_tokens", "completion_tokens", "total_tokens")):
                raise ValueError("Scored Code8 usage differs from the actual raw SSE usage")
            request = json.loads(gzip.decompress(artifact(args.proxy_root, wire["request_body_file"])))
            if (request.get("model") != args.model or request.get("messages") != sample["messages"]
                    or request.get("max_tokens") != 1024 or request.get("temperature") != 0
                    or request.get("seed") != sample["seed"] or request.get("ignore_eos", False) is not False):
                raise ValueError("Code8 actual HTTP payload differs from its frozen prompt/generation semantics")
            if streamed_content(artifact(args.proxy_root, wire["response_body_file"])) != text:
                raise ValueError("Code8 scored text differs from the actual raw SSE output")
        if response.get("http_status") != 200 or response.get("finish_reason") != "stop":
            result = {"status": "request_failure", "passed": 0,
                      "total": len(sample["unit_cases"]),
                      "failed_case_ids": [case["id"] for case in sample["unit_cases"]],
                      "error": response.get("error") or "HTTP/finish contract failed"}
        else:
            result = workloads.score_code_output(sample["task_id"], text,
                timeout_seconds=args.timeout, memory_mb=args.memory_mb,
                cpu_seconds=args.cpu_seconds)
        results.append({"sample_id": sample["sample_id"], "task_id": sample["task_id"],
                        "split_artifact_sha256": sample["artifact_sha256"],
                        "output_sha256": response["output_sha256"],
                        "unit_cases_sha256": sample["unit_cases_sha256"],
                        "score": result})
    passed_tasks = sum(row["score"].get("status") == "pass"
                       and row["score"].get("passed") == row["score"].get("total")
                       for row in results)
    passed_cases = sum(row["score"].get("passed", 0) for row in results)
    total_cases = sum(row["score"].get("total", 0) for row in results)
    cases_passed = passed_tasks == 8 and passed_cases == total_cases == 128
    summary = {"schema": "native_dp_code8_score_v1",
               "service_instance_id": args.sid, "runtime_fingerprint": args.fingerprint,
               "served_model": args.model, "layout": args.layout,
               "physical_chips": list(topology.chips),
               "split_manifest_sha256": inputs["manifest_sha256"],
               "workloads_source_sha256": WORKLOADS_SHA256,
               "tasks": len(results), "passed_tasks": passed_tasks,
               "passed_cases": passed_cases, "total_cases": total_cases,
               "request_scope": args.scope,
               "all_unit_cases_passed": cases_passed,
               "code8_quality_gate_passed": cases_passed and wire_result is not None,
               "all_quality_suites_passed": False,
               "isolation": "frozen workloads.score_code_output; bwrap --unshare-net; read-only runtime; 512MiB; 2 CPU seconds; 3s wall"}
    document = {"summary": summary, "wire_audit": wire_result, "results": results}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with destination.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--workloads", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    for name in ("sid", "fingerprint", "model", "layout"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--scope", required=True, choices=("ordinary_quality", "armed_performance"))
    parser.add_argument("--proxy-root", type=Path)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--memory-mb", type=int, default=512)
    parser.add_argument("--cpu-seconds", type=int, default=2)
    parser.add_argument("--physical-chips", default=DEFAULT_CHIP_LIST)
    args = parser.parse_args()
    if args.timeout <= 0 or args.memory_mb <= 0 or args.cpu_seconds <= 0:
        parser.error("Scorer resource limits must be positive")
    score(args)


if __name__ == "__main__":
    main()
