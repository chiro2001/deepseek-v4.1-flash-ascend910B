"""Read pinned historical prompt bytes without inheriting their measurements.

This is a CPU-only, fail-closed input gate. It never imports a tokenizer,
connects to a service, or distributes the copyrighted prompt text.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any


MAX_MANIFEST_BYTES = 512 * 1024
MAX_COMPRESSED_BYTES = 999_999
MAX_REQUEST_BYTES = 8 * 1024 * 1024
INPUT_METADATA_KEYS = ("sample_id", "suffix_name", "corpus_offset_chars", "corpus_slice_chars",
                       "corpus_slice_stride_chars", "tokenizer_count", "input_tokens_exact")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("Frozen Dihuo input gate: " + message)


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _integer(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _json(data: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Frozen Dihuo input gate: non-finite JSON constant " + value)
    return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def load_frozen_dihuo_inputs(manifest_path: str | Path, expected_manifest_sha256: str,
                             input_tokens: int, output_tokens: int) -> dict[str, Any]:
    """Verify all eight gzip/raw/message hashes before returning any prompts.

    Only input messages and original input metadata are returned. Historical
    outputs, latency, acceptance, counters, and quality grants are not reused.
    Historical token counts are provenance, not a new tokenizer measurement;
    the existing client's actual SSE usage gate must still pass on this run.
    """
    _require(_digest(expected_manifest_sha256), "an explicit manifest SHA-256 is required")
    _require(not Path(manifest_path).is_symlink(), "manifest must not be a file symlink")
    path = Path(manifest_path).resolve(strict=True)
    _require(path.is_file() and 0 < path.stat().st_size <= MAX_MANIFEST_BYTES,
             "manifest size is outside the bounded input contract")
    with path.open("rb") as stream:
        raw_manifest = stream.read(MAX_MANIFEST_BYTES + 1)
    _require(len(raw_manifest) <= MAX_MANIFEST_BYTES, "manifest grew beyond the input bound")
    _require(_sha(raw_manifest) == expected_manifest_sha256, "manifest SHA-256 mismatch")
    manifest = _json(raw_manifest)
    _require(isinstance(manifest, dict), "manifest must be an object")
    _require(manifest.get("schema_version") == "delivery_matrix_split_manifest_v1"
             and manifest.get("artifact_compression") == "gzip"
             and _integer(manifest.get("artifact_count"), 8), "unsupported archive schema/count")
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, list) and len(artifacts) == 8, "exactly eight artifacts required")
    source = manifest.get("source")
    _require(isinstance(source, dict), "missing historical input source")
    workload = source.get("workload_manifest")
    _require(isinstance(workload, dict) and workload.get("tasks") == ["dihuo"]
             and _integer(workload.get("input_tokens_target"), input_tokens)
             and _integer(workload.get("dihuo_output_tokens"), output_tokens),
             "historical workload differs from the requested Dihuo budget")
    _require(_canonical_sha(workload) == source.get("workload_fingerprint"),
             "historical workload fingerprint mismatch")
    expected_prompts = workload.get("prompt_sha256_by_task", {}).get("dihuo")
    _require(isinstance(expected_prompts, list) and len(expected_prompts) == 8
             and all(_digest(value) for value in expected_prompts)
             and len(set(expected_prompts)) == 8, "eight distinct pinned prompt hashes required")
    declared_request_root = path.parent / "requests"
    request_root = declared_request_root.resolve(strict=True)
    _require(request_root == declared_request_root, "requests directory escapes its manifest root")
    _require(request_root.is_dir(), "requests must be a directory")
    samples, receipts = [], []
    for index, artifact in enumerate(artifacts):
        name = f"dihuo_round00_request{index:02d}.json.gz"
        _require(isinstance(artifact, dict) and artifact.get("path") == name
                 and artifact.get("compression") == "gzip", "artifact order/path/compression mismatch")
        compressed_size = artifact.get("size_bytes")
        raw_size = artifact.get("uncompressed_size_bytes")
        _require(type(compressed_size) is int and 0 < compressed_size <= MAX_COMPRESSED_BYTES
                 and type(raw_size) is int and 0 < raw_size <= MAX_REQUEST_BYTES,
                 "artifact size is outside the bounded input contract")
        request_path = (request_root / name).resolve(strict=True)
        _require(request_path.parent == request_root and request_path.is_file(),
                 "artifact escapes the request directory")
        _require(request_path.stat().st_size == compressed_size, "compressed size mismatch")
        with request_path.open("rb") as stream:
            compressed = stream.read(MAX_COMPRESSED_BYTES + 1)
        _require(len(compressed) == compressed_size, "compressed file changed during verification")
        _require(_sha(compressed) == artifact.get("sha256"), "compressed SHA-256 mismatch")
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            raw = stream.read(MAX_REQUEST_BYTES + 1)
        _require(len(raw) == raw_size and len(raw) <= MAX_REQUEST_BYTES,
                 "uncompressed size mismatch")
        _require(_sha(raw) == artifact.get("uncompressed_sha256"), "uncompressed SHA-256 mismatch")
        document = _json(raw)
        _require(isinstance(document, dict)
                 and document.get("schema_version") == "delivery_matrix_request_receipt_v1"
                 and document.get("source") == source, "request receipt/source mismatch")
        _require(document.get("task") == "dihuo"
                 and _integer(document.get("round_index"), 0)
                 and _integer(document.get("request_index"), index), "request receipt order mismatch")
        request = document.get("request")
        _require(isinstance(request, dict), "request payload must be an object")
        text, messages = request.get("prompt_text"), request.get("prompt_messages")
        _require(isinstance(text, str) and text.startswith(f"[delivery-matrix-dihuo-{index + 1:02d}]\n")
                 and messages == [{"role": "user", "content": text}], "prompt/messages contract mismatch")
        prompt_sha = _sha(text.encode("utf-8"))
        _require(prompt_sha == request.get("prompt_sha256") == artifact.get("prompt_sha256")
                 == expected_prompts[index], "prompt SHA-256 mismatch")
        messages_sha = _canonical_sha(messages)
        _require(messages_sha == request.get("prompt_messages_sha256"), "messages SHA-256 mismatch")
        sample_id = f"dihuo-{index + 1:02d}"
        _require(request.get("task") == "dihuo" and request.get("sample_id") == sample_id
                 and _integer(request.get("round_index"), 0)
                 and _integer(request.get("request_index"), index)
                 and _integer(request.get("seed"), index)
                 and request.get("ignore_eos") is True
                 and _integer(request.get("requested_max_tokens"), output_tokens),
                 "historical request generation contract mismatch")
        usage, metadata = request.get("usage_tokens"), request.get("source_metadata")
        _require(isinstance(usage, dict) and _integer(usage.get("prompt_tokens"), input_tokens)
                 and _integer(usage.get("completion_tokens"), output_tokens)
                 and _integer(usage.get("total_tokens"), input_tokens + output_tokens),
                 "historical exact usage gate did not pass")
        _require(request.get("http_status") == 200 and request.get("finish_reason") == "length"
                 and request.get("error") is None and request.get("sse_error_event") is None
                 and isinstance(request.get("clean"), dict) and request["clean"].get("clean") is True,
                 "historical request did not complete cleanly")
        _require(isinstance(metadata, dict) and metadata.get("sample_id") == sample_id
                 and metadata.get("input_tokens_exact") is True
                 and _integer(metadata.get("tokenizer_count"), input_tokens),
                 "historical tokenizer provenance mismatch")
        sample = copy.deepcopy({key: metadata[key] for key in INPUT_METADATA_KEYS if key in metadata})
        sample.update(messages=copy.deepcopy(messages), prompt_text=text,
                      tokenizer_count_scope="historical verified archive receipt; not counted by a tokenizer this run",
                      input_tokens_exact_scope="historical usage; current SSE exact-usage gate remains required")
        samples.append(sample)
        receipts.append({"path": name, "compressed_sha256": _sha(compressed),
                         "uncompressed_sha256": _sha(raw), "prompt_sha256": prompt_sha,
                         "prompt_messages_sha256": messages_sha})
    return {"samples": samples, "workload_manifest": copy.deepcopy(workload),
            "dihuo_source": copy.deepcopy(source.get("dihuo_source")),
            "proof": {"schema": "frozen_dihuo_input_gate_v1", "manifest_path": str(path),
                      "manifest_sha256": _sha(raw_manifest), "artifact_count": 8,
                      "all_inputs_verified_before_service_requests": True,
                      "historical_measurements_reused": False,
                      "token_count_scope": "historical usage/tokenizer provenance; current SSE usage gate remains required",
                      "artifacts": receipts}}
