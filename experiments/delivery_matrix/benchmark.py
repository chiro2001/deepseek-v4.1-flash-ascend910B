#!/usr/bin/env python3
"""Reproducible serial streaming benchmark for delivery_matrix service forms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from frozen_dihuo_inputs import load_frozen_dihuo_inputs
from workloads import (
    CODE_TASKS,
    CODE_TASK_IDS,
    DIHUO_PATH,
    DIHUO_SUFFIX_DIR,
    build_code_prompt,
    code_padding_prefix,
    dihuo_prompt,
    fixed_unit_cases,
    load_dihuo_material,
    output_cleanliness,
    repeated_corpus_slice,
    score_code_output,
    sha256_bytes,
    sha256_text,
)


COUNTER_SCHEMA = "delivery_runtime_metrics_v1"
COUNTER_SCOPE = "actual runner scheduled/request CPU-token bookkeeping"
CLIENT_DECODE_MS_PER_STEP_SCOPE = (
    "HTTP SSE first-content to last-content decode_s, normalized by trusted delivery-counter decode_rounds; excludes TTFT."
)
SERVER_FENCED_MS_PER_STEP_SCOPE = (
    "GET /delivery/metrics ranks[].cumulative.decode_seconds delta, normalized by trusted decode_rounds; "
    "terminal prefill CPU bookkeeping -> worker completion fence; may include metrics RPC delay."
)
CED_FORM = "a3-ced-pd"
CED_PREFILL_REPORTING = "sum of independently observed P encoder and D replay/tail wall times"
RANK_COUNT = 8
TASK_OPTIONS = ("dihuo", "code", "all")
INPUT_TOKEN_OPTIONS = (2048, 32768, 147456)
CODE_OUTPUT_TOKENS_DEFAULT = 1024
_CURRENT_STAGE = "startup"


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _auth_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _post_json(url: str, payload: dict[str, Any], timeout: float, api_key: str | None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_auth_headers(api_key),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, timeout: float, api_key: str | None) -> dict[str, Any]:
    headers = _auth_headers(api_key)
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def verify_model_ownership(base_url: str, expected_model: str, timeout: float, api_key: str | None) -> dict[str, Any]:
    """Require an exact model-id match from /v1/models; /health is not used."""
    response = _get_json(_url(base_url, "/v1/models"), timeout, api_key)
    entries = response.get("data")
    if not isinstance(entries, list):
        raise ValueError("GET /v1/models returned no data[] model list")
    ids = [item.get("id") for item in entries if isinstance(item, dict) and isinstance(item.get("id"), str)]
    matches = [model_id for model_id in ids if model_id == expected_model]
    if len(matches) != 1:
        raise ValueError(
            f"model ownership mismatch: expected exactly one id {expected_model!r}; returned ids={ids!r}"
        )
    return {
        "endpoint": _url(base_url, "/v1/models"),
        "expected_model": expected_model,
        "model_ids": ids,
        "exact_match_count": len(matches),
        "verified": True,
        "response_sha256": canonical_json_sha256(response),
    }


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_ced_rank_descriptions(ranks: Any, label: str) -> None:
    if not isinstance(ranks, list) or len(ranks) != RANK_COUNT:
        raise ValueError(f"CED {label} identity must describe exactly eight ranks")
    rank_ids = [row.get("rank") for row in ranks if isinstance(row, dict)]
    if len(rank_ids) != RANK_COUNT or any(type(rank) is not int for rank in rank_ids) or set(rank_ids) != set(range(RANK_COUNT)):
        raise ValueError(f"CED {label} identity rank set is incomplete: {rank_ids!r}")
    if any(row.get("form") != CED_FORM or row.get("tp") != RANK_COUNT or row.get("dcp") != 1 for row in ranks):
        raise ValueError(f"CED {label} ranks must agree on form, TP8, and DCP=1")


def _validate_ced_identity_pair(identity: dict[str, Any], expected_model: str) -> tuple[str, str]:
    if identity.get("topology") != "P8+D8; DCP=1" or identity.get("role") != "decode":
        raise ValueError("CED proxy identity must expose the P8+D8; DCP=1 decode pair")
    prefill = identity.get("prefill_identity")
    decode = identity.get("decode_identity")
    if not isinstance(prefill, dict) or not isinstance(decode, dict):
        raise ValueError("CED proxy identity is missing the nested P/D identities")

    for value, role in ((prefill, "prefill"), (decode, "decode")):
        if (
            value.get("served_model") != expected_model
            or value.get("form") != CED_FORM
            or value.get("role") != role
            or value.get("engine_initialized") is not True
            or value.get("ASGI_startup_completed") is not True
        ):
            raise ValueError(f"CED {role} identity is not the selected initialized/ready model")
        model_kwargs = value.get("model_kwargs")
        if (
            not isinstance(model_kwargs, dict)
            or model_kwargs.get("enable_prefix_caching") is not False
            or model_kwargs.get("decode_context_parallel_size") != 1
        ):
            raise ValueError(f"CED {role} identity does not verify PREFIX=0 and DCP=1")
        if not isinstance(value.get("service_instance_id"), str) or not value["service_instance_id"].strip():
            raise ValueError(f"CED {role} identity has no service_instance_id")
        if not _is_sha256_hex(value.get("runtime_fingerprint")):
            raise ValueError(f"CED {role} identity has no runtime_fingerprint")
        _validate_ced_rank_descriptions(value.get("rank_descriptions"), role)

    prefill_sid = prefill["service_instance_id"]
    decode_sid = decode["service_instance_id"]
    expected_pair_sid = hashlib.sha256(f"{prefill_sid}:{decode_sid}".encode("utf-8")).hexdigest()
    if identity.get("service_instance_id") != expected_pair_sid:
        raise ValueError("CED root service_instance_id is not the SHA-256 P/D SID pair")
    expected_pair_runtime = hashlib.sha256(
        (prefill["runtime_fingerprint"] + decode["runtime_fingerprint"]).encode("utf-8")
    ).hexdigest()
    if identity.get("runtime_fingerprint") != expected_pair_runtime:
        raise ValueError("CED root runtime_fingerprint is not bound to the P/D runtime pair")
    if identity.get("rank_descriptions") != decode.get("rank_descriptions"):
        raise ValueError("CED root rank descriptions differ from the bound D identity")

    prefill_manifest = prefill.get("ced_overlay_manifest_sha256")
    decode_manifest = decode.get("ced_overlay_manifest_sha256")
    if not _is_sha256_hex(prefill_manifest) or prefill_manifest != decode_manifest:
        raise ValueError("CED P/D must use the same pinned CED overlay manifest")
    prefill_opp = (prefill.get("native_opp_receipt") or {}).get("artifact_fingerprint")
    decode_opp = (decode.get("native_opp_receipt") or {}).get("artifact_fingerprint")
    if not _is_sha256_hex(prefill_opp) or prefill_opp != decode_opp:
        raise ValueError("CED P/D must use the same verified native OPP fingerprint")
    return prefill_sid, decode_sid


def verify_service_identity(
    base_url: str,
    expected_model: str,
    expected_form: str | None,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    """Bind measurements to the running service's immutable identity snapshot."""
    endpoint = _url(base_url, "/delivery/identity")
    identity = _get_json(endpoint, timeout, api_key)
    if identity.get("served_model") != expected_model:
        raise ValueError(
            f"delivery identity served_model mismatch: expected {expected_model!r}, got {identity.get('served_model')!r}"
        )
    form = identity.get("form")
    if not isinstance(form, str) or (expected_form is not None and form != expected_form):
        raise ValueError(f"delivery identity form mismatch: expected {expected_form!r}, got {form!r}")
    if identity.get("engine_initialized") is not True or identity.get("ASGI_startup_completed") is not True:
        raise ValueError("delivery identity does not report initialized engine and completed ASGI startup")
    ranks = identity.get("rank_descriptions")
    if not isinstance(ranks, list) or len(ranks) != RANK_COUNT:
        raise ValueError("delivery identity does not contain eight rank descriptions")
    rank_ids = [rank.get("rank") for rank in ranks if isinstance(rank, dict)]
    if len(rank_ids) != RANK_COUNT or set(rank_ids) != set(range(RANK_COUNT)):
        raise ValueError(f"delivery identity rank descriptions are incomplete: {rank_ids!r}")
    if any(rank.get("form") != form or rank.get("tp") != RANK_COUNT for rank in ranks):
        raise ValueError("delivery identity rank descriptions disagree on form or TP8 placement")
    if form == "a3-tp8-dcp8" and any(rank.get("dcp") != RANK_COUNT for rank in ranks):
        raise ValueError("delivery identity rank descriptions do not verify DCP8 placement")
    model_kwargs = identity.get("model_kwargs")
    if not isinstance(model_kwargs, dict) or model_kwargs.get("enable_prefix_caching") is not False:
        raise ValueError("delivery identity does not verify enable_prefix_caching=false (PREFIX=0)")
    if not identity.get("service_instance_id") or not identity.get("runtime_fingerprint"):
        raise ValueError("delivery identity is missing service_instance_id or runtime_fingerprint")
    prefill_service_instance_id = None
    decode_service_instance_id = None
    if form == CED_FORM:
        prefill_service_instance_id, decode_service_instance_id = _validate_ced_identity_pair(identity, expected_model)
    return {
        "endpoint": endpoint,
        "verified": True,
        "served_model": expected_model,
        "form": form,
        "service_instance_id": identity["service_instance_id"],
        "runtime_fingerprint": identity["runtime_fingerprint"],
        "prefill_service_instance_id": prefill_service_instance_id,
        "decode_service_instance_id": decode_service_instance_id,
        "source_candidate_fingerprint": identity.get("source_candidate_fingerprint"),
        "rank_descriptions": ranks,
        "identity_sha256": canonical_json_sha256(identity),
        "raw_identity": identity,
    }


class LocalChatTokenizer:
    def __init__(self, model_or_path: str, tokenizer_path: str | None = None):
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        source = tokenizer_path or model_or_path
        self.source = source
        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        if not callable(getattr(self.tokenizer, "apply_chat_template", None)):
            raise ValueError(f"local tokenizer at {source!r} has no chat template")

    def count(self, messages: list[dict[str, str]]) -> int:
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(tokens)

    def info(self) -> dict[str, Any]:
        return {"method": "local_transformers_chat_template", "source": self.source, "exact_chat_template": True}


class RemoteTokenizer:
    """Use server /tokenize for chat messages, then exact completion usage as fallback."""

    def __init__(self, base_url: str, model: str, timeout: float, api_key: str | None):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self._message_tokenize_supported: bool | None = None
        self._raw_tokenize_supported: bool | None = None
        self.query_count = 0
        self.completion_fallback_count = 0
        self.last_method = "remote_tokenize_messages"
        self.frame_overhead: int | None = None
        self.search_method: str | None = None

    @staticmethod
    def _token_count(response: dict[str, Any]) -> int | None:
        count = response.get("count")
        if isinstance(count, int) and not isinstance(count, bool):
            return count
        for key in ("tokens", "token_ids", "input_ids"):
            value = response.get(key)
            if isinstance(value, list):
                return len(value)
        return None

    def _tokenize_request(self, payload: dict[str, Any]) -> int | None:
        self.query_count += 1
        try:
            response = _post_json(_url(self.base_url, "/tokenize"), payload, self.timeout, self.api_key)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            return None
        return self._token_count(response)

    def _completion_usage_count(self, messages: list[dict[str, str]]) -> int:
        self.query_count += 1
        self.completion_fallback_count += 1
        response = _post_json(
            _url(self.base_url, "/v1/chat/completions"),
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 1,
                "stream": False,
            },
            max(self.timeout, 120.0),
            self.api_key,
        )
        usage = response.get("usage")
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool):
            raise RuntimeError("remote completion fallback returned no integer usage.prompt_tokens")
        self.last_method = "remote_chat_completion_usage_fallback"
        return prompt_tokens

    def count(self, messages: list[dict[str, str]]) -> int:
        if self._message_tokenize_supported is not False:
            try:
                self.query_count += 1
                response = _post_json(
                    _url(self.base_url, "/tokenize"),
                    {"model": self.model, "messages": messages},
                    self.timeout,
                    self.api_key,
                )
                count = self._token_count(response)
                if count is not None:
                    self._message_tokenize_supported = True
                    self.last_method = "remote_tokenize_messages"
                    return count
            except (urllib.error.URLError, TimeoutError, ValueError, OSError):
                pass
            self._message_tokenize_supported = False

        # Check for a fast raw-prompt tokenization endpoint before invoking the
        # generation endpoint for every binary-search probe.  The raw count is
        # never treated as the final chat count; it only gives a useful search
        # location.  Exact counts still come from completion usage below.
        if self._raw_tokenize_supported is not False:
            raw_prompt = "\n".join(message.get("content", "") for message in messages)
            raw_count = self._tokenize_request({"model": self.model, "prompt": raw_prompt})
            if raw_count is not None:
                self._raw_tokenize_supported = True
                # Chat framing is model-specific; calibrate its offset with the
                # actual server usage and never report the raw text count as an
                # exact chat prompt length.
                exact = self._completion_usage_count(messages)
                self.frame_overhead = exact - raw_count
                self.search_method = "remote_prompt_tokenize_plus_measured_chat_offset"
                return exact
            self._raw_tokenize_supported = False
        return self._completion_usage_count(messages)

    def estimate(self, messages: list[dict[str, str]]) -> int:
        """Fast search count; only a final count() may certify the prompt length."""
        if self._message_tokenize_supported is not False:
            try:
                self.query_count += 1
                response = _post_json(
                    _url(self.base_url, "/tokenize"),
                    {"model": self.model, "messages": messages},
                    self.timeout,
                    self.api_key,
                )
                count = self._token_count(response)
                if count is not None:
                    self._message_tokenize_supported = True
                    self.search_method = "remote_tokenize_messages_exact"
                    self.last_method = "remote_tokenize_messages"
                    return count
            except (urllib.error.URLError, TimeoutError, ValueError, OSError):
                pass
            self._message_tokenize_supported = False

        if self._raw_tokenize_supported is not False:
            raw_prompt = "\n".join(message.get("content", "") for message in messages)
            raw_count = self._tokenize_request({"model": self.model, "prompt": raw_prompt})
            if raw_count is not None:
                self._raw_tokenize_supported = True
                if self.frame_overhead is None:
                    exact = self._completion_usage_count(messages)
                    self.frame_overhead = exact - raw_count
                    self.search_method = "remote_prompt_tokenize_plus_measured_chat_offset"
                    return exact
                self.search_method = "remote_prompt_tokenize_plus_measured_chat_offset"
                return raw_count + self.frame_overhead
            self._raw_tokenize_supported = False
        self.search_method = "remote_chat_completion_usage_fallback"
        return self._completion_usage_count(messages)

    def info(self) -> dict[str, Any]:
        return {
            "method": self.last_method,
            "source": _url(self.base_url, "/tokenize") if self.last_method.startswith("remote_tokenize") else "chat-completion usage.prompt_tokens",
            "query_count": self.query_count,
            "completion_fallback_count": self.completion_fallback_count,
            "search_method": self.search_method,
            "measured_chat_frame_overhead_tokens": self.frame_overhead,
            "exact_chat_template": True,
        }


class TokenCounter:
    def __init__(self, local: LocalChatTokenizer | None, remote: RemoteTokenizer):
        self.local = local
        self.remote = remote
        self.method = "local" if local else "remote"

    def count(self, messages: list[dict[str, str]]) -> int:
        if self.local is not None:
            try:
                return self.local.count(messages)
            except Exception:
                # If the local tokenizer cannot count this model's exact chat
                # template, fall back to the live server tokenizer/usage path.
                self.local = None
                self.method = "remote_fallback_after_local_error"
        return self.remote.count(messages)

    def estimate(self, messages: list[dict[str, str]]) -> int:
        if self.local is not None:
            try:
                return self.local.count(messages)
            except Exception:
                self.local = None
                self.method = "remote_fallback_after_local_error"
        return self.remote.estimate(messages)

    def info(self) -> dict[str, Any]:
        if self.local is not None:
            return self.local.info()
        remote_info = self.remote.info()
        if self.method == "remote_fallback_after_local_error":
            remote_info["local_fallback"] = "local tokenizer failed; remote exact counter used"
        return remote_info


def make_token_counter(
    base_url: str,
    model: str,
    timeout: float,
    api_key: str | None,
    tokenizer_path: str | None,
) -> TokenCounter:
    candidates: list[str] = []
    if tokenizer_path:
        candidates.append(tokenizer_path)
    env_path = os.environ.get("TOKENIZER_PATH")
    if env_path and env_path not in candidates:
        candidates.append(env_path)
    if model not in candidates:
        candidates.append(model)
    local: LocalChatTokenizer | None = None
    failures: list[str] = []
    for candidate in candidates:
        try:
            local = LocalChatTokenizer(model, candidate)
            break
        except Exception as exc:  # local-only probe; remote fallback is mandatory
            failures.append(f"{candidate}: {type(exc).__name__}: {exc}")
    remote = RemoteTokenizer(base_url, model, timeout, api_key)
    counter = TokenCounter(local, remote)
    counter.local_probe_failures = failures  # type: ignore[attr-defined]
    return counter


def _messages(user_text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": user_text}]


def exact_prompt_text(
    text_factory: Callable[[int], str],
    counter: TokenCounter,
    target_tokens: int,
    *,
    initial_chars: int | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Find text whose full chat-template count exactly matches target_tokens."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    fixed_count = counter.count(_messages(text_factory(0)))
    if fixed_count > target_tokens:
        raise ValueError(f"fixed prompt alone has {fixed_count} tokens, target is {target_tokens}")
    start_high = initial_chars if initial_chars is not None else max(32, target_tokens * 2)
    limit = max_chars if max_chars is not None else max(target_tokens * 12, start_high)
    offset_error = 0
    closest: tuple[int, int] = (0, fixed_count)
    for _attempt in range(8):
        search_target = target_tokens - offset_error
        low = 0
        high = min(start_high, limit)
        high_count = counter.estimate(_messages(text_factory(high)))
        while high_count < search_target and high < limit:
            high = min(max(high + 1, high * 2), limit)
            high_count = counter.estimate(_messages(text_factory(high)))
        if high_count < search_target:
            raise ValueError(
                f"could not reach {target_tokens} tokens before {limit} padding characters "
                f"(estimated count {high_count})"
            )

        best_estimate: tuple[int, int] = (high, high_count)
        while low <= high:
            middle = (low + high) // 2
            estimated = counter.estimate(_messages(text_factory(middle)))
            if abs(estimated - search_target) < abs(best_estimate[1] - search_target):
                best_estimate = (middle, estimated)
            if estimated == search_target:
                best_estimate = (middle, estimated)
                break
            if estimated < search_target:
                low = middle + 1
            else:
                high = middle - 1

        # Check the search result with the exact local chat template or actual
        # server usage.  Remote raw-prompt tokenization is only a fast search
        # aid and can never certify the final count.
        candidate_chars = best_estimate[0]
        exact = counter.count(_messages(text_factory(candidate_chars)))
        if abs(exact - target_tokens) < abs(closest[1] - target_tokens):
            closest = (candidate_chars, exact)
        if exact == target_tokens:
            return {
                "text": text_factory(candidate_chars),
                "chars": candidate_chars,
                "tokens": exact,
                "exact": True,
                "search_estimated_tokens": best_estimate[1],
            }
        exact_estimate = counter.estimate(_messages(text_factory(candidate_chars)))
        offset_error = exact - exact_estimate

        # Boundary tokenization can make the estimate-to-chat offset vary by a
        # token at the join.  Try a small deterministic neighborhood before
        # recalibrating the estimate offset.
        left = max(0, candidate_chars - 128)
        right = min(limit, candidate_chars + 128)
        for chars in range(left, right + 1):
            estimated = counter.estimate(_messages(text_factory(chars)))
            if estimated != target_tokens - offset_error:
                continue
            exact = counter.count(_messages(text_factory(chars)))
            if abs(exact - target_tokens) < abs(closest[1] - target_tokens):
                closest = (chars, exact)
            if exact == target_tokens:
                return {
                    "text": text_factory(chars),
                    "chars": chars,
                    "tokens": exact,
                    "exact": True,
                    "search_estimated_tokens": estimated,
                }
            offset_error = exact - counter.estimate(_messages(text_factory(chars)))

    raise ValueError(
        f"tokenizer could not construct an exact {target_tokens}-token prompt; closest was "
        f"{closest[1]} tokens at {closest[0]} padding characters"
    )


def prepare_dihuo_prompts(
    corpus: str,
    suffixes: list[dict[str, str]],
    counter: TokenCounter,
    target_tokens: int,
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    corpus_len = len(corpus)
    estimated_slice_chars = int(target_tokens * 1.3)
    if corpus_len <= estimated_slice_chars:
        slice_stride = max(1, corpus_len // 4)
    else:
        slice_stride = corpus_len // 8
        if slice_stride < 64:
            slice_stride = max(64, estimated_slice_chars // 4)
        slice_stride = min(slice_stride, corpus_len - estimated_slice_chars)
        slice_stride = max(1, slice_stride)
    offset_span = max(1, corpus_len - min(corpus_len // 4, 4000))
    for index in range(8):
        offset = (index * slice_stride) % offset_span
        suffix = suffixes[index % len(suffixes)]
        calibrated = None
        last_error: Exception | None = None
        used_offset = offset
        for retry in range(8):
            used_offset = (offset + retry * 37) % offset_span

            def factory(chars: int, *, i: int = index, off: int = used_offset, s: str = suffix["text"]) -> str:
                body = repeated_corpus_slice(corpus, off, chars)
                return dihuo_prompt(i, body, s)

            try:
                calibrated = exact_prompt_text(factory, counter, target_tokens)
                break
            except ValueError as exc:
                last_error = exc
        if calibrated is None:
            raise ValueError(f"could not calibrate Dihuo sample {index + 1}: {last_error}")
        prompts.append(
            {
                "sample_id": f"dihuo-{index + 1:02d}",
                "suffix_name": suffix["name"],
                "corpus_offset_chars": used_offset,
                "corpus_slice_stride_chars": slice_stride,
                "corpus_slice_chars": calibrated["chars"],
                "tokenizer_count": calibrated["tokens"],
                "input_tokens_exact": calibrated["exact"],
                "messages": _messages(calibrated["text"]),
                "prompt_text": calibrated["text"],
            }
        )
    return prompts


def prepare_code_prompts(counter: TokenCounter, target_tokens: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for index, task in enumerate(CODE_TASKS):

        calibrated = None
        last_error: Exception | None = None
        padding_seed = index
        for retry in range(8):
            padding_seed = index + retry * 3

            def factory(chars: int, *, i: int = padding_seed, task_id: str = task.task_id) -> str:
                padding = code_padding_prefix(chars, i)
                return build_code_prompt(task_id, padding)

            try:
                calibrated = exact_prompt_text(factory, counter, target_tokens)
                break
            except ValueError as exc:
                last_error = exc
        if calibrated is None:
            raise ValueError(f"could not calibrate code task {task.task_id}: {last_error}")
        cases = fixed_unit_cases(task.task_id)
        prompts.append(
            {
                "sample_id": f"code-{index + 1:02d}-{task.task_id}",
                "task_id": task.task_id,
                "function_name": task.function_name,
                "padding_chars": calibrated["chars"],
                "padding_seed": padding_seed,
                "tokenizer_count": calibrated["tokens"],
                "input_tokens_exact": calibrated["exact"],
                "unit_cases": cases,
                "unit_cases_sha256": canonical_json_sha256(cases),
                "messages": _messages(calibrated["text"]),
                "prompt_text": calibrated["text"],
            }
        )
    return prompts


def _event_content(delta: Any) -> str:
    if isinstance(delta, str):
        return delta
    if isinstance(delta, list):
        parts = []
        for item in delta:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def stream_chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    api_key: str | None,
    seed: int,
    request_id: str,
    ignore_eos: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": seed,
        "request_id": request_id,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    request = urllib.request.Request(
        _url(base_url, "/v1/chat/completions"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={**_auth_headers(api_key), "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    first_content_at: float | None = None
    last_content_at: float | None = None
    chunks = 0
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    sse_error_event: Any = None
    http_status: int | None = None
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            http_status = getattr(response, "status", 200)
            while True:
                raw_line = response.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                event_data = line[5:].strip()
                if event_data == "[DONE]":
                    break
                try:
                    event = json.loads(event_data)
                except json.JSONDecodeError:
                    continue
                chunks += 1
                if "error" in event:
                    sse_error_event = event.get("error")
                    error = "server SSE error event: " + json.dumps(sse_error_event, ensure_ascii=False)[:2000]
                    break
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if isinstance(choice.get("finish_reason"), str):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta")
                content = _event_content(delta.get("content")) if isinstance(delta, dict) else ""
                if content:
                    now = time.monotonic()
                    if first_content_at is None:
                        first_content_at = now
                    last_content_at = now
                    text_parts.append(content)
    except urllib.error.HTTPError as exc:
        error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:2000]}"
        http_status = exc.code
    except Exception as exc:  # noqa: BLE001 - evidence keeps the exact client failure
        error = f"{type(exc).__name__}: {exc}"

    ended = time.monotonic()
    output_text = "".join(text_parts)
    if not output_text and usage is None and finish_reason is None and error is None:
        error = "SSE stream ended without content, usage, finish_reason, or server error event"
    raw_usage = usage or {}
    prompt_tokens = raw_usage.get("prompt_tokens")
    completion_tokens = raw_usage.get("completion_tokens")
    total_tokens = raw_usage.get("total_tokens")
    usage_tokens = {
        "prompt_tokens": prompt_tokens if isinstance(prompt_tokens, int) else None,
        "completion_tokens": completion_tokens if isinstance(completion_tokens, int) else None,
        "total_tokens": total_tokens if isinstance(total_tokens, int) else None,
        "source": "stream_options.include_usage final SSE usage object" if usage is not None else None,
    }
    ttft_s = first_content_at - started if first_content_at is not None else None
    decode_s = last_content_at - first_content_at if last_content_at is not None and first_content_at is not None else None
    completion_for_rate = usage_tokens["completion_tokens"]
    rate_tokens = max(0, completion_for_rate - 1) if isinstance(completion_for_rate, int) else None
    decode_tok_s = (
        rate_tokens / decode_s
        if isinstance(rate_tokens, int) and decode_s is not None and decode_s > 0
        else None
    )
    tpot_s = (
        decode_s / rate_tokens
        if isinstance(rate_tokens, int) and rate_tokens > 0 and decode_s is not None
        else None
    )
    return {
        "started_at_utc": started_at,
        "http_status": http_status,
        "error": error,
        "sse_error_event": sse_error_event,
        "output_text": output_text,
        "output_sha256": sha256_text(output_text),
        "output_utf8_bytes": len(output_text.encode("utf-8")),
        "usage_tokens": usage_tokens,
        "finish_reason": finish_reason,
        "sse_json_events": chunks,
        "request_wall_s": ended - started,
        "ttft_s": ttft_s,
        "decode_s": decode_s,
        "decode_rate_completion_tokens": rate_tokens,
        "decode_tok_s": decode_tok_s,
        "tpot_s": tpot_s,
    }


def _metrics_snapshot(base_url: str, timeout: float, api_key: str | None) -> dict[str, Any]:
    endpoint = _url(base_url, "/delivery/metrics")
    try:
        response = _get_json(endpoint, timeout, api_key)
        return {"ok": True, "endpoint": endpoint, "received_at_utc": datetime.now(timezone.utc).isoformat(), "data": response}
    except urllib.error.HTTPError as exc:  # preserve server-side fence failures in the receipt
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        return {
            "ok": False,
            "endpoint": endpoint,
            "received_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"HTTP {exc.code}: {body}",
            "data": None,
        }
    except Exception as exc:  # no metrics is represented as null downstream, never zero
        return {
            "ok": False,
            "endpoint": endpoint,
            "received_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "data": None,
        }


def _arm_delivery_window(
    base_url: str,
    max_tokens: int,
    tag: str,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    endpoint = _url(base_url, "/delivery/arm")
    try:
        response = _post_json(
            endpoint,
            {"max_tokens": max_tokens, "tag": tag},
            timeout,
            api_key,
        )
        return {
            "ok": True,
            "endpoint": endpoint,
            "sent": {"max_tokens": max_tokens, "tag": tag},
            "data": response,
            "response_sha256": canonical_json_sha256(response),
        }
    except Exception as exc:  # keep a failed arm as evidence, then counters stay null
        return {
            "ok": False,
            "endpoint": endpoint,
            "sent": {"max_tokens": max_tokens, "tag": tag},
            "error": f"{type(exc).__name__}: {exc}",
            "data": None,
        }


def _valid_metric_data(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(snapshot, dict) or not snapshot.get("ok"):
        return None
    data = snapshot.get("data")
    if not isinstance(data, dict):
        return None
    return data


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _milliseconds_per_step(seconds: Any, steps: Any) -> float | None:
    if not _numeric(seconds) or not _numeric(steps) or steps <= 0:
        return None
    return float(seconds) * 1000.0 / float(steps)


def _http_rate_from_A(A: Any, client_decode_ms_per_step: Any) -> float | None:
    if not _numeric(A) or not _numeric(client_decode_ms_per_step) or client_decode_ms_per_step <= 0:
        return None
    return float(A) * 1000.0 / float(client_decode_ms_per_step)


def _rank_counters(data: dict[str, Any], ranks_field: str = "ranks") -> dict[str, dict[str, Any]] | None:
    ranks = data.get(ranks_field)
    if not isinstance(ranks, list) or len(ranks) != RANK_COUNT:
        return None
    mapping: dict[str, dict[str, Any]] = {}
    for rank in ranks:
        if not isinstance(rank, dict) or "rank" not in rank or not isinstance(rank.get("cumulative"), dict):
            return None
        key = str(rank["rank"])
        if key in mapping:
            return None
        mapping[key] = rank["cumulative"]
    return mapping if len(mapping) == RANK_COUNT else None


def _rank_rows(data: dict[str, Any], ranks_field: str = "ranks") -> dict[str, dict[str, Any]] | None:
    ranks = data.get(ranks_field)
    if not isinstance(ranks, list) or len(ranks) != RANK_COUNT:
        return None
    mapping: dict[str, dict[str, Any]] = {}
    for rank in ranks:
        if not isinstance(rank, dict) or "rank" not in rank or not isinstance(rank.get("requests"), list):
            return None
        key = str(rank["rank"])
        if key in mapping:
            return None
        mapping[key] = rank
    return mapping if len(mapping) == RANK_COUNT else None


def counter_metrics_delta(
    before_snapshot: dict[str, Any] | None,
    after_snapshot: dict[str, Any] | None,
    expected_model: str,
    target_prompt_tokens: int,
    usage_tokens: dict[str, Any] | None,
    expected_service_instance_id: str,
    expected_request_id: str,
    expected_max_tokens: int,
    arm_receipt: dict[str, Any] | None,
    expected_form: str | None = None,
    expected_prefill_service_instance_id: str | None = None,
    expected_decode_service_instance_id: str | None = None,
) -> dict[str, Any]:
    """Validate one armed request fence and its actual per-rank metric delta."""
    empty = {
        "available": False,
        "steps": None,
        "A": None,
        "ms_step": None,
        "server_fenced_ms_per_step": None,
        "server_fenced_ms_per_step_scope": SERVER_FENCED_MS_PER_STEP_SCOPE,
        "ms_step_legacy_alias_of": "server_fenced_ms_per_step",
        "decode_tokens": None,
        "decode_tokens_raw": None,
        "final_decode_tokens": None,
        "decode_seconds_s": None,
        "server_prefill_s": None,
        "server_prefill_components": None,
        "server_prefill_source": None,
        "server_prefill_scope": None,
        "counter_deltas_by_rank": None,
        "request_rows_by_rank": None,
        "request_id": None,
        "request_tag": None,
        "window_fingerprint": None,
        "async_mode": None,
        "reason": "missing_or_unreadable_metrics_snapshot",
    }
    is_ced = expected_form == CED_FORM
    if is_ced and (
        not expected_prefill_service_instance_id
        or not expected_decode_service_instance_id
    ):
        return {**empty, "reason": "ced_prefill_decode_identity_pair_missing"}
    if not isinstance(arm_receipt, dict) or not arm_receipt.get("ok") or not isinstance(arm_receipt.get("data"), dict):
        return {**empty, "reason": "measurement_window_arm_failed"}
    arm = arm_receipt["data"]
    if (
        arm.get("schema") != "delivery_counter_arm_v1"
        or arm.get("service_instance_id") != expected_service_instance_id
        or arm.get("tag") != expected_request_id
        or arm.get("max_tokens") != expected_max_tokens
    ):
        return {**empty, "reason": "arm_receipt_identity_mismatch"}
    arm_ranks = arm.get("ranks")
    if not isinstance(arm_ranks, list) or len(arm_ranks) != RANK_COUNT:
        return {**empty, "reason": "arm_receipt_missing_eight_rank_acknowledgements"}
    if {row.get("rank") for row in arm_ranks if isinstance(row, dict)} != set(range(RANK_COUNT)):
        return {**empty, "reason": "arm_receipt_rank_set_invalid"}
    if any(
        not isinstance(row, dict)
        or row.get("tag") != expected_request_id
        or row.get("max_tokens") != expected_max_tokens
        or row.get("armed") is not True
        or not isinstance(row.get("async_mode"), bool)
        for row in arm_ranks
    ):
        return {**empty, "reason": "arm_receipt_rank_ack_mismatch"}
    arm_modes = {row["async_mode"] for row in arm_ranks}
    if len(arm_modes) != 1:
        return {**empty, "reason": "async_mode_disagrees_across_arm_ranks"}
    arm_async_mode = next(iter(arm_modes))
    if is_ced:
        prefill_arm = arm.get("prefill_ack")
        if (
            not isinstance(prefill_arm, dict)
            or prefill_arm.get("schema") != "delivery_counter_arm_v1"
            or prefill_arm.get("service_instance_id") != expected_prefill_service_instance_id
            or prefill_arm.get("tag") != expected_request_id
            or prefill_arm.get("max_tokens") != expected_max_tokens
        ):
            return {**empty, "reason": "ced_prefill_arm_identity_mismatch"}
        prefill_arm_ranks = prefill_arm.get("ranks")
        if not isinstance(prefill_arm_ranks, list) or len(prefill_arm_ranks) != RANK_COUNT:
            return {**empty, "reason": "ced_prefill_arm_missing_eight_rank_acknowledgements"}
        if {row.get("rank") for row in prefill_arm_ranks if isinstance(row, dict)} != set(range(RANK_COUNT)):
            return {**empty, "reason": "ced_prefill_arm_rank_set_invalid"}
        if any(
            not isinstance(row, dict)
            or row.get("tag") != expected_request_id
            or row.get("max_tokens") != expected_max_tokens
            or row.get("armed") is not True
            or not isinstance(row.get("async_mode"), bool)
            or row.get("async_mode") != arm_async_mode
            for row in prefill_arm_ranks
        ):
            return {**empty, "reason": "ced_prefill_arm_rank_ack_mismatch"}

    before = _valid_metric_data(before_snapshot)
    after = _valid_metric_data(after_snapshot)
    if before is None or after is None:
        return empty
    for label, data in (("before", before), ("after", after)):
        if data.get("schema") != COUNTER_SCHEMA:
            return {**empty, "reason": f"{label}_schema_untrusted"}
        if data.get("trusted") is not True or data.get("counter_scope") != COUNTER_SCOPE:
            return {**empty, "reason": f"{label}_trust_or_scope_unverified"}
        if data.get("served_model") != expected_model:
            return {**empty, "reason": f"{label}_served_model_mismatch"}
        if expected_form is not None and data.get("form") != expected_form:
            return {**empty, "reason": f"{label}_form_mismatch"}
    if before.get("service_instance_id") != after.get("service_instance_id"):
        return {**empty, "reason": "service_instance_changed"}
    if before.get("service_instance_id") != expected_service_instance_id:
        return {**empty, "reason": "service_instance_id_mismatch"}
    if before.get("form") != after.get("form") or before.get("served_model") != after.get("served_model"):
        return {**empty, "reason": "service_identity_changed"}

    before_ranks = _rank_counters(before)
    after_ranks = _rank_counters(after)
    before_rank_rows = _rank_rows(before)
    after_rank_rows = _rank_rows(after)
    if (
        before_ranks is None
        or after_ranks is None
        or before_rank_rows is None
        or after_rank_rows is None
        or set(before_ranks) != set(after_ranks)
        or set(before_rank_rows) != set(after_rank_rows)
    ):
        return {**empty, "reason": "rank_set_incomplete_or_changed"}

    if is_ced:
        if any(
            data.get("prefill_reporting") != CED_PREFILL_REPORTING
            or (
                data.get("prefill_service_instance_id") is not None
                and data.get("prefill_service_instance_id") != expected_prefill_service_instance_id
            )
            or (
                data.get("decode_service_instance_id") is not None
                and data.get("decode_service_instance_id") != expected_decode_service_instance_id
            )
            for data in (before, after)
        ):
            return {**empty, "reason": "ced_prefill_reporting_or_pair_binding_mismatch"}
        before_prefill_counters = _rank_counters(before, "prefill_ranks")
        after_prefill_counters = _rank_counters(after, "prefill_ranks")
        before_prefill_rows = _rank_rows(before, "prefill_ranks")
        after_prefill_rows = _rank_rows(after, "prefill_ranks")
        if (
            before_prefill_counters is None
            or after_prefill_counters is None
            or before_prefill_rows is None
            or after_prefill_rows is None
            or set(before_prefill_counters) != set(before_ranks)
            or set(after_prefill_counters) != set(after_ranks)
            or set(before_prefill_rows) != set(before_rank_rows)
            or set(after_prefill_rows) != set(after_rank_rows)
            or any(row.get("role") != "prefill" for row in before_prefill_rows.values())
            or any(row.get("role") != "prefill" for row in after_prefill_rows.values())
            or any(row.get("role") == "prefill" for row in before_rank_rows.values())
            or any(row.get("role") == "prefill" for row in after_rank_rows.values())
        ):
            return {**empty, "reason": "ced_prefill_ranks_missing_or_not_independent_tp8"}
    else:
        before_prefill_counters = after_prefill_counters = None
        before_prefill_rows = after_prefill_rows = None

    deltas: dict[str, dict[str, float | int | None]] = {}
    request_rows: dict[str, dict[str, Any]] = {}
    prefill_request_rows: dict[str, dict[str, Any]] = {}
    ced_prefill_deltas: dict[str, float] = {}
    ced_decode_replay_deltas: dict[str, float] = {}
    keys = ("decode_rounds", "decode_tokens", "decode_seconds", "prefill_seconds")
    for rank_id in sorted(before_ranks, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)):
        row: dict[str, float | int | None] = {}
        for key in keys:
            left = before_ranks[rank_id].get(key)
            right = after_ranks[rank_id].get(key)
            if _numeric(left) and _numeric(right) and float(right) >= float(left):
                row[key] = right - left
            else:
                row[key] = None
        deltas[rank_id] = row

        before_rank = before_rank_rows[rank_id]
        after_rank = after_rank_rows[rank_id]
        if (
            before_rank.get("pending") is not False
            or after_rank.get("pending") is not False
            or before_rank.get("errors") != []
            or after_rank.get("errors") != []
        ):
            return {
                **empty,
                "reason": "counter_pending_or_errors_nonempty",
                "counter_deltas_by_rank": deltas,
            }
        if (
            before_rank.get("counter_scope") != COUNTER_SCOPE
            or after_rank.get("counter_scope") != COUNTER_SCOPE
            or not str(after_rank.get("decode_tokens_scope", "")).startswith("HTTP final usage required for async")
            and after_rank.get("decode_tokens_scope") != "raw worker tokens before terminal scheduler trimming"
        ):
            return {
                **empty,
                "reason": "rank_counter_scope_unverified",
                "counter_deltas_by_rank": deltas,
            }
        before_requests = before_rank["requests"]
        after_requests = after_rank["requests"]
        if any(not isinstance(item, dict) or not isinstance(item.get("request_id"), str) for item in before_requests + after_requests):
            return {
                **empty,
                "reason": "request_rows_missing_ids",
                "counter_deltas_by_rank": deltas,
            }
        old_ids = {item["request_id"] for item in before_requests}
        new_rows = [item for item in after_requests if item["request_id"] not in old_ids]
        if len(new_rows) != 1:
            return {
                **empty,
                "reason": "expected_one_new_serial_request_row_per_rank",
                "counter_deltas_by_rank": deltas,
            }
        request_row = new_rows[0]
        if request_row.get("tag") != expected_request_id:
            return {
                **empty,
                "reason": "request_row_tag_mismatch",
                "counter_deltas_by_rank": deltas,
            }
        last_window = after_rank.get("last_window")
        if (
            not isinstance(last_window, dict)
            or last_window.get("tag") != expected_request_id
            or last_window.get("window_fingerprint") != request_row.get("window_fingerprint")
        ):
            return {
                **empty,
                "reason": "terminal_window_tag_or_fingerprint_mismatch",
                "counter_deltas_by_rank": deltas,
            }
        request_rows[rank_id] = request_row

        if is_ced:
            p_before = before_prefill_rows[rank_id]
            p_after = after_prefill_rows[rank_id]
            if (
                p_before.get("pending") is not False
                or p_after.get("pending") is not False
                or p_before.get("errors") != []
                or p_after.get("errors") != []
            ):
                return {**empty, "reason": "ced_prefill_counter_pending_or_errors_nonempty", "counter_deltas_by_rank": deltas}
            p_before_requests = p_before["requests"]
            p_after_requests = p_after["requests"]
            if any(
                not isinstance(item, dict) or not isinstance(item.get("request_id"), str) or not item["request_id"]
                for item in p_before_requests + p_after_requests
            ):
                return {**empty, "reason": "ced_prefill_request_rows_missing_ids", "counter_deltas_by_rank": deltas}
            p_before_ids = [item["request_id"] for item in p_before_requests]
            p_after_ids = [item["request_id"] for item in p_after_requests]
            if len(set(p_before_ids)) != len(p_before_ids) or len(set(p_after_ids)) != len(p_after_ids):
                return {**empty, "reason": "ced_prefill_request_ids_are_not_unique", "counter_deltas_by_rank": deltas}
            p_old_ids = set(p_before_ids)
            new_prefill_rows = [item for item in p_after_requests if item["request_id"] not in p_old_ids]
            if len(new_prefill_rows) != 1:
                return {**empty, "reason": "ced_expected_one_new_prefill_row_per_rank", "counter_deltas_by_rank": deltas}
            p_request = new_prefill_rows[0]
            if (
                p_request.get("tag") != expected_request_id
                or p_request.get("prompt_tokens") != target_prompt_tokens
                or p_request.get("profile_excluded") is not True
                or p_request.get("terminal_fence_observed") is not True
                or not _numeric(p_request.get("start_ns"))
                or not _numeric(p_request.get("end_ns"))
                or p_request["end_ns"] < p_request["start_ns"]
                or not _numeric(p_request.get("calls"))
                or p_request["calls"] <= 0
                or not isinstance(p_request.get("prefill_wall_scope"), str)
                or not p_request["prefill_wall_scope"].startswith("P first scheduled prompt -> terminal P CPU bookkeeping")
            ):
                return {**empty, "reason": "ced_prefill_request_row_tag_or_timing_invalid", "counter_deltas_by_rank": deltas}

            p_before_seconds = before_prefill_counters[rank_id].get("prefill_seconds")
            p_after_seconds = after_prefill_counters[rank_id].get("prefill_seconds")
            if not _numeric(p_before_seconds) or not _numeric(p_after_seconds) or p_after_seconds < p_before_seconds:
                return {**empty, "reason": "ced_prefill_seconds_missing_or_decreased", "counter_deltas_by_rank": deltas}
            p_delta = float(p_after_seconds) - float(p_before_seconds)
            p_request_seconds = (p_request["end_ns"] - p_request["start_ns"]) / 1e9
            if abs(p_delta - p_request_seconds) > 0.005:
                return {**empty, "reason": "ced_prefill_counter_disagrees_with_request_timestamps", "counter_deltas_by_rank": deltas}

            combined_delta = row.get("prefill_seconds")
            if not _numeric(combined_delta):
                return {**empty, "reason": "ced_combined_prefill_delta_missing", "counter_deltas_by_rank": deltas}
            d_delta = float(combined_delta) - p_delta
            if d_delta < -1e-6:
                return {**empty, "reason": "ced_decode_prefill_delta_is_negative", "counter_deltas_by_rank": deltas}
            ced_prefill_deltas[rank_id] = p_delta
            ced_decode_replay_deltas[rank_id] = max(0.0, d_delta)
            prefill_request_rows[rank_id] = p_request

    if is_ced:
        prefill_engine_ids = {row.get("request_id") for row in prefill_request_rows.values()}
        prefill_tags = {row.get("tag") for row in prefill_request_rows.values()}
        if len(prefill_engine_ids) != 1 or prefill_tags != {expected_request_id}:
            return {
                **empty,
                "reason": "ced_prefill_request_id_or_tag_disagrees_across_ranks",
                "counter_deltas_by_rank": deltas,
                "request_rows_by_rank": request_rows,
                "prefill_request_rows_by_rank": prefill_request_rows,
            }
        if any(
            not str(row.get("prefill_wall_scope", "")).startswith(
                "P scheduled prefill + D bounded replay/tail CPU bookkeeping"
            )
            for row in request_rows.values()
        ):
            return {
                **empty,
                "reason": "ced_decode_prefill_scope_does_not_include_p_and_d",
                "counter_deltas_by_rank": deltas,
                "request_rows_by_rank": request_rows,
                "prefill_request_rows_by_rank": prefill_request_rows,
            }

    rounds = [row["decode_rounds"] for row in deltas.values()]
    tokens = [row["decode_tokens"] for row in deltas.values()]
    if any(value is None for value in rounds) or len(set(rounds)) != 1:
        return {**empty, "reason": "decode_rounds_missing_or_rank_counts_disagree", "counter_deltas_by_rank": deltas}
    steps_raw = rounds[0]
    if not _numeric(steps_raw) or steps_raw <= 0:
        return {**empty, "reason": "no_decode_round_counter_increment", "counter_deltas_by_rank": deltas}
    seconds_values = [row["decode_seconds"] for row in deltas.values()]
    if any(value is None for value in seconds_values):
        return {**empty, "reason": "decode_seconds_missing", "counter_deltas_by_rank": deltas}
    seconds = max(float(value) for value in seconds_values if value is not None)
    prefill_values = [row["prefill_seconds"] for row in deltas.values()]
    prefill_s = (
        max(float(value) for value in prefill_values if value is not None)
        if all(value is not None for value in prefill_values)
        else None
    )
    prefill_components = None
    prefill_source = (
        "GET /delivery/metrics delta: ranks[].cumulative.prefill_seconds; max rank delta"
        if prefill_s is not None
        else None
    )
    if is_ced:
        p_stage_s = max(ced_prefill_deltas.values())
        d_stage_s = max(ced_decode_replay_deltas.values())
        prefill_s = p_stage_s + d_stage_s
        prefill_source = (
            "per-request max-rank P prefill_ranks prefill_seconds delta + "
            "per-request max-rank D replay/tail delta (combined ranks prefill_seconds delta minus P delta)"
        )
        prefill_components = {
            "prefill_encoder_s": p_stage_s,
            "decode_replay_tail_s": d_stage_s,
            "combined_service_prefill_s": prefill_s,
            "aggregation": "max per-rank P delta + max per-rank D-only delta; computed inside this request window",
            "prefill_encoder_delta_by_rank": ced_prefill_deltas,
            "decode_replay_tail_delta_by_rank": ced_decode_replay_deltas,
            "combined_delta_by_rank": {
                rank_id: ced_prefill_deltas[rank_id] + ced_decode_replay_deltas[rank_id]
                for rank_id in ced_prefill_deltas
            },
            "prefill_reporting": CED_PREFILL_REPORTING,
        }
    request_ids = {row.get("request_id") for row in request_rows.values()}
    request_tags = {row.get("tag") for row in request_rows.values()}
    if len(request_ids) != 1 or request_tags != {expected_request_id}:
        return {
            **empty,
            "reason": "request_id_or_tag_disagrees_across_ranks",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
        }
    required_request_fields = (
        "prompt_tokens",
        "seed_tokens",
        "decode_rounds",
    )
    if any(any(not _numeric(row.get(key)) for key in required_request_fields) for row in request_rows.values()):
        return {
            **empty,
            "reason": "request_row_missing_token_or_round_count",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    first_row = next(iter(request_rows.values()))
    if any(
        not isinstance(row.get("async_mode"), bool)
        or row.get("terminal_fence_observed") is not True
        or row.get("profile_excluded") is not True
        or not isinstance(row.get("terminal_tp_join_observed"), bool)
        or not _numeric(row.get("decode_start_ns"))
        or not _numeric(row.get("terminal_sample_end_ns"))
        or not _numeric(row.get("fenced_end_ns"))
        or not isinstance(row.get("window_fingerprint"), str)
        for row in request_rows.values()
    ):
        return {
            **empty,
            "reason": "request_row_missing_mode_fingerprint_profile_or_terminal_fence",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    request_fields_agree = all(
        all(row.get(key) == first_row.get(key) for key in required_request_fields)
        for row in request_rows.values()
    )
    if not request_fields_agree:
        return {
            **empty,
            "reason": "request_token_or_round_counts_disagree_across_ranks",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    request_rounds = first_row["decode_rounds"]
    async_mode = first_row["async_mode"]
    if not isinstance(async_mode, bool) or async_mode != arm_async_mode:
        return {
            **empty,
            "reason": "async_mode_disagrees_between_arm_and_request_rows",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    if first_row["terminal_fence_observed"] is not True:
        return {
            **empty,
            "reason": "terminal_completion_fence_not_observed",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    if async_mode and first_row["terminal_tp_join_observed"] is not True:
        return {
            **empty,
            "reason": "async_terminal_tp_join_not_observed",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    if async_mode and any(row["terminal_tp_join_observed"] is not True for row in request_rows.values()):
        return {
            **empty,
            "reason": "async_terminal_tp_join_disagrees_across_ranks",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    if any(
        row["fenced_end_ns"] < row["terminal_sample_end_ns"]
        or row["terminal_sample_end_ns"] < row["decode_start_ns"]
        for row in request_rows.values()
    ):
        return {
            **empty,
            "reason": "request_timing_boundaries_are_not_monotonic",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    if any(
        abs(
            deltas[rank_id]["decode_seconds"]
            - (row["fenced_end_ns"] - row["decode_start_ns"]) / 1e9
        ) > 0.005
        for rank_id, row in request_rows.items()
    ):
        return {
            **empty,
            "reason": "decode_seconds_delta_disagrees_with_request_fence_timestamps",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    for scope_field in ("prefill_wall_scope", "decode_wall_scope"):
        scope_values = {row.get(scope_field) for row in request_rows.values()}
        if len(scope_values) != 1 or not isinstance(first_row.get(scope_field), str):
            return {
                **empty,
                "reason": f"{scope_field}_missing_or_disagrees_across_ranks",
                "counter_deltas_by_rank": deltas,
                "request_rows_by_rank": request_rows,
                "request_id": next(iter(request_ids)),
            }
    window_fingerprints = {row.get("window_fingerprint") for row in request_rows.values()}
    if (
        len(window_fingerprints) != 1
        or not isinstance(first_row.get("window_fingerprint"), str)
        or len(first_row["window_fingerprint"]) != 64
        or any(char not in "0123456789abcdef" for char in first_row["window_fingerprint"])
    ):
        return {
            **empty,
            "reason": "window_fingerprint_disagrees_across_ranks",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    request_raw_decode_tokens = first_row.get("raw_decode_tokens")
    last_round_tokens = first_row.get("last_round_tokens")
    if any(
        deltas[rank_id]["decode_rounds"] != row["decode_rounds"]
        for rank_id, row in request_rows.items()
    ):
        return {
            **empty,
            "reason": "request_row_and_cumulative_delta_disagree",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    if async_mode:
        if any(
            row.get("raw_decode_tokens") is not None or row.get("last_round_tokens") is not None
            for row in request_rows.values()
        ) or any(
            before_ranks[rank_id].get("decode_tokens") is not None
            or after_ranks[rank_id].get("decode_tokens") is not None
            for rank_id in request_rows
        ):
            return {
                **empty,
                "reason": "async_raw_token_counter_must_remain_null",
                "counter_deltas_by_rank": deltas,
                "request_rows_by_rank": request_rows,
                "request_id": next(iter(request_ids)),
            }
    else:
        if any(
            not _numeric(row.get("raw_decode_tokens")) or not _numeric(row.get("last_round_tokens"))
            for row in request_rows.values()
        ):
            return {
                **empty,
                "reason": "sync_raw_token_counter_missing",
                "counter_deltas_by_rank": deltas,
                "request_rows_by_rank": request_rows,
                "request_id": next(iter(request_ids)),
            }
        if any(
            deltas[rank_id]["decode_tokens"] != row["raw_decode_tokens"]
            for rank_id, row in request_rows.items()
        ) or any(value is None for value in tokens) or len(set(tokens)) != 1:
            return {
                **empty,
                "reason": "sync_raw_decode_counts_missing_or_rank_disagreement",
                "counter_deltas_by_rank": deltas,
                "request_rows_by_rank": request_rows,
                "request_id": next(iter(request_ids)),
            }
        if len({row["last_round_tokens"] for row in request_rows.values()}) != 1:
            return {
                **empty,
                "reason": "sync_last_round_token_counts_disagree_across_ranks",
                "counter_deltas_by_rank": deltas,
                "request_rows_by_rank": request_rows,
                "request_id": next(iter(request_ids)),
            }
    if first_row["prompt_tokens"] != target_prompt_tokens:
        return {
            **empty,
            "reason": "worker_prompt_tokens_mismatch",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    if int(first_row["seed_tokens"]) != 1:
        return {
            **empty,
            "reason": "unexpected_seed_token_count",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    if not isinstance(usage_tokens, dict) or not _usage_is_real(usage_tokens):
        return {
            **empty,
            "reason": "actual_stream_usage_missing_or_inconsistent",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    if usage_tokens.get("prompt_tokens") != target_prompt_tokens or first_row["prompt_tokens"] != usage_tokens.get("prompt_tokens"):
        return {
            **empty,
            "reason": "server_usage_and_worker_prompt_tokens_disagree",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    final_decode_tokens = usage_tokens["completion_tokens"] - 1
    raw_trimmed = None
    if not async_mode:
        raw_trimmed = request_raw_decode_tokens - final_decode_tokens
        if (
            request_raw_decode_tokens < final_decode_tokens
            or raw_trimmed < 0
            or raw_trimmed >= last_round_tokens
        ):
            return {
                **empty,
                "reason": "http_completion_usage_not_explainable_by_terminal_round_trimming",
                "counter_deltas_by_rank": deltas,
                "request_rows_by_rank": request_rows,
                "request_id": next(iter(request_ids)),
                "raw_decode_tokens": request_raw_decode_tokens,
                "final_decode_tokens": final_decode_tokens,
                "last_round_tokens": last_round_tokens,
            }
    elif final_decode_tokens < 0:
        return {
            **empty,
            "reason": "async_http_usage_has_no_seed_token",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
            "final_decode_tokens": final_decode_tokens,
        }
    if not _numeric(request_rounds) or request_rounds <= 0:
        return {
            **empty,
            "reason": "request_has_no_decode_rounds",
            "counter_deltas_by_rank": deltas,
            "request_rows_by_rank": request_rows,
            "request_id": next(iter(request_ids)),
        }
    steps = int(steps_raw) if int(steps_raw) == steps_raw else float(steps_raw)
    decode_tokens = int(final_decode_tokens)
    raw_decode_tokens = (
        int(tokens[0]) if int(tokens[0]) == tokens[0] else float(tokens[0])
    ) if not async_mode else None
    return {
        "available": True,
        "steps": int(request_rounds) if int(request_rounds) == request_rounds else float(request_rounds),
        "A": float(decode_tokens) / float(steps),
        "ms_step": seconds * 1000.0 / float(steps),
        "server_fenced_ms_per_step": seconds * 1000.0 / float(steps),
        "server_fenced_ms_per_step_scope": (
            f"GET /delivery/metrics delta: ranks[].cumulative.decode_seconds; "
            f"{first_row.get('decode_wall_scope') or 'worker completion fence'}; may include metrics RPC delay"
        ),
        "ms_step_legacy_alias_of": "server_fenced_ms_per_step",
        "decode_tokens": decode_tokens,
        "decode_tokens_raw": None if async_mode else raw_decode_tokens,
        "final_decode_tokens": decode_tokens,
        "decode_seconds_s": seconds,
        "server_prefill_s": prefill_s,
        "server_prefill_components": prefill_components,
        "server_prefill_source": prefill_source,
        "server_prefill_scope": first_row.get("prefill_wall_scope"),
        "prefill_reporting": after.get("prefill_reporting") if is_ced else None,
        "prefill_request_rows_by_rank": prefill_request_rows if is_ced else None,
        "counter_deltas_by_rank": deltas,
        "request_rows_by_rank": request_rows,
        "request_id": next(iter(request_ids)),
        "request_tag": expected_request_id,
        "window_fingerprint": first_row["window_fingerprint"],
        "async_mode": async_mode,
        "raw_tokens_trimmed_at_terminal": raw_trimmed,
        "ms_step_scope": first_row.get("decode_wall_scope"),
        "server_counter_decode_wall_may_include_metrics_rpc_delay": True,
        "reason": None,
        "trusted_source": {
            "endpoint": "/delivery/metrics",
            "schema": COUNTER_SCHEMA,
            "counter_scope": COUNTER_SCOPE,
            "service_instance_id": before.get("service_instance_id"),
            "prefill_service_instance_id": expected_prefill_service_instance_id if is_ced else None,
            "decode_service_instance_id": expected_decode_service_instance_id if is_ced else None,
            "form": after.get("form"),
            "served_model": after.get("served_model"),
            "rank_count": RANK_COUNT,
        },
    }


def _usage_is_real(usage: Any) -> bool:
    return (
        isinstance(usage, dict)
        and isinstance(usage.get("prompt_tokens"), int)
        and isinstance(usage.get("completion_tokens"), int)
        and isinstance(usage.get("total_tokens"), int)
        and usage.get("total_tokens") == usage.get("prompt_tokens") + usage.get("completion_tokens")
    )


def summarize_group(group: dict[str, Any], expected_requests: int, target_input_tokens: int) -> dict[str, Any]:
    requests = group.get("requests", [])
    actual_outputs = sum(1 for row in requests if row.get("output_text") is not None and row.get("error") is None)
    real_usage_count = sum(1 for row in requests if _usage_is_real(row.get("usage_tokens")))
    finish_reason_count = sum(1 for row in requests if isinstance(row.get("finish_reason"), str))
    ttft_count = sum(1 for row in requests if isinstance(row.get("ttft_s"), (int, float)))
    exact_input_count = sum(
        1 for row in requests
        if _usage_is_real(row.get("usage_tokens")) and row["usage_tokens"].get("prompt_tokens") == target_input_tokens
    )
    clean_count = sum(
        bool(row.get("clean", {}).get("clean"))
        for row in requests
        if row.get("output_text") is not None and row.get("error") is None
    )
    all_outputs_consistent = len(requests) == expected_requests and actual_outputs == expected_requests
    valid = (
        all_outputs_consistent
        and real_usage_count == expected_requests
        and exact_input_count == expected_requests
        and finish_reason_count == expected_requests
        and ttft_count == expected_requests
    )
    completion_values = [row["usage_tokens"]["completion_tokens"] for row in requests if _usage_is_real(row.get("usage_tokens"))]
    ttft_values = [row["ttft_s"] for row in requests if isinstance(row.get("ttft_s"), (int, float))]
    decode_seconds = [row["decode_s"] for row in requests if isinstance(row.get("decode_s"), (int, float))]
    rate_tokens = [
        row["decode_rate_completion_tokens"]
        for row in requests
        if isinstance(row.get("decode_rate_completion_tokens"), int)
    ]
    sum_decode_s = sum(decode_seconds) if decode_seconds else None
    sum_rate_tokens = sum(rate_tokens) if rate_tokens else None
    counter_rows = [row.get("delivery_counter_delta", {}) for row in requests]
    all_counter_rows = len(counter_rows) == expected_requests and all(row.get("available") for row in counter_rows)
    metric_steps = sum(row["steps"] for row in counter_rows) if all_counter_rows else None
    metric_decode_tokens = sum(row["decode_tokens"] for row in counter_rows) if all_counter_rows else None
    metric_decode_s = sum(row["decode_seconds_s"] for row in counter_rows) if all_counter_rows else None
    metric_prefill_s = (
        sum(row["server_prefill_s"] for row in counter_rows)
        if all_counter_rows and all(row.get("server_prefill_s") is not None for row in counter_rows)
        else None
    )
    ced_prefill_component_rows = [row.get("server_prefill_components") for row in counter_rows]
    server_prefill_components = None
    if (
        all_counter_rows
        and all(isinstance(value, dict) for value in ced_prefill_component_rows)
        and metric_prefill_s is not None
    ):
        server_prefill_components = {
            "prefill_encoder_s": sum(value["prefill_encoder_s"] for value in ced_prefill_component_rows),
            "decode_replay_tail_s": sum(value["decode_replay_tail_s"] for value in ced_prefill_component_rows),
            "server_prefill_s": metric_prefill_s,
            "aggregation": "sum of per-request max-rank P delta + max-rank D-only delta",
            "prefill_reporting": CED_PREFILL_REPORTING,
        }
    request_prefill_sources = {
        row.get("server_prefill_source") for row in counter_rows if row.get("server_prefill_source")
    }
    request_prefill_scopes = {
        row.get("server_prefill_scope") for row in counter_rows if row.get("server_prefill_scope")
    }
    server_prefill_scope = next(iter(request_prefill_scopes)) if len(request_prefill_scopes) == 1 else None
    server_prefill_source = None
    if metric_prefill_s is not None:
        if server_prefill_components is not None:
            server_prefill_source = (
                "sum of per-request P prefill_ranks max delta + D-only max delta "
                "(combined ranks delta minus P delta)"
            )
        elif len(request_prefill_sources) == 1:
            server_prefill_source = (
                "sum of per-request GET /delivery/metrics deltas; "
                + next(iter(request_prefill_sources))
            )
    client_decode_ms_per_step = (
        _milliseconds_per_step(sum_decode_s, metric_steps)
        if valid and all_counter_rows
        else None
    )
    server_fenced_ms_per_step = _milliseconds_per_step(metric_decode_s, metric_steps)
    server_fenced_scope = next(
        (row.get("server_fenced_ms_per_step_scope") or SERVER_FENCED_MS_PER_STEP_SCOPE for row in counter_rows),
        SERVER_FENCED_MS_PER_STEP_SCOPE,
    )
    http_rate_from_A = _http_rate_from_A(
        (metric_decode_tokens / metric_steps) if metric_steps else None,
        client_decode_ms_per_step,
    )
    http_decode_tok_s = (
        sum_rate_tokens / sum_decode_s
        if valid and sum_rate_tokens is not None and sum_decode_s is not None and sum_decode_s > 0
        else None
    )

    code_case_passed = sum(int(row.get("code_score", {}).get("passed", 0)) for row in requests)
    code_case_total = sum(int(row.get("code_score", {}).get("total", 0)) for row in requests)
    code_tasks_passed = sum(row.get("code_score", {}).get("status") == "pass" for row in requests)
    is_code = group.get("task") == "code"
    code_evaluated_tasks = sum(
        row.get("code_score", {}).get("status") in ("pass", "fail", "timeout", "parse_error", "execution_error")
        for row in requests
    )
    code_evaluation_errors = sum(
        row.get("code_score", {}).get("status") == "sandbox_error"
        for row in requests
    )
    return {
        "expected_requests": expected_requests,
        "actual_requests": len(requests),
        "actual_outputs": actual_outputs,
        "real_usage_requests": real_usage_count,
        "exact_input_length_requests": exact_input_count,
        "finish_reason_requests": finish_reason_count,
        "ttft_requests": ttft_count,
        "actual_output_tokens": sum(completion_values) if valid else None,
        "valid_for_aggregation": valid,
        "invalid_reasons": [
            reason
            for reason, condition in (
                ("request_or_output_count_mismatch", not all_outputs_consistent),
                ("missing_actual_server_usage", real_usage_count != expected_requests),
                ("input_token_count_mismatch", exact_input_count != expected_requests),
                ("missing_finish_reason", finish_reason_count != expected_requests),
                ("missing_ttft", ttft_count != expected_requests),
            )
            if condition
        ],
        "clean_count": clean_count,
        "clean_rate": clean_count / actual_outputs if actual_outputs else None,
        "mean_ttft_s": statistics.fmean(ttft_values) if ttft_values else None,
        "median_ttft_s": statistics.median(ttft_values) if ttft_values else None,
        "sum_decode_s": sum_decode_s,
        "decode_tok_s": (
            sum_rate_tokens / sum_decode_s
            if valid and sum_rate_tokens is not None and sum_decode_s is not None and sum_decode_s > 0
            else None
        ),
        "tpot_s": (
            sum_decode_s / sum_rate_tokens
            if valid and sum_rate_tokens is not None and sum_rate_tokens > 0 and sum_decode_s is not None
            else None
        ),
        "decode_rate_tokens_sum": sum_rate_tokens if valid else None,
        "client_e2e_ms_per_round": (
            sum_decode_s * 1000.0 / metric_steps
            if valid and all_counter_rows and metric_steps and sum_decode_s is not None
            else None
        ),
        "client_decode_ms_per_step": client_decode_ms_per_step,
        "client_decode_ms_per_step_scope": CLIENT_DECODE_MS_PER_STEP_SCOPE,
        "client_e2e_ms_per_round_legacy_alias_of": "client_decode_ms_per_step",
        "steps": metric_steps,
        "A": (metric_decode_tokens / metric_steps) if metric_steps else None,
        "ms_step": (metric_decode_s * 1000.0 / metric_steps) if metric_steps else None,
        "server_fenced_ms_per_step": server_fenced_ms_per_step,
        "server_fenced_ms_per_step_scope": server_fenced_scope,
        "ms_step_legacy_alias_of": "server_fenced_ms_per_step",
        "http_rate_from_A_tok_s": http_rate_from_A,
        "http_rate_consistency_abs_delta_tok_s": (
            abs(http_rate_from_A - http_decode_tok_s)
            if http_rate_from_A is not None and http_decode_tok_s is not None
            else None
        ),
        "counter_decode_tokens": metric_decode_tokens,
        "counter_decode_seconds_s": metric_decode_s,
        "server_prefill_s": metric_prefill_s,
        "server_prefill_components": server_prefill_components,
        "server_prefill_source": server_prefill_source,
        "server_prefill_scope": server_prefill_scope,
        "code_passed_tasks": code_tasks_passed if is_code else None,
        "code_total_tasks": len(requests) if is_code else None,
        "code_evaluated_tasks": code_evaluated_tasks if is_code else None,
        "code_evaluation_errors": code_evaluation_errors if is_code else None,
        "code_passed_cases": code_case_passed if is_code else None,
        "code_total_cases": code_case_total if is_code else None,
    }


def aggregate_rounds(groups: list[dict[str, Any]], expected_rounds: int) -> dict[str, Any]:
    usable = [group for group in groups if group.get("aggregate", {}).get("valid_for_aggregation")]
    aggregates = [group["aggregate"] for group in usable]
    outputs = sum(row["actual_outputs"] for row in aggregates)
    output_tokens = sum(row["actual_output_tokens"] for row in aggregates)
    decode_s_values = [row["sum_decode_s"] for row in aggregates if row.get("sum_decode_s") is not None]
    decode_s = sum(decode_s_values) if decode_s_values else None
    rate_tokens = sum(row["decode_rate_tokens_sum"] for row in aggregates if row.get("decode_rate_tokens_sum") is not None)
    clean_count = sum(row["clean_count"] for row in aggregates)
    code_tasks_passed = sum(row["code_passed_tasks"] or 0 for row in aggregates)
    code_tasks_total = sum(row["code_total_tasks"] or 0 for row in aggregates)
    code_cases_passed = sum(row["code_passed_cases"] or 0 for row in aggregates)
    code_cases_total = sum(row["code_total_cases"] or 0 for row in aggregates)
    all_metrics = bool(aggregates) and all(row.get("steps") is not None for row in aggregates)
    metric_steps = sum(row["steps"] for row in aggregates) if all_metrics else None
    metric_tokens = sum(row["counter_decode_tokens"] for row in aggregates) if all_metrics else None
    metric_seconds = sum(row["counter_decode_seconds_s"] for row in aggregates) if all_metrics else None
    prefill_vals = [row["server_prefill_s"] for row in aggregates]
    server_prefill_s = sum(prefill_vals) if prefill_vals and all(value is not None for value in prefill_vals) else None
    prefill_component_rows = [row.get("server_prefill_components") for row in aggregates]
    server_prefill_components = None
    if server_prefill_s is not None and all(isinstance(value, dict) for value in prefill_component_rows):
        server_prefill_components = {
            "prefill_encoder_s": sum(value["prefill_encoder_s"] for value in prefill_component_rows),
            "decode_replay_tail_s": sum(value["decode_replay_tail_s"] for value in prefill_component_rows),
            "server_prefill_s": server_prefill_s,
            "aggregation": "sum of valid-round per-request max-rank P delta + max-rank D-only delta",
            "prefill_reporting": CED_PREFILL_REPORTING,
        }
    prefill_sources = {row.get("server_prefill_source") for row in aggregates if row.get("server_prefill_source")}
    server_prefill_source = None
    if server_prefill_s is not None and len(prefill_sources) == 1:
        server_prefill_source = "sum of valid rounds; " + next(iter(prefill_sources))
    prefill_scopes = {row.get("server_prefill_scope") for row in aggregates if row.get("server_prefill_scope")}
    server_prefill_scope = next(iter(prefill_scopes)) if len(prefill_scopes) == 1 else None
    client_decode_ms_per_step = _milliseconds_per_step(decode_s, metric_steps)
    server_fenced_ms_per_step = _milliseconds_per_step(metric_seconds, metric_steps)
    A = (metric_tokens / metric_steps) if metric_tokens is not None and metric_steps else None
    http_rate_from_A = _http_rate_from_A(A, client_decode_ms_per_step)
    http_decode_rate = rate_tokens / decode_s if decode_s is not None and decode_s > 0 else None
    server_fenced_scope = (
        aggregates[0].get("server_fenced_ms_per_step_scope", SERVER_FENCED_MS_PER_STEP_SCOPE)
        if aggregates
        else SERVER_FENCED_MS_PER_STEP_SCOPE
    )
    return {
        "expected_rounds": expected_rounds,
        "actual_valid_rounds": len(usable),
        "valid_round_indices": [group.get("round_index") for group in usable],
        "actual_outputs": outputs,
        "actual_output_tokens": output_tokens if usable else None,
        "clean_rate": clean_count / outputs if outputs else None,
        "decode_tok_s": rate_tokens / decode_s if decode_s is not None and decode_s > 0 else None,
        "tpot_s": decode_s / rate_tokens if decode_s is not None and rate_tokens > 0 else None,
        "steps": metric_steps,
        "A": A,
        "ms_step": metric_seconds * 1000.0 / metric_steps if metric_steps else None,
        "server_fenced_ms_per_step": server_fenced_ms_per_step,
        "server_fenced_ms_per_step_scope": server_fenced_scope,
        "ms_step_legacy_alias_of": "server_fenced_ms_per_step",
        "client_e2e_ms_per_round": client_decode_ms_per_step,
        "client_decode_ms_per_step": client_decode_ms_per_step,
        "client_decode_ms_per_step_scope": CLIENT_DECODE_MS_PER_STEP_SCOPE,
        "client_e2e_ms_per_round_legacy_alias_of": "client_decode_ms_per_step",
        "http_rate_from_A_tok_s": http_rate_from_A,
        "http_rate_consistency_abs_delta_tok_s": (
            abs(http_rate_from_A - http_decode_rate)
            if http_rate_from_A is not None and http_decode_rate is not None
            else None
        ),
        "server_prefill_s": server_prefill_s,
        "server_prefill_components": server_prefill_components,
        "server_prefill_source": server_prefill_source,
        "server_prefill_scope": server_prefill_scope,
        "code_passed_tasks": code_tasks_passed,
        "code_total_tasks": code_tasks_total,
        "code_passed_cases": code_cases_passed,
        "code_total_cases": code_cases_total,
        "only_actual_consistent_rounds_included": True,
    }


def _record_response(
    sample: dict[str, Any],
    response: dict[str, Any],
    metric_delta: dict[str, Any],
    task: str,
    round_index: int,
    request_index: int,
    code_timeout: float,
    code_memory_mb: int,
    code_cpu_seconds: int,
) -> dict[str, Any]:
    prompt_text = sample["prompt_text"]
    output_text = response["output_text"]
    row = {
        "task": task,
        "round_index": round_index,
        "request_index": request_index,
        "sample_id": sample["sample_id"],
        "prompt_text": prompt_text,
        "prompt_sha256": sha256_text(prompt_text),
        "prompt_messages": sample["messages"],
        "prompt_messages_sha256": canonical_json_sha256(sample["messages"]),
        "output_text": output_text,
        "output_sha256": response["output_sha256"],
        "output_utf8_bytes": response["output_utf8_bytes"],
        "usage_tokens": response["usage_tokens"],
        "finish_reason": response["finish_reason"],
        "http_status": response["http_status"],
        "error": response["error"],
        "sse_error_event": response["sse_error_event"],
        "sse_json_events": response["sse_json_events"],
        "request_wall_s": response["request_wall_s"],
        "ttft_s": response["ttft_s"],
        "decode_s": response["decode_s"],
        "decode_rate_completion_tokens": response["decode_rate_completion_tokens"],
        "decode_tok_s": response["decode_tok_s"],
        "tpot_s": response["tpot_s"],
        "clean": output_cleanliness(output_text),
        "delivery_counter_delta": metric_delta,
        "source_metadata": {
            key: value
            for key, value in sample.items()
            if key not in ("messages", "prompt_text", "unit_cases")
        },
    }
    counter_steps = metric_delta.get("steps")
    client_decode_ms_per_step = _milliseconds_per_step(response.get("decode_s"), counter_steps)
    rate_from_A = _http_rate_from_A(metric_delta.get("A"), client_decode_ms_per_step)
    server_fenced_ms_per_step = metric_delta.get(
        "server_fenced_ms_per_step",
        metric_delta.get("ms_step"),
    )
    server_fenced_scope = metric_delta.get(
        "server_fenced_ms_per_step_scope",
        SERVER_FENCED_MS_PER_STEP_SCOPE,
    )
    row["client_e2e_ms_per_round"] = client_decode_ms_per_step
    row["client_decode_ms_per_step"] = client_decode_ms_per_step
    row["client_decode_ms_per_step_scope"] = CLIENT_DECODE_MS_PER_STEP_SCOPE
    row["client_e2e_ms_per_round_legacy_alias_of"] = "client_decode_ms_per_step"
    row["server_fenced_ms_per_step"] = server_fenced_ms_per_step
    row["server_fenced_ms_per_step_scope"] = server_fenced_scope
    row["ms_step_legacy_alias_of"] = "server_fenced_ms_per_step"
    row["http_rate_from_A_tok_s"] = rate_from_A
    row["http_rate_consistency_abs_delta_tok_s"] = (
        abs(rate_from_A - response["decode_tok_s"])
        if isinstance(rate_from_A, (int, float)) and isinstance(response["decode_tok_s"], (int, float))
        else None
    )
    if task == "code":
        score = score_code_output(
            sample["task_id"],
            output_text,
            timeout_seconds=code_timeout,
            memory_mb=code_memory_mb,
            cpu_seconds=code_cpu_seconds,
        )
        row["code_score"] = score
        row["code_task_id"] = sample["task_id"]
        row["unit_cases"] = sample["unit_cases"]
        row["unit_cases_sha256"] = sample["unit_cases_sha256"]
    return row


def _run_warmup(
    base_url: str,
    model: str,
    sample: dict[str, Any],
    timeout: float,
    api_key: str | None,
    task: str,
    warmup_index: int,
    request_id: str,
) -> dict[str, Any]:
    result = stream_chat_completion(
        base_url,
        model,
        sample["messages"],
        1,
        timeout,
        api_key,
        seed=0,
        request_id=request_id,
        ignore_eos=(task == "dihuo"),
    )
    return {
        "excluded_from_metrics": True,
        "task": task,
        "warmup_index": warmup_index,
        "request_id": request_id,
        "sample_id": sample["sample_id"],
        "prompt_text": sample["prompt_text"],
        "prompt_sha256": sha256_text(sample["prompt_text"]),
        "output_text": result["output_text"],
        "output_sha256": result["output_sha256"],
        "usage_tokens": result["usage_tokens"],
        "finish_reason": result["finish_reason"],
        "error": result["error"],
        "sse_error_event": result["sse_error_event"],
        "request_wall_s": result["request_wall_s"],
    }


def _run_counter_canary(
    base_url: str,
    model: str,
    form: str,
    service_instance_id: str,
    sample: dict[str, Any],
    input_tokens: int,
    output_tokens: int,
    timeout: float,
    api_key: str | None,
    expected_prefill_service_instance_id: str | None = None,
    expected_decode_service_instance_id: str | None = None,
) -> dict[str, Any]:
    request_id = f"dm-canary-{uuid.uuid4().hex}"
    before = _metrics_snapshot(base_url, timeout, api_key)
    arm = _arm_delivery_window(base_url, output_tokens, request_id, timeout, api_key)
    response = stream_chat_completion(
        base_url,
        model,
        sample["messages"],
        output_tokens,
        timeout,
        api_key,
        seed=0,
        request_id=request_id,
        ignore_eos=False,
    )
    after = _metrics_snapshot(base_url, timeout, api_key)
    delta = counter_metrics_delta(
        before,
        after,
        model,
        input_tokens,
        response["usage_tokens"],
        service_instance_id,
        request_id,
        output_tokens,
        arm,
        form,
        expected_prefill_service_instance_id,
        expected_decode_service_instance_id,
    )
    client_decode_ms_per_step = _milliseconds_per_step(response.get("decode_s"), delta.get("steps"))
    return {
        "excluded_from_formal_8": True,
        "task": "code",
        "sample_id": sample["sample_id"],
        "code_task_id": sample["task_id"],
        "request_id": request_id,
        "request_tag": request_id,
        "engine_request_id": delta.get("request_id"),
        "requested_max_tokens": output_tokens,
        "prompt_text": sample["prompt_text"],
        "prompt_sha256": sha256_text(sample["prompt_text"]),
        "output_text": response["output_text"],
        "output_sha256": response["output_sha256"],
        "usage_tokens": response["usage_tokens"],
        "finish_reason": response["finish_reason"],
        "http_status": response["http_status"],
        "error": response["error"],
        "sse_error_event": response["sse_error_event"],
        "ttft_s": response["ttft_s"],
        "decode_s": response["decode_s"],
        "decode_tok_s": response["decode_tok_s"],
        "tpot_s": response["tpot_s"],
        "client_e2e_ms_per_round": client_decode_ms_per_step,
        "client_decode_ms_per_step": client_decode_ms_per_step,
        "client_decode_ms_per_step_scope": CLIENT_DECODE_MS_PER_STEP_SCOPE,
        "client_e2e_ms_per_round_legacy_alias_of": "client_decode_ms_per_step",
        "server_fenced_ms_per_step": delta.get("server_fenced_ms_per_step", delta.get("ms_step")),
        "server_fenced_ms_per_step_scope": delta.get(
            "server_fenced_ms_per_step_scope",
            SERVER_FENCED_MS_PER_STEP_SCOPE,
        ),
        "ms_step_legacy_alias_of": "server_fenced_ms_per_step",
        "delivery_arm": arm,
        "delivery_metrics_before": before,
        "delivery_metrics_after": after,
        "delivery_counter_delta": delta,
        "counter_schema_valid": bool(delta.get("available")),
        "score_deferred_to_formal_code_run": True,
    }


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results/delivery_matrix") / f"delivery_matrix_{timestamp}.json"


def write_json_no_overwrite(path: str | Path, evidence: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible service root, e.g. http://host:8000")
    parser.add_argument("--model", required=True, help="Exact expected model id from GET /v1/models")
    parser.add_argument("--expected-form", help="Optional exact delivery form expected in /delivery/metrics")
    parser.add_argument("--tasks", choices=TASK_OPTIONS, default="all")
    parser.add_argument("--input-tokens", type=int, choices=INPUT_TOKEN_OPTIONS, default=2048)
    parser.add_argument("--output-tokens", type=int, default=256, help="Dihuo output-token cap (default: 256)")
    parser.add_argument("--code-output-tokens", type=int, default=CODE_OUTPUT_TOKENS_DEFAULT)
    parser.add_argument("--canary-only", action="store_true", help="Run one armed code request to validate the counter schema; excluded from formal 8")
    parser.add_argument("--canary-output-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=1, help="Measured rounds; each selected task has exactly 8 serial requests per round")
    parser.add_argument("--warmups", type=int, default=1, help="Excluded one-token warmup requests per selected task (default: 1)")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--tokenizer", help="Optional local HF tokenizer path; remote exact tokenizer is the fallback")
    parser.add_argument("--corpus", default=str(DIHUO_PATH), help="Local cleaned Dihuo source; never fetched by this script")
    parser.add_argument("--suffix-dir", default=str(DIHUO_SUFFIX_DIR))
    parser.add_argument("--frozen-dihuo-manifest", help="Pinned split manifest with eight historical request.gz inputs; avoids rebuilding prompts")
    parser.add_argument("--frozen-dihuo-manifest-sha256", help="Required SHA-256 pin for --frozen-dihuo-manifest")
    parser.add_argument("--expected-service-instance-id", help="Exact instance binding before warmup/generation; required for frozen replay")
    parser.add_argument("--expected-runtime-fingerprint", help="Exact runtime binding before warmup/generation; required for frozen replay")
    parser.add_argument("--json-out", help="Evidence path; created exclusively and never overwritten")
    parser.add_argument("--code-timeout", type=float, default=3.0)
    parser.add_argument("--code-memory-mb", type=int, default=512)
    parser.add_argument("--code-cpu-seconds", type=int, default=2)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    global _CURRENT_STAGE
    _CURRENT_STAGE = "validate_options"
    if args.rounds <= 0 or args.warmups < 0:
        raise ValueError("--rounds must be positive and --warmups non-negative")
    if args.output_tokens < 2 or args.code_output_tokens < 2:
        raise ValueError("output token limits must be >= 2 for an armed request window")
    if args.canary_output_tokens < 2:
        raise ValueError("--canary-output-tokens must be >= 2")
    if args.canary_only and args.tasks != "code":
        raise ValueError("--canary-only requires --tasks code")
    if args.request_timeout <= 0 or args.code_timeout <= 0 or args.code_memory_mb <= 0 or args.code_cpu_seconds <= 0:
        raise ValueError("timeouts and code resource limits must be positive")
    frozen_path = getattr(args, "frozen_dihuo_manifest", None)
    frozen_sha = getattr(args, "frozen_dihuo_manifest_sha256", None)
    if (frozen_path is None) != (frozen_sha is None):
        raise ValueError("--frozen-dihuo-manifest and its explicit SHA-256 pin must be supplied together")
    frozen_inputs = None
    if frozen_path is not None:
        if not isinstance(frozen_path, str) or not frozen_path.strip() or not isinstance(frozen_sha, str) or not frozen_sha:
            raise ValueError("Frozen Dihuo path and SHA-256 pin must not be empty")
        if args.tasks != "dihuo" or args.rounds != 1 or args.canary_only:
            raise ValueError("Frozen Dihuo replay requires --tasks dihuo --rounds 1 without a schema canary")
        expected_sid = getattr(args, "expected_service_instance_id", None)
        expected_fp = getattr(args, "expected_runtime_fingerprint", None)
        if (not isinstance(expected_sid, str) or len(expected_sid) != 32
                or any(c not in "0123456789abcdef" for c in expected_sid)
                or not _is_sha256_hex(expected_fp)):
            raise ValueError("Frozen Dihuo replay requires the current 32-hex service instance and 64-hex runtime fingerprint")
        _CURRENT_STAGE = "verify eight frozen Dihuo inputs before any service request"
        frozen_inputs = load_frozen_dihuo_inputs(frozen_path, frozen_sha, args.input_tokens, args.output_tokens)
    base_url = args.base_url.rstrip("/")
    _CURRENT_STAGE = "GET /v1/models exact model ownership"
    model_check = verify_model_ownership(base_url, args.model, args.request_timeout, args.api_key)
    _CURRENT_STAGE = "GET /delivery/identity service/form/PREFIX=0 binding"
    service_identity = verify_service_identity(
        base_url,
        args.model,
        args.expected_form,
        args.request_timeout,
        args.api_key,
    )
    if (getattr(args, "expected_service_instance_id", None) is not None
            and service_identity["service_instance_id"] != args.expected_service_instance_id):
        raise ValueError("delivery identity does not match the pinned service instance")
    if (getattr(args, "expected_runtime_fingerprint", None) is not None
            and service_identity["runtime_fingerprint"] != args.expected_runtime_fingerprint):
        raise ValueError("delivery identity does not match the pinned runtime fingerprint")
    verified_form = service_identity["form"]
    configured_max_len = service_identity["raw_identity"].get("model_kwargs", {}).get("max_model_len")
    output_caps = []
    if args.tasks in ("dihuo", "all"):
        output_caps.append(args.output_tokens)
    if args.tasks in ("code", "all"):
        output_caps.append(args.canary_output_tokens if args.canary_only else args.code_output_tokens)
    required_sequence_len = args.input_tokens + max(output_caps or [0])
    if isinstance(configured_max_len, int) and required_sequence_len > configured_max_len:
        raise ValueError(
            f"requested exact input plus output budget ({required_sequence_len}) exceeds "
            f"service model_kwargs.max_model_len={configured_max_len}"
        )
    tokenizer = None
    if frozen_inputs is None:
        _CURRENT_STAGE = "local tokenizer probe"
        tokenizer = make_token_counter(base_url, args.model, args.request_timeout, args.api_key, args.tokenizer)

    requested_tasks = ("dihuo", "code") if args.tasks == "all" else (args.tasks,)
    dih_material: dict[str, Any] | None = None
    prepared: dict[str, list[dict[str, Any]]] = {}
    if "dihuo" in requested_tasks:
        if frozen_inputs is not None:
            prepared["dihuo"] = frozen_inputs["samples"]
        else:
            _CURRENT_STAGE = "load verified local Dihuo corpus/suffix files"
            dih_material = load_dihuo_material(args.corpus, args.suffix_dir, verify_sha256=True)
            _CURRENT_STAGE = "calibrate eight Dihuo prompts to exact tokenizer length"
            prepared["dihuo"] = prepare_dihuo_prompts(
                dih_material["corpus"], dih_material["suffixes"], tokenizer, args.input_tokens
            )
    if "code" in requested_tasks:
        _CURRENT_STAGE = "calibrate eight code prompts to exact tokenizer length"
        prepared["code"] = prepare_code_prompts(tokenizer, args.input_tokens)
    if frozen_inputs is not None:
        tokenizer_info = {"backend": "frozen historical messages; no tokenizer calls",
                          "token_count_scope": frozen_inputs["proof"]["token_count_scope"]}
    else:
        tokenizer_info = tokenizer.info()
        tokenizer_info["local_probe_failures"] = getattr(tokenizer, "local_probe_failures", [])
        if tokenizer.remote.query_count:
            tokenizer_info["remote_query_count"] = tokenizer.remote.query_count
            tokenizer_info["remote_completion_fallback_count"] = tokenizer.remote.completion_fallback_count
    prompt_hashes = {
        task: [sha256_text(sample["prompt_text"]) for sample in prepared[task]]
        for task in requested_tasks
    }
    workload_definition = {
        "tasks": requested_tasks,
        "input_tokens_target": args.input_tokens,
        "dihuo_output_tokens": args.output_tokens if "dihuo" in requested_tasks else None,
        "code_output_tokens": args.code_output_tokens if "code" in requested_tasks else None,
        "prompt_sha256_by_task": prompt_hashes,
        "code_unit_cases_sha256": {
            sample["task_id"]: sample["unit_cases_sha256"]
            for sample in prepared.get("code", [])
        },
        "dihuo_corpus_sha256": dih_material["corpus_sha256"] if dih_material else None,
        "dihuo_suffix_sha256": dih_material["suffix_sha256"] if dih_material else None,
    }
    if frozen_inputs is not None:
        workload_definition = frozen_inputs["workload_manifest"]
    workload_fingerprint = canonical_json_sha256(workload_definition)
    client_source_sha256 = {
        "benchmark.py": sha256_bytes(Path(__file__).read_bytes()),
        "workloads.py": sha256_bytes(Path(__file__).with_name("workloads.py").read_bytes()),
    }
    if frozen_inputs is not None:
        client_source_sha256["frozen_dihuo_inputs.py"] = sha256_bytes(
            Path(__file__).with_name("frozen_dihuo_inputs.py").read_bytes())

    evidence: dict[str, Any] = {
        "schema_version": "delivery_matrix_client_evidence_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": base_url,
            "model": args.model,
            "expected_form": verified_form,
            "tasks": requested_tasks,
            "input_tokens_target": args.input_tokens,
            "dihuo_output_tokens": args.output_tokens if "dihuo" in requested_tasks else None,
            "code_output_tokens": args.code_output_tokens if "code" in requested_tasks else None,
            "canary_only": args.canary_only,
            "canary_output_tokens": args.canary_output_tokens if args.canary_only else None,
            "rounds": args.rounds,
            "formal_requests_per_task_per_round": 8,
            "serial": True,
            "prefix": 0,
            "prefix_policy": "unique request marker at start of each user prompt; service identity verified enable_prefix_caching=false",
            "warmups": args.warmups,
            "warmups_excluded": True,
            "streaming": True,
            "temperature": 0,
            "seed": 0,
            "output_usage_source": "OpenAI chat-completion stream usage object (stream_options.include_usage=true)",
            "input_length_source": tokenizer_info,
            "frozen_input_gate": frozen_inputs["proof"] if frozen_inputs else None,
            "expected_service_instance_id": getattr(args, "expected_service_instance_id", None),
            "expected_runtime_fingerprint": getattr(args, "expected_runtime_fingerprint", None),
            "client_source_sha256": client_source_sha256,
            "request_timeout_s": args.request_timeout,
            "code_sandbox": {
                "runtime": "bwrap --unshare-net; read-only root; RLIMIT_CPU/RLIMIT_AS/RLIMIT_FSIZE; wall timeout",
                "timeout_s": args.code_timeout,
                "memory_mb": args.code_memory_mb,
                "cpu_seconds": args.code_cpu_seconds,
            },
        },
        "model_ownership": model_check,
        "service_identity": service_identity,
        "workload_fingerprint": workload_fingerprint,
        "workload_manifest": workload_definition,
        "dihuo_source": (
            {
                "corpus_path": dih_material["corpus_path"],
                "corpus_sha256": dih_material["corpus_sha256"],
                "suffix_dir": dih_material["suffix_dir"],
                "suffix_sha256": dih_material["suffix_sha256"],
                "copyright_text_loaded_from_local_file": True,
            }
            if dih_material
            else None
        ),
        "code_task_manifest": (
            [
                {
                    "task_id": task.task_id,
                    "function_name": task.function_name,
                    "signature": task.signature.strip(),
                    "description": task.description,
                    "example": task.example,
                    "unit_cases": sample["unit_cases"],
                    "unit_cases_sha256": sample["unit_cases_sha256"],
                }
                for task, sample in zip(CODE_TASKS, prepared.get("code", []), strict=False)
            ]
            if "code" in requested_tasks
            else None
        ),
        "warmups": [],
        "groups": [],
        "summary_by_task": {},
        "summary": {},
        "prefill_reporting": {
            "client_fields": ["usage_tokens.prompt_tokens", "ttft_s"],
            "server_prefill_seconds": "only a sourced GET /delivery/metrics cumulative prefill_seconds delta; otherwise null",
            "server_prefill_source_field": "ranks[].cumulative.prefill_seconds",
            "ced_prefill_components": "prefill_ranks[].cumulative.prefill_seconds and (ranks[].cumulative.prefill_seconds - prefill_ranks[].cumulative.prefill_seconds), per request; transfer wait excluded",
        },
    }
    if frozen_inputs is not None:
        evidence["dihuo_source"] = {"original_source": frozen_inputs["dihuo_source"],
                                    "copyright_text_loaded_from_frozen_requests": True,
                                    "corpus_file_read_this_run": False}

    if args.canary_only:
        _CURRENT_STAGE = "armed code schema canary"
        canary = _run_counter_canary(
            base_url,
            args.model,
            verified_form,
            service_identity["service_instance_id"],
            prepared["code"][0],
            args.input_tokens,
            args.canary_output_tokens,
            args.request_timeout,
            args.api_key,
            service_identity.get("prefill_service_instance_id"),
            service_identity.get("decode_service_instance_id"),
        )
        evidence["canary"] = canary
        evidence["summary"] = {
            "canary_only": True,
            "schema_canary_passed": canary["counter_schema_valid"],
            "formal_8_started": False,
            "actual_outputs": int(canary.get("output_text") is not None and canary.get("error") is None),
            "steps": canary["delivery_counter_delta"].get("steps"),
            "A": canary["delivery_counter_delta"].get("A"),
            "ms_step": canary["delivery_counter_delta"].get("ms_step"),
            "server_fenced_ms_per_step": canary.get("server_fenced_ms_per_step"),
            "server_fenced_ms_per_step_scope": canary.get("server_fenced_ms_per_step_scope"),
            "ms_step_legacy_alias_of": "server_fenced_ms_per_step",
            "client_e2e_ms_per_round": canary.get("client_e2e_ms_per_round"),
            "client_decode_ms_per_step": canary.get("client_decode_ms_per_step"),
            "client_decode_ms_per_step_scope": canary.get("client_decode_ms_per_step_scope"),
            "client_e2e_ms_per_round_legacy_alias_of": "client_decode_ms_per_step",
        }
        return evidence

    for task in requested_tasks:
        task_samples = prepared[task]
        for warmup_index in range(args.warmups):
            _CURRENT_STAGE = f"excluded {task} warmup {warmup_index}"
            sample = task_samples[warmup_index % len(task_samples)]
            request_id = f"dm-{uuid.uuid4().hex}"
            evidence["warmups"].append(
                _run_warmup(
                    base_url,
                    args.model,
                    sample,
                    args.request_timeout,
                    args.api_key,
                    task,
                    warmup_index,
                    request_id,
                )
            )

    for task in requested_tasks:
        task_samples = prepared[task]
        groups: list[dict[str, Any]] = []
        output_cap = args.output_tokens if task == "dihuo" else args.code_output_tokens
        for round_index in range(args.rounds):
            group: dict[str, Any] = {
                "task": task,
                "round_index": round_index,
                "serial": True,
                "prefix": 0,
                "expected_requests": 8,
                "counter_window": "before/after every measured single request",
                "requests": [],
            }
            for request_index, sample in enumerate(task_samples):
                _CURRENT_STAGE = f"arm/send/fence {task} round {round_index} request {request_index}"
                request_id = f"dm-{uuid.uuid4().hex}"
                before = _metrics_snapshot(base_url, args.request_timeout, args.api_key)
                arm = _arm_delivery_window(
                    base_url,
                    output_cap,
                    request_id,
                    args.request_timeout,
                    args.api_key,
                )
                response = stream_chat_completion(
                    base_url,
                    args.model,
                    sample["messages"],
                    output_cap,
                    args.request_timeout,
                    args.api_key,
                    seed=request_index,
                    request_id=request_id,
                    ignore_eos=(task == "dihuo"),
                )
                after = _metrics_snapshot(base_url, args.request_timeout, args.api_key)
                metric_delta = counter_metrics_delta(
                    before,
                    after,
                    args.model,
                    args.input_tokens,
                    response["usage_tokens"],
                    service_identity["service_instance_id"],
                    request_id,
                    output_cap,
                    arm,
                    verified_form,
                    service_identity.get("prefill_service_instance_id"),
                    service_identity.get("decode_service_instance_id"),
                )
                row = _record_response(
                    sample,
                    response,
                    metric_delta,
                    task,
                    round_index,
                    request_index,
                    args.code_timeout,
                    args.code_memory_mb,
                    args.code_cpu_seconds,
                )
                row["delivery_metrics_before"] = before
                row["delivery_metrics_after"] = after
                row["delivery_arm"] = arm
                row["requested_max_tokens"] = output_cap
                row["seed"] = request_index
                row["request_id"] = request_id
                row["engine_request_id"] = metric_delta.get("request_id")
                row["request_tag"] = request_id
                row["ignore_eos"] = task == "dihuo"
                group["requests"].append(row)
            group["aggregate"] = summarize_group(group, 8, args.input_tokens)
            groups.append(group)
            evidence["groups"].append(group)
        evidence["summary_by_task"][task] = aggregate_rounds(groups, args.rounds)

    all_measured = [row for group in evidence["groups"] for row in group["requests"]]
    all_actual = [
        row for row in all_measured
        if row.get("output_text") is not None and row.get("error") is None
    ]
    clean = sum(bool(row.get("clean", {}).get("clean")) for row in all_actual)
    task_prefill_values = [
        evidence["summary_by_task"][task].get("server_prefill_s")
        for task in requested_tasks
    ]
    total_server_prefill = (
        sum(task_prefill_values)
        if task_prefill_values and all(value is not None for value in task_prefill_values)
        else None
    )
    task_prefill_sources = {
        task: evidence["summary_by_task"][task].get("server_prefill_source")
        for task in requested_tasks
        if evidence["summary_by_task"][task].get("server_prefill_source")
    }
    task_prefill_scopes = {
        task: evidence["summary_by_task"][task].get("server_prefill_scope")
        for task in requested_tasks
        if evidence["summary_by_task"][task].get("server_prefill_scope")
    }
    task_prefill_components = [
        evidence["summary_by_task"][task].get("server_prefill_components")
        for task in requested_tasks
    ]
    total_server_prefill_components = None
    if (
        total_server_prefill is not None
        and task_prefill_components
        and all(isinstance(value, dict) for value in task_prefill_components)
    ):
        total_server_prefill_components = {
            "prefill_encoder_s": sum(value["prefill_encoder_s"] for value in task_prefill_components),
            "decode_replay_tail_s": sum(value["decode_replay_tail_s"] for value in task_prefill_components),
            "server_prefill_s": total_server_prefill,
            "aggregation": "sum of per-task server prefill totals; each request uses max-rank P + max-rank D-only",
            "prefill_reporting": CED_PREFILL_REPORTING,
        }
    server_prefill_source = None
    if total_server_prefill is not None and task_prefill_sources:
        server_prefill_source = "sum of per-task server_prefill_s; " + "; ".join(
            f"{task}: {source}" for task, source in task_prefill_sources.items()
        )
    server_prefill_scope = None
    if total_server_prefill is not None and task_prefill_scopes:
        server_prefill_scope = "; ".join(
            f"{task}: {scope}" for task, scope in task_prefill_scopes.items()
        )
    evidence["summary"] = {
        "expected_measured_rounds": args.rounds * len(requested_tasks),
        "actual_measured_rounds": len(evidence["groups"]),
        "expected_outputs": args.rounds * len(requested_tasks) * 8,
        "actual_outputs": len(all_actual),
        "clean_outputs": clean,
        "clean_rate": clean / len(all_actual) if all_actual else None,
        "valid_task_rounds_for_aggregation": sum(
            group.get("aggregate", {}).get("valid_for_aggregation", False) for group in evidence["groups"]
        ),
        "server_prefill_s": total_server_prefill,
        "server_prefill_components": total_server_prefill_components,
        "server_prefill_source": server_prefill_source,
        "server_prefill_scope": server_prefill_scope,
        "note": "No prompt prefill duration is inferred from prompt_tokens or TTFT; server prefill appears only with its explicit counter source and scope.",
    }
    for task in requested_tasks:
        evidence["summary"][f"{task}_aggregate"] = evidence["summary_by_task"][task]
    _CURRENT_STAGE = "complete"
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.input_tokens not in INPUT_TOKEN_OPTIONS:
        parser.error(f"--input-tokens must be one of {INPUT_TOKEN_OPTIONS}")
    output_path = Path(args.json_out) if args.json_out else _default_output_path()
    try:
        evidence = run(args)
        written = write_json_no_overwrite(output_path, evidence)
    except FileExistsError:
        print(f"[benchmark] refusing to overwrite existing evidence: {output_path}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[benchmark] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        failure = {
            "schema_version": "delivery_matrix_client_failure_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": _CURRENT_STAGE,
            "config": {
                "base_url": args.base_url.rstrip("/"),
                "model": args.model,
                "expected_form": args.expected_form,
                "tasks": args.tasks,
                "input_tokens": args.input_tokens,
                "output_tokens": args.output_tokens,
                "code_output_tokens": args.code_output_tokens,
                "canary_only": args.canary_only,
                "corpus_path": args.corpus,
                "suffix_dir": args.suffix_dir,
            },
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=12),
            },
            "client_source_sha256": {
                "benchmark.py": sha256_bytes(Path(__file__).read_bytes()),
                "workloads.py": sha256_bytes(Path(__file__).with_name("workloads.py").read_bytes()),
            },
        }
        try:
            written = write_json_no_overwrite(output_path, failure)
            print(f"[benchmark] failure receipt: {written}", file=sys.stderr)
        except Exception as receipt_exc:  # preserve no-overwrite even for failure evidence
            print(f"[benchmark] could not write failure receipt: {type(receipt_exc).__name__}: {receipt_exc}", file=sys.stderr)
        return 1
    print(json.dumps({"evidence": str(written), "summary": evidence["summary"]}, ensure_ascii=False, indent=2))
    if args.canary_only and not evidence["summary"].get("schema_canary_passed"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

