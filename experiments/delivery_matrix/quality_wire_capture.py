"""Observe the original evaluator's consumed HTTP bytes without changing them.

The pinned evaluator, prompts, scoring, output budget and retry policy stay
unchanged. A separate JSONL records each HTTP attempt's finish_reason/usage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid


class CapturedResponse:
    def __init__(self, response, receipt, append):
        self._response, self._receipt, self._append = response, receipt, append
        self._digest = hashlib.sha256()
        self._closed = False

    def _object(self, value):
        if not isinstance(value, dict):
            return
        if value.get("id") is not None:
            self._receipt["response_id"] = value["id"]
        if isinstance(value.get("usage"), dict):
            self._receipt["usage"] = value["usage"]
        for choice in value.get("choices") or []:
            if isinstance(choice, dict) and choice.get("finish_reason") is not None:
                self._receipt["finish_reasons"].append({"index": choice.get("index", 0),
                                                        "finish_reason": choice["finish_reason"]})

    def _observe(self, raw, streamed):
        self._digest.update(raw)
        self._receipt["consumed_wire_bytes"] += len(raw)
        if streamed:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                return
            value = line[5:].strip()
            if value == "[DONE]":
                self._receipt["done_consumed"] = True
                return
        else:
            value = raw.decode("utf-8", "ignore")
        try:
            self._object(json.loads(value))
        except (ValueError, TypeError):
            self._receipt["unparsed_consumed_chunks"] += 1

    def __iter__(self):
        self._iterator = iter(self._response)
        return self

    def __next__(self):
        raw = next(self._iterator)
        self._observe(raw, streamed=True)
        return raw

    def read(self, *args, **kwargs):
        # The pinned evaluator uses a single complete read for non-stream mode.
        raw = self._response.read(*args, **kwargs)
        self._observe(raw, streamed=False)
        return raw

    def close(self):
        if not self._closed:
            try:
                self._response.close()
            finally:
                self._closed = True
                self._receipt.update(consumed_wire_sha256=self._digest.hexdigest(),
                                     closed_monotonic_ns=time.monotonic_ns(), response_closed=True)
                self._append(dict(self._receipt))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getattr__(self, name):
        return getattr(self._response, name)


class WireCapture:
    def __init__(self, original, append):
        self.original, self.append = original, append

    def __call__(self, request, *args, **kwargs):
        body = None
        if (isinstance(request, urllib.request.Request) and request.data is not None
                and urllib.parse.urlsplit(request.full_url).path == "/v1/completions"):
            try:
                body = json.loads(request.data)
            except (ValueError, TypeError):
                pass
        if not isinstance(body, dict) or not isinstance(body.get("prompt"), str):
            return self.original(request, *args, **kwargs)
        receipt = {"schema": "delivery_quality_wire_attempt_v1", "attempt_id": uuid.uuid4().hex,
                   "thread": threading.current_thread().name, "endpoint": request.full_url,
                   "model": body.get("model"), "max_tokens": body.get("max_tokens"),
                   "temperature": body.get("temperature"), "stream": body.get("stream", False),
                   "request_body_sha256": hashlib.sha256(request.data).hexdigest(),
                   "prompt_utf8_sha256": hashlib.sha256(body["prompt"].encode()).hexdigest(),
                   "started_monotonic_ns": time.monotonic_ns(), "finish_reasons": [], "usage": None,
                   "response_id": None, "consumed_wire_bytes": 0, "unparsed_consumed_chunks": 0,
                   "done_consumed": False, "response_closed": False,
                   "scope": "only bytes consumed by the unchanged pinned evaluator; not an extra request"}
        try:
            response = self.original(request, *args, **kwargs)
        except BaseException as exc:
            receipt.update(open_error_type=type(exc).__name__, open_error=str(exc)[:500])
            self.append(receipt)
            raise
        receipt["status"] = getattr(response, "status", None)
        return CapturedResponse(response, receipt, self.append)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--wire-output", required=True, type=Path)
    parser.add_argument("evaluator_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    source = args.evaluator.resolve(strict=True)
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != args.evaluator_sha256:
        raise ValueError("Original quality evaluator SHA mismatch; do not thaw or substitute it")
    forwarded = args.evaluator_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    args.wire_output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    with args.wire_output.open("x") as sidecar:
        def append(receipt):
            receipt["evaluator_sha256"] = actual
            with lock:
                sidecar.write(json.dumps(receipt, ensure_ascii=False) + "\n")
                sidecar.flush()
        original, argv = urllib.request.urlopen, sys.argv
        urllib.request.urlopen = WireCapture(original, append)
        sys.argv = [str(source), *forwarded]
        try:
            runpy.run_path(str(source), run_name="__main__")
        finally:
            urllib.request.urlopen, sys.argv = original, argv


if __name__ == "__main__":
    main()
