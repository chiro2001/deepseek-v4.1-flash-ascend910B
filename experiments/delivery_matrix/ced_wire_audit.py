"""Opt-in raw P/D wire journal around the byte-pinned, unchanged CED proxy.

Only P response bodies are read (P is already non-streaming). D response/SSE
bodies are never observed here. No requests, retries or headers are added.
Journal I/O precedes the D request and belongs to TTFT, not decode wall time.
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import uuid

PROXY_SHA256 = "813c5a0e1a8b7e7988ee9f02fd3a83d8fa38ede9a88713a9c65e5cf0cf99eb33"
MAX_RAW_BYTES = 8 * 1024 * 1024
MAX_GZIP_BYTES = 999999  # Every journal artifact is below the SSH 1 MB limit.
GENERATION_PATHS = ("/v1/chat/completions", "/v1/completions")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_frozen_proxy(path):
    path = Path(path).resolve(strict=True)
    if sha(path.read_bytes()) != PROXY_SHA256:
        raise ValueError("The wire journal requires the unchanged frozen CED proxy")
    spec = importlib.util.spec_from_file_location("_delivery_frozen_ced_proxy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exclusive_bytes(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(raw)


class WireJournal:
    def __init__(self, root, prefill_url, decode_url, model, proxy_source):
        self.root = Path(root).resolve()
        self.urls = {"prefill": prefill_url.rstrip("/"), "decode": decode_url.rstrip("/")}
        self.model = model
        source = Path(proxy_source).resolve(strict=True)
        if sha(source.read_bytes()) != PROXY_SHA256:
            raise ValueError("Proxy source drift before wire-journal installation")
        self.root.mkdir(parents=True, exist_ok=False)
        info = {"schema": "delivery_ced_wire_journal_v1", "served_model": model,
                "proxy_source": str(source), "proxy_source_sha256": PROXY_SHA256,
                "journal_source_sha256": sha(Path(__file__).read_bytes()),
                "backend_urls": self.urls, "max_raw_bytes": MAX_RAW_BYTES,
                "max_gzip_bytes": MAX_GZIP_BYTES, "adds_requests_or_retries": False,
                "D_response_or_SSE_body_consumed": False,
                "scope": "raw outbound generation requests and P non-streaming response; not performance counters"}
        exclusive_bytes(self.root / "journal_info.json", json.dumps(info, indent=2).encode())

    def role(self, request):
        if request.method != "POST":
            return None
        url = str(request.url)
        for role, base in self.urls.items():
            if url in (base + path for path in GENERATION_PATHS):
                return role
        return None

    def record(self, phase, request, raw, *, http_status=None, linked_request=None):
        if len(raw) > MAX_RAW_BYTES:
            raise ValueError("CED audit body exceeds the explicit raw-byte budget")
        compressed = gzip.compress(raw, mtime=0)
        if len(compressed) > MAX_GZIP_BYTES:
            raise ValueError("CED audit artifact exceeds the single-file transfer budget")
        # Caller-controlled request IDs must never become path components.
        name = phase + "-" + uuid.uuid4().hex
        body_path = self.root / (name + ".raw.json.gz")
        metadata = {"schema": "delivery_ced_wire_body_v1", "phase": phase,
                    "url": str(request.url), "method": request.method,
                    "request_id": request.headers.get("X-Request-Id"),
                    "served_model": self.model, "http_status": http_status,
                    "raw_sha256": sha(raw), "raw_bytes": len(raw),
                    "gzip_sha256": sha(compressed), "gzip_bytes": len(compressed),
                    "body_file": body_path.name, "linked_request": linked_request,
                    "captured_monotonic_ns": time.monotonic_ns()}
        encoded = json.dumps(metadata, indent=2).encode()
        if len(encoded) > MAX_GZIP_BYTES:
            raise ValueError("CED audit metadata exceeds the single-file transfer budget")
        exclusive_bytes(body_path, compressed)
        exclusive_bytes(self.root / (name + ".exchange.json"), encoded)
        return name + ".exchange.json"

    async def observe_request(self, request):
        role = self.role(request)
        if role is not None:
            # json= requests are buffered already; do not iterate a request
            # stream or mutate its actual bytes/headers/extensions.
            await asyncio.to_thread(self.record, role + "-request", request, request.content)

    async def observe_response(self, response):
        request = response.request
        if self.role(request) == "prefill":
            # This matches the frozen proxy's post(stream=False): its result
            # would be fully buffered before parsing kv_transfer_params.
            raw = await response.aread()
            await asyncio.to_thread(self.record, "prefill-response", request, raw,
                                    http_status=response.status_code)

    def attach(self, client):
        if getattr(client, "_delivery_ced_wire_journal", False):
            raise RuntimeError("Do not install duplicate wire hooks on a live client")
        client.event_hooks["request"].append(self.observe_request)
        client.event_hooks["response"].append(self.observe_response)
        client._delivery_ced_wire_journal = True


def create_audited_app(proxy, journal, prefill_url, decode_url, model):
    app = proxy.create_app(prefill_url, decode_url, model)
    original = app.router.lifespan_context

    @asynccontextmanager
    async def audited_lifespan(current):
        async with original(current) as state:
            journal.attach(current.state.client)
            yield state

    app.router.lifespan_context = audited_lifespan
    return app


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--proxy-source", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--port", type=int, default=18992)
    args = parser.parse_args()
    proxy = load_frozen_proxy(args.proxy_source)
    journal = WireJournal(args.audit_output, args.prefill_url, args.decode_url,
                          args.served_model, args.proxy_source)
    import uvicorn
    uvicorn.run(create_audited_app(proxy, journal, args.prefill_url, args.decode_url,
                                 args.served_model), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
