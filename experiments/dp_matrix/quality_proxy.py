"""Loopback-only transparent rank injector and streaming quality wire journal."""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import uuid


GENERATION_PATHS = {"/v1/chat/completions", "/v1/completions"}
HOP_BY_HOP = {b"connection", b"keep-alive", b"proxy-authenticate",
              b"proxy-authorization", b"te", b"trailer", b"transfer-encoding",
              b"upgrade"}


def loopback_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("The quality proxy backend must be a plain HTTP loopback URL")
    return value.rstrip("/")


def _json_event(line: bytes, metadata: dict):
    line = line.strip()
    if not line.startswith(b"data:"):
        return
    data = line[5:].strip()
    if data == b"[DONE]":
        metadata["done_consumed"] = True
        return
    try:
        event = json.loads(data)
    except (ValueError, TypeError):
        metadata["unparsed_sse_data_lines"] += 1
        return
    if isinstance(event, dict):
        if "error" in event:
            metadata["api_error_event"] = event["error"] if event["error"] is not None else {"null_error_event": True}
        if event.get("usage") is not None:
            metadata["usage"] = event["usage"]
        for choice in event.get("choices") or []:
            if isinstance(choice, dict) and choice.get("finish_reason") is not None:
                metadata["finish_reasons"].append({
                    "index": choice.get("index", 0),
                    "finish_reason": choice["finish_reason"],
                })


class QualityProxy:
    """Minimal ASGI proxy. Request bytes stay unchanged; response bytes stream through."""

    def __init__(self, backend_url: str, output: Path, dp_count: int,
                 served_model: str, *, backend_transport=None):
        if type(dp_count) is not int or dp_count not in (2, 4):
            raise ValueError("Only the requested DP2/DP4 layouts are supported")
        self.backend_url = loopback_url(backend_url)
        self.output = Path(output).resolve()
        self.dp_count = dp_count
        self.served_model = served_model
        self.backend_transport = backend_transport
        self.next_rank = 0
        self.body_rank = {}
        self.file = None
        self.lock = asyncio.Lock()
        self.client = None

    async def _record(self, row):
        encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
        async with self.lock:
            await asyncio.to_thread(self.file.write, encoded)
            await asyncio.to_thread(self.file.flush)

    async def startup(self):
        self.output.mkdir(parents=True, exist_ok=False)
        self.file = (self.output / "attempts.jsonl").open("xb", buffering=0)
        import httpx
        self.client = httpx.AsyncClient(transport=self.backend_transport, timeout=None)
        # Disable httpx's default semantic headers; the client request plus our
        # single native-DP header define the forwarded request.
        self.client.headers.clear()

    async def shutdown(self):
        if self.client is not None:
            await self.client.aclose()
        if self.file is not None:
            self.file.close()

    def _select_rank(self, digest):
        if digest not in self.body_rank:
            self.body_rank[digest] = self.next_rank
            self.next_rank = (self.next_rank + 1) % self.dp_count
        return self.body_rank[digest]

    async def _body(self, receive):
        chunks = []
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                raise ConnectionError("Client disconnected before completing the request body")
            if event["type"] != "http.request":
                continue
            chunks.append(event.get("body", b""))
            if not event.get("more_body", False):
                return b"".join(chunks)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    await self.startup()
                    await send({"type": "lifespan.startup.complete"})
                elif event["type"] == "lifespan.shutdown":
                    await self.shutdown()
                    await send({"type": "lifespan.shutdown.complete"})
                    return

        if scope["type"] != "http":
            return
        import httpx

        body = await self._body(receive)
        headers = list(scope.get("headers", []))
        names = [key.lower() for key, _ in headers]
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        generation = method == "POST" and path in GENERATION_PATHS
        bad_route_headers = any(key in names for key in
                                (b"x-data-parallel-rank", b"x-dp-matrix-tag"))
        if generation and bad_route_headers:
            raw = json.dumps({"error": {"message": "Quality client must not set private DP headers"}}).encode()
            await send({"type": "http.response.start", "status": 400,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": raw})
            return

        request_hash = hashlib.sha256(body).hexdigest()
        rank = self._select_rank(request_hash) if generation else None
        forwarded = [(key, value) for key, value in headers
                     if key.lower() not in HOP_BY_HOP and key.lower() != b"host"]
        if rank is not None:
            forwarded.append((b"x-data-parallel-rank", str(rank).encode()))
        raw_path = scope.get("raw_path")
        wire_path = (raw_path.decode("ascii") if isinstance(raw_path, bytes)
                     else urllib.parse.quote(path, safe="/%:@!$&'()*+,;=-._~"))
        query_bytes = scope.get("query_string", b"")
        query = query_bytes.decode("latin-1")
        target = self.backend_url + wire_path + ("?" + query if query else "")
        request_headers = [(key.decode("latin-1"), value.decode("latin-1"))
                           for key, value in forwarded]
        row = {"schema": "dp_quality_proxy_attempt_v1", "attempt_id": uuid.uuid4().hex,
               "method": method, "path": path, "generation": generation,
               "served_model": self.served_model, "request_body_sha256": request_hash,
               "request_body_bytes": len(body), "selected_dp_rank": rank,
               "retry_key_sha256": request_hash if generation else None,
               "started_monotonic_ns": time.monotonic_ns(), "http_status": None,
               "response_dp_rank": None, "response_body_sha256": None,
               "route_matches_response": None,
               "response_body_bytes": 0, "finish_reasons": [], "usage": None,
               "done_consumed": False, "unparsed_sse_data_lines": 0,
               "response_closed": False, "error": None, "api_error_event": None}
        request_compressed = gzip.compress(body, mtime=0)
        request_file = row["attempt_id"] + ".request.raw.gz"
        response_file = row["attempt_id"] + ".response.raw"
        with (self.output / request_file).open("xb") as artifact:
            artifact.write(request_compressed)
        row.update(request_body_file=request_file,
                   request_gzip_sha256=hashlib.sha256(request_compressed).hexdigest(),
                   response_body_file=response_file)
        response_artifact = (self.output / response_file).open("xb", buffering=0)
        digest = hashlib.sha256()
        pending = b""
        nonstream = bytearray()
        try:
            async with self.client.stream(method, target, headers=request_headers,
                                          content=body) as response:
                row["http_status"] = response.status_code
                raw_rank = response.headers.get("X-DP-Matrix-Rank")
                try:
                    row["response_dp_rank"] = int(raw_rank) if raw_rank is not None else None
                except ValueError:
                    row["response_dp_rank"] = "invalid"
                if generation:
                    row["route_matches_response"] = row["response_dp_rank"] == rank
                response_headers = [(key.lower().encode(), value.encode("latin-1"))
                                    for key, value in response.headers.items()
                                    if key.lower().encode() not in HOP_BY_HOP]
                await send({"type": "http.response.start", "status": response.status_code,
                            "headers": response_headers})
                is_sse = "text/event-stream" in response.headers.get("content-type", "").lower()
                row["response_is_sse"] = is_sse
                async for chunk in response.aiter_raw():
                    digest.update(chunk)
                    row["response_body_bytes"] += len(chunk)
                    await asyncio.to_thread(response_artifact.write, chunk)
                    if is_sse:
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            _json_event(line, row)
                    elif len(nonstream) <= 4 * 1024 * 1024:
                        nonstream.extend(chunk)
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                if pending:
                    _json_event(pending, row)
                if not is_sse and nonstream:
                    try:
                        decoded = json.loads(nonstream)
                        if isinstance(decoded, dict) and "error" in decoded:
                            row["api_error_event"] = decoded["error"] if decoded["error"] is not None else {"null_error_event": True}
                        row["usage"] = decoded.get("usage") if isinstance(decoded, dict) else None
                        choices = decoded.get("choices", []) if isinstance(decoded, dict) else []
                        row["finish_reasons"] = [
                            {"index": item.get("index", 0), "finish_reason": item["finish_reason"]}
                            for item in choices if isinstance(item, dict)
                            and item.get("finish_reason") is not None]
                    except (ValueError, TypeError):
                        row["unparsed_nonstream"] = True
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                row["response_closed"] = True
        except Exception as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
            if row["http_status"] is None:
                raw = json.dumps({"error": {"message": "Backend request failed"}}).encode()
                await send({"type": "http.response.start", "status": 502,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": raw})
            else:
                raise
        finally:
            response_artifact.close()
            row["response_body_sha256"] = digest.hexdigest()
            row["finished_monotonic_ns"] = time.monotonic_ns()
            await self._record(row)


def create_app(backend_url, output, dp_count, served_model, *, backend_transport=None):
    return QualityProxy(backend_url, output, dp_count, served_model,
                        backend_transport=backend_transport)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dp-count", required=True, type=int, choices=(2, 4))
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    backend = loopback_url(args.backend)
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be unprivileged")
    import uvicorn
    uvicorn.run(create_app(backend, args.output, args.dp_count, args.served_model),
                host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
