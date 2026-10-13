"""CPU-only tests for the transparent quality DP header proxy."""
import hashlib
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from experiments.dp_matrix.quality_proxy import create_app, loopback_url
from experiments.dp_matrix.quality_audit import audit


TASK_TMP_ROOT = Path.home() / "tmp" / "dsv41-dp-quality-proxy-tests"
TASK_TMP_ROOT.mkdir(parents=True, exist_ok=True)


class Backend:
    def __init__(self):
        self.received = []
        self.response_rank_override = None

    async def __call__(self, scope, receive, send):
        chunks = []
        while True:
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
        self.received.append((scope["method"], scope["path"], scope["headers"], b"".join(chunks)))
        if scope["path"] in ("/v1/completions", "/v1/chat/completions"):
            raw = (b'data: {"choices":[{"index":0,"finish_reason":"stop"}],'
                   b'"usage":{"prompt_tokens":7,"completion_tokens":3}}\n'
                   b"data: [DONE]\n")
            status = 200
            content_type = b"text/event-stream"
        else:
            raw, status, content_type = b'{"data":[]}', 200, b"application/json"
        rank = self.response_rank_override
        if rank is None:
            headers = {key.lower(): value for key, value in scope["headers"]}
            rank = headers.get(b"x-data-parallel-rank", b"0")
        if isinstance(rank, int):
            rank = str(rank).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", content_type),
                                (b"x-dp-matrix-rank", rank),
                                (b"content-length", str(len(raw)).encode())]})
        await send({"type": "http.response.body", "body": raw, "more_body": False})


class QualityProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dp-quality-proxy-", dir=TASK_TMP_ROOT)
        self.root = Path(self.temp.name)
        self.backend = Backend()
        self.transport = httpx.ASGITransport(app=self.backend)
        self.proxy = create_app("http://127.0.0.1:18960", self.root / "audit", 2,
                                "dsv41-dp-test", backend_transport=self.transport)
        await self.proxy.startup()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.proxy),
                                        base_url="http://127.0.0.1:18961")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.proxy.shutdown()
        self.temp.cleanup()

    async def test_generation_payload_is_byte_identical_and_only_dp_rank_is_added(self):
        body = b'{"model":"deepseek-v41","prompt":"a","stream":true}'
        response = await self.client.post("/v1/completions", content=body,
                                          headers={"Content-Type": "application/json",
                                                   "X-Test-Semantic": "keep"})
        raw = response.content
        self.assertEqual(response.status_code, 200)
        self.assertEqual(raw, (b'data: {"choices":[{"index":0,"finish_reason":"stop"}],'
                               b'"usage":{"prompt_tokens":7,"completion_tokens":3}}\n'
                               b"data: [DONE]\n"))
        _, _, headers, received_body = self.backend.received[0]
        header_map = {key.lower(): value for key, value in headers}
        self.assertEqual(received_body, body)
        self.assertEqual(header_map[b"x-data-parallel-rank"], b"0")
        self.assertEqual(header_map[b"x-test-semantic"], b"keep")
        self.assertNotIn(b"x-dp-matrix-tag", header_map)
        row = json.loads((self.root / "audit/attempts.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["request_body_sha256"], hashlib.sha256(body).hexdigest())
        self.assertEqual(row["response_body_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(row["response_body_bytes"], len(raw))
        self.assertEqual((self.root / "audit" / row["response_body_file"]).read_bytes(), raw)
        self.assertEqual(gzip.decompress((self.root / "audit" / row["request_body_file"]).read_bytes()), body)
        self.assertEqual(row["selected_dp_rank"], 0)
        self.assertEqual(row["response_dp_rank"], 0)
        self.assertEqual(row["finish_reasons"][0]["finish_reason"], "stop")
        self.assertEqual(row["usage"]["completion_tokens"], 3)
        self.assertTrue(row["done_consumed"])
        self.assertTrue(row["response_closed"])

    async def test_retries_keep_rank_and_new_body_rotates(self):
        first = b'{"prompt":"retry-me"}'
        second = b'{"prompt":"next"}'
        for payload in (first, first, second):
            await self.client.post("/v1/completions", content=payload)
        ranks = []
        for _, _, headers, _ in self.backend.received:
            ranks.append({key.lower(): value for key, value in headers}[b"x-data-parallel-rank"])
        self.assertEqual(ranks, [b"0", b"0", b"1"])

    async def test_response_rank_mismatch_is_explicitly_recorded_as_failure(self):
        self.backend.response_rank_override = 1
        response = await self.client.post("/v1/completions", content=b'{"prompt":"first"}')
        self.assertEqual(response.status_code, 200)  # Proxy preserves the backend response.
        row = json.loads((self.root / "audit/attempts.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["selected_dp_rank"], 0)
        self.assertEqual(row["response_dp_rank"], 1)
        self.assertFalse(row["route_matches_response"])

    async def test_independent_wire_audit_checks_raw_bytes_and_all_dp_ranks(self):
        for prompt in ("a", "b"):
            await self.client.post("/v1/completions", content=json.dumps({"prompt": prompt}).encode())
        result = audit(self.root / "audit", dp_count=2, min_requests=2, min_unique=2)
        self.assertTrue(result["wire_audit_passed"])
        self.assertFalse(result["semantic_quality_passed"])
        journal = self.root / "audit/attempts.jsonl"
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        rows[1]["route_matches_response"] = False
        journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "rank/closure"):
            audit(self.root / "audit", dp_count=2, min_requests=2, min_unique=2)

    async def test_non_generation_passthrough_has_no_dp_header(self):
        body = b'{"model":"deepseek-v41"}'
        response = await self.client.post("/tokenize", content=body)
        self.assertEqual(response.status_code, 200)
        _, _, headers, received_body = self.backend.received[0]
        header_map = {key.lower(): value for key, value in headers}
        self.assertEqual(received_body, body)
        self.assertNotIn(b"x-data-parallel-rank", header_map)
        record = json.loads((self.root / "audit/attempts.jsonl").read_text().splitlines()[0])
        self.assertFalse(record["generation"])
        self.assertIsNone(record["selected_dp_rank"])

    async def test_private_routing_headers_from_client_are_rejected(self):
        response = await self.client.post("/v1/chat/completions", content=b"{}",
                                          headers={"X-Data-Parallel-Rank": "1"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.backend.received, [])

    def test_only_loopback_backend_is_admitted(self):
        self.assertEqual(loopback_url("http://localhost:18960/"), "http://localhost:18960")
        for value in ("https://127.0.0.1:18960", "http://192.0.2.4:18960",
                      "http://user@127.0.0.1:18960", "http://127.0.0.1:18960?q=1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                loopback_url(value)


if __name__ == "__main__":
    unittest.main()
