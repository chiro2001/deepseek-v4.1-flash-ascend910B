"""CPU transparency, raw-contract, provenance and failure controls."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
import test_ced_proxy as proxy_tests

MODEL, STREAM, transfer_contract = proxy_tests.MODEL, proxy_tests.STREAM, proxy_tests.transfer_contract

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ced_wire_audit_cpu", HERE / "ced_wire_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class WireAuditTests(proxy_tests.ProxyTests):
    # Reuse the complete frozen-proxy tests, now with hooks attached.
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.task = TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="ced-wire-")
        self.journal = audit.WireJournal(Path(self.task.name) / "wire", "http://p", "http://d",
                                         MODEL, HERE / "ced_proxy.py")
        self.journal.attach(self.backend)

    async def asyncTearDown(self):
        await super().asyncTearDown()
        self.task.cleanup()

    def records(self):
        result = []
        for path in self.journal.root.glob("*.exchange.json"):
            row = json.loads(path.read_text())
            compressed = (path.parent / row["body_file"]).read_bytes()
            raw = gzip.decompress(compressed)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["raw_sha256"])
            self.assertEqual(hashlib.sha256(compressed).hexdigest(), row["gzip_sha256"])
            self.assertEqual(len(raw), row["raw_bytes"])
            self.assertEqual(len(compressed), row["gzip_bytes"])
            result.append((row, json.loads(raw)))
        return result

    async def test_journal_keeps_exact_dynamic_transfer_and_no_extra_requests(self):
        response = await self.client.post("/v1/chat/completions", json=self.payload())
        self.assertEqual(response.content, STREAM)
        records = self.records()
        self.assertEqual(sorted(row["phase"] for row, _ in records),
                         ["decode-request", "prefill-request", "prefill-response"])
        p = next(body for row, body in records if row["phase"] == "prefill-response")
        d = next(body for row, body in records if row["phase"] == "decode-request")
        self.assertEqual(p["kv_transfer_params"], transfer_contract())
        self.assertEqual(p["kv_transfer_params"], d["kv_transfer_params"])
        self.assertEqual(len(self.calls), 4)  # 2 original identity GET + P/D POST.

    async def test_d_response_stream_is_not_read_by_hook(self):
        request = httpx.Request("POST", "http://d/v1/chat/completions", json=self.payload())
        response = httpx.Response(200, request=request, stream=httpx.ByteStream(STREAM))
        await self.journal.observe_response(response)
        self.assertFalse(response.is_stream_consumed)
        self.assertEqual(self.records(), [])
        await response.aclose()

    async def test_nonstream_D_body_is_also_not_journaled(self):
        payload = self.payload(); payload["stream"] = False
        response = await self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "answer")
        self.assertEqual(len(self.records()), 3)

    async def test_duplicate_install_path_ids_and_existing_directory(self):
        with self.assertRaises(RuntimeError): self.journal.attach(self.backend)
        with self.assertRaises(FileExistsError):
            audit.WireJournal(self.journal.root, "http://p", "http://d", MODEL, HERE / "ced_proxy.py")
        request = httpx.Request("POST", "http://p/v1/completions", json={"prompt": "a"},
                               headers={"X-Request-Id": "../../outside"})
        await self.journal.observe_request(request)
        self.assertEqual(self.records()[0][0]["request_id"], "../../outside")
        self.assertFalse((Path(self.task.name) / "outside").exists())

    async def test_source_drift_and_oversize_fail_closed(self):
        path = Path(self.task.name) / "drift.py"
        path.write_text((HERE / "ced_proxy.py").read_text() + "\n# source drift\n")
        with self.assertRaises(ValueError): audit.load_frozen_proxy(path)
        request = httpx.Request("POST", "http://p/v1/completions", json={"prompt": "a"})
        with patch.object(audit, "MAX_RAW_BYTES", 1), self.assertRaises(ValueError):
            await self.journal.observe_request(request)
        with patch.object(audit, "MAX_GZIP_BYTES", 1), self.assertRaises(ValueError):
            await self.journal.observe_request(request)
        self.assertEqual(self.records(), [])

    async def test_actual_lifespan_installs_hooks_then_closes_backend(self):
        proxy = audit.load_frozen_proxy(HERE / "ced_proxy.py")
        app = audit.create_audited_app(proxy, self.journal, "http://p", "http://d", MODEL)
        backend = httpx.AsyncClient(transport=httpx.MockTransport(self.handle), trust_env=False)
        frontend = httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy")
        try:
            with patch.object(httpx, "AsyncClient", return_value=backend):
                async with app.router.lifespan_context(app):
                    self.assertIs(app.state.client, backend)
                    response = await frontend.post("/v1/chat/completions", json=self.payload())
                    self.assertEqual(response.content, STREAM)
                    self.assertEqual(len(self.records()), 3)
            self.assertTrue(backend.is_closed)
        finally:
            await frontend.aclose()
            await backend.aclose()


if __name__ == "__main__":
    unittest.main()
