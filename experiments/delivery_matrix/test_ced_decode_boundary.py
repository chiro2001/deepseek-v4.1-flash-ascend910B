import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI

package = ModuleType("delivery_matrix")
package.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules.setdefault("delivery_matrix", package)
from delivery_matrix.ced_decode_boundary import ced_decode_guard


def params():
    return {"do_remote_prefill": True, "do_remote_decode": False, "remote_engine_id": "engine",
            "remote_request_id": "p-request", "remote_host": "192.0.2.1", "remote_port": 19090,
            "remote_ptp_size": 8, "remote_block_ids": [[1]] + [[] for _ in range(11)],
            "ced_replay_tokens": 128, "ced_missing_swa_groups": [7, 8, 9, 10, 11],
            "ced_prefix_tokens": 2047, "num_prompt_blocks": 16}


class DecodeBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        app = FastAPI()
        app.middleware("http")(ced_decode_guard)
        @app.post("/v1/chat/completions")
        @app.post("/tokenize")
        async def engine(payload: dict):
            self.calls.append(payload)
            return {"forwarded": True}
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://d")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def call(self, payload, role="decode", path="/v1/chat/completions"):
        with patch.dict(os.environ, {"V41_CED_ROLE": role, "V41_DECODE_API_GUARD": "1"}):
            return await self.client.post(path, json=payload)

    async def test_nonempty_but_bad_contract_is_rejected_before_engine(self):
        response = await self.call({"kv_transfer_params": {"do_remote_prefill": True}})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.calls, [])
        self.assertEqual(response.json()["error"]["code"], "ced_invalid_transfer_contract")

    async def test_complete_proxy_contract_is_forwarded_without_mutation(self):
        payload = {"kv_transfer_params": params(), "messages": [{"role": "user", "content": "actual prompt"}]}
        self.assertEqual((await self.call(payload)).status_code, 200)
        self.assertEqual(self.calls, [payload])

    async def test_missing_params_preserve_original_boundary_guard(self):
        self.assertEqual((await self.call({"messages": []})).status_code, 400)
        self.assertEqual(self.calls, [])

    async def test_p_and_nongeneration_routes_are_unaffected(self):
        self.assertEqual((await self.call({}, role="prefill")).status_code, 200)
        self.assertEqual((await self.call({}, path="/tokenize")).status_code, 200)

    async def test_malformed_json_remains_framework_validation(self):
        with patch.dict(os.environ, {"V41_CED_ROLE": "decode", "V41_DECODE_API_GUARD": "1"}):
            response = await self.client.post("/v1/chat/completions", content=b"{invalid", headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
