"""CPU tests for the real P-to-D wire contract and identity/timing binding."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import httpx

spec = importlib.util.spec_from_file_location("ced_proxy_test", Path(__file__).with_name("ced_proxy.py"))
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)

MODEL = "ced-test-model"
STREAM = b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n'


def transfer_contract():
    return {"do_remote_prefill": True, "do_remote_decode": False, "remote_engine_id": "P-engine",
            "remote_request_id": "p-request", "remote_host": "192.0.2.1", "remote_port": 19090,
            "remote_ptp_size": 8, "remote_block_ids": [[1, 2]] + [[1]] * 6 + [[] for _ in range(5)],
            "ced_replay_tokens": 128, "ced_missing_swa_groups": [7, 8, 9, 10, 11],
            "ced_prefix_tokens": 2047, "num_prompt_blocks": 16}


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        self.bad_transfer = False
        self.bad_model = False
        self.bad_window = False
        self.bad_metrics_instance = False
        self.duplicate_rank = False
        self.bad_source_manifest = False
        self.bad_opp = False
        self.app = proxy.create_app("http://p", "http://d", MODEL)
        self.backend = httpx.AsyncClient(transport=httpx.MockTransport(self.handle), trust_env=False)
        self.app.state.client = self.backend
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(self.app), base_url="http://proxy")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.backend.aclose()

    def handle(self, request):
        role = "prefill" if request.url.host == "p" else "decode"
        body = json.loads(request.content) if request.content else None
        self.calls.append((role, request.url.path, body))
        if request.url.path == "/delivery/identity":
            manifest = "b" * 64 if self.bad_source_manifest and role == "prefill" else "a" * 64
            opp = "c" * 64 if self.bad_opp and role == "prefill" else "d" * 64
            return httpx.Response(200, json={"role": role, "form": "a3-ced-pd", "served_model": MODEL,
                                           "engine_initialized": True, "ASGI_startup_completed": True,
                                           "service_instance_id": role + "-nonce", "runtime_fingerprint": role + "-config",
                                           "ced_overlay_manifest_sha256": manifest,
                                           "native_opp_receipt": {"artifact_fingerprint": opp},
                                           "rank_descriptions": [{"rank": rank} for rank in range(8)]})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "foreign-model" if self.bad_model and role == "prefill" else MODEL}]})
        if request.url.path == "/delivery/metrics":
            seconds = 0.2 if role == "prefill" else 0.03
            nonce = "foreign-nonce" if self.bad_metrics_instance else role + "-nonce"
            tag = "other-request" if self.bad_window and role == "prefill" else "task-0"
            ranks = list(range(8)) + ([0] if self.duplicate_rank else [])
            return httpx.Response(200, json={"service_instance_id": nonce,
                                           "ranks": [{"rank": rank, **({"role": "prefill"} if role == "prefill" else {}),
                                                      "cumulative": {"prefill_seconds": seconds},
                                                      "requests": [{"tag": tag, "prefill_wall_scope": "original"}],
                                                      "pending": False, "errors": []} for rank in ranks]})
        if request.url.path == "/delivery/arm":
            return httpx.Response(200, json={"schema": "delivery_counter_arm_v1", "tag": body["tag"],
                                           "ranks": [{"rank": rank, "armed": True} for rank in range(8)]})
        if request.url.path == "/v1/chat/completions":
            if role == "prefill":
                params = {} if self.bad_transfer else transfer_contract()
                return httpx.Response(200, json={"kv_transfer_params": params})
            if body["stream"]:
                return httpx.Response(200, stream=httpx.ByteStream(STREAM), headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})
        raise AssertionError("Unexpected backend route")

    def payload(self):
        return {"model": MODEL, "messages": [{"role": "user", "content": "actual user prompt"}],
                "max_tokens": 256, "stream": True, "stream_options": {"include_usage": True}, "request_id": "client-id"}

    async def test_stream_bytes_and_original_decoder_prompt_are_unchanged(self):
        original = self.payload()
        response = await self.client.post("/v1/chat/completions", json=original)
        self.assertEqual(response.content, STREAM)
        p = next(body for role, path, body in self.calls if role == "prefill" and path.endswith("completions"))
        d = next(body for role, path, body in self.calls if role == "decode" and path.endswith("completions"))
        self.assertEqual(p["max_tokens"], 1)
        self.assertEqual(p["min_tokens"], 1)
        self.assertFalse(p["stream"])
        self.assertNotIn("stream_options", p)
        self.assertEqual(d["messages"], original["messages"])
        self.assertEqual(d["max_tokens"], 256)
        self.assertTrue(d["kv_transfer_params"]["do_remote_prefill"])

    async def test_invalid_producer_contract_never_reaches_decoder(self):
        self.bad_transfer = True
        response = await self.client.post("/v1/chat/completions", json=self.payload())
        self.assertEqual(response.status_code, 502)
        self.assertFalse(any(role == "decode" and path.endswith("completions") for role, path, _ in self.calls))

    async def test_both_model_catalogues_must_match(self):
        self.bad_model = True
        self.assertEqual((await self.client.get("/v1/models")).status_code, 503)

    async def test_pair_identity_changes_if_either_side_changes(self):
        first = proxy.pair_id({"service_instance_id": "P1"}, {"service_instance_id": "D1"})
        self.assertNotEqual(first, proxy.pair_id({"service_instance_id": "P2"}, {"service_instance_id": "D1"}))
        result = (await self.client.get("/delivery/identity")).json()
        self.assertEqual(result["topology"], "P8+D8; DCP=1")
        self.assertIn("prefill_identity", result)
        self.assertEqual(
            result["service_instance_id"],
            hashlib.sha256(
                f"{result['prefill_identity']['service_instance_id']}:{result['decode_identity']['service_instance_id']}".encode()
            ).hexdigest(),
        )
        self.assertEqual(result["decode_identity"]["service_instance_id"], "decode-nonce")
        self.assertEqual(result["prefill_identity"]["service_instance_id"], "prefill-nonce")

    async def test_p_and_d_source_manifests_must_match(self):
        self.bad_source_manifest = True
        self.assertEqual((await self.client.get("/delivery/identity")).status_code, 503)

    async def test_p_and_d_actual_opp_must_match(self):
        self.bad_opp = True
        self.assertEqual((await self.client.get("/delivery/identity")).status_code, 503)

    async def test_prefill_times_are_actual_p_plus_d_with_explicit_scope(self):
        result = (await self.client.get("/delivery/metrics")).json()
        self.assertEqual(len(result["prefill_ranks"]), 8)
        self.assertIsNot(result["prefill_ranks"], result["ranks"])
        self.assertEqual(result["prefill_ranks"][0]["role"], "prefill")
        self.assertAlmostEqual(result["prefill_ranks"][0]["cumulative"]["prefill_seconds"], 0.2)
        self.assertAlmostEqual(result["ranks"][0]["cumulative"]["prefill_seconds"], 0.23)
        self.assertIn("transfer wait excluded", result["ranks"][0]["requests"][0]["prefill_wall_scope"])

    async def test_producer_payload_does_not_mutate_callers_message(self):
        original = self.payload()
        saved = copy.deepcopy(original)
        modified = proxy.prefill_payload(original)
        modified["messages"][0]["content"] = "changed"
        self.assertEqual(original, saved)

    async def test_prefill_metrics_cannot_be_joined_to_another_request(self):
        self.bad_window = True
        self.assertEqual((await self.client.get("/delivery/metrics")).status_code, 502)

    async def test_prefill_metrics_cannot_be_joined_to_another_instance(self):
        self.bad_metrics_instance = True
        self.assertEqual((await self.client.get("/delivery/metrics")).status_code, 502)

    async def test_duplicate_rank_is_not_a_complete_tp8_receipt(self):
        self.duplicate_rank = True
        self.assertEqual((await self.client.get("/delivery/metrics")).status_code, 502)

    async def test_transfer_contract_rejects_bad_fields_and_uncomputed_swa(self):
        valid = transfer_contract()
        self.assertEqual(valid["remote_ptp_size"], 8)
        self.assertEqual(len(valid["remote_block_ids"]), 12)
        self.assertEqual(valid["ced_replay_tokens"], 128)
        proxy.validate_transfer(valid)
        bad_fields = {"remote_engine_id": "", "remote_request_id": None, "remote_host": " ",
                      "remote_port": 65529, "ced_prefix_tokens": True, "num_prompt_blocks": 0,
                      "remote_ptp_size": 4, "ced_replay_tokens": 256, "do_remote_decode": True,
                      "ced_missing_swa_groups": [8, 9, 10, 11]}
        for field, bad in bad_fields.items():
            params = transfer_contract()
            params[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                proxy.validate_transfer(params)
        for mutate in (lambda groups: groups.pop(), lambda groups: groups[0].append(-1),
                       lambda groups: groups[7].append(1)):
            params = transfer_contract()
            mutate(params["remote_block_ids"])
            with self.assertRaises(ValueError):
                proxy.validate_transfer(params)


if __name__ == "__main__":
    unittest.main()
