"""CPU-only ASGI-scope tests for native data-parallel routing."""
from __future__ import annotations

import asyncio
import copy
import os
import unittest
from unittest.mock import patch

from experiments.dp_matrix.http_routes import DPRouteMiddleware, WINDOWS


class FakeASGIApp:
    def __init__(self, delay: float = 0, *, status: int = 200, body_messages=None):
        self.scopes = []
        self.delay = delay
        self.status = status
        self.body_messages = body_messages or [
            {"body": b"fake-response", "more_body": False},
        ]

    async def __call__(self, scope, receive, send):
        self.scopes.append(scope)
        if self.delay:
            await asyncio.sleep(self.delay)
        await send({"type": "http.response.start", "status": self.status, "headers": []})
        for body in self.body_messages:
            await send({"type": "http.response.body", **body})


class DPRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_windows = copy.deepcopy(WINDOWS)
        WINDOWS.clear()
        self._environment = patch.dict(
            os.environ,
            {"DP_MATRIX_LAYOUT": "dp4-tp2"},
            clear=False,
        )
        self._environment.start()
        self.app = FakeASGIApp()
        self.middleware = DPRouteMiddleware(self.app)

    async def asyncTearDown(self):
        WINDOWS.clear()
        WINDOWS.update(self._old_windows)
        self._environment.stop()

    @staticmethod
    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    @staticmethod
    def _scope(headers=(), *, path="/v1/chat/completions", method="POST"):
        return {
            "type": "http",
            "method": method,
            "path": path,
            "headers": list(headers),
        }

    async def _request(self, headers=(), *, path="/v1/chat/completions", method="POST",
                       send_hook=None):
        messages = []

        async def send(message):
            if send_hook is not None:
                await send_hook(message)
            messages.append(message)

        await self.middleware(self._scope(headers, path=path, method=method), self._receive, send)
        return messages

    @staticmethod
    def _status(messages):
        return next(message["status"] for message in messages if message["type"] == "http.response.start")

    @staticmethod
    def _response_headers(messages):
        start = next(message for message in messages if message["type"] == "http.response.start")
        return start["headers"]

    @staticmethod
    def _scope_headers(scope):
        return [(key.lower(), value) for key, value in scope["headers"]]

    async def test_unmeasured_quality_requests_without_header_use_round_robin(self):
        observed = []
        for expected in (0, 1, 2, 3, 0):
            messages = await self._request()
            self.assertEqual(self._status(messages), 200)
            self.assertEqual(len(self.app.scopes), len(observed) + 1)
            scope_headers = self._scope_headers(self.app.scopes[-1])
            routed = [value for key, value in scope_headers if key == b"x-data-parallel-rank"]
            self.assertEqual(routed, [str(expected).encode()])
            response_rank = [
                value for key, value in self._response_headers(messages)
                if key.lower() == b"x-dp-matrix-rank"
            ]
            self.assertEqual(response_rank, [str(expected).encode()])
            observed.append(expected)

        self.assertEqual(observed, [0, 1, 2, 3, 0])
        self.assertEqual(WINDOWS, {})

    async def test_explicit_unmeasured_rank_is_preserved_and_echoed(self):
        messages = await self._request(
            [(b"x-data-parallel-rank", b"3"), (b"content-type", b"application/json")]
        )

        self.assertEqual(self._status(messages), 200)
        headers = self._scope_headers(self.app.scopes[0])
        routed = [value for key, value in headers if key == b"x-data-parallel-rank"]
        self.assertEqual(routed, [b"3"])
        echoed = [
            value for key, value in self._response_headers(messages)
            if key.lower() == b"x-dp-matrix-rank"
        ]
        self.assertEqual(echoed, [b"3"])

    async def test_duplicate_malformed_and_out_of_range_rank_headers_are_rejected(self):
        cases = (
            [(b"x-data-parallel-rank", b"1"), (b"x-data-parallel-rank", b"1")],
            [(b"x-data-parallel-rank", b"-1")],
            [(b"x-data-parallel-rank", b"+1")],
            [(b"x-data-parallel-rank", b" 1")],
            [(b"x-data-parallel-rank", b"1,1")],
            [(b"x-data-parallel-rank", b"4")],
            [(b"x-data-parallel-rank", b"999")],
        )
        for headers in cases:
            with self.subTest(headers=headers):
                messages = await self._request(headers)
                self.assertEqual(self._status(messages), 400)
        self.assertEqual(self.app.scopes, [])

    async def test_duplicate_and_invalid_utf8_tags_are_rejected(self):
        cases = (
            [
                (b"x-data-parallel-rank", b"1"),
                (b"x-dp-matrix-tag", b"tag-a"),
                (b"x-dp-matrix-tag", b"tag-a"),
            ],
            [(b"x-data-parallel-rank", b"1"), (b"x-dp-matrix-tag", b"\xff")],
        )
        for headers in cases:
            with self.subTest(headers=headers):
                messages = await self._request(headers)
                self.assertEqual(self._status(messages), 400)
        self.assertEqual(self.app.scopes, [])

    async def test_active_counter_window_requires_explicit_rank_and_matching_tag(self):
        tag = "dp1-window-你好"
        WINDOWS[1] = {"tag": tag, "budget": 256, "generation_started": False}

        cases = (
            [],
            [(b"x-dp-matrix-tag", tag.encode("utf-8"))],
            [(b"x-data-parallel-rank", b"1")],
            [(b"x-data-parallel-rank", b"1"), (b"x-dp-matrix-tag", b"wrong")],
            [(b"x-data-parallel-rank", b"2"), (b"x-dp-matrix-tag", tag.encode("utf-8"))],
            [(b"x-data-parallel-rank", b"1"), (b"x-dp-matrix-tag", b"\xff")],
        )
        for headers in cases:
            with self.subTest(headers=headers):
                messages = await self._request(headers)
                self.assertEqual(self._status(messages), 400)
        self.assertFalse(WINDOWS[1]["generation_started"])
        self.assertEqual(self.app.scopes, [])

        matching = [
            (b"x-data-parallel-rank", b"1"),
            (b"x-dp-matrix-tag", tag.encode("utf-8")),
        ]
        first = await self._request(matching)
        second = await self._request(matching)
        self.assertEqual(self._status(first), 200)
        self.assertEqual(self._status(second), 400)
        self.assertTrue(WINDOWS[1]["generation_started"])
        self.assertEqual(len(self.app.scopes), 1)

    async def test_unarmed_dp_cannot_generate_while_another_dp_window_is_armed(self):
        WINDOWS[1] = {"tag": "armed-dp1", "budget": 256, "generation_started": False}

        # An explicit rank alone must not turn an unarmed replica into a normal
        # quality route while any measured counter window is open.
        messages = await self._request([(b"x-data-parallel-rank", b"2")])
        self.assertEqual(self._status(messages), 400)
        self.assertEqual(self.app.scopes, [])
        self.assertFalse(WINDOWS[1]["generation_started"])

    async def test_different_dp_windows_can_run_concurrently_without_broadcast(self):
        tag1, tag3 = "dp1-concurrent-β", "dp3-concurrent-γ"
        WINDOWS[1] = {"tag": tag1, "budget": 256, "generation_started": False}
        WINDOWS[3] = {"tag": tag3, "budget": 256, "generation_started": False}
        self.app.delay = 0.001

        headers1 = [(b"x-data-parallel-rank", b"1"), (b"x-dp-matrix-tag", tag1.encode())]
        headers3 = [(b"x-data-parallel-rank", b"3"), (b"x-dp-matrix-tag", tag3.encode())]
        response1, response3 = await asyncio.gather(
            self._request(headers1), self._request(headers3)
        )

        self.assertEqual([self._status(response1), self._status(response3)], [200, 200])
        self.assertEqual(len(self.app.scopes), 2)
        observed = {}
        for scope in self.app.scopes:
            headers = self._scope_headers(scope)
            rank = next(value for key, value in headers if key == b"x-data-parallel-rank")
            self.assertEqual(len([value for key, value in headers if key == b"x-dp-matrix-tag"]), 1)
            observed[rank] = next(value for key, value in headers if key == b"x-dp-matrix-tag")
        self.assertEqual(observed, {b"1": tag1.encode(), b"3": tag3.encode()})
        self.assertTrue(WINDOWS[1]["generation_started"])
        self.assertTrue(WINDOWS[3]["generation_started"])
        for messages, expected_rank in ((response1, b"1"), (response3, b"3")):
            echoed = [
                value for key, value in self._response_headers(messages)
                if key.lower() == b"x-dp-matrix-rank"
            ]
            self.assertEqual(echoed, [expected_rank])

    async def test_generation_completes_only_after_successful_final_sse_send(self):
        tag = "dp0-complete-sse"
        WINDOWS[0] = {"tag": tag, "budget": 256, "generation_started": False}
        self.app = FakeASGIApp(body_messages=[
            {"body": b'data: {"choices":["partial"]}\n\n', "more_body": True},
            {"body": b"data: [DONE]\n\n", "more_body": False},
        ])
        self.middleware = DPRouteMiddleware(self.app)
        state_during_send = []

        async def inspect_send(message):
            if message["type"] == "http.response.body":
                state_during_send.append((message.get("more_body", False),
                                          WINDOWS[0]["generation_complete"]))

        messages = await self._request(
            [(b"x-data-parallel-rank", b"0"), (b"x-dp-matrix-tag", tag.encode())],
            send_hook=inspect_send,
        )

        self.assertEqual(self._status(messages), 200)
        self.assertEqual(state_during_send, [(True, False), (False, False)])
        self.assertEqual(WINDOWS[0]["generation_http_status"], 200)
        self.assertIs(WINDOWS[0]["generation_complete"], True)
        self.assertTrue(WINDOWS[0]["generation_complete"]
                        and WINDOWS[0]["generation_http_status"] == 200)

    async def test_partial_sse_does_not_admit_counter_metrics(self):
        tag = "dp2-partial-sse"
        WINDOWS[2] = {"tag": tag, "budget": 256, "generation_started": False}
        self.app = FakeASGIApp(body_messages=[
            {"body": b"data: partial\n\n", "more_body": True},
        ])
        self.middleware = DPRouteMiddleware(self.app)

        messages = await self._request(
            [(b"x-data-parallel-rank", b"2"), (b"x-dp-matrix-tag", tag.encode())]
        )

        self.assertEqual(self._status(messages), 200)
        self.assertEqual(WINDOWS[2]["generation_http_status"], 200)
        self.assertIs(WINDOWS[2]["generation_complete"], False)
        self.assertFalse(WINDOWS[2]["generation_complete"]
                         and WINDOWS[2]["generation_http_status"] == 200)

    async def test_failed_final_body_send_does_not_complete_counter_window(self):
        tag = "dp3-send-failure"
        WINDOWS[3] = {"tag": tag, "budget": 256, "generation_started": False}
        self.app = FakeASGIApp(body_messages=[
            {"body": b"data: partial\n\n", "more_body": True},
            {"body": b"data: [DONE]\n\n", "more_body": False},
        ])
        self.middleware = DPRouteMiddleware(self.app)
        sent = []

        async def fail_final_body(message):
            if (message["type"] == "http.response.body"
                    and not message.get("more_body", False)):
                raise RuntimeError("simulated downstream send failure")
            sent.append(message)

        with self.assertRaisesRegex(RuntimeError, "downstream send failure"):
            await self._request(
                [(b"x-data-parallel-rank", b"3"), (b"x-dp-matrix-tag", tag.encode())],
                send_hook=fail_final_body,
            )

        self.assertEqual(WINDOWS[3]["generation_http_status"], 200)
        self.assertIs(WINDOWS[3]["generation_complete"], False)
        self.assertFalse(WINDOWS[3]["generation_complete"]
                         and WINDOWS[3]["generation_http_status"] == 200)
        self.assertEqual(sum(row["type"] == "http.response.body" for row in sent), 1)

    async def test_non_200_complete_body_is_not_metrics_eligible(self):
        tag = "dp1-http-error"
        WINDOWS[1] = {"tag": tag, "budget": 256, "generation_started": False}
        self.app = FakeASGIApp(status=500)
        self.middleware = DPRouteMiddleware(self.app)

        messages = await self._request(
            [(b"x-data-parallel-rank", b"1"), (b"x-dp-matrix-tag", tag.encode())]
        )

        self.assertEqual(self._status(messages), 500)
        self.assertIs(WINDOWS[1]["generation_complete"], True)
        self.assertEqual(WINDOWS[1]["generation_http_status"], 500)
        self.assertFalse(WINDOWS[1]["generation_complete"]
                         and WINDOWS[1]["generation_http_status"] == 200)

    async def test_non_generation_scopes_are_passed_through(self):
        messages = await self._request(
            [(b"x-data-parallel-rank", b"3")],
            path="/dp/status",
            method="GET",
        )
        self.assertEqual(self._status(messages), 200)
        self.assertEqual(len(self.app.scopes), 1)
        self.assertEqual(self.app.scopes[0]["path"], "/dp/status")


if __name__ == "__main__":
    unittest.main()
