"""CPU-only tests for per-EngineCore data-parallel RPC routing."""
from __future__ import annotations

import asyncio
import unittest

from experiments.dp_matrix.native_rpc import NativeDPRPC
from experiments.dp_matrix.topology import Topology


def _rows(dp_rank: int, tp_size: int) -> list[dict[str, int]]:
    return [{"rank": rank, "dp_rank": dp_rank} for rank in range(tp_size)]


def _fake_client_type():
    """Give a pure-Python fake the exact pinned native client's type identity."""

    class FakeNativeClient:
        def __init__(self, dp_size: int, tp_size: int):
            self.engine_ranks_managed = list(range(dp_size))
            self.core_engines = [rank.to_bytes(2, "little") for rank in range(dp_size)]
            self.calls = []
            self.tp_size = tp_size
            self.response_factory = lambda dp_rank: _rows(dp_rank, tp_size)

        async def _call_utility_async(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            engine = kwargs.get("engine")
            if not isinstance(engine, bytes) or len(engine) != 2:
                return self.response_factory(-1)
            dp_rank = int.from_bytes(engine, "little")
            return self.response_factory(dp_rank)

    FakeNativeClient.__module__ = "vllm.v1.engine.core_client"
    FakeNativeClient.__name__ = "DPLBAsyncMPClient"
    return FakeNativeClient


class NativeDPRPCTests(unittest.TestCase):
    def setUp(self):
        self.topology = Topology.named("dp4-tp2")
        self.client = _fake_client_type()(self.topology.dp, self.topology.tp)
        self.rpc = NativeDPRPC(self.client, self.topology)

    def test_call_routes_to_exact_engine_without_broadcast(self):
        result = asyncio.run(
            self.rpc.call(
                2,
                "dp_capacity",
                args=("capacity-tag",),
                kwargs={"detail": True},
                timeout=19,
            )
        )

        self.assertEqual(result, _rows(2, self.topology.tp))
        self.assertEqual(len(self.client.calls), 1)
        args, kwargs = self.client.calls[0]
        self.assertEqual(
            args,
            ("collective_rpc", "dp_capacity", 19, ("capacity-tag",), {"detail": True}),
        )
        self.assertIs(kwargs["engine"], self.client.core_engines[2])

    def test_all_preserves_every_dp_replica_reply(self):
        results = asyncio.run(self.rpc.all("dp_describe"))

        self.assertEqual(len(results), self.topology.dp)
        self.assertEqual(results, [_rows(dp, self.topology.tp) for dp in range(self.topology.dp)])
        self.assertEqual(len(self.client.calls), self.topology.dp)
        self.assertEqual(
            [kwargs["engine"] for _, kwargs in self.client.calls],
            self.client.core_engines,
        )

    def test_rejects_wrong_native_client_type(self):
        class OtherClient:
            engine_ranks_managed = list(range(4))
            core_engines = [rank.to_bytes(2, "little") for rank in range(4)]

            async def _call_utility_async(self, *args, **kwargs):
                return []

        with self.assertRaisesRegex(ValueError, "actual internal native DP client"):
            NativeDPRPC(OtherClient(), self.topology)

    def test_rejects_incomplete_or_reordered_engine_identities(self):
        for attr, value in (
            ("engine_ranks_managed", [0, 1, 3]),
            ("engine_ranks_managed", [1, 0, 2, 3]),
            ("core_engines", [b"\x00\x00", b"\x01\x00", b"\x03\x00", b"\x02\x00"]),
            ("core_engines", [b"\x00\x00", b"\x01\x00", b"\x02\x00"]),
        ):
            client = _fake_client_type()(self.topology.dp, self.topology.tp)
            setattr(client, attr, value)
            with self.subTest(attr=attr, value=value), self.assertRaisesRegex(
                ValueError, "do not cover each local DP replica"
            ):
                NativeDPRPC(client, self.topology)

    def test_rejects_missing_pinned_utility_abi(self):
        client = _fake_client_type()(self.topology.dp, self.topology.tp)
        client._call_utility_async = None
        with self.assertRaisesRegex(ValueError, "utility ABI is unavailable"):
            NativeDPRPC(client, self.topology)

    def test_rejects_invalid_dp_rank_or_method_before_rpc(self):
        for dp_rank, method in (
            (-1, "dp_status"),
            (self.topology.dp, "dp_status"),
            (True, "dp_status"),
            (1.0, "dp_status"),
            (0, "not_a_worker_method"),
        ):
            with self.subTest(dp_rank=dp_rank, method=method), self.assertRaises(ValueError):
                asyncio.run(self.rpc.call(dp_rank, method))
        self.assertEqual(self.client.calls, [])

    def test_rejects_missing_extra_or_non_list_tp_results(self):
        bad_replies = (
            _rows(1, self.topology.tp - 1),
            _rows(1, self.topology.tp + 1),
            tuple(_rows(1, self.topology.tp)),
            {"ranks": _rows(1, self.topology.tp)},
        )
        for reply in bad_replies:
            self.client.response_factory = lambda _dp, reply=reply: reply
            with self.subTest(reply=reply), self.assertRaisesRegex(
                ValueError, "complete actual TP subgroup reply"
            ):
                asyncio.run(self.rpc.call(1, "dp_status"))

    def test_rejects_duplicate_missing_foreign_and_bool_tp_ranks(self):
        valid = _rows(1, self.topology.tp)
        bad_replies = []

        duplicate = [dict(row) for row in valid]
        duplicate[1]["rank"] = duplicate[0]["rank"]
        bad_replies.append(duplicate)

        missing = [dict(row) for row in valid]
        missing[1]["rank"] = 99
        bad_replies.append(missing)

        foreign_dp = [dict(row) for row in valid]
        foreign_dp[0]["dp_rank"] = 0
        bad_replies.append(foreign_dp)

        bool_dp = [dict(row) for row in valid]
        bool_dp[1]["dp_rank"] = True
        bad_replies.append(bool_dp)

        bool_tp = [dict(row) for row in valid]
        bool_tp[1]["rank"] = True
        bad_replies.append(bool_tp)

        for reply in bad_replies:
            self.client.response_factory = lambda _dp, reply=reply: reply
            with self.subTest(reply=reply), self.assertRaisesRegex(
                ValueError, "RPC replies differ from the addressed native DP/TP group"
            ):
                asyncio.run(self.rpc.call(1, "dp_status"))


if __name__ == "__main__":
    unittest.main()
