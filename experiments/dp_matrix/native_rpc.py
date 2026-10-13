"""Preserve native DP routing while addressing each EngineCore explicitly."""
import asyncio

from .topology import integer


METHODS = {"dp_describe", "dp_status", "dp_capacity", "delivery_begin",
           "delivery_arm", "delivery_metrics"}


class NativeDPRPC:
    def __init__(self, client, topology):
        actual = type(client).__module__ + "." + type(client).__name__
        if actual != "vllm.v1.engine.core_client.DPLBAsyncMPClient":
            raise ValueError("The actual internal native DP client is required")
        ranks = list(range(topology.dp))
        engines = [rank.to_bytes(2, "little") for rank in ranks]
        if client.engine_ranks_managed != ranks or client.core_engines != engines:
            raise ValueError("Native engine identities do not cover each local DP replica")
        if not callable(getattr(client, "_call_utility_async", None)):
            raise ValueError("Pinned native per-engine utility ABI is unavailable")
        self.client, self.topology = client, topology

    async def call(self, dp_rank, method, *, args=(), kwargs=None, timeout=120):
        integer(dp_rank, "dp_rank")
        if dp_rank >= self.topology.dp or method not in METHODS:
            raise ValueError("Foreign DP rank or unsupported worker RPC")
        results = await self.client._call_utility_async(
            "collective_rpc", method, timeout, args, kwargs,
            engine=self.client.core_engines[dp_rank])
        if not isinstance(results, list) or len(results) != self.topology.tp:
            raise ValueError("A complete actual TP subgroup reply is required")
        expected = set(range(self.topology.tp))
        if (any(not isinstance(row, dict) for row in results)
                or {row.get("rank") for row in results} != expected
                or any(type(row.get("rank")) is not int for row in results)
                or any(type(row.get("dp_rank")) is not int for row in results)
                or any(row.get("dp_rank") != dp_rank for row in results)):
            raise ValueError("RPC replies differ from the addressed native DP/TP group")
        return results

    async def all(self, method, *, args=(), kwargs=None):
        # Unlike native call_utility_async, never discard DP1..DPn replies.
        return await asyncio.gather(*[
            self.call(rank, method, args=args, kwargs=kwargs)
            for rank in range(self.topology.dp)
        ])
