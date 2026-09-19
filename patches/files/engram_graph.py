# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the device Engram step from inside ACLGraph, one graph per batch shape.

`prepare_engram_inputs()` runs *before* `run_model()`, i.e. outside the model's
ACLGraph capture, so the device path is issued **eagerly**.  Measured on A3-node1
that costs ~1.5 ms of pure launch overhead for the hash alone, and the cost is
independent of batch size (1.587 / 1.507 / 1.513 ms for n = 6 / 32 / 192) --
the signature of ~50 tiny ops: nothing is compute-bound, everything is
dispatch-bound.  Captured, the same work replays in 0.405 ms (0.285 ms of which
is the fixed ACLGraph floor).

**Zero-copy by construction.** A vLLM step's inputs already live in persistent
buffers (``input_ids.gpu``, ``positions``, ``query_start_loc.gpu`` and the block
table), so the graph is captured directly over *those* tensors and no per-step
copy exists.  An earlier version copied the step's values into private buffers
instead; that looked safer but was strictly worse: the copies are host-to-device,
so they block until the device queue drains, and the cost then grows with the
batch (measured `route` = 2.0 ms at n = 6 but 5.4 ms at n = 24, versus 0.35 ms
when nothing is copied).

The price of zero copies is that a graph latches addresses.  So the cache records
every input's ``data_ptr`` at capture time and re-checks it on every replay; a
mismatch falls back to eager rather than to a silently wrong lookup.

The token -> request mapping is computed *inside* the graph with a
`searchsorted` over the (persistent) ``query_start_loc`` slice, so even that
needs no host round trip.
"""

from __future__ import annotations

import torch

__all__ = ["EngramGraphCache"]


def _is_capturing() -> bool:
    """True when the current stream is already being captured.

    Nested capture is illegal, so the cache must stay eager in that case.
    """
    try:
        return bool(torch.npu.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 - older torch_npu has no such helper
        return False


class _Bucket:
    """The captured graph plus the identity of the tensors it reads."""

    __slots__ = ("size", "n_reqs", "ptrs", "graph", "lookups", "mask", "captured")

    def __init__(self, size, n_reqs, ptrs):
        self.size = int(size)
        self.n_reqs = int(n_reqs)
        self.ptrs = tuple(ptrs)
        self.graph = torch.npu.NPUGraph()
        self.lookups = None
        self.mask = None
        self.captured = False


class EngramGraphCache:
    """One ACLGraph per batch size for the whole device Engram step.

    ``run`` is a drop-in replacement for the eager
    ``device_engram_lookup`` call: same arguments, same return values.
    """

    def __init__(
        self,
        hash_impl,
        tables,
        layer_ids,
        device,
        max_graphs: int = 16,
        max_tokens: int = 192,
    ):
        self.hash_impl = hash_impl
        self.tables = tables
        self.layer_ids = tuple(layer_ids)
        self.device = torch.device(device)
        self.max_graphs = int(max_graphs)
        # Only decode-scale batches get a graph.  Prefill is ~1.1 s, so its 2.5 ms
        # of launch overhead does not matter, while its outputs are huge
        # (2 layers x 8192 x 6144 bf16 = 201 MB per graph) and would eat HBM.
        self.max_tokens = int(max_tokens)
        self._buckets: dict[tuple[int, int, int], _Bucket] = {}
        self._order: list[tuple[int, int, int]] = []
        self.eager_fallbacks = 0
        self.graph_replays = 0
        self.captures = 0
        # Reason for the last fallback, so a silent regression is visible.
        self.last_fallback_reason = ""
        # Diagnostics: how often the harness had to fall back, so a silent
        # regression cannot hide behind a "it works" report.
        self.last_size: int | None = None
        # Capturing while a capture is already in progress is not allowed, and
        # `prepare_engram_inputs` is *called* from outside the model's capture —
        # but a future refactor could move it inside, so guard explicitly.
        self.skipped_inside_capture = 0
        # A zero-copy capture is only valid while the inputs keep their
        # addresses; this counts the times they did not.
        self.pointer_mismatches = 0

    # ------------------------------------------------------------------ eager
    @torch.no_grad()
    def _eager(self, input_ids, positions, start_loc, block_table):
        from .engram_device_index import device_engram_lookup

        self.eager_fallbacks += 1
        return device_engram_lookup(
            self.hash_impl, self.tables, self.layer_ids,
            input_ids, positions, start_loc, block_table,
        )

    # ---------------------------------------------------------------- capture
    @torch.no_grad()
    def _capture(self, size, n_reqs, ids, pos, start_loc, block_table) -> _Bucket:
        ptrs = (
            ids.data_ptr(), pos.data_ptr(), start_loc.data_ptr(),
            block_table.data_ptr(), block_table.shape,
        )
        b = _Bucket(size, n_reqs, ptrs)

        def node():
            # Token -> request, on device, from the persistent start_loc slice.
            req = torch.searchsorted(
                start_loc[: n_reqs + 1],
                torch.arange(size, dtype=start_loc.dtype, device=self.device),
                right=True,
            ).to(torch.int64) - 1
            hashes, mask = self.hash_impl(ids, pos, req, block_table)
            lookups = {
                lid: self.tables[lid].lookup(hashes[:, slot])
                for slot, lid in enumerate(self.layer_ids)
            }
            return lookups, mask

        with torch.npu.graph(b.graph):
            b.lookups, b.mask = node()
        b.captured = True
        # A capture only records work, so replay once to fill the outputs.
        #
        # Do NOT synchronize here.  This runs in the middle of a serving step,
        # and a device-wide synchronize at that point breaks the engine's
        # `device_metadata` submit/release pairing: the next step then dies with
        # "The previous device metadata submission has not been released"
        # (the defect documented in DMQ-LEAK.md, but here triggered by us).
        # Stream order already guarantees the model sees the replayed values.
        b.graph.replay()
        self.captures += 1
        return b

    # ------------------------------------------------------------------- run
    @torch.no_grad()
    def run(self, input_ids, positions, start_loc, block_table):
        """All four arguments must be **device tensors backed by persistent
        buffers**; that is what makes the zero-copy capture valid."""
        n = int(input_ids.numel())
        if n <= 0 or n > self.max_tokens:
            self.last_fallback_reason = f"n={n} exceeds max_tokens={self.max_tokens}"
            return self._eager(input_ids, positions, start_loc, block_table)
        n_reqs = int(start_loc.numel()) - 1
        # The metadata's request axis must match query_start_loc, or the
        # per-request page lookup would be misaligned.
        if n_reqs <= 0 or block_table.shape[0] != n_reqs:
            self.last_fallback_reason = (
                f"block_table rows {block_table.shape[0]} != n_reqs {n_reqs}"
            )
            return self._eager(input_ids, positions, start_loc, block_table)

        # The key carries the full shape identity: the same token count can come
        # from a different batch layout (n=12 as 2 requests x 6 or 12 x 1), and
        # the captured graph has those shapes baked in.
        key = (n, n_reqs, int(block_table.shape[1]))
        b = self._buckets.get(key)
        if b is None:
            if len(self._buckets) >= self.max_graphs:
                self.last_fallback_reason = f"graph cache full ({self.max_graphs})"
                return self._eager(input_ids, positions, start_loc, block_table)
            if _is_capturing():
                self.skipped_inside_capture += 1
                self.last_fallback_reason = "already inside a capture"
                return self._eager(input_ids, positions, start_loc, block_table)
            try:
                b = self._capture(n, n_reqs, input_ids, positions, start_loc, block_table)
            except Exception as exc:  # noqa: BLE001 - never break the model over this
                self.last_fallback_reason = f"capture failed: {type(exc).__name__}: {exc}"
                return self._eager(input_ids, positions, start_loc, block_table)
            self._buckets[key] = b
            self._order.append(key)

        # The graph reads specific addresses, so verify them on every replay.
        # This pointer comparison is the entire safety net for the zero-copy
        # design, and it costs nothing measurable.
        got = (
            input_ids.data_ptr(), positions.data_ptr(), start_loc.data_ptr(),
            block_table.data_ptr(), block_table.shape,
        )
        if got != b.ptrs:
            self.pointer_mismatches += 1
            self.last_fallback_reason = f"input buffers changed for key {key}"
            return self._eager(input_ids, positions, start_loc, block_table)
        b.graph.replay()
        self.graph_replays += 1
        self.last_size = n
        return {lid: v for lid, v in b.lookups.items()}, b.mask

    # ----------------------------------------------------------------- stats
    def stats(self) -> str:
        return (
            f"graphs={len(self._buckets)} replays={self.graph_replays} "
            f"captures={self.captures} eager_fallbacks={self.eager_fallbacks} "
            f"ptr_mismatch={self.pointer_mismatches} "
            f"keys={sorted(self._buckets)}"
            + (f" last_fallback='{self.last_fallback_reason}'" if self.last_fallback_reason else "")
        )
