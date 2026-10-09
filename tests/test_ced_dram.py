"""Offline contracts; real native/DMA and numerical checks remain hardware gates."""

from __future__ import annotations

import ast
import os
import unittest
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace as NS

from experimental.ced.dram.config import make_config
from experimental.ced.dram.contract import (
    alignment_unit, bound_store_keys, prefill_groups, tail_boundary,
)
from tools.ced_mock_decode import page_plan


ROOT = Path(__file__).resolve().parents[1]


def compile_class(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class GeometryTests(unittest.TestCase):
    def test_noncacheable_state_does_not_constrain_native_hash_alignment(self):
        path = ROOT / "a2/patches/kv8-offload-pool/offloading_config.py"
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "validate_group_hash_alignment")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        check = namespace[node.name]
        groups = [NS(tokens_per_block=128), NS(tokens_per_block=32), NS(tokens_per_block=128)]
        check(groups, 128, [1])
        with self.assertRaises(AssertionError):
            check(groups, 128)
        with self.assertRaises(AssertionError):
            check([NS(tokens_per_block=32)], 128, [])

    def test_shared_spec_does_not_exclude_encoder_groups(self):
        full = NS(block_size=128)
        recurrent = NS(block_size=32, prefix_cacheable=False)
        swa = NS(block_size=32, sliding_window=128)
        groups = [NS(kv_cache_spec=full, layer_names=["model.layers.20.attn"]),
                  NS(kv_cache_spec=recurrent, layer_names=["model.layers.0.state"])]
        for layer in (0, 4, 8, 12, 16, 20, 24, 28, 32, 36):
            groups.append(NS(kv_cache_spec=NS(kv_cache_specs={"s": swa}),
                             layer_names=[f"model.layers.{layer}.swa"]))
        result = prefill_groups(groups)
        self.assertEqual(result.participating, (0, 2, 3, 4, 5, 6))
        self.assertEqual(result.missing_swa, (7, 8, 9, 10, 11))
        self.assertEqual(result.recurrent, (1,))

    def test_full_hit_keeps_nonzero_aligned_tail_across_boundaries(self):
        for n in (1, 127, 128, 129, 1023, 1024, 1025, 2048, 2049):
            for unit in (128, 1024):
                boundary = tail_boundary(n, n, unit)
                self.assertLess(boundary, n)
                self.assertEqual(boundary % unit, 0)
                self.assertLessEqual(n - boundary, unit)
        self.assertEqual(alignment_unit([32, 128, 1024]), 1024)
        self.assertEqual(alignment_unit([96, 128]), 384)

    def test_backpressure_counts_shared_physical_blocks_once(self):
        selected, pinned = bound_store_keys(
            ["a", "b", "c"], {"a": {1, 2}, "b": {2, 3}, "c": {4}}, {1}, 3,
        )
        self.assertEqual(selected, ["a", "b"])
        self.assertEqual(pinned, {1, 2, 3})
        self.assertEqual(bound_store_keys(["c"], {"c": {4}}, {1, 2, 3}, 3)[0], [])

    def test_composite_is_p_producer_then_native_dram(self):
        children = make_config(19290, 2 << 30, 16384)["kv_connector_extra_config"]["connectors"]
        self.assertEqual(children[0]["kv_role"], "kv_producer")
        self.assertEqual(children[1]["kv_connector"], "CEDOffloadingConnector")
        with self.assertRaises(ValueError):
            make_config(65535, 1, 8)

    def test_mock_reads_partial_tail_but_compares_only_full_pages(self):
        blocks = [[] for _ in range(12)]
        blocks[0] = [5, 7]
        params = {"remote_block_ids": blocks, "ced_prefix_tokens": 129,
                  "ced_missing_swa_groups": [7, 8, 9, 10, 11]}
        metadata = {"mock_registered_regions": [(1000, 2048)], "mock_segments": [{"group": 0, "component": "kv",
                    "base": 1000, "stride": 256, "page_bytes": 128,
                    "num_blocks": 8, "tokens_per_block": 128, "prefix_cacheable": True}]}
        rows = list(page_plan(metadata, params))
        self.assertEqual([r["remote"] for r in rows], [2280, 2792])
        self.assertEqual([r["compare"] for r in rows], [True, False])
        blocks[0] = [8]
        with self.assertRaises(ValueError):
            list(page_plan(metadata, params))

    def test_mock_rejects_unregistered_payload_before_native_read(self):
        blocks = [[1]] + [[] for _ in range(11)]
        params = {"remote_block_ids": blocks, "ced_prefix_tokens": 128,
                  "ced_missing_swa_groups": [7, 8, 9, 10, 11]}
        metadata = {"mock_registered_regions": [(1000, 256)], "mock_segments": [{
            "group": 0, "component": "kv", "base": 1000, "stride": 256,
            "page_bytes": 128, "num_blocks": 2, "tokens_per_block": 128,
            "prefix_cacheable": True,
        }]}
        with self.assertRaisesRegex(ValueError, "registered"):
            list(page_plan(metadata, params))


class BarrierTests(unittest.TestCase):
    def setUp(self):
        @dataclass
        class Metadata:
            load_jobs: dict
            store_jobs: dict
            jobs_to_flush: set | None = None
            finished_store_requests: set = field(default_factory=set)

        class Native:
            def handle_preemptions(self, metadata):
                pass

            def get_finished(self, finished):
                return set(), set()

        namespace = {"AscendOffloadingConnector": Native, "CEDOffloadingMetadata": Metadata,
                     "os": os, "alignment_unit": alignment_unit, "prefill_groups": prefill_groups}
        cls = compile_class(ROOT / "experimental/ced/dram/connector.py", "CEDOffloadingConnector", namespace)
        self.connector = cls.__new__(cls)
        self.connector._ced_worker_barriers = set()
        self.connector._ced_job_requests = {}
        self.connector._ced_request_jobs = {}
        self.connector._ced_seen_metadata = None
        self.connector.connector_worker = NS(_connector_worker_meta=NS(completed_jobs={}))
        self.Metadata = Metadata

    def test_final_barrier_waits_for_all_stores_and_reports_once(self):
        c = self.connector
        early = self.Metadata({}, {1: NS(req_id="r")})
        c.handle_preemptions(early)
        c._connector_metadata = self.Metadata({}, {2: NS(req_id="r")}, finished_store_requests={"r"})
        self.assertEqual(c.get_finished({"r"})[0], set())
        c.connector_worker._connector_worker_meta.completed_jobs = {1: 1}
        self.assertEqual(c.get_finished(set())[0], set())
        c.connector_worker._connector_worker_meta.completed_jobs = {2: 1}
        self.assertEqual(c.get_finished(set())[0], {"r"})
        self.assertEqual(c.get_finished(set())[0], set())

    def test_final_barrier_with_no_stores_completes(self):
        c = self.connector
        c._connector_metadata = self.Metadata({}, {}, finished_store_requests={"r"})
        self.assertEqual(c.get_finished({"r"})[0], {"r"})

    def test_abort_barrier_waits_for_pending_load_destinations(self):
        c = self.connector
        c._connector_metadata = self.Metadata({4: NS(req_id="aborted")}, {})
        c.handle_preemptions(c._connector_metadata)
        c._connector_metadata = self.Metadata({}, {}, finished_store_requests={"aborted"})
        self.assertEqual(c.get_finished({"aborted"})[0], set())
        c.connector_worker._connector_worker_meta.completed_jobs = {4: 1}
        self.assertEqual(c.get_finished(set())[0], {"aborted"})


class AsyncResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = os.environ.get("CED_TEST_CORE")
        if not source:
            raise unittest.SkipTest("Set CED_TEST_CORE to the patched image scheduler for runtime method tests")
        tree = ast.parse(Path(source).read_text())
        scheduler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        method = next(n for n in scheduler.body if isinstance(n, ast.FunctionDef) and n.name == "_update_waiting_for_remote_kv")
        method.decorator_list = []
        namespace = {"os": os, "Request": object}
        exec(compile(ast.Module(body=[method], type_ignores=[]), source, "exec"), namespace)
        cls.method = staticmethod(namespace[method.name])

    def test_async_p_boundary_and_invalid_full_hit(self):
        saved = []
        request = NS(request_id="r", num_tokens=1025, num_computed_tokens=1024,
                     kv_transfer_params={"_ced_dram_load_boundary": 1024, "_ced_dram_alignment": 1024})
        scheduler = NS(connector=object(), failed_recving_kv_req_ids=set(),
                       finished_recving_kv_req_ids={"r"},
                       kv_cache_manager=NS(cache_blocks=lambda req, n: saved.append(n)))
        old = os.environ.get("V41_CED_ROLE")
        os.environ["V41_CED_ROLE"] = "prefill"
        try:
            self.method(scheduler, request)
            self.assertEqual(saved, [1024])
            self.assertEqual(request.num_computed_tokens, 1024)
            self.assertEqual(request.kv_transfer_params, {})
            request.kv_transfer_params = {"_ced_dram_load_boundary": 1025, "_ced_dram_alignment": 1024}
            request.num_computed_tokens = 1025
            with self.assertRaises(RuntimeError):
                self.method(scheduler, request)
            self.assertEqual(saved, [1024])
        finally:
            if old is None:
                os.environ.pop("V41_CED_ROLE", None)
            else:
                os.environ["V41_CED_ROLE"] = old


class CompositeCompletionTests(unittest.TestCase):
    def test_native_multi_waits_for_both_children_in_either_order(self):
        source = os.environ.get("CED_TEST_MULTI")
        if not source:
            self.skipTest("Set CED_TEST_MULTI to the image MultiConnector source")
        tree = ast.parse(Path(source).read_text())
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiConnector")
        methods = [n for n in original.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("get_finished", "_aggregate_request_finished")]
        node = ast.ClassDef(name="Multi", bases=[], keywords=[], body=methods, decorator_list=[])
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        namespace = {}
        module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
        exec(compile(module, source, "exec"), namespace)

        class Child:
            def __init__(self):
                self.done = set()

            def get_finished(self, finished):
                result, self.done = self.done, set()
                return result, set()

        for first in (0, 1):
            multi = namespace["Multi"]()
            multi._connectors = [Child(), Child()]
            multi._extra_async_saves = {}
            multi._requests_to_connector = {"r": 1}
            delayed, _ = multi._aggregate_request_finished(NS(request_id="r"), lambda c: (True, None))
            self.assertTrue(delayed)
            multi._connectors[first].done = {"r"}
            self.assertEqual(multi.get_finished({"r"}), (None, None))
            multi._connectors[1 - first].done = {"r"}
            self.assertEqual(multi.get_finished(set()), ({"r"}, None))
            self.assertEqual(multi.get_finished(set()), (None, None))


if __name__ == "__main__":
    unittest.main()
