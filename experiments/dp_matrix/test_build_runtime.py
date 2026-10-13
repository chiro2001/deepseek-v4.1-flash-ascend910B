"""CPU-only tests for the bounded DP runtime derivation.

No torch, vLLM, NPU, HTTP, native extension, or service is imported here.
Temporary build products are kept under the task-specific TMPDIR (or the
documented ~/tmp fallback), never the system /tmp directory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

TASK_TMP_ROOT = Path(
    os.environ.get("TMPDIR")
    or (Path.home() / "tmp" / "dsv41-dp-build-runtime-tests")
)
TASK_TMP_ROOT.mkdir(parents=True, exist_ok=True)

from dp_matrix import build_runtime  # noqa: E402
from dp_matrix.topology import Topology  # noqa: E402


class _FakeDistributed:
    @staticmethod
    def get_backend(cpu_group):
        return "gloo" if cpu_group == "cpu-gloo" else "hccl"


class _FakeTPGroup:
    def __init__(self, world_size, rank, barrier):
        self.world_size = world_size
        self.rank_in_group = rank
        self.cpu_group = "cpu-gloo"
        self._barrier = barrier
        self.barrier_calls = 0

    def barrier(self):
        self.barrier_calls += 1
        self._barrier.wait(timeout=5)


class _FakeCounter:
    def __init__(self, rank, *, fenced=True, already_joined=False):
        self.rank = rank
        self.pending = None
        self._clock = 1000 + 10 * rank
        self.requests = {
            "request-1": {
                "rounds": [{
                    "terminal_device_synced": fenced,
                    "sample_end_ns": self._clock,
                    **({"terminal_tp_join": {}} if already_joined else {}),
                }]
            }
        }

    def now(self):
        self._clock += 1
        return self._clock


class BuildRuntimeTests(unittest.TestCase):
    def temporary_root(self):
        temporary = tempfile.TemporaryDirectory(
            prefix="dp-runtime-", dir=TASK_TMP_ROOT
        )
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def build_runtime_dir(self):
        output = self.temporary_root() / "runtime"
        manifest = build_runtime.build(output)
        return output, manifest

    def load_derived_counter(self, output):
        name = "_dp_matrix_counter_" + uuid.uuid4().hex
        path = output / "formal_dspark_counters.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            self.fail("Could not load the derived counter as a CPU-only module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def source(name):
        return (build_runtime.FROZEN / name).read_bytes()

    def test_exact_actual_source_hashes_are_required(self):
        for name in build_runtime.BASE_HASHES:
            with self.subTest(name=name):
                modified = self.source(name) + b"\n# changed source\n"
                with self.assertRaisesRegex(ValueError, "exact actual best async source"):
                    build_runtime.derive(name, modified)
        with self.assertRaisesRegex(ValueError, "exact actual best async source"):
            build_runtime.derive("unreviewed.py", b"source")

    def test_every_worker_and_counter_patch_anchor_is_pinned(self):
        cases = [("worker.py", build_runtime.WORKER_EDIT[0]),
                 ("worker.py", build_runtime.WORKER_PHYSICAL_EDIT[0])]
        cases.extend(("formal_dspark_counters.py", old)
                     for old, _new, _count in build_runtime.COUNTER_EDITS)
        for name, anchor in cases:
            with self.subTest(name=name, anchor=anchor[:48]):
                original = self.source(name)
                anchor_bytes = anchor.encode()
                self.assertIn(anchor_bytes, original)
                changed = original.replace(anchor_bytes, b"# altered derivation anchor\n", 1)
                digest = hashlib.sha256(changed).hexdigest()
                with patch.dict(build_runtime.BASE_HASHES, {name: digest}):
                    with self.assertRaisesRegex(ValueError, "Unexpected derivation anchor count"):
                        build_runtime.derive(name, changed)

    def test_build_verify_and_only_expected_contract_changes(self):
        output, manifest = self.build_runtime_dir()
        verified = build_runtime.verify(output)
        self.assertEqual(verified, manifest)
        self.assertFalse(manifest["actual_quality_passed"])
        self.assertFalse(manifest["service_started"])
        self.assertEqual(
            manifest["terminal_join_scope"],
            "actual active-DP TP group, not global EP",
        )

        worker = (output / "worker.py").read_text()
        self.assertNotIn("STACK_TP8_ARM", worker)
        self.assertNotIn("assert os.environ['STACK_PHYSICAL_CHIPS']=='8,9,10,11,12,13,14,15'", worker)
        self.assertIn("validate_worker_config(self.vllm_config, os.environ)", worker)
        for retained_quality_fence in (
            "assert cfg.hidden_size==5120 and cfg.num_hidden_layers==40",
            "assert cfg.n_routed_experts==384 and cfg.num_experts_per_tok==6",
            "assert spec is not None and spec.method=='dspark'",
            "assert os.environ['HCCL_DETERMINISTIC']=='strict'",
            "assert os.getenv('STACK_FP32_DECODE_REDUCTION')=='0'",
            "inspect_native_factory()",
            "route_capture_patch.install()",
            "formal_dspark_counters.install_runner(runner",
        ):
            with self.subTest(fence=retained_quality_fence):
                self.assertIn(retained_quality_fence, worker)
        counter = (output / "formal_dspark_counters.py").read_text()
        self.assertIn("Expected all actual active-DP TP ranks", counter)
        self.assertIn("CPU Gloo active-DP TP join (not a global EP barrier)", counter)
        self.assertIn("tp_group.world_size != tp_size", counter)
        self.assertIn("'world_size': tp_size", counter)
        self.assertIn("device_fence_preceded_entry", counter)
        self.assertIn("actual_draft_graph_observer", counter)

    def test_verify_rejects_tampered_source_and_manifest(self):
        output, _manifest = self.build_runtime_dir()
        source = output / "worker.py"
        source.write_bytes(source.read_bytes() + b"\n# tamper\n")
        with self.assertRaisesRegex(ValueError, "differs from the bounded derivation"):
            build_runtime.verify(output)

        output, _manifest = self.build_runtime_dir()
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["worker.py"]["bytes"] += 1
        manifest_path.write_text(json.dumps(manifest) + "\n")
        with self.assertRaisesRegex(ValueError, "differs from the bounded derivation"):
            build_runtime.verify(output)

    def test_verify_rejects_source_symlink_escape(self):
        output, _manifest = self.build_runtime_dir()
        outside = self.temporary_root() / "outside.py"
        outside.write_bytes((output / "worker.py").read_bytes())
        source = output / "worker.py"
        source.unlink()
        source.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes its pinned component"):
            build_runtime.verify(output)

    def test_cpu_group_barrier_runs_for_tp4_and_tp2(self):
        output, _manifest = self.build_runtime_dir()
        counter_module = self.load_derived_counter(output)
        for layout in ("dp2-tp4", "dp4-tp2"):
            topology = Topology.named(layout)
            with self.subTest(layout=layout), patch.dict(os.environ, {"DP_MATRIX_LAYOUT": layout}):
                barrier = threading.Barrier(topology.tp)
                groups = [_FakeTPGroup(topology.tp, rank, barrier)
                          for rank in range(topology.tp)]
                counters = [_FakeCounter(rank) for rank in range(topology.tp)]
                errors = []

                def join(rank):
                    try:
                        counter_module.terminal_tp_completion_join(
                            counters[rank], groups[rank], _FakeDistributed
                        )
                    except BaseException as exc:  # collect thread failures in the test thread
                        errors.append(exc)

                threads = [threading.Thread(target=join, args=(rank,))
                           for rank in range(topology.tp)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=6)
                self.assertFalse(any(thread.is_alive() for thread in threads))
                self.assertEqual(errors, [])
                for rank, (counter, group) in enumerate(zip(counters, groups)):
                    row = counter.requests["request-1"]["rounds"][-1]
                    receipt = row["terminal_tp_join"]
                    self.assertEqual(group.barrier_calls, 1)
                    self.assertEqual(receipt["backend"], "gloo")
                    self.assertEqual(receipt["world_size"], topology.tp)
                    self.assertEqual(receipt["rank"], rank)
                    self.assertLessEqual(row["sample_end_ns"], receipt["entered_ns"])

    def test_terminal_join_rejects_world8_foreign_rank_missing_fence_and_duplicate(self):
        output, _manifest = self.build_runtime_dir()
        counter_module = self.load_derived_counter(output)
        cases = (
            ("dp2-tp4", _FakeTPGroup(8, 0, threading.Barrier(1)), _FakeCounter(0),
             "actual active-DP CPU Gloo TP group"),
            ("dp2-tp4", _FakeTPGroup(4, 1, threading.Barrier(1)), _FakeCounter(0),
             "actual active-DP CPU Gloo TP group"),
            ("dp4-tp2", _FakeTPGroup(2, 0, threading.Barrier(1)), _FakeCounter(0, fenced=False),
             "Terminal join missing a device fence"),
            ("dp4-tp2", _FakeTPGroup(2, 0, threading.Barrier(1)), _FakeCounter(0, already_joined=True),
             "Terminal join missing a device fence"),
        )
        for layout, group, counter, message in cases:
            with self.subTest(layout=layout, world_size=group.world_size,
                              rank=group.rank_in_group,
                              fenced=counter.requests["request-1"]["rounds"][-1]["terminal_device_synced"]):
                with patch.dict(os.environ, {"DP_MATRIX_LAYOUT": layout}):
                    with self.assertRaisesRegex(ValueError, message):
                        counter_module.terminal_tp_completion_join(
                            counter, group, _FakeDistributed
                        )
            self.assertEqual(group.barrier_calls, 0)

    @staticmethod
    def async_receipt(rank, tp_size, *, join_world_size=None, join_rank=None, fenced=True):
        clock = {"clock": "CLOCK_MONOTONIC", "boot_id": "cpu-test", "time_namespace": "time:[1]"}
        sample_end = 110
        join = {
            "backend": "gloo",
            "world_size": tp_size if join_world_size is None else join_world_size,
            "rank": rank if join_rank is None else join_rank,
            "entered_ns": sample_end,
            "returned_ns": sample_end + 1,
            "device_fence_preceded_entry": True,
            "sample_timestamp_preceded_entry": True,
        }
        proposal = {"k": 5, "output_shape": [1, 5], "runtime_graph": True, "method": "dspark"}
        request = {
            "decode_start_ns": 100,
            "initial_draft": [],
            "rounds": [{
                "scheduled_tokens": 1,
                "scheduled_draft_token_ids": [-1],
                "draft_calls": [proposal],
                "sample_end_ns": sample_end,
                "bookkeeping_end_ns": 105,
                "sampler_called": True,
                "verified_draft_tokens": 5,
                "terminal_tp_join": join,
            }],
        }
        return {
            "rank": rank,
            "clock_scope": clock,
            "errors": [],
            "requests": {"request-1": request},
            "async_mode": True,
            "terminal_fence": {"sync_returned_ns": 115, "phase": "cpu-test"} if fenced else None,
            "zero_token_executions": [],
        }

    def test_async_summarizer_uses_active_tp_cardinality(self):
        output, _manifest = self.build_runtime_dir()
        counter_module = self.load_derived_counter(output)
        for layout in ("dp2-tp4", "dp4-tp2"):
            topology = Topology.named(layout)
            with self.subTest(layout=layout), patch.dict(os.environ, {"DP_MATRIX_LAYOUT": layout}):
                clock = {"clock": "CLOCK_MONOTONIC", "boot_id": "cpu-test", "time_namespace": "time:[1]"}
                receipts = [self.async_receipt(rank, topology.tp) for rank in range(topology.tp)]
                summary = counter_module.summarize(
                    receipts, [10, 11], 120, clock, draft_graph=True
                )
                self.assertEqual(len(summary["rank_counters"]), topology.tp)
                self.assertTrue(summary["terminal_tp_completion_join_observed"])

                bad_world = [self.async_receipt(
                    rank, topology.tp, join_world_size=8
                ) for rank in range(topology.tp)]
                if topology.tp != 8:
                    with self.assertRaisesRegex(ValueError, "terminal completion join"):
                        counter_module.summarize(bad_world, [10, 11], 120, clock, draft_graph=True)

                bad_rank = [self.async_receipt(rank, topology.tp) for rank in range(topology.tp)]
                bad_rank[-1] = self.async_receipt(
                    topology.tp - 1, topology.tp, join_rank=topology.tp
                )
                with self.assertRaisesRegex(ValueError, "terminal completion join"):
                    counter_module.summarize(bad_rank, [10, 11], 120, clock, draft_graph=True)

                with self.assertRaisesRegex(ValueError, "terminal device fence"):
                    unfenced = [self.async_receipt(rank, topology.tp, fenced=False)
                                for rank in range(topology.tp)]
                    counter_module.summarize(unfenced, [10, 11], 120, clock, draft_graph=True)

                with self.assertRaisesRegex(ValueError, "Expected all actual active-DP TP ranks"):
                    counter_module.summarize(
                        [self.async_receipt(rank, topology.tp) for rank in range(8)],
                        [10, 11], 120, clock, draft_graph=True,
                    )


if __name__ == "__main__":
    unittest.main()
