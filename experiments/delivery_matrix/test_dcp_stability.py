"""CPU-only tensor/collective spies; no vendor imports or NPU execution."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import shutil
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
reduce = load("dcp_reduction_cpu", "dcp_reduction.py")
builder = load("dcp_stability_cpu", "build_dcp_stability.py")


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape, self.dtype, self.device = self.values.shape, self.values.dtype, "cpu-spy"
    def element_size(self): return self.values.dtype.itemsize
    def numel(self): return self.values.size
    def contiguous(self): return Tensor(np.ascontiguousarray(self.values))
    def reshape(self, *shape): return Tensor(self.values.reshape(*shape))
    view = reshape
    def __getitem__(self, key): return Tensor(self.values[key])
    def __add__(self, other): return Tensor(self.values + other.values)
    def copy_(self, other): self.values[...] = other.values; return self
    def zero_(self): self.values.fill(0); return self


class CollectiveSpy:
    def __init__(self, ranks, chunks):
        self.ranks, self.chunks, self.calls, self.peak_bytes = ranks, chunks, 0, 0
        self.source_buffers, self.output_buffers = set(), set()
    def gather(self, output, source, group):
        start, stop, count, padded = self.chunks[self.calls]
        self.calls += 1
        self.peak_bytes = max(self.peak_bytes, output.values.nbytes)
        self.source_buffers.add(id(source.values))
        self.output_buffers.add(id(output.values))
        if source.values.nbytes % 512 or output.values.ndim != 1:
            raise AssertionError("Collective input must be aligned and flattened")
        np.testing.assert_array_equal(source.values[:count], self.ranks[0][start:stop].reshape(-1))
        np.testing.assert_array_equal(source.values[count:], 0)
        result = output.values.reshape(len(self.ranks), source.values.size)
        result.fill(0)
        for rank, values in enumerate(self.ranks):
            result[rank, :count] = values[start:stop].reshape(-1)


def fake_torch(spy):
    module = ModuleType("torch")
    module.empty = lambda shape, dtype, device: Tensor(np.empty(shape, dtype=dtype))
    module.zeros = lambda shape, dtype, device: Tensor(np.zeros(shape, dtype=dtype))
    module.distributed = NS(all_gather_into_tensor=spy.gather)
    return module


class StableReductionTests(unittest.TestCase):
    def test_bounded_chunks_equal_fixed_rank_reference_with_tail(self):
        ranks = [np.random.default_rng(rank).normal(size=(137, 3, 5)).astype(np.float32) for rank in range(8)]
        budget = 8 * 512 * 3
        chunks = reduce.plan_chunks(ranks[0].shape, 4, 8, budget, 64)
        spy = CollectiveSpy(ranks, chunks)
        with patch.dict(sys.modules, {"torch": fake_torch(spy)}):
            result = reduce.ordered_allreduce(Tensor(ranks[0]), NS(world_size=8, device_group="group"), budget)
        reference = ranks[0].copy()
        for rank in ranks[1:]: reference = reference + rank
        np.testing.assert_array_equal(result.values, reference)
        self.assertLessEqual(spy.peak_bytes, budget)
        self.assertGreater(spy.calls, 1)
        self.assertEqual(len(spy.source_buffers), 1)
        self.assertEqual(len(spy.output_buffers), 1)
        self.assertEqual(chunks[-1][1], 137)

    def test_cancellation_negative_control_distinguishes_rank_order(self):
        ranks = [np.full((1, 1, 1), value, np.float32) for value in (1e20, 3, -1e20)]
        chunks = reduce.plan_chunks((1, 1, 1), 4, 3)
        spy = CollectiveSpy(ranks, chunks)
        with patch.dict(sys.modules, {"torch": fake_torch(spy)}):
            result = reduce.ordered_allreduce(Tensor(ranks[0]), NS(world_size=3, device_group="group"))
        self.assertEqual(result.values.item(), 0)
        self.assertEqual(((ranks[0] + ranks[2]) + ranks[1]).item(), 3)

    def test_max_prefill_geometry_has_bounded_aligned_gather(self):
        chunks = reduce.plan_chunks((8192, 64, 640), 4, 8)
        self.assertEqual(len(chunks), 128)
        self.assertTrue(all(8 * padded * 4 <= reduce.MAX_GATHER_BYTES for _, _, _, padded in chunks))
        self.assertEqual(max(8 * padded * 4 for _, _, _, padded in chunks), 80 * 1024 * 1024)

    def test_unfurled_pack_and_bad_budgets_fail_closed(self):
        chunks = reduce.plan_chunks((8192, 64, 1026), 4, 8)
        self.assertTrue(all(8 * padded * 4 <= reduce.MAX_GATHER_BYTES for _, _, _, padded in chunks))
        with self.assertRaises(ValueError): reduce.plan_chunks((1, 64, 640), 4, 8, 1024)
        with self.assertRaises(ValueError): reduce.plan_chunks((1, 64, 640), 4, 8, True)
        self.assertEqual(reduce.plan_chunks((0, 64, 640), 4, 8), [])

    def test_all_five_gates_use_one_env_or_file_predicate(self):
        source = (HERE / "dcp_runtime_20261013_v3/vllm_ascend/attention/dsa_v41.py").read_text()
        derived = builder.derive(source)
        tree = ast.parse(derived)
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_v41_det_reduce_enabled")
        merge = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_v41_dcp_merge_attention")
        guards = [node.test for node in ast.walk(merge) if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name)
                  and node.test.func.id == "_v41_det_reduce_enabled"]
        self.assertEqual(len(guards), 5)
        for env, file, expected in ((False, {}, False), (True, {}, True), (False, {"det_reduce": "1"}, True)):
            scope = {"_DCP_DET_REDUCE": env, "_perf_flags": lambda: file}
            exec(compile(ast.Module(body=[helper], type_ignores=[]), "predicate", "exec"), scope)
            self.assertEqual(scope["_v41_det_reduce_enabled"](), expected)
        # The old denominator-only file guard is the precise coverage mutant.
        mutant = derived.replace("if _v41_det_reduce_enabled():", 'if _perf_flags().get("det_reduce") == "1":', 1)
        changed_merge = next(node for node in ast.parse(mutant).body if isinstance(node, ast.FunctionDef) and node.name == merge.name)
        self.assertNotEqual(sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                                and node.func.id == "_v41_det_reduce_enabled" for node in ast.walk(changed_merge)), 5)

    def test_all_branches_are_bounded_and_kernel_keeps_resident_address(self):
        source = (HERE / "dcp_runtime_20261013_v3/vllm_ascend/attention/dsa_v41.py").read_text()
        derived = builder.derive(source)
        builder.verify_bounded_branches(derived)
        merge = next(node for node in ast.parse(derived).body if isinstance(node, ast.FunctionDef)
                     and node.name == "_v41_dcp_merge_attention")
        branches = [node for node in ast.walk(merge) if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name)
                    and node.test.func.id == "_v41_det_reduce_enabled"]
        calls = [node for branch in branches for statement in branch.body for node in ast.walk(statement)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "_v41_ordered_allreduce"]
        self.assertEqual(len(calls), 8)
        for payload in ("_pk", "_pack"):
            branch = next(branch for branch in branches
                          if any(isinstance(node, ast.Name) and node.id == payload
                                 for statement in branch.body for node in ast.walk(statement)))
            original = Tensor(np.arange(18, dtype=np.float32).reshape(2, 3, 3))
            scope = {payload: original, "group": "actual-group"}
            seen = []
            def ordered(value, group):
                seen.append((value, group))
                return Tensor(value.values + 100)
            scope["_v41_ordered_allreduce"] = ordered
            exec(compile(ast.Module(body=branch.body, type_ignores=[]), "bounded-body", "exec"), scope)
            self.assertEqual(seen, [(original, "actual-group")])
            self.assertEqual(scope[payload] is original, payload == "_pk")
            np.testing.assert_array_equal(scope[payload].values,
                                          np.arange(18, dtype=np.float32).reshape(2, 3, 3) + 100)
        for payload in ("_pk", "_pack"):
            pattern = ("_pk.copy_(_v41_ordered_allreduce(_pk, group))" if payload == "_pk"
                       else "_pack = _v41_ordered_allreduce(_pack, group)")
            mutant = derived.replace(pattern,
                                     f"torch.distributed.all_gather_into_tensor({payload}, {payload}, group=group)")
            with self.assertRaises(ValueError): builder.verify_bounded_branches(mutant)

    def test_source_drift_is_rejected_and_original_overlay_stays_frozen(self):
        original = HERE / "dcp_runtime_20261013_v3"
        original_dsa = original / "vllm_ascend/attention/dsa_v41.py"
        original_manifest = original / "manifest.json"
        source = original_dsa.read_text()
        with self.assertRaises(ValueError): builder.derive(source + "\n# drift\n")
        before_manifest = builder.sha(original_manifest)
        before_dsa = builder.sha(original_dsa)
        with TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="dcp-stability-") as task:
            base = Path(task) / "base"
            dsa_relative = "vllm_ascend/attention/dsa_v41.py"
            fixture_dsa = base / dsa_relative
            fixture_dsa.parent.mkdir(parents=True)
            shutil.copy2(original_dsa, fixture_dsa)
            manifest = {
                "schema": "delivery_dcp_overlay_v2",
                "files": {dsa_relative: {"sha256": builder.sha(fixture_dsa)}},
                "modules": {"vllm_ascend.attention.dsa_v41": dsa_relative},
            }
            fixture_manifest = base / "manifest.json"
            fixture_manifest.write_text(json.dumps(manifest))
            before_fixture_manifest = builder.sha(fixture_manifest)
            before_fixture_dsa = builder.sha(fixture_dsa)
            result = builder.build(base, Path(task) / "candidate")
            self.assertFalse(result["actual_quality_passed"])
            self.assertFalse(result["service_launch_requested"])
            self.assertTrue(builder.verify_candidate(Path(task) / "candidate")["all_five_reduction_gates_covered"])
            self.assertEqual(builder.sha(fixture_manifest), before_fixture_manifest)
            self.assertEqual(builder.sha(fixture_dsa), before_fixture_dsa)
        self.assertEqual(builder.sha(original_manifest), before_manifest)
        self.assertEqual(builder.sha(original_dsa), before_dsa)

    def test_old_partial_unbounded_candidate_cannot_enable_software_ordering(self):
        with self.assertRaises(ValueError):
            builder.verify_candidate(HERE / "dcp_runtime_20261013_v3")
        with self.assertRaises(ValueError):
            builder.verify_candidate(HERE / "dcp_runtime_20261013_v5")


if __name__ == "__main__":
    unittest.main()
