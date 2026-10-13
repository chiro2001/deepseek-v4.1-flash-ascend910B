"""CPU address invariant/math/provenance controls; no vendor imports."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
package = ModuleType("delivery_matrix")
package.__path__ = [str(HERE)]
sys.modules.setdefault("delivery_matrix", package)

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

builder = load("ced_compressor_builder_cpu", "build_ced_compressor.py")
overlay = load("delivery_matrix.ced_compressor_overlay", "ced_compressor_overlay.py")
COMPONENT = HERE / "ced_compressor_20261013_v1"
CLAMP = "(seg_row_base + tl.maximum(seg_off, 0))[:, None] * HEAD_DIM"
RAW = "(seg_row_base + seg_off)[:, None] * HEAD_DIM"


class Pointer:
    def __init__(self, values): self.values = np.asarray(values).reshape(-1)
    def __add__(self, offsets): return Address(self, np.asarray(offsets))


class Address:
    def __init__(self, pointer, offsets): self.pointer, self.offsets = pointer, offsets
    def __add__(self, offsets): return Address(self.pointer, self.offsets + offsets)


class TritonSpy:
    constexpr, float32 = int, np.float32
    full, zeros, arange = staticmethod(np.full), staticmethod(np.zeros), staticmethod(np.arange)
    maximum, where, exp = staticmethod(np.maximum), staticmethod(np.where), staticmethod(np.exp)
    max, sum = staticmethod(np.max), staticmethod(np.sum)
    @staticmethod
    def load(address, mask=None, other=0):
        pointer, offsets = address.pointer, address.offsets
        # Strong address invariant: check even masked lanes, as the candidate
        # is specifically designed not to form pre-allocation DMA addresses.
        if offsets.min() < 0 or offsets.max() >= len(pointer.values):
            raise ValueError("A masked lane still formed an invalid projection address")
        values = pointer.values[offsets]
        return values if mask is None else np.where(np.broadcast_to(mask, offsets.shape), values, other)


def pooled(source):
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                and node.name == "_pooled_blocked")
    node.decorator_list = []
    scope = {"tl": TritonSpy}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "actual-pooled-blocked", "exec"), scope)
    return scope[node.name]


class CompressorTests(unittest.TestCase):
    def sources(self):
        derived = (COMPONENT / builder.RELATIVE).read_text()
        return derived.replace(CLAMP, RAW), derived

    def test_odd_tail_and_ratio_boundaries_keep_math_and_all_addresses_valid(self):
        original, derived = self.sources()
        self.assertEqual(builder.derive(original), derived)
        function = pooled(derived)
        rng = np.random.default_rng(419)
        for ratio in (2, 4, 8):
            for start in range(2041, 2049):
                with self.subTest(ratio=ratio, start=start):
                    group, dim, cache_size, cache_row = start // ratio, 4, 32, 1
                    residual = start % ratio
                    current_rows = ratio - residual
                    kv = rng.normal(size=(current_rows, dim)).astype(np.float32)
                    score = rng.normal(size=(current_rows, dim)).astype(np.float32)
                    ring = rng.normal(size=(2, cache_size, 2, dim)).astype(np.float32)
                    actual = function(group, start, 0, cache_row, Pointer(kv), Pointer(score), Pointer(ring),
                                      np.arange(dim), cache_size, dim, ratio, max(2, ratio),
                                      min(max(2, ratio), 8), bool(residual))
                    history_slots = np.arange(group * ratio, start) % cache_size
                    reference_kv = np.concatenate([ring[cache_row, history_slots, 0], kv])
                    reference_score = np.concatenate([ring[cache_row, history_slots, 1], score])
                    exponent = np.exp(reference_score - reference_score.max(axis=0))
                    reference = (exponent / exponent.sum(axis=0) * reference_kv).sum(axis=0)
                    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)

    def test_original_odd_ratio2_mask_is_not_a_safe_address_negative_control(self):
        original, _ = self.sources()
        with self.assertRaisesRegex(ValueError, "invalid projection address"):
            pooled(original)(1023, 2047, 0, 1, Pointer(np.ones((1, 4))), Pointer(np.ones((1, 4))),
                             Pointer(np.ones((2, 32, 2, 4))), np.arange(4), 32, 4, 2, 2, 2, True)

    def test_only_four_projection_addresses_change_not_masks_or_ring_math(self):
        before, after = self.sources()
        def functions(source):
            return {node.name: ast.dump(node) for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef)}
        old, new = functions(before), functions(after)
        self.assertEqual({name for name in old if old[name] != new[name]}, {"_pooled_blocked"})
        def loads(source):
            node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                        and node.name == "_pooled_blocked")
            return [node for node in ast.walk(node) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "load"]
        old_loads, new_loads = loads(before), loads(after)
        self.assertEqual([ast.dump(ast.Tuple(elts=x.keywords, ctx=ast.Load())) for x in old_loads],
                         [ast.dump(ast.Tuple(elts=x.keywords, ctx=ast.Load())) for x in new_loads])
        self.assertEqual(sum(ast.dump(x.args[0]) != ast.dump(y.args[0])
                             for x, y in zip(old_loads, new_loads)), 4)

    def test_component_tamper_and_base_drift_fail_closed(self):
        original, _ = self.sources()
        with self.assertRaises(ValueError): builder.derive(original + "\n# drift\n")
        with TemporaryDirectory(dir=os.environ["TMPDIR"]) as task:
            base, root = Path(task) / "actual.py", Path(task) / "candidate"
            base.write_text(original)
            builder.build(base, root)
            self.assertEqual(builder.verify(root)["source_sha256"], builder.BASE_SHA)
            target = root / builder.RELATIVE
            target.write_text(target.read_text().replace(CLAMP, RAW, 1))
            with self.assertRaises(ValueError): builder.verify(root)

    def test_d_only_preimport_selection_and_actual_sdk_gate(self):
        with self.assertRaises(ValueError): overlay.install(COMPONENT, "prefill")
        with patch.dict(sys.modules, {builder.MODULE: ModuleType(builder.MODULE)}):
            with self.assertRaises(RuntimeError): overlay.install(COMPONENT, "decode")
        finder = overlay.CompressorFinder(COMPONENT)
        with TemporaryDirectory(dir=os.environ["TMPDIR"]) as task:
            actual = Path(task) / "actual.py"
            actual.write_text(self.sources()[0])
            with patch.object(overlay.PathFinder, "find_spec", return_value=NS(origin=str(actual))):
                self.assertEqual(finder.find_spec(builder.MODULE).origin, str(COMPONENT / builder.RELATIVE))
                actual.write_text(actual.read_text() + "\n# drift\n")
                with self.assertRaises(ImportError): finder.find_spec(builder.MODULE)
        self.assertIsNone(finder.find_spec("vllm_ascend.models.deepseek_v41.model"))

    def test_selected_actual_source_is_reported_and_mismatch_rejected(self):
        from test_worker import stub
        with patch.dict(os.environ, {"DELIVERY_BASE_WORKER": stub.__name__ + ".Worker"}):
            worker_module = load("delivery_matrix.compressor_worker_cpu", "worker.py")
        distributed = ModuleType("vllm.distributed")
        distributed.get_tensor_model_parallel_rank = lambda: 3
        worker = worker_module.DeliveryWorker()
        worker.vllm_config = NS(load_config=NS(load_format="auto"),
                               model_config=NS(hf_text_config=NS(num_hidden_layers=40)),
                               parallel_config=NS(tensor_parallel_size=8, decode_context_parallel_size=1),
                               speculative_config=NS(num_speculative_tokens=5),
                               scheduler_config=NS(async_scheduling=True))
        worker.model_runner = NS(drafter=NS(use_cuda_graph=True))
        env = {"DELIVERY_FORM": "a3-ced-pd", "DELIVERY_CED_COMPRESSOR": str(COMPONENT),
               "DELIVERY_CED_COMPRESSOR_MANIFEST_SHA256": builder.sha(COMPONENT / "manifest.json")}
        with patch.dict(os.environ, env), patch.dict(sys.modules, {"vllm.distributed": distributed,
                builder.MODULE: NS(__file__=str(COMPONENT / builder.RELATIVE))}):
            actual = worker.delivery_describe()
            self.assertEqual(actual["ced_compressor_source_sha256"], builder.sha(COMPONENT / builder.RELATIVE))
            self.assertEqual(actual["ced_compressor_manifest_sha256"], env["DELIVERY_CED_COMPRESSOR_MANIFEST_SHA256"])
            with patch.dict(sys.modules, {builder.MODULE: NS(__file__=str(HERE / "worker.py"))}):
                with self.assertRaises(ValueError): worker.delivery_describe()


if __name__ == "__main__":
    unittest.main()
