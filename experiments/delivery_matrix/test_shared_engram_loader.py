"""AST-execute only loader methods with CPU spies; no vendor/Torch imports."""
import ast
from collections import OrderedDict
import gc
import importlib.util
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from test_shared_engram import checkpoint, shared, KEY

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("shared_engram_builder_cpu", HERE / "build_shared_engram.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape, self.dtype, self.device = self.values.shape, self.values.dtype, NS(type="cpu")
    def reshape(self, shape): return Tensor(self.values.reshape(shape))


class FakeModule:
    def register_buffer(self, name, value): setattr(self, name, value)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="engram-loader-", dir=os.environ["TMPDIR"])
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.codes, self.scales = checkpoint(self.root)
        self.allocations = []
        torch = ModuleType("torch")
        def empty(*shape, dtype, **kwargs):
            self.allocations.append((shape, dtype))
            return Tensor(np.empty(shape, dtype=dtype))
        torch.empty, torch.zeros = empty, lambda size, dtype, **kwargs: Tensor(np.zeros(size, dtype=dtype))
        torch.device = lambda _: NS(type="cpu")
        torch.int8, torch.int64, torch.float32 = np.int8, np.int64, np.float32
        torch.frombuffer = lambda buffer, dtype: Tensor(np.frombuffer(buffer, dtype=dtype))
        self.torch = torch
        derived = builder.derive((HERE.parent.parent / "patches/files/engram_hbm.py").read_text())
        cls = next(node for node in ast.parse(derived).body if isinstance(node, ast.ClassDef) and node.name == "NodeShardedEngram")
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("__init__", "set_rows", "load_checkpoint")]
        node = ast.ClassDef(name=cls.name, bases=[ast.Name(id="FakeModule", ctx=ast.Load())],
                            keywords=[], body=methods, decorator_list=[])
        self.scope = {"FakeModule": FakeModule, "torch": torch, "nn": NS(Parameter=lambda value, **kwargs: value),
                      "os": os, "OrderedDict": OrderedDict, "_ENGRAM_DEVICE_TRIM_SHARD": False,
                      "_ENGRAM_DEVICE_INDEX": False, "_OFFLOAD_BUFFER_BYTES_LIMIT": 512 * 1024 * 1024,
                      "__package__": "delivery_matrix", "__name__": "delivery_matrix.engram_loader_cpu"}
        tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(tree, "best-derived-loader", "exec"), self.scope)
        self.cls = self.scope[cls.name]

    def construct(self, enabled="1", host="1", storage="int8"):
        with patch.dict(os.environ, {"V41_ENGRAM_SHARED_HOST_MMAP": enabled, "V41_ENGRAM_HOST_RESIDENT": host}):
            return self.cls(64, 256, NS(size=8, rank=3), storage_format=storage)

    def load(self, model):
        package = ModuleType("delivery_matrix")
        package.__path__ = [str(HERE)]
        with patch.dict(sys.modules, {"torch": self.torch, "delivery_matrix": package,
                                      "delivery_matrix.engram_shared_host": shared}):
            model.load_checkpoint(self.root, KEY)

    def cleanup_model(self, model):
        model.weight = model.weight_scale = None
        gc.collect()
        for mapping in model._shared_host_mappings: mapping.close()

    def test_constructor_skips_private_shard_then_binds_readonly_zero_copy_views(self):
        model = self.construct()
        self.assertEqual(model.weight.shape, (1, 256))
        self.assertEqual(model.weight_scale.shape, (1, 8))
        self.assertNotIn(((8, 256), np.int8), self.allocations)
        self.load(model)
        np.testing.assert_array_equal(model.weight.values, self.codes[24:32])
        np.testing.assert_array_equal(model.weight_scale.values, self.scales[24:32])
        self.assertFalse(model.weight.values.flags.writeable)
        self.assertEqual(len(model._shared_host_receipt), 2)
        self.cleanup_model(model)

    def test_unset_feature_retains_original_private_shape(self):
        model = self.construct(enabled="0")
        self.assertFalse(model.shared_host_mmap)
        self.assertEqual(model.weight.shape, (8, 256))
        self.assertEqual(model.weight_scale.shape, (8, 8))

    def test_device_index_and_missing_host_resident_are_rejected(self):
        with self.assertRaises(ValueError): self.construct(host="0")
        self.scope["_ENGRAM_DEVICE_INDEX"] = True
        with self.assertRaises(ValueError): self.construct()

    def test_mutating_loader_and_live_remap_are_rejected(self):
        model = self.construct()
        with self.assertRaises(RuntimeError): model.set_rows(0, self.codes[:1])
        self.load(model)
        with self.assertRaises(RuntimeError): self.load(model)
        self.cleanup_model(model)

    def test_only_storage_methods_change_and_base_drift_is_rejected(self):
        original = (HERE.parent.parent / "patches/files/engram_hbm.py").read_text()
        derived = builder.derive(original)
        def functions(source):
            cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "NodeShardedEngram")
            return {node.name: ast.dump(node) for node in cls.body if isinstance(node, ast.FunctionDef)}
        before, after = functions(original), functions(derived)
        changed = {name for name in before if before[name] != after[name]}
        self.assertEqual(changed, {"__init__", "set_rows", "load_checkpoint"})
        with self.assertRaises(ValueError): builder.derive(original + "\n# drift\n")


if __name__ == "__main__":
    unittest.main()
