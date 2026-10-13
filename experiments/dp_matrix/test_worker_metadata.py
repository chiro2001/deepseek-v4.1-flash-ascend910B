"""CPU metadata fixtures for loaded Engram inspection and group-aware KV accounting."""
import ast
import hashlib
import inspect
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
from collections.abc import Mapping


ROOT = Path.home() / "tmp" / "dsv41-dp-worker-metadata-tests"
ROOT.mkdir(parents=True, exist_ok=True)
SOURCE = Path(__file__).with_name("worker.py")


def source_file(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def method(name):
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DPWorker")
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = dict(Path=Path, sys=sys, source_file=source_file, re=re, os=os,
                 Mapping=Mapping, inspect=inspect, math=math,
                 backing_storages=lambda caches: [{"size_bytes": caches}])
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope[name]


def modules(names):
    result = {}
    for name in names:
        module = ModuleType(name)
        module.__path__ = []
        result[name] = module
        parent, _, attr = name.rpartition(".")
        if parent in result:
            setattr(result[parent], attr, module)
    return result


class Device:
    index = 0
    def __str__(self):
        return "cpu"


class WorkerMetadataTests(unittest.TestCase):
    def test_describes_actual_engram_objects_and_boolean_fastpath_semantics(self):
        fake = modules(["torch", "torch.distributed", "vllm_ascend", "vllm_ascend.ascend_config"])
        fake["torch.distributed"].get_process_group_ranks = lambda group: group
        fake["vllm_ascend.ascend_config"].get_ascend_config = lambda: NS(enable_engram=True, engram_storage="int8")
        query = NS(rank=0, size=8, group=list(range(8)), tp_group=list(range(4)), is_source=True)
        table = NS(query_group=query, weight=NS(shape=(4, 256), dtype="int8", device=Device()), storage_format="int8")
        config = NS(engram_layer_ids=[1, 14])
        model = NS(config=config, named_modules=lambda: [
            ("model.layers.1.engram.embed", table), ("model.layers.14.engram.embed", table)])
        fn = method("_engram_runtime_description")
        with patch.dict(sys.modules, fake), patch.dict(os.environ, {"V41_ENGRAM_DEVICE_INDEX": "0"}):
            row = fn(NS(model_runner=NS(model=model)))
        self.assertEqual(row["loaded_layer_ids"], [1, 14])
        self.assertTrue(row["enabled_in_actual_ascend_config"])
        self.assertFalse(row["device_index_fastpath_enabled"])
        self.assertFalse(row["tensor_data_read"])
        self.assertEqual(row["shards"][0]["engram_group_global_ranks"], list(range(8)))

    def test_enabled_engram_rejects_missing_actual_table_objects(self):
        fake = modules(["torch", "torch.distributed", "vllm_ascend", "vllm_ascend.ascend_config"])
        fake["vllm_ascend.ascend_config"].get_ascend_config = lambda: NS(enable_engram=True)
        model = NS(config=NS(engram_layer_ids=[1, 14]), named_modules=lambda: [])
        with patch.dict(sys.modules, fake), self.assertRaisesRegex(ValueError, "incomplete"):
            method("_engram_runtime_description")(NS(model_runner=NS(model=model)))

    def capacity_fixture(self, source, *, helper_demand=13):
        fake = modules(["vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.kv_cache_utils",
                        "vllm_ascend", "vllm_ascend.core", "vllm_ascend.core.deepseek_v41"])
        core = fake["vllm_ascend.core.deepseek_v41"]
        core.request_blocks = lambda runtime, groups: helper_demand
        core.pool_bytes_per_block = lambda groups: 1024
        fake["vllm.v1.core.kv_cache_utils"].get_max_concurrency_for_kv_cache_config = lambda runtime, cfg: 99 / helper_demand
        groups = []
        for demand in (10, 1, 2):
            leaf = NS(block_size=128, page_size_bytes=1024, store_on_host=False,
                      max_memory_usage_bytes=lambda runtime, d=demand: d * 1024)
            groups.append(NS(layer_names=["layer"], kv_cache_spec=NS(kv_cache_specs={"layer": leaf})))
        config = NS(num_blocks=100, kv_cache_groups=groups,
                    kv_cache_tensors=[NS(size=102400, shared_by=["layer"], offset=0, block_stride=1024)])
        runtime = NS(model_config=NS(max_model_len=196608), scheduler_config=NS(max_num_seqs=1),
                     cache_config=NS(kv_cache_memory_bytes=None), max_in_flight_tokens=8192)
        worker = NS(_dp_kv_cache_config=config, _dp_fields=lambda: {"dp_rank": 0, "tp_rank": 0},
                    vllm_config=runtime, model_runner=NS(kv_caches=102400, model_memory_usage=1))
        return fake, worker

    def test_capacity_uses_group_demand_null_block_and_actual_override(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            source = Path(temporary) / "patch_kv_cache_utils.py"
            source.write_text("# CPU source fixture\n")
            fake, worker = self.capacity_fixture(source)
            with patch.dict(sys.modules, fake), patch.object(inspect, "getsourcefile", return_value=str(source)):
                row = method("dp_capacity")(worker)
            self.assertEqual(row["request_blocks_by_group"], [10, 1, 2])
            self.assertEqual(row["request_blocks_at_max_model_len"], 13)
            self.assertTrue(row["null_block_reserved"])
            self.assertEqual(row["logical_token_capacity"], int(99 / 13 * 196608))
            self.assertFalse(row["long_request_validated"])

    def test_capacity_rejects_helper_that_disagrees_with_actual_groups(self):
        fake, worker = self.capacity_fixture(None, helper_demand=14)
        with patch.dict(sys.modules, fake), self.assertRaisesRegex(ValueError, "block accounting"):
            method("dp_capacity")(worker)


if __name__ == "__main__":
    unittest.main()
