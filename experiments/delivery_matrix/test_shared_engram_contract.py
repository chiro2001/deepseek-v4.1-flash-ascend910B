"""CPU backing-pointer/16-rank identity tests; no vendor or Torch imports."""
import copy
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from test_shared_engram import checkpoint, shared, KEY

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("shared_contract_cpu", HERE / "shared_engram_contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
COMPONENT = HERE / "shared_engram_20261013_v1"


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape, self.dtype = values.shape, values.dtype
        self.device, self.requires_grad = NS(type="cpu"), False
    def data_ptr(self): return self.values.__array_interface__["data"][0]


class Table:
    pass


class SharedContractTests(unittest.TestCase):
    def setUp(self):
        self.task = TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="shared-contract-")
        self.addCleanup(self.task.cleanup)
        self.root = Path(self.task.name)
        checkpoint(self.root)
        self.tables = []

    def table(self, rank):
        table = Table()
        table.rows, table.width, table.start, table.end = 64, 256, rank * 8, (rank + 1) * 8
        table.shared_host_mmap, table.device_index = True, False
        maps = shared.map_checkpoint_rows(self.root, KEY, 64, 256, table.start, table.end)
        for mapping in maps:
            def torch_view(m=mapping):
                dtype = np.int8 if m.dtype == "I8" else np.float32
                return Tensor(np.frombuffer(m.buffer(), dtype=dtype).reshape(m.shape))
            mapping.torch_view = torch_view
        table._shared_host_mappings = maps
        table._shared_host_receipt = [m.receipt for m in maps]
        table.weight, table.weight_scale = [m.torch_view() for m in maps]
        self.tables.append(table)
        return table

    def describe(self, table):
        modules = {contract.MODULE: NS(__file__=str(COMPONENT / contract.FILES[contract.MODULE]), NodeShardedEngram=Table),
                   contract.HELPER: NS(__file__=str(COMPONENT / contract.FILES[contract.HELPER]), SharedTensorRows=shared.SharedTensorRows)}
        with patch.dict(sys.modules, modules):
            return contract.describe_tables(NS(named_modules=lambda: [("table", table)]),
                                            COMPONENT, contract.MANIFEST_SHA)

    def tearDown(self):
        for table in self.tables:
            table.weight = table.weight_scale = None
        gc.collect()
        for table in self.tables:
            for mapping in table._shared_host_mappings:
                mapping.close()

    def pair(self):
        rows = [{"rank": rank, "shared_engram": self.describe(self.table(rank))} for rank in range(8)]
        return ({"role": "prefill", "rank_descriptions": rows},
                {"role": "decode", "rank_descriptions": copy.deepcopy(rows)})

    def test_pinned_component_and_actual_weight_scale_aliasing(self):
        self.assertEqual(contract.verify_component(COMPONENT)["source_sha256"], contract.BASE_SHA)
        result = self.describe(self.table(3))
        self.assertTrue(result["actual_tensor_mmap_aliasing_passed"])
        self.assertFalse(result["actual_PSS_capacity_or_quality_passed"])
        self.assertEqual(result["tables"][0]["mappings"][0]["row_start"], 24)

    def test_same_shaped_private_clone_is_rejected(self):
        table = self.table(0)
        table.weight = Tensor(table.weight.values.copy())
        with self.assertRaisesRegex(ValueError, "does not alias"):
            self.describe(table)

    def test_missing_actual_module_selection_or_device_storage_fails(self):
        table = self.table(0)
        with patch.dict(sys.modules, {contract.MODULE: None}):
            with self.assertRaises(ValueError):
                contract.describe_tables(NS(named_modules=lambda: [("x", table)]), COMPONENT, contract.MANIFEST_SHA)
        table.device_index = True
        with self.assertRaises(ValueError): self.describe(table)
        with self.assertRaises(ValueError):
            contract.describe_tables(NS(named_modules=lambda: []), COMPONENT, "a" * 64)

    def test_source_manifest_and_payload_tamper_are_rejected(self):
        root = self.root / "component"
        root.mkdir()
        (root / "manifest.json").write_bytes((COMPONENT / "manifest.json").read_bytes())
        with self.assertRaises(FileNotFoundError): contract.verify_component(root)
        (root / "manifest.json").write_text(json.dumps({"schema": "delivery_shared_engram_component_v1"}))
        with self.assertRaises(ValueError): contract.verify_component(root)

    def test_sixteen_rank_same_file_ranges_and_complete_coverage(self):
        p, d = self.pair()
        result = contract.verify_pair_shared_tables(p, d)
        self.assertTrue(result["PD_shared_file_and_range_contract_passed"])
        self.assertEqual(result["mapped_tensors_per_rank"], 2)
        self.assertFalse(result["actual_PSS_capacity_or_quality_passed"])

    def test_cross_side_file_difference_and_partial_selection_are_rejected(self):
        p, d = self.pair()
        original = copy.deepcopy(d)
        d["rank_descriptions"][0]["shared_engram"]["tables"][0]["mappings"][0]["inode"] += 1
        with self.assertRaises(ValueError): contract.verify_pair_shared_tables(p, d)
        d = original
        d["rank_descriptions"][7].pop("shared_engram")
        with self.assertRaises(ValueError): contract.verify_pair_shared_tables(p, d)

    def test_same_wrong_flags_duplicate_ranks_and_cross_rank_clones_fail(self):
        p, d = self.pair()
        original = copy.deepcopy((p, d))
        for side in (p, d):
            side["rank_descriptions"][0]["shared_engram"]["tables"][0]["mappings"][0]["MAP_SHARED"] = False
        with self.assertRaises(ValueError): contract.verify_pair_shared_tables(p, d)
        p, d = copy.deepcopy(original)
        d["rank_descriptions"][7]["rank"] = 0
        with self.assertRaises(ValueError): contract.verify_pair_shared_tables(p, d)
        p, d = copy.deepcopy(original)
        for side in (p, d):
            side["rank_descriptions"][1]["shared_engram"]["tables"][0]["mappings"][0]["inode"] += 1
        with self.assertRaises(ValueError): contract.verify_pair_shared_tables(p, d)


if __name__ == "__main__":
    unittest.main()
