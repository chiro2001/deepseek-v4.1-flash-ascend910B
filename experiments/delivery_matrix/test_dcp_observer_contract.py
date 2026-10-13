import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace as NS
import unittest

spec = importlib.util.spec_from_file_location("dcp_observer_test", Path(__file__).with_name("dcp_observer_contract.py"))
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
ROOT = Path(__file__).with_name("dcp_draft_20261013_v2")


class ObserverContractTests(unittest.TestCase):
    def test_new_candidate_admits_exact_derived_digest(self):
        observer = NS(STOCK_PROPOSER_SHA256=contract.STOCK_BASE_SHA256,
                      OVERLAY_PROPOSER_SHA256=contract.BEST_SOURCES["llm_base_proposer.py"])
        receipt = contract.install(ROOT, observer)
        self.assertEqual(observer.STOCK_PROPOSER_SHA256, receipt["derived_sources"]["llm_base_proposer.py"])
        self.assertFalse(receipt["original_stock_claim"])
        self.assertTrue(receipt["graph_execution_checks_unchanged"])

    def test_unknown_observer_guard_is_not_disabled(self):
        with self.assertRaises(ValueError):
            contract.install(ROOT, NS(STOCK_PROPOSER_SHA256="0" * 64))

    def test_payload_changes_are_rejected(self):
        task_tmp = Path(os.environ.get("TMPDIR", Path.home() / "tmp/dsv41-delivery-matrix"))
        task_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=task_tmp) as temp:
            root = Path(temp) / "payload"
            shutil.copytree(ROOT, root)
            (root / "llm_base_proposer.py").write_text("altered\n")
            with self.assertRaises(ValueError):
                contract.verify(root)

    def test_wrong_original_source_provenance_is_rejected(self):
        task_tmp = Path(os.environ.get("TMPDIR", Path.home() / "tmp/dsv41-delivery-matrix"))
        task_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=task_tmp) as temp:
            root = Path(temp) / "payload"
            shutil.copytree(ROOT, root)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["files"]["dspark_proposer.py"]["source_sha256"] = "0" * 64
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                contract.verify(root)


if __name__ == "__main__":
    unittest.main()
