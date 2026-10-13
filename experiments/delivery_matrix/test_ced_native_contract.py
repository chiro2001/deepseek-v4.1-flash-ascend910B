import hashlib
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

spec = importlib.util.spec_from_file_location("ced_native_contract_test", Path(__file__).with_name("ced_native_contract.py"))
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


class OPPTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="ced-opp-", dir=os.environ["TMPDIR"])
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.expected = {}
        for name, value in (("version.info", b"A3-v1"), ("op_api/lib/api.so", b"ABI-v1")):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            self.expected[name] = hashlib.sha256(value).hexdigest()

    def test_actual_files_are_hashed_with_explicit_limited_scope(self):
        receipt = contract.inspect_opp(self.root, self.expected)
        self.assertEqual(len(receipt["files"]), 2)
        self.assertFalse(receipt["full_tree_verified_by_this_gate"])
        self.assertEqual(len(receipt["artifact_fingerprint"]), 64)

    def test_missing_opp_is_a_startup_failure(self):
        with self.assertRaises(FileNotFoundError):
            contract.inspect_opp(self.root / "image-missing-vendor", self.expected)

    def test_changed_abi_library_is_rejected(self):
        (self.root / "op_api/lib/api.so").write_bytes(b"different-ABI")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            contract.inspect_opp(self.root, self.expected)

    def test_external_symlink_cannot_substitute_a_vendor_file(self):
        (self.root / "outside").mkdir()
        (self.root / "outside/api.so").write_bytes(b"ABI-v1")
        selected = self.root / "vendor"
        selected.mkdir()
        (selected / "api.so").symlink_to(self.root / "outside/api.so")
        with self.assertRaisesRegex(ValueError, "within"):
            contract.inspect_opp(selected, {"api.so": hashlib.sha256(b"ABI-v1").hexdigest()})

    def test_empty_contract_is_not_validation(self):
        with self.assertRaises(ValueError):
            contract.inspect_opp(self.root, {})


if __name__ == "__main__":
    unittest.main()
