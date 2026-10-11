"""CPU snapshot integrity checks using real frozen SDK/prepack dependencies.

The v10 driver and base are explicitly synthetic CPU fixtures, so these tests
cannot certify that the real, unfinished full43 driver is executable.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import prepare_full43_audit_snapshot as snapshot


ROOT = Path(__file__).resolve().parent


class SnapshotCPU(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dsv41_full43_snapshot_cpu_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.payload = {}
        for prefix, (name, field, expected, extras) in snapshot.DEPENDENCIES.items():
            path = ROOT / prefix.removeprefix("stack/")
            raw = (path / name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
            manifest = json.loads(raw)
            for rel in set(manifest[field]) | {name} | set(extras):
                self.payload[prefix + rel] = (path / rel).read_bytes()
        self.driver = b"# CPU fixture: no SDK, model or NPU runtime.\n"
        self.candidate = {"files": {"formal_audit_driver.py": hashlib.sha256(self.driver).hexdigest()},
                          "startup_env": {"HCCL_DETERMINISTIC": "strict"},
                          "strict_uncovered_points": ["CPU_fixture_all_device_consumers_uncovered"],
                          "implemented_points": [], "full_consumer_audit_passed": False,
                          "formal_model_validated": False}
        self.payload[snapshot.V10 + "formal_audit_driver.py"] = self.driver
        self.payload[snapshot.V10 + "SHA256SUMS"] = (hashlib.sha256(self.driver).hexdigest() + "  formal_audit_driver.py\n").encode()
        base = self.root / "base"
        for name in ("stack", "operator", "up950"): (base / name).mkdir(parents=True)
        for name in ("run_stack.sh", "sitecustomize.py", "formal_model_contract.py"):
            (base / "stack" / name).write_text("# CPU fixture only\n")
        frozen = base / "stack/parallel_tiny/dspark_formal_prep"; frozen.mkdir(parents=True)
        (frozen / "freeze.json").write_text(json.dumps({"files": {"formal_model_contract.py": snapshot.sha(base / "stack/formal_model_contract.py")}}))

    def archive(self, changes=None, candidate=None, extra_members=()):
        payload = copy.deepcopy(self.payload)
        raw = json.dumps(candidate if candidate is not None else self.candidate).encode()
        payload[snapshot.V10 + "freeze.json"] = raw
        if changes: changes(payload)
        bundle = self.root / "bundle.tgz"
        with tarfile.open(bundle, "w:gz") as archive:
            for name, data in payload.items():
                member = tarfile.TarInfo(name); member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            for member, data in extra_members: archive.addfile(member, io.BytesIO(data) if data is not None else None)
        return argparse.Namespace(root=str(self.root), base_name="base", source_name="new_source",
                                  bundle_name=bundle.name, full43_freeze_sha256=hashlib.sha256(raw).hexdigest(), mode="diagnostic")

    def test_actual_frozen_dependencies_and_synthetic_driver_copy(self):
        result = snapshot.prepare(self.archive())
        self.assertEqual(result["NPU_calls"], 0)
        target = self.root / "new_source"
        self.assertEqual((target / "stack" / snapshot.DRIVER).read_bytes(), self.driver)
        manifest = json.loads((target / "full43_snapshot.json").read_text())
        self.assertFalse(manifest["full_consumer_audit_passed"])
        self.assertFalse(manifest["formal_receipt_granted"])
        for rel, expected in manifest["files"].items(): self.assertEqual(snapshot.sha(target / rel), expected)

    def test_reusing_snapshot_is_rejected(self):
        args = self.archive(); snapshot.prepare(args)
        with self.assertRaisesRegex(ValueError, "already exists"): snapshot.prepare(args)

    def test_partial_coverage_cannot_use_audit_mode(self):
        args = self.archive(); args.mode = "audit"
        with self.assertRaisesRegex(ValueError, "omit actual consumer"): snapshot.prepare(args)

    def test_cpu_fixture_cannot_grant_model_admission(self):
        candidate = copy.deepcopy(self.candidate); candidate["formal_model_validated"] = True
        with self.assertRaisesRegex(ValueError, "unsupported device admission"): snapshot.prepare(self.archive(candidate=candidate))

    def test_original_dependency_corruption_is_rejected(self):
        key = "stack/parallel_tiny/dspark_formal_audit/formal_consumer_sdk_bridge_v9_cpu/evidence/actual_pair_protocol.json"
        args = self.archive(lambda p: p.__setitem__(key, p[key] + b"\n"))
        with self.assertRaisesRegex(ValueError, "Frozen payload differs"): snapshot.prepare(args)

    def test_base_corruption_is_rejected(self):
        args = self.archive(); (self.root / "base/stack/formal_model_contract.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Base frozen source changed"): snapshot.prepare(args)

    def test_unlisted_overlay_is_rejected(self):
        args = self.archive(lambda p: p.__setitem__("stack/unknown_live_override.py", b"pass\n"))
        with self.assertRaisesRegex(ValueError, "Unexpected/missing"): snapshot.prepare(args)

    def test_non_strict_environment_is_rejected(self):
        candidate = copy.deepcopy(self.candidate); candidate["startup_env"]["HCCL_DETERMINISTIC"] = "off"
        with self.assertRaisesRegex(ValueError, "startup environment"): snapshot.prepare(self.archive(candidate=candidate))

    def test_traversal_archive_is_rejected(self):
        member = tarfile.TarInfo("stack/../../foreign"); member.size = 1
        args = self.archive(extra_members=[(member, b"x")])
        with self.assertRaisesRegex(ValueError, "safe stack"): snapshot.prepare(args)

    def test_symlink_archive_is_rejected(self):
        member = tarfile.TarInfo("stack/link"); member.type = tarfile.SYMTYPE; member.linkname = "/etc/passwd"
        with self.assertRaisesRegex(ValueError, "regular"): snapshot.prepare(self.archive(extra_members=[(member, None)]))

    def test_duplicate_archive_members_are_rejected(self):
        member = tarfile.TarInfo(snapshot.V10 + "formal_audit_driver.py"); member.size = len(self.driver)
        with self.assertRaisesRegex(ValueError, "duplicate"): snapshot.prepare(self.archive(extra_members=[(member, self.driver)]))

    def test_one_mb_archive_member_is_rejected(self):
        member = tarfile.TarInfo("stack/large"); member.size = 1_000_000
        with self.assertRaisesRegex(ValueError, "bounded"): snapshot.prepare(self.archive(extra_members=[(member, b"x" * 1_000_000)]))


if __name__ == "__main__": unittest.main(verbosity=2)
