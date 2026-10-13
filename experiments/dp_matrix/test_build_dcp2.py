"""CPU-only tests for deriving the DCP2 package from frozen DCP8 sources."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

TASK_TMP_ROOT = Path(
    os.environ.get("TMPDIR")
    or (Path.home() / "tmp" / "dsv41-dp-matrix-dcp2-builder-tests")
)
TASK_TMP_ROOT.mkdir(parents=True, exist_ok=True)

from dp_matrix import build_dcp2  # noqa: E402
from dp_matrix.dcp2_contract import derive_engram  # noqa: E402


FROZEN_DCP8 = PROJECT_ROOT / "experiments/delivery_matrix/dcp_runtime_20261013_v6"
ENGRAM_RELATIVE = "vllm_ascend/models/deepseek_v41/engram_hbm.py"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BuildDCP2Tests(unittest.TestCase):
    def temporary_root(self):
        temporary = tempfile.TemporaryDirectory(prefix="dcp2-builder-", dir=TASK_TMP_ROOT)
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def copy_base(self, root):
        base = root / "frozen-dcp8"
        shutil.copytree(FROZEN_DCP8, base)
        return base

    @staticmethod
    def read_manifest(base):
        manifest_path = base / "manifest.json"
        return manifest_path, json.loads(manifest_path.read_text())

    @staticmethod
    def write_manifest(path, manifest):
        path.write_text(json.dumps(manifest, indent=2) + "\n")

    def test_builds_only_into_absent_output_and_never_inherits_quality(self):
        root = self.temporary_root()
        base = self.copy_base(root)
        _, base_manifest = self.read_manifest(base)

        output = root / "derived-dcp2"
        self.assertFalse(output.exists())
        result = build_dcp2.build(base, output)
        self.assertTrue(output.is_dir())
        self.assertEqual(result["output"], str(output.resolve()))
        self.assertEqual(result["manifest_sha256"], _sha(output / "manifest.json"))
        self.assertFalse(result["service_started"])
        self.assertFalse(result["quality_inherited"])
        self.assertEqual(build_dcp2.verify(output)["native_dp_geometry"]["layout"],
                         "dp4-tp2-dcp2")

        derived_manifest = json.loads((output / "manifest.json").read_text())
        geometry = derived_manifest["native_dp_geometry"]
        self.assertEqual(
            {key: geometry[key] for key in ("layout", "dp", "tp", "dcp", "ep")},
            {"layout": "dp4-tp2-dcp2", "dp": 4, "tp": 2, "dcp": 2, "ep": 8},
        )
        self.assertTrue(geometry["engram_config_and_actual_groups_required"])
        self.assertTrue(geometry["engram_lookup_owner_metadata_and_broadcast_unchanged"])
        self.assertFalse(geometry["quality_inherited"])
        self.assertFalse(geometry["actual_quality_passed"])
        self.assertFalse(geometry["actual_performance_observed"])
        self.assertFalse(geometry["actual_capacity_observed"])
        self.assertFalse(geometry["service_started"])
        self.assertFalse(derived_manifest["actual_DCP_BI_async_quality_passed"])
        self.assertFalse(derived_manifest["actual_DCP_BI_async_performance_observed"])
        self.assertFalse(
            derived_manifest["engram_DCP8_guard_narrowly_enabled_for_single_node_TP8_EP8"]
        )

        # Every frozen component other than the narrowly derived Engram gate
        # remains byte-identical; the manifest records the exact derivation.
        for relative in base_manifest["files"]:
            with self.subTest(relative=relative):
                source = base / relative
                target = output / relative
                self.assertTrue(target.is_file())
                if relative == ENGRAM_RELATIVE:
                    expected = derive_engram(source.read_text()).encode()
                    self.assertEqual(target.read_bytes(), expected)
                    record = derived_manifest["files"][relative]
                    self.assertEqual(record["sha256"], _sha(target))
                    self.assertEqual(record["bytes"], target.stat().st_size)
                    self.assertEqual(
                        record["derivation_base_sha256"], build_dcp2.ENGRAM_SHA
                    )
                else:
                    self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_build_rejects_manifest_drift_even_when_payloads_still_match(self):
        root = self.temporary_root()
        base = self.copy_base(root)
        manifest_path, manifest = self.read_manifest(base)
        manifest["actual_DCP_BI_async_quality_passed"] = True
        self.write_manifest(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "exact reviewed DCP v6 manifest"):
            build_dcp2.build(base, root / "derived-dcp2")

    def test_runtime_verify_rejects_a_tampered_derived_payload(self):
        root = self.temporary_root()
        base = self.copy_base(root)
        output = root / "derived-dcp2"
        build_dcp2.build(base, output)
        path = output / ENGRAM_RELATIVE
        path.write_bytes(path.read_bytes() + b"\n# runtime tamper\n")
        with self.assertRaisesRegex(ValueError, "payload SHA/size mismatch"):
            build_dcp2.verify(output)

    def test_build_refuses_a_preexisting_output_directory(self):
        root = self.temporary_root()
        base = self.copy_base(root)
        output = root / "already-exists"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            build_dcp2.build(base, output)
        self.assertEqual(list(output.iterdir()), [])

    def test_build_rejects_a_base_source_tampered_against_frozen_manifest(self):
        root = self.temporary_root()
        base = self.copy_base(root)
        tampered = base / ENGRAM_RELATIVE
        tampered.write_bytes(tampered.read_bytes() + b"\n# tampered base\n")
        output = root / "derived-dcp2"
        with self.assertRaisesRegex(ValueError, "Frozen DCP base changed"):
            build_dcp2.build(base, output)
        self.assertFalse(output.exists())

    def test_build_rejects_tampered_engram_even_if_manifest_hash_is_rewritten(self):
        root = self.temporary_root()
        base = self.copy_base(root)
        tampered = base / ENGRAM_RELATIVE
        tampered.write_bytes(tampered.read_bytes() + b"\n# tampered base\n")
        manifest_path, manifest = self.read_manifest(base)
        manifest["files"][ENGRAM_RELATIVE]["sha256"] = _sha(tampered)
        manifest["files"][ENGRAM_RELATIVE]["bytes"] = tampered.stat().st_size
        self.write_manifest(manifest_path, manifest)

        output = root / "derived-dcp2"
        with self.assertRaisesRegex(ValueError, "Unreviewed Engram source base"):
            build_dcp2.build(base, output)
        self.assertFalse(output.exists())

    def test_build_requires_the_reviewed_bounded_reduction_source_manifest(self):
        cases = (
            ("schema", "unreviewed"),
            ("best_model_unchanged", False),
            ("best_engram_lookup_algorithm_preserved", False),
        )
        for field, value in cases:
            root = self.temporary_root()
            base = self.copy_base(root)
            manifest_path, manifest = self.read_manifest(base)
            manifest[field] = value
            self.write_manifest(manifest_path, manifest)
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "frozen bounded-reduction DCP8 source candidate"
            ):
                build_dcp2.build(base, root / "derived-dcp2")

        root = self.temporary_root()
        base = self.copy_base(root)
        manifest_path, manifest = self.read_manifest(base)
        manifest["stable_reduction_candidate"]["all_five_gates_use_bounded_helper"] = False
        self.write_manifest(manifest_path, manifest)
        with self.assertRaisesRegex(
            ValueError, "Software ordered reduction requires the all-branch bounded candidate"
        ):
            build_dcp2.build(base, root / "derived-dcp2")


if __name__ == "__main__":
    unittest.main()
