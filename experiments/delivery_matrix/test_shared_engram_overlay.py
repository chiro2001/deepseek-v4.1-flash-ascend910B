"""CPU import-chain/opt-in/plan controls for the independent mmap candidate."""
import importlib.util
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from test_ced_plan import candidate, args, service

HERE = Path(__file__).resolve().parent
package = ModuleType("delivery_matrix")
package.__path__ = [str(HERE)]
sys.modules.setdefault("delivery_matrix", package)
spec = importlib.util.spec_from_file_location("delivery_matrix.shared_engram_overlay", HERE / "shared_engram_overlay.py")
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)
COMPONENT = HERE / "shared_engram_20261013_v1"
BASE = HERE.parent.parent / "patches/files/engram_hbm.py"


class SharedOverlayTests(unittest.TestCase):
    def env(self, role="decode"):
        return {"V41_CED_ROLE": role, "V41_ENGRAM_DEVICE_INDEX": "0", "V41_ENGRAM_HOST_RESIDENT": "1",
                "V41_ENGRAM_SHARED_HOST_MMAP": "1", "DELIVERY_SHARED_ENGRAM_MANIFEST_SHA256": overlay.MANIFEST_SHA}

    def test_native_resolution_delegates_actual_meta_chain_not_only_PathFinder(self):
        calls = []
        class ScopedDraft:
            def find_spec(self, name, path=None, target=None):
                calls.append(("scoped", name)); return None
        class ActualSDK:
            def find_spec(self, name, path=None, target=None):
                calls.append(("sdk", name)); return NS(origin=str(BASE))
        finder = overlay.SharedEngramFinder(COMPONENT)
        with patch.object(sys, "meta_path", [ScopedDraft(), finder, ActualSDK()]):
            result = finder.find_spec(overlay.MODULE, ["actual-parent"])
        self.assertEqual(result.origin, str(COMPONENT / overlay.FILES[overlay.MODULE]))
        self.assertEqual([kind for kind, _ in calls], ["scoped", "sdk"])
        self.assertFalse(finder.resolving)
        self.assertIsNone(finder.find_spec("vllm_ascend.models.deepseek_v41.model"))

    def test_actual_base_drift_and_missing_source_are_hard_failures(self):
        finder = overlay.SharedEngramFinder(COMPONENT)
        with patch.object(sys, "meta_path", [finder]), self.assertRaises(ImportError):
            finder.find_spec(overlay.MODULE)
        with TemporaryDirectory(dir=os.environ["TMPDIR"]) as task:
            path = Path(task) / "wrong.py"; path.write_text("# not the best SDK\n")
            sdk = NS(find_spec=lambda *args: NS(origin=str(path)))
            with patch.object(sys, "meta_path", [finder, sdk]), self.assertRaises(ImportError):
                finder.find_spec(overlay.MODULE)
        self.assertFalse(finder.resolving)

    def test_role_index_host_flag_and_early_import_are_rejected(self):
        for key, value in (("V41_CED_ROLE", "prefill"), ("V41_ENGRAM_DEVICE_INDEX", "1"),
                           ("V41_ENGRAM_HOST_RESIDENT", "0"), ("V41_ENGRAM_SHARED_HOST_MMAP", "0"),
                           ("DELIVERY_SHARED_ENGRAM_MANIFEST_SHA256", "a" * 64)):
            env = self.env(); env[key] = value
            with patch.dict(os.environ, env), self.assertRaises(ValueError): overlay.install(COMPONENT, "decode")
        with patch.dict(os.environ, self.env()), patch.dict(sys.modules, {overlay.MODULE: ModuleType(overlay.MODULE)}):
            with self.assertRaises(RuntimeError): overlay.install(COMPONENT, "decode")

    def test_install_before_best_scoped_finders_and_duplicate_install(self):
        with patch.dict(os.environ, self.env()), patch.object(sys, "meta_path", []):
            overlay.install(COMPONENT, "decode")
            shared_finder = sys.meta_path[0]
            sys.meta_path.insert(0, NS(find_spec=lambda *args: None))  # best BI/draft hook.
            sys.meta_path.append(NS(find_spec=lambda *args: NS(origin=str(BASE))))
            result = next(spec for finder in list(sys.meta_path)
                          if (spec := finder.find_spec(overlay.MODULE, ["actual-parent"])) is not None)
            self.assertEqual(result.origin, str(COMPONENT / overlay.FILES[overlay.MODULE]))
            self.assertIsInstance(shared_finder, overlay.SharedEngramFinder)
            with self.assertRaises(RuntimeError): overlay.install(COMPONENT, "decode")

    def test_opt_in_plan_keeps_math_kwargs_and_new_fingerprint(self):
        for role in ("prefill", "decode"):
            value = candidate(); value["startup_env"]["V41_ENGRAM_HOST_RESIDENT"] = "1"
            value["model_kwargs"]["additional_config"]["engram_storage"] = "int8"
            config = args(role)
            old = service.make_ced_plan(value, config)
            config.shared_engram_overlay = COMPONENT
            new = service.make_ced_plan(value, config)
            self.assertEqual(new["model_kwargs"], old["model_kwargs"])
            self.assertEqual(new["environment"]["DELIVERY_SHARED_ENGRAM_MANIFEST_SHA256"], overlay.MANIFEST_SHA)
            self.assertEqual(new["environment"]["V41_ENGRAM_SHARED_HOST_MMAP"], "1")
            self.assertNotEqual(new["runtime_fingerprint"], old["runtime_fingerprint"])
            self.assertFalse(new["actual_quality_passed"])

    def test_plan_refuses_unsupported_storage_and_clears_stale_opt_in(self):
        value = candidate(); value["startup_env"].update(self.env(), DELIVERY_SHARED_ENGRAM="/old/component")
        cleared = service.make_ced_plan(value, args("decode"))
        self.assertNotIn("DELIVERY_SHARED_ENGRAM", cleared["environment"])
        config = args("decode"); config.shared_engram_overlay = COMPONENT
        with self.assertRaises(ValueError): service.make_ced_plan(value, config)
        value["model_kwargs"]["additional_config"]["engram_storage"] = "int8"
        config.device_index = 1
        with self.assertRaises(ValueError): service.make_ced_plan(value, config)


if __name__ == "__main__":
    unittest.main()
