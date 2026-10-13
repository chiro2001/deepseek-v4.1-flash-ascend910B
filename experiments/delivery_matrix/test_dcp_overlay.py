"""CPU-only controls for minimal DCP source selection and platform patching."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("delivery_dcp_test", Path(__file__).with_name("dcp_overlay.py"))
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)


class OverlayTests(unittest.TestCase):
    def platform(self, active=True):
        m = NS(calls=0)
        def original(config):
            m.calls += 1
            return "original-result"
        m._validate_parallel_config = original
        guard = ModuleType("vllm_ascend.patch.platform.patch_v41_dcp")
        guard.v41_dcp_active = lambda: active
        core = ModuleType("vllm_ascend.core.deepseek_v41")
        core.STATE_RING_ROWS = 32
        return m, {guard.__name__: guard, core.__name__: core}

    def test_reused_dcp_normalizes_interleave_without_changing_base_result(self):
        module, fake = self.platform()
        config = NS(parallel_config=NS(decode_context_parallel_size=8, cp_kv_cache_interleave_size=128),
                    cache_config=NS(block_size=128))
        with patch.dict(sys.modules, fake):
            overlay.patch_platform(module)
            self.assertEqual(module._validate_parallel_config(config), "original-result")
        self.assertEqual(module.calls, 1)
        self.assertEqual(config.parallel_config.cp_kv_cache_interleave_size, 32)

    def test_non_v41_parallel_configuration_is_unchanged(self):
        module, fake = self.platform(active=False)
        config = NS(parallel_config=NS(decode_context_parallel_size=8, cp_kv_cache_interleave_size=128),
                    cache_config=NS(block_size=128))
        with patch.dict(sys.modules, fake):
            overlay.patch_platform(module)
            module._validate_parallel_config(config)
        self.assertEqual(config.parallel_config.cp_kv_cache_interleave_size, 128)

    def test_compression_group_cannot_be_split(self):
        module, fake = self.platform()
        config = NS(parallel_config=NS(decode_context_parallel_size=8), cache_config=NS(block_size=7))
        with patch.dict(sys.modules, fake):
            overlay.patch_platform(module)
            with self.assertRaises(ValueError):
                module._validate_parallel_config(config)

    def test_manifest_preserves_worker_context_and_main_model(self):
        root = Path(__file__).with_name("dcp_runtime_20261013_v3")
        manifest = json.loads((root / "manifest.json").read_text())
        protected = ["vllm_ascend.worker.model_runner_v1", "vllm_ascend.worker.worker",
                     "vllm_ascend.ascend_forward_context", "vllm_ascend.platform",
                     "vllm_ascend.models.deepseek_v41.model"]
        for name in protected:
            self.assertNotIn(name, manifest["modules"])
        self.assertTrue(manifest["best_QLI_no_candidate_preserved"])

    def test_platform_import_before_overlay_is_rejected(self):
        root = Path(__file__).with_name("dcp_runtime_20261013_v3")
        with patch.dict(sys.modules, {"vllm_ascend.platform": ModuleType("vllm_ascend.platform")}):
            with self.assertRaises(RuntimeError):
                overlay.install(root)

    def test_pinned_payload_tampering_is_rejected(self):
        task_tmp = Path(os.environ.get("TMPDIR", Path.home() / "tmp/dsv41-delivery-matrix"))
        task_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=task_tmp) as directory:
            root = Path(directory)
            (root / "payload.py").write_text("tampered\n")
            manifest = {"schema": "delivery_dcp_overlay_v2", "best_model_unchanged": True,
                        "best_engram_lookup_algorithm_preserved": True,
                        "files": {"payload.py": {"sha256": "0" * 64}}, "modules": {"test_payload": "payload.py"}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                overlay.install(root)


if __name__ == "__main__":
    unittest.main()
