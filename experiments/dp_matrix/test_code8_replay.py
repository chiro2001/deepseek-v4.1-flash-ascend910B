"""CPU-only verification of the frozen Code8 split input gate."""
from pathlib import Path
import tempfile
import unittest

from experiments.dp_matrix.code8_replay import (
    MANIFEST_SHA256,
    load_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE8_ROOT = (PROJECT_ROOT
              / "experiments/delivery_matrix/evidence/a2sim_tp8_v4/client/code_formal8")
TASK_TMP_ROOT = Path.home() / "tmp" / "dsv41-dp-code8-input-tests"
TASK_TMP_ROOT.mkdir(parents=True, exist_ok=True)


class Code8ReplayInputTests(unittest.TestCase):
    def test_loads_all_eight_exact_frozen_prompts_and_cases(self):
        loaded = load_inputs(CODE8_ROOT / "manifest.json", CODE8_ROOT / "requests")
        self.assertEqual(loaded["manifest_sha256"], MANIFEST_SHA256)
        self.assertEqual(len(loaded["samples"]), 8)
        self.assertEqual([row["index"] for row in loaded["samples"]], list(range(8)))
        self.assertTrue(all(row["messages"] == [{"role": "user", "content": row["prompt_text"]}]
                            for row in loaded["samples"]))
        self.assertEqual(len({row["prompt_sha256"] for row in loaded["samples"]}), 8)

    def test_manifest_digest_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
            load_inputs(CODE8_ROOT / "manifest.json", CODE8_ROOT / "requests", "0" * 64)

    def test_request_artifact_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="code8-tamper-", dir=TASK_TMP_ROOT) as temp:
            root = Path(temp) / "code_formal8"
            requests = root / "requests"
            requests.mkdir(parents=True)
            manifest = root / "manifest.json"
            manifest.write_bytes((CODE8_ROOT / "manifest.json").read_bytes())
            for source in sorted((CODE8_ROOT / "requests").glob("*.json")):
                (requests / source.name).write_bytes(source.read_bytes())
            artifact = requests / "code_round00_request03.json"
            artifact.write_bytes(artifact.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "request size/SHA mismatch"):
                load_inputs(manifest, requests)


if __name__ == "__main__":
    unittest.main()
