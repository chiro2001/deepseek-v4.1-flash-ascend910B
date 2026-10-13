"""Explicit chip selection and pre-import role-contamination negative controls."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
package = ModuleType("delivery_matrix")
package.__path__ = [str(HERE)]
sys.modules.setdefault("delivery_matrix", package)
spec = importlib.util.spec_from_file_location("delivery_matrix.generic_service_cpu", HERE / "service.py")
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


def candidate():
    return {"model_kwargs": {"tensor_parallel_size": 8, "load_format": "auto", "async_scheduling": True,
                             "worker_cls": "parallel_tiny.dspark_formal_prep.worker.FormalDSparkWorker", "model": "/real/model",
                             "speculative_config": {"method": "dspark", "num_speculative_tokens": 5},
                             "compilation_config": {"cudagraph_capture_sizes": [6]},
                             "additional_config": {"enable_engram": True, "multistream_dsv4_dsa_overlap": True,
                                                   "multistream_overlap_shared_expert": True}},
            "startup_env": {"VLLM_BATCH_INVARIANT": "1", "DRAFT_GRAPH": "1",
                            "DSPARK_GRAPH_CAPTURE_METADATA": "1", "ASCEND_CUSTOM_OPP_PATH": "/real/opp",
                            "STACK_PHYSICAL_CHIPS": "8,9,10,11,12,13,14,15"}}


def args(chips=None, form="a2-tp8-a3sim"):
    return NS(source=Path("/work/best"), output=Path("/work/new/service"), served_model="new-a2",
              max_model_len=196608, batched_tokens=8192, device_index=0, physical_chips=chips,
              physical_worker=HERE / "tp8_physical_0_7_20261013_v1" if chips == "0,1,2,3,4,5,6,7" else None,
              dcp_det_reduce=None, form=form)


class TopologyTests(unittest.TestCase):
    def test_explicit_subset_is_validated_never_silently_selected(self):
        for value in ("0,1,2,3,4,5,6,7", "8,9,10,11,12,13,14,15", "0, 2,4,6,8,10,12,14"):
            self.assertEqual(len(service.physical_chips(value).split(",")), 8)
        for value in (None, True, "0,1,2,3,4,5,6", "0,1,2,3,4,5,6,6", "0,1,2,3,4,5,6,16",
                      "0,1,2,3,4,5,6,-1", "0,1,2,3,4,5,6,07"):
            with self.assertRaises(ValueError): service.physical_chips(value)

    def test_move_is_a_new_unverified_candidate_not_a_model_or_draft_change(self):
        original = candidate()
        old = service.make_plan(original, args())
        new = service.make_plan(original, args("0,1,2,3,4,5,6,7"))
        self.assertEqual(old["model_kwargs"], new["model_kwargs"])
        self.assertNotEqual(old["runtime_fingerprint"], new["runtime_fingerprint"])
        self.assertEqual(original, candidate())
        for key in ("STACK_PHYSICAL_CHIPS", "STACK_EXPECTED_CHIPS", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
            self.assertEqual(new["environment"][key], "0,1,2,3,4,5,6,7")
        self.assertFalse(new["actual_quality_passed"])
        self.assertEqual(new["environment"]["DRAFT_GRAPH"], "1")
        self.assertTrue(new["model_kwargs"]["additional_config"]["multistream_dsv4_dsa_overlap"])

    def test_a2_explicitly_has_dcp1_and_cannot_inherit_ced_or_dcp(self):
        source = candidate()
        source["model_kwargs"]["decode_context_parallel_size"] = 8
        source["startup_env"].update(V41_CED_ROLE="prefill", DELIVERY_CED_OVERLAY="/old/ced",
                                     DELIVERY_CED_COMPRESSOR="/old/compressor", V41_DCP="1",
                                     DELIVERY_DCP_OVERLAY="/old/dcp", DELIVERY_DCP_DRAFT="/old/draft")
        plan = service.make_plan(source, args("0,1,2,3,4,5,6,7"))
        self.assertEqual(plan["model_kwargs"]["decode_context_parallel_size"], 1)
        self.assertFalse(any(key.startswith("DELIVERY_CED_") or key.startswith("DELIVERY_DCP_")
                             or key in ("V41_CED_ROLE", "V41_DCP") for key in plan["environment"]))

    def test_dcp8_still_reuses_the_selected_eight_ranks(self):
        with patch.dict(os.environ, {"DELIVERY_DCP_OVERLAY": "/pinned/dcp", "DELIVERY_DCP_DRAFT": "/pinned/draft"}):
            plan = service.make_plan(candidate(), args("0,1,2,3,4,5,6,7", "a3-tp8-dcp8"))
        self.assertEqual(plan["model_kwargs"]["decode_context_parallel_size"], 8)
        self.assertEqual(plan["model_kwargs"]["tensor_parallel_size"], 8)
        self.assertEqual(plan["environment"]["V41_DCP"], "1")

    def test_direct_script_launcher_clears_container_role_before_python(self):
        with TemporaryDirectory(dir=os.environ["TMPDIR"]) as task:
            root = Path(task)
            contract, output = root / "candidate.json", root / "runner.sh"
            contract.write_text(json.dumps(candidate()))
            result = subprocess.run([sys.executable, str(HERE / "prepare_launch.py"),
                "--candidate", str(contract), "--source", "/work/best", "--package", "/work/generic",
                "--job-root", "/work/new", "--output", str(output), "--port", "18960",
                "--served-model", "new-a2", "--form", "a2-tp8-a3sim", "--device-index", "0",
                "--physical-chips", "0,1,2,3,4,5,6,7", "--physical-worker", "/work/pinned-physical"],
                check=True, text=True, capture_output=True)
            report = json.loads(result.stdout)
            self.assertFalse(report["service_launch_requested"])
            shell = output.read_text()
            self.assertIn("unset V41_DCP DELIVERY_DCP_OVERLAY", shell)
            self.assertIn("DELIVERY_CED_COMPRESSOR", shell.splitlines()[3])
            self.assertIn("--physical-chips 0,1,2,3,4,5,6,7", shell)
            self.assertNotIn("export V41_CED_ROLE", shell)
            self.assertLess(shell.index("unset "), shell.index("export "))
            subprocess.run(["bash", "-n", str(output)], check=True)

    def test_environment_only_physical_move_cannot_bypass_frozen_guard(self):
        config = args("0,1,2,3,4,5,6,7")
        config.physical_worker = None
        with self.assertRaises(ValueError): service.make_plan(candidate(), config)


if __name__ == "__main__":
    unittest.main()
