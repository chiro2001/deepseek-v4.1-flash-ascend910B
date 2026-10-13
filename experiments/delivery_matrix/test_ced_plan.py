"""CPU negative controls for P/D topology, draft and best-quality configuration."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
package = ModuleType("delivery_matrix")
package.__path__ = [str(HERE)]
sys.modules.setdefault("delivery_matrix", package)
spec = importlib.util.spec_from_file_location("delivery_matrix.ced_service", HERE / "ced_service.py")
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


def candidate():
    return {"model_kwargs": {"tensor_parallel_size": 8, "load_format": "auto", "async_scheduling": True,
                              "worker_cls": "best.NativeWorker", "model": "/real/model",
                              "speculative_config": {"method": "dspark", "num_speculative_tokens": 5},
                              "compilation_config": {"cudagraph_capture_sizes": [6]},
                              "additional_config": {"enable_engram": True, "multistream_dsv4_dsa_overlap": True,
                                                    "multistream_overlap_shared_expert": True}},
            "startup_env": {"VLLM_BATCH_INVARIANT": "1", "DRAFT_GRAPH": "1",
                            "ASCEND_CUSTOM_OPP_PATH": "/real/opp", "DSPARK_GRAPH_CAPTURE_METADATA": "1",
                            "DSPARK_DRAFT_USE_CUDAGRAPH": "1", "HCCL_DETERMINISTIC": "strict"}}


def args(role):
    return NS(role=role, output=Path("/work/results/ced-test/service"), source=Path("/work/best"),
              overlay=Path("/work/ced-overlay"), max_model_len=1048576, kv_bytes=15728022528,
              device_index=0, kv_port=19090 if role == "prefill" else 19091, served_model="ced-test")


class CEDPlanTests(unittest.TestCase):
    def test_p_is_real_encoder_and_cannot_claim_architecturally_impossible_draft(self):
        plan = service.make_ced_plan(candidate(), args("prefill"))
        self.assertIsNone(plan["model_kwargs"]["speculative_config"])
        self.assertEqual(plan["environment"]["DRAFT_GRAPH"], "0")
        self.assertEqual(plan["environment"]["STACK_PHYSICAL_CHIPS"], "0,1,2,3,4,5,6,7")
        self.assertEqual(plan["model_kwargs"]["kv_transfer_config"]["kv_role"], "kv_producer")

    def test_d_retains_best_draft_bi_async_and_both_streams(self):
        plan = service.make_ced_plan(candidate(), args("decode"))
        self.assertEqual(plan["model_kwargs"]["speculative_config"]["num_speculative_tokens"], 5)
        self.assertEqual(plan["environment"]["VLLM_BATCH_INVARIANT"], "1")
        self.assertEqual(plan["environment"]["DSPARK_GRAPH_CAPTURE_METADATA"], "1")
        self.assertTrue(plan["model_kwargs"]["async_scheduling"])
        self.assertTrue(plan["model_kwargs"]["additional_config"]["multistream_dsv4_dsa_overlap"])
        self.assertTrue(plan["model_kwargs"]["additional_config"]["multistream_overlap_shared_expert"])
        self.assertEqual(plan["environment"]["V41_CED_GRAPH_PROMPT_TAIL_EAGER"], "1")
        self.assertEqual(plan["http_middleware"], ["delivery_matrix.ced_decode_boundary.ced_decode_guard"])

    def test_dummy_weights_are_rejected(self):
        value = candidate()
        value["model_kwargs"]["load_format"] = "dummy"
        with self.assertRaises(ValueError):
            service.make_ced_plan(value, args("decode"))

    def test_disabled_engram_or_bi_are_rejected(self):
        for key in ("engram", "bi"):
            value = candidate()
            if key == "engram":
                value["model_kwargs"]["additional_config"]["enable_engram"] = False
            else:
                value["startup_env"]["VLLM_BATCH_INVARIANT"] = "0"
            with self.assertRaises(ValueError):
                service.make_ced_plan(value, args("decode"))

    def test_ced_roles_do_not_silently_inherit_dcp_overlay(self):
        value = candidate()
        value["startup_env"].update(V41_DCP="1", DELIVERY_DCP_OVERLAY="/old/dcp",
                                    DELIVERY_DCP_DRAFT="/old/proposer")
        plan = service.make_ced_plan(value, args("decode"))
        self.assertEqual(plan["model_kwargs"]["decode_context_parallel_size"], 1)
        self.assertNotIn("V41_DCP", plan["environment"])
        self.assertNotIn("DELIVERY_DCP_OVERLAY", plan["environment"])
        self.assertNotIn("DELIVERY_DCP_DRAFT", plan["environment"])

    def test_missing_draft_metadata_or_custom_opp_are_rejected(self):
        for key in ("DRAFT_GRAPH", "DSPARK_GRAPH_CAPTURE_METADATA", "ASCEND_CUSTOM_OPP_PATH"):
            value = candidate()
            value["startup_env"].pop(key)
            with self.assertRaises(ValueError):
                service.make_ced_plan(value, args("decode"))

    def test_stream_fallback_is_a_new_unverified_candidate(self):
        original = service.make_ced_plan(candidate(), args("decode"))
        config = args("decode")
        config.decode_multistream = "off"
        fallback = service.make_ced_plan(candidate(), config)
        extra = fallback["model_kwargs"]["additional_config"]
        self.assertFalse(extra["multistream_dsv4_dsa_overlap"])
        self.assertFalse(extra["multistream_overlap_shared_expert"])
        self.assertNotEqual(original["runtime_fingerprint"], fallback["runtime_fingerprint"])
        self.assertFalse(fallback["actual_quality_passed"])
        self.assertEqual(fallback["environment"]["DRAFT_GRAPH"], "1")

    def test_d_compressor_component_is_fingerprinted_and_does_not_change_streams(self):
        original = service.make_ced_plan(candidate(), args("decode"))
        config = args("decode")
        config.compressor_overlay = HERE / "ced_compressor_20261013_v1"
        changed = service.make_ced_plan(candidate(), config)
        self.assertNotEqual(original["runtime_fingerprint"], changed["runtime_fingerprint"])
        self.assertEqual(original["model_kwargs"], changed["model_kwargs"])
        self.assertEqual(len(changed["ced_compressor_manifest_sha256"]), 64)
        self.assertEqual(changed["environment"]["DELIVERY_CED_COMPRESSOR"], str(config.compressor_overlay))
        self.assertFalse(changed["actual_quality_passed"])
        self.assertEqual(changed["environment"]["DRAFT_GRAPH"], "1")
        config.role = "prefill"
        with self.assertRaises(ValueError): service.make_ced_plan(candidate(), config)

    def test_no_old_quality_receipt_is_inherited(self):
        plan = service.make_ced_plan(candidate(), args("decode"))
        self.assertFalse(plan["actual_quality_passed"])
        self.assertFalse(plan["actual_performance_observed"])

    def test_p_has_no_decode_boundary_and_d_boundary_is_fingerprinted(self):
        p = service.make_ced_plan(candidate(), args("prefill"))
        d = service.make_ced_plan(candidate(), args("decode"))
        self.assertEqual(p["http_middleware"], [])
        from delivery_matrix.service import fingerprint
        self.assertNotEqual(d["runtime_fingerprint"], fingerprint({"model_kwargs": d["model_kwargs"],
                                                                 "environment": d["environment"]}))

    def test_overlay_manifest_bytes_are_bound_into_runtime_identity(self):
        config = args("decode")
        config.overlay_manifest_sha256 = "a" * 64
        first = service.make_ced_plan(candidate(), config)
        config.overlay_manifest_sha256 = "b" * 64
        second = service.make_ced_plan(candidate(), config)
        self.assertEqual(first["ced_overlay_manifest_sha256"], "a" * 64)
        self.assertNotEqual(first["runtime_fingerprint"], second["runtime_fingerprint"])


if __name__ == "__main__":
    unittest.main()
