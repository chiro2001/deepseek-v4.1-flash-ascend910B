import copy
import unittest

from experiments.dp_matrix.launch_plan import WORKER, make_plan, DCP2_DRAFT_MANIFEST_SHA256


class LaunchPlanTests(unittest.TestCase):
    def candidate(self):
        # CPU fixture only, not a source/SDK or real-weight receipt.
        return {"model_kwargs": {
            "tensor_parallel_size": 8, "load_format": "auto", "worker_cls": "frozen.Worker",
            "enable_expert_parallel": True, "async_scheduling": True,
            "speculative_config": {"method": "dspark", "num_speculative_tokens": 5},
            "additional_config": {"enable_engram": True},
            "kv_cache_memory_bytes": 4 * 1024**3,
            "compilation_config": {"cudagraph_capture_sizes": [6]},
        }, "startup_env": {
            "STACK_PHYSICAL_CHIPS": "8,9,10,11,12,13,14,15",
            "DRAFT_GRAPH": "1", "DSPARK_GRAPH_CAPTURE_METADATA": "1",
            "VLLM_BATCH_INVARIANT": "1", "ASCEND_CUSTOM_OPP_PATH": "/pinned/opp",
            "DSPARK_ASYNC_COUNTER": "1",
        }}

    def plan(self, candidate=None, **kwargs):
        return make_plan(candidate or self.candidate(), layout="dp2-tp4",
                         source="/frozen/source", output="/new/job",
                         served_model="dsv41-dp-new", **kwargs)

    def test_new_plan_does_not_mutate_or_admit_base(self):
        original = self.candidate()
        before = copy.deepcopy(original)
        plan = self.plan(original)
        self.assertEqual(original, before)
        kw = plan["model_kwargs"]
        self.assertEqual((kw["data_parallel_size"], kw["tensor_parallel_size"]), (2, 4))
        self.assertEqual(kw["data_parallel_size_local"], 2)
        self.assertEqual(kw["decode_context_parallel_size"], 1)
        self.assertEqual(kw["worker_cls"], WORKER)
        self.assertTrue(kw["enable_expert_parallel"])
        self.assertIsNone(kw["kv_cache_memory_bytes"])
        self.assertEqual(plan["kv_budget_mode"], "automatic_profiling")
        self.assertEqual(kw["compilation_config"], before["model_kwargs"]["compilation_config"])
        for key in ("service_started", "actual_quality_passed", "actual_performance_observed",
                    "actual_kv_capacity_observed", "independent_tp_replicas"):
            self.assertFalse(plan[key])
        self.assertEqual(plan["environment"]["V41_ENGRAM_DEVICE_INDEX"], "0")

    def test_explicit_low_chip_plan_overwrites_all_visible_envs_and_binds_fingerprint(self):
        candidate = self.candidate()
        before = copy.deepcopy(candidate)
        selected = "0,1,2,3,4,5,6,7"
        legacy = self.plan(candidate)
        low = self.plan(candidate, physical_chips=selected)

        self.assertEqual(candidate, before)
        self.assertEqual(low["topology"]["physical_chips"], list(range(8)))
        for key in (
            "STACK_PHYSICAL_CHIPS", "STACK_EXPECTED_CHIPS",
            "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES",
        ):
            with self.subTest(key=key):
                self.assertEqual(low["environment"][key], selected)
                self.assertEqual(legacy["environment"][key], "8,9,10,11,12,13,14,15")
        self.assertNotEqual(low["runtime_fingerprint"], legacy["runtime_fingerprint"])
        self.assertFalse(low["actual_quality_passed"])

    def test_explicit_chip_plan_rejects_mixed_or_noncanonical_group(self):
        for selected in ("0,1,2,3,4,5,6,8", "8,9,10,11,12,13,14,14",
                         "9,8,10,11,12,13,14,15"):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                self.plan(physical_chips=selected)

    def test_stale_env_kv_pin_is_not_reintroduced(self):
        candidate = self.candidate()
        candidate["startup_env"]["KV_CACHE_MEMORY_BYTES"] = "4294967296"
        plan = self.plan(candidate)
        self.assertNotIn("KV_CACHE_MEMORY_BYTES", plan["environment"])
        self.assertIsNone(plan["model_kwargs"]["kv_cache_memory_bytes"])

    def test_loading_threads_reduction_changes_only_loader_budget(self):
        candidate = self.candidate()
        candidate["model_kwargs"]["model_loader_extra_config"] = {
            "enable_multithread_load": True, "num_threads": 128, "extra": "preserved"}
        candidate["model_kwargs"]["safetensors_load_strategy"] = "lazy"
        baseline = self.plan(candidate)
        reduced = self.plan(candidate, loading_threads=16)
        expected = copy.deepcopy(baseline["model_kwargs"])
        expected["model_loader_extra_config"]["num_threads"] = 16
        self.assertEqual(reduced["model_kwargs"], expected)
        self.assertEqual(reduced["environment"], baseline["environment"])
        self.assertNotEqual(reduced["runtime_fingerprint"], baseline["runtime_fingerprint"])
        self.assertEqual(candidate["model_kwargs"]["model_loader_extra_config"]["num_threads"], 128)
        self.assertFalse(reduced["actual_quality_passed"])
        for invalid in (True, 0, 129, 2.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.plan(candidate, loading_threads=invalid)
        candidate["model_kwargs"]["model_loader_extra_config"]["num_threads"] = 8
        with self.assertRaises(ValueError):
            self.plan(candidate, loading_threads=16)

    def test_loading_threads_does_not_silently_switch_loader(self):
        with self.assertRaisesRegex(ValueError, "only reduce"):
            self.plan(loading_threads=8)
        candidate = self.candidate()
        candidate["model_kwargs"]["model_loader_extra_config"] = {
            "enable_multithread_load": False, "num_threads": 128}
        with self.assertRaisesRegex(ValueError, "only reduce"):
            self.plan(candidate, loading_threads=8)

    def test_numa_policy_is_explicit_fingerprinted_and_not_inherited(self):
        candidate = self.candidate()
        candidate["startup_env"]["DP_MATRIX_NUMA_INTERLEAVE_NODES"] = "6,7"
        default = self.plan(candidate)
        self.assertNotIn("DP_MATRIX_NUMA_INTERLEAVE_NODES", default["environment"])
        explicit = self.plan(candidate, host_numa_nodes="0,1,2,3,7")
        self.assertEqual(explicit["environment"]["DP_MATRIX_NUMA_INTERLEAVE_NODES"], "0,1,2,3,7")
        self.assertEqual(explicit["model_kwargs"], default["model_kwargs"])
        self.assertNotEqual(explicit["runtime_fingerprint"], default["runtime_fingerprint"])

    def test_fixed_pool_not_maximum_capacity(self):
        plan = self.plan(kv_cache_memory_bytes=2 * 1024**3)
        self.assertEqual(plan["kv_budget_mode"], "fixed_pool")
        self.assertFalse(plan["actual_kv_capacity_observed"])

    def test_non_best_or_non_real_candidate_rejected(self):
        for key, value in (("tensor_parallel_size", 4), ("load_format", "dummy"),
                           ("enable_expert_parallel", False), ("async_scheduling", False)):
            candidate = self.candidate()
            candidate["model_kwargs"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.plan(candidate)

    def test_unsafe_budget_or_shape_rejected(self):
        for kwargs in ({"max_model_len": True}, {"max_model_len": 256},
                       {"gpu_memory_utilization": True}, {"gpu_memory_utilization": 0.99},
                       {"kv_cache_memory_bytes": 0}, {"batched_tokens": False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plan(**kwargs)

    def test_dcp2_not_admitted_by_old_dcp8_recipe(self):
        with self.assertRaisesRegex(ValueError, "exact pinned"):
            make_plan(self.candidate(), layout="dp4-tp2-dcp2", source="/frozen/source",
                      output="/new/job", served_model="dsv41-dp-new")

    def test_dcp2_plan_requires_explicit_manifest_pins_and_sets_only_requested_geometry(self):
        plan = make_plan(self.candidate(), layout="dp4-tp2-dcp2", source="/frozen/source",
                         output="/new/job", served_model="dsv41-dp-new",
                         dcp_overlay="/new/package/dcp2_runtime", dcp_draft="/new/package/draft",
                         dcp2_overlay_manifest_sha256="a" * 64,
                         dcp_draft_manifest_sha256=DCP2_DRAFT_MANIFEST_SHA256)
        kw, env = plan["model_kwargs"], plan["environment"]
        self.assertEqual((kw["data_parallel_size"], kw["tensor_parallel_size"],
                          kw["decode_context_parallel_size"]), (4, 2, 2))
        self.assertEqual(env["V41_DCP"], "1")
        self.assertEqual(env["V41_DCP_REPLICATE_INDEXER"], "1")
        self.assertEqual(env["DELIVERY_DCP_OVERLAY"], "/new/package/dcp2_runtime")
        self.assertEqual(env["DP_MATRIX_DCP2_OVERLAY_MANIFEST_SHA256"], "a" * 64)
        self.assertFalse(plan["actual_quality_passed"])


if __name__ == "__main__":
    unittest.main()
