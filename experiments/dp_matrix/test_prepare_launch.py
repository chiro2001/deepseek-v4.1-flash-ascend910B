"""CPU-only launch/preflight contract tests; no launch, HTTP, native or NPU."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from experiments.dp_matrix import freeze_package, service as dp_service
from experiments.delivery_matrix.service import fingerprint
from experiments.dp_matrix.launch_plan import WORKER, DCP2_DRAFT_MANIFEST_SHA256
from experiments.dp_matrix.prepare_launch import CLEAR, shell
from experiments.dp_matrix.service import verify_native_files


NATIVE_NAMES = ("core_client.py", "input_processor.py", "serving.py")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _candidate():
    stale_environment = {key: "stale-value" for key in CLEAR}
    stale_environment.update({
        "STACK_PHYSICAL_CHIPS": "8,9,10,11,12,13,14,15",
        "STACK_EXPECTED_CHIPS": "8,9,10,11,12,13,14,15",
        "ASCEND_RT_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15",
        "ASCEND_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15",
        "DRAFT_GRAPH": "1",
        "DSPARK_GRAPH_CAPTURE_METADATA": "1",
        "VLLM_BATCH_INVARIANT": "1",
        "ASCEND_CUSTOM_OPP_PATH": "/pinned/opp with spaces",
        "DSPARK_ASYNC_COUNTER": "1",
        "V41_ENGRAM_DEVICE_INDEX": "0",
        "HCCL_DETERMINISTIC": "strict",
        "SDK_PIN": "sdk-v1.2.3",
    })
    return {
        "model_kwargs": {
            "model": "/models/full w4a8 model",
            "tokenizer": "/models/full w4a8 model",
            "load_format": "auto",
            "dtype": "bfloat16",
            "model_loader_extra_config": {"enable_multithread_load": True, "num_threads": 128},
            "trust_remote_code": True,
            "tensor_parallel_size": 8,
            "worker_cls": "frozen.best.Worker",
            "enable_expert_parallel": True,
            "async_scheduling": True,
            "enable_prefix_caching": False,
            "speculative_config": {"method": "dspark", "num_speculative_tokens": 5},
            "additional_config": {"enable_engram": True, "engram_storage": "int8"},
            "kv_cache_memory_bytes": 4 * 1024**3,
            "compilation_config": {"cudagraph_capture_sizes": [6]},
        },
        "startup_env": stale_environment,
    }


def _call_shell(candidate=None, *, port=18960, job_root="/tmp/job with spaces"):
    return shell(
        candidate or _candidate(),
        layout="dp2-tp4",
        source="/frozen source/tree with spaces",
        package="/delivery package with spaces",
        generic_package="/generic package with spaces",
        runtime="/runtime derived with spaces",
        native_guards="/guards/native sources.json",
        job_root=job_root,
        served_model="dsv41-dp-test-contract",
        port=port,
        max_model_len=196608,
        gpu_memory_utilization=0.75,
    )


class PrepareLaunchTests(unittest.TestCase):
    def test_explicit_numa_policy_cli_matches_reexport_after_clear(self):
        body, plan = shell(_candidate(), layout="dp2-tp4", source="/frozen/source",
                           package="/new/package/dp_matrix", generic_package="/pinned/generic",
                           runtime="/new/package/runtime", native_guards="/new/guards.json",
                           job_root="/new/job", served_model="dsv41-dp-test-numa", port=18960,
                           host_numa_nodes="0,1,2,3,7")
        self.assertIn("--host-numa-nodes 0,1,2,3,7", body)
        self.assertIn("export DP_MATRIX_NUMA_INTERLEAVE_NODES=0,1,2,3,7", body)
        self.assertLess(body.index("unset "), body.index("export DP_MATRIX_NUMA_INTERLEAVE_NODES="))
        self.assertEqual(plan["environment"]["DP_MATRIX_NUMA_INTERLEAVE_NODES"], "0,1,2,3,7")

    def test_loading_threads_cli_and_plan_are_consistent(self):
        body, plan = shell(_candidate(), layout="dp2-tp4", source="/frozen/source",
                           package="/new/package/dp_matrix", generic_package="/pinned/generic",
                           runtime="/new/package/runtime", native_guards="/new/guards.json",
                           job_root="/new/job", served_model="dsv41-dp-test-threads", port=18960,
                           loading_threads=16)
        self.assertIn("--loading-threads 16", body)
        self.assertEqual(plan["model_kwargs"]["model_loader_extra_config"]["num_threads"], 16)
        default_body, default_plan = _call_shell()
        self.assertNotIn("--loading-threads", default_body)
        self.assertEqual(default_plan["model_kwargs"]["model_loader_extra_config"]["num_threads"], 128)

    def test_explicit_physical_chips_cli_and_launch_environment_are_consistent(self):
        candidate = _candidate()
        before = json.loads(json.dumps(candidate))
        selected = "0,1,2,3,4,5,6,7"
        body, plan = shell(
            candidate, layout="dp2-tp4", source="/frozen/source",
            package="/new/package/dp_matrix", generic_package="/pinned/generic",
            runtime="/new/package/runtime", native_guards="/new/guards.json",
            job_root="/new/job", served_model="dsv41-dp-test-physical", port=18960,
            physical_chips=selected,
        )
        self.assertEqual(candidate, before)
        self.assertIn(f"--physical-chips {selected}", body)
        self.assertEqual(plan["topology"]["physical_chips"], list(range(8)))
        for key in (
            "STACK_PHYSICAL_CHIPS", "STACK_EXPECTED_CHIPS",
            "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES",
        ):
            with self.subTest(key=key):
                self.assertIn(f"export {key}={selected}", body)
                self.assertEqual(plan["environment"][key], selected)

    def test_prepare_launch_command_line_accepts_and_propagates_physical_chips(self):
        selected = "0,1,2,3,4,5,6,7"
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            candidate_path = root / "candidate.json"
            candidate_path.write_text(json.dumps(_candidate()))
            output = root / "launch.sh"
            command = [
                sys.executable, "-m", "experiments.dp_matrix.prepare_launch",
                "--candidate", str(candidate_path), "--source", "/frozen/source",
                "--package", "/new/package/dp_matrix", "--generic-package", "/pinned/generic",
                "--runtime", "/new/package/runtime", "--native-guards", "/new/guards.json",
                "--job-root", "/new/job", "--served-model", "dsv41-dp-cli-physical",
                "--output", str(output), "--layout", "dp2-tp4", "--port", "18960",
                "--physical-chips", selected,
            ]
            env = dict(os.environ)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True,
                                    capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            body = output.read_text()
        self.assertIn(f"--physical-chips {selected}", body)
        for key in (
            "STACK_PHYSICAL_CHIPS", "STACK_EXPECTED_CHIPS",
            "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES",
        ):
            self.assertIn(f"export {key}={selected}", body)

    def test_freeze_package_cli_binds_explicit_physical_chips(self):
        selected = "0,1,2,3,4,5,6,7"
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            candidate_path = root / "candidate.json"
            candidate_path.write_text(json.dumps(_candidate()))
            guard_map = {}
            for index, name in enumerate(NATIVE_NAMES):
                path = root / f"guard-{index}" / name
                path.parent.mkdir()
                path.write_text(f"# CPU fixture for {name}\n")
                guard_map[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
            guards_path = root / "native-guards.json"
            guards_path.write_text(json.dumps(guard_map))
            output = root / "frozen-package"
            arguments = [
                "--output", str(output), "--native-guards", str(guards_path),
                "--candidate", str(candidate_path), "--remote-package", "/remote/package",
                "--remote-job", "/remote/job", "--source", "/frozen/source",
                "--generic-package", "/generic/package", "--served-model", "dsv41-dp-freeze",
                "--layout", "dp2-tp4", "--port", "18960", "--physical-chips", selected,
            ]
            with patch.object(sys, "argv", ["freeze-package", *arguments]), \
                    redirect_stdout(io.StringIO()):
                freeze_package.main()
            plan = json.loads((output / "prepared_plan.json").read_text())
            launch = (output / "launch.sh").read_text()
        self.assertEqual(plan["topology"]["physical_chips"], list(range(8)))
        self.assertIn(f"--physical-chips {selected}", launch)
        for key in (
            "STACK_PHYSICAL_CHIPS", "STACK_EXPECTED_CHIPS",
            "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES",
        ):
            self.assertEqual(plan["environment"][key], selected)
            self.assertIn(f"export {key}={selected}", launch)

    def test_service_cli_rejects_environment_not_matching_explicit_chip_argument(self):
        selected = "0,1,2,3,4,5,6,7"
        legacy = "8,9,10,11,12,13,14,15"
        arguments = [
            "--source", "/missing/source", "--output", "/missing/output",
            "--runtime", "/missing/runtime", "--native-source-guards", "/missing/guards.json",
            "--layout", "dp2-tp4", "--served-model", "dsv41-dp-service-cli",
            "--port", "18960", "--physical-chips", selected,
        ]
        environment = {
            "STACK_PHYSICAL_CHIPS": legacy, "STACK_EXPECTED_CHIPS": legacy,
            "ASCEND_RT_VISIBLE_DEVICES": legacy, "ASCEND_VISIBLE_DEVICES": legacy,
        }
        with patch.dict(os.environ, environment, clear=True), \
                patch.object(sys, "argv", ["dp-service", *arguments]):
            with self.assertRaisesRegex(ValueError, "Physical launch environment and service CLI differ"):
                dp_service.main()

    def test_dcp2_clears_inherited_settings_then_reexports_only_pinned_dcp_plan(self):
        body, plan = shell(_candidate(), layout="dp4-tp2-dcp2", source="/frozen/source",
                           package="/new/package/dp_matrix", generic_package="/pinned/generic",
                           runtime="/new/package/runtime", native_guards="/new/guards.json",
                           job_root="/new/job", served_model="dsv41-dp-test-dcp2", port=18960,
                           dcp_overlay="/new/package/dcp2_runtime", dcp_draft="/new/package/dcp_draft",
                           dcp2_overlay_manifest_sha256="a" * 64,
                           dcp_draft_manifest_sha256=DCP2_DRAFT_MANIFEST_SHA256)
        self.assertIn("export V41_DCP=1", body)
        self.assertIn("export V41_DCP_REPLICATE_INDEXER=1", body)
        self.assertIn("export DELIVERY_DCP_OVERLAY=/new/package/dcp2_runtime", body)
        self.assertIn("export DP_MATRIX_DCP2_BOOTSTRAP=/new/package/dp_matrix/dcp2_bootstrap", body)
        self.assertIn("--dcp2-overlay /new/package/dcp2_runtime", body)
        self.assertNotIn("export V41_CED_ROLE=", body)
        self.assertNotIn("export V41_DCP_ALLOW_CAPACITY_PROBE=", body)
        self.assertEqual(plan["environment"]["V41_DCP_DET_REDUCE"], "1")

    def test_shell_clears_legacy_environment_before_python_entry(self):
        body, plan = _call_shell()
        lines = body.splitlines()
        unset_index = next(i for i, line in enumerate(lines) if line.startswith("unset "))
        export_index = next(i for i, line in enumerate(lines) if line.startswith("export "))
        command_index = next(i for i, line in enumerate(lines) if "run_service.sh" in line)
        unset = set(lines[unset_index].split()[1:])

        self.assertEqual(unset, set(CLEAR))
        required_clear = {
            "STACK_TP8_ARM", "V41_CED_ROLE", "DELIVERY_CED_OVERLAY",
            "DELIVERY_CED_COMPRESSOR", "V41_DCP", "DELIVERY_DCP_OVERLAY",
            "DELIVERY_DCP_DRAFT", "DELIVERY_SHARED_ENGRAM",
            "DELIVERY_SHARED_ENGRAM_MANIFEST_SHA256", "V41_ENGRAM_SHARED_HOST_MMAP",
            "KV_CACHE_MEMORY_BYTES",
        }
        self.assertTrue(required_clear.issubset(unset))
        self.assertLess(unset_index, export_index)
        self.assertLess(unset_index, command_index)
        self.assertLess(export_index, command_index)
        for key in CLEAR:
            self.assertNotIn(f"export {key}=", body)
        self.assertIn("DELIVERY_FORM=dp2-tp4", body)
        self.assertIn("STACK_SRC_ROOT='/frozen source/tree with spaces'", body)
        self.assertFalse(plan["service_started"])

    def test_plan_fingerprints_dp_kwargs_and_retains_quality_critical_source(self):
        candidate = _candidate()
        before = {
            "model_kwargs": dict(candidate["model_kwargs"]),
            "startup_env": dict(candidate["startup_env"]),
        }
        body, plan = _call_shell(candidate)

        self.assertEqual(candidate, before)
        kw = plan["model_kwargs"]
        self.assertEqual(
            (kw["data_parallel_size"], kw["data_parallel_size_local"],
             kw["tensor_parallel_size"], kw["decode_context_parallel_size"]),
            (2, 2, 4, 1),
        )
        self.assertTrue(kw["enable_expert_parallel"])
        self.assertIsNone(kw["kv_cache_memory_bytes"])
        self.assertEqual(plan["kv_budget_mode"], "automatic_profiling")
        self.assertEqual(kw["max_model_len"], 196608)
        self.assertEqual(kw["max_num_batched_tokens"], 8192)
        self.assertEqual(kw["gpu_memory_utilization"], 0.75)
        self.assertFalse(kw["enable_prefix_caching"])
        self.assertEqual(kw["load_format"], "auto")
        self.assertEqual(kw["worker_cls"], WORKER)
        self.assertEqual(kw["model"], candidate["model_kwargs"]["model"])
        self.assertEqual(kw["tokenizer"], candidate["model_kwargs"]["tokenizer"])
        self.assertEqual(kw["model_loader_extra_config"], candidate["model_kwargs"]["model_loader_extra_config"])
        self.assertEqual(kw["dtype"], "bfloat16")
        self.assertTrue(kw["trust_remote_code"])
        self.assertTrue(kw["additional_config"]["enable_engram"])
        self.assertEqual(kw["additional_config"]["engram_storage"], "int8")
        self.assertEqual(kw["speculative_config"], {"method": "dspark", "num_speculative_tokens": 5})

        env = plan["environment"]
        for key, value in (
            ("DRAFT_GRAPH", "1"),
            ("DSPARK_GRAPH_CAPTURE_METADATA", "1"),
            ("VLLM_BATCH_INVARIANT", "1"),
            ("ASCEND_CUSTOM_OPP_PATH", "/pinned/opp with spaces"),
            ("DSPARK_ASYNC_COUNTER", "1"),
            ("V41_ENGRAM_DEVICE_INDEX", "0"),
            ("HCCL_DETERMINISTIC", "strict"),
            ("SDK_PIN", "sdk-v1.2.3"),
        ):
            self.assertEqual(env[key], value)
        self.assertEqual(plan["source_candidate_fingerprint"], fingerprint(candidate))
        self.assertEqual(plan["source_root"], "/frozen source/tree with spaces")
        self.assertTrue(plan["native_data_parallel_required"])
        self.assertRegex(plan["base_tp8_plan_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertRegex(plan["runtime_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(plan["runtime_fingerprint"], plan["base_tp8_plan_fingerprint"])
        self.assertIn("service_started", plan)
        self.assertFalse(plan["service_started"])
        for key in ("actual_quality_passed", "actual_performance_observed",
                    "actual_kv_capacity_observed"):
            self.assertFalse(plan[key])
        self.assertIn("--native-source-guards", body)
        self.assertIn("--execute", body)

    def test_runtime_fingerprint_binds_dp_layout(self):
        _body2, dp2tp4 = _call_shell()
        candidate = _candidate()
        _body4, dp4tp2 = shell(
            candidate,
            layout="dp4-tp2",
            source="/frozen source/tree with spaces",
            package="/delivery package with spaces",
            generic_package="/generic package with spaces",
            runtime="/runtime derived with spaces",
            native_guards="/guards/native sources.json",
            job_root="/tmp/job with spaces",
            served_model="dsv41-dp-test-contract",
            port=18960,
            max_model_len=196608,
            gpu_memory_utilization=0.75,
        )
        self.assertEqual(dp4tp2["model_kwargs"]["data_parallel_size"], 4)
        self.assertEqual(dp4tp2["model_kwargs"]["tensor_parallel_size"], 2)
        self.assertNotEqual(dp2tp4["runtime_fingerprint"], dp4tp2["runtime_fingerprint"])

    def test_generated_script_with_spaces_passes_bash_syntax_check(self):
        tmp_root = os.environ.get("TMPDIR")
        with tempfile.TemporaryDirectory(dir=tmp_root) as directory:
            script = Path(directory) / "generated launch script.sh"
            body, _ = _call_shell(job_root=str(Path(directory) / "new job root"))
            script.write_text(body)
            result = subprocess.run(
                ["bash", "-n", str(script)],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_non_integer_and_out_of_range_ports(self):
        for port in (True, False, 80, 0, -1, 65536, 18960.0, "18960", None):
            with self.subTest(port=port), self.assertRaisesRegex(ValueError, "unprivileged port"):
                _call_shell(port=port)

    def test_accepts_unprivileged_port_boundaries(self):
        for port in (1024, 65535):
            with self.subTest(port=port):
                body, _ = _call_shell(port=port)
                self.assertIn(f"--port {port}", body)


class NativeSourceGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "native sources with spaces"
        self.root.mkdir()
        self.sources = {}
        for index, name in enumerate(NATIVE_NAMES):
            source_dir = self.root / f"module copy {index}"
            source_dir.mkdir()
            path = source_dir / name
            path.write_text(f"# fake immutable source: {name}\nVALUE = {index}\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.sources[name] = {"path": str(path.resolve()), "sha256": digest}

    def _nested(self, sources=None):
        return {"schema": "native_source_guard_v1", "sources": sources or self.sources}

    def _flat(self):
        return {record["path"]: record["sha256"] for record in self.sources.values()}

    def test_accepts_site_nested_guard_and_normalizes_to_flat_map(self):
        result = verify_native_files(self._nested())
        self.assertEqual(result, self._flat())

    def test_accepts_flat_guard_map(self):
        result = verify_native_files(self._flat())
        self.assertEqual(result, self._flat())

    def test_accepts_additional_kv_guard_in_nested_and_flat_maps(self):
        extra_dir = self.root / "additional kv metadata"
        extra_dir.mkdir()
        extra_path = extra_dir / "kv_cache_metadata.py"
        extra_path.write_text("# fake extra metadata source\n")
        extra = {"path": str(extra_path.resolve()),
                 "sha256": hashlib.sha256(extra_path.read_bytes()).hexdigest()}

        nested_sources = dict(self.sources)
        nested_sources["kv_cache_metadata.py"] = extra
        nested = verify_native_files(self._nested(nested_sources))

        flat = self._flat()
        flat[extra["path"]] = extra["sha256"]
        flat_result = verify_native_files(flat)

        self.assertEqual(nested, flat)
        self.assertEqual(flat_result, flat)

    def test_rejects_single_unrelated_flat_source(self):
        unrelated = self.root / "misc" / "unrelated.py"
        unrelated.parent.mkdir()
        unrelated.write_text("# not a guarded native source\n")
        guards = {str(unrelated.resolve()): hashlib.sha256(unrelated.read_bytes()).hexdigest()}
        with self.assertRaisesRegex(ValueError, "must cover core-client"):
            verify_native_files(guards)

    def test_rejects_missing_nested_source(self):
        sources = dict(self.sources)
        sources.pop("input_processor.py")
        with self.assertRaisesRegex(ValueError, "All three actual native"):
            verify_native_files(self._nested(sources))

    def test_rejects_non_absolute_path_and_malformed_hash(self):
        relative = {
            f"relative/{name}": record["sha256"]
            for name, record in self.sources.items()
        }
        with self.assertRaisesRegex(ValueError, "absolute path"):
            verify_native_files(relative)

        malformed = self._flat()
        first = next(iter(malformed))
        malformed[first] = "z" * 64
        with self.assertRaisesRegex(ValueError, "absolute path"):
            verify_native_files(malformed)

    def test_rejects_valid_but_wrong_hash(self):
        wrong = self._flat()
        first = next(iter(wrong))
        wrong[first] = "0" * 64 if wrong[first] != "0" * 64 else "1" * 64
        with self.assertRaisesRegex(ValueError, "changed since preflight"):
            verify_native_files(wrong)

    def test_rejects_source_modified_after_guard_was_created(self):
        path = Path(self.sources["serving.py"]["path"])
        path.write_text("# changed after site preflight\n")
        with self.assertRaisesRegex(ValueError, "changed since preflight"):
            verify_native_files(self._nested())

    def test_missing_file_fails_closed(self):
        guards = self._flat()
        missing = str((self.root / "missing source.py").resolve())
        guards[missing] = hashlib.sha256(b"expected bytes").hexdigest()
        with self.assertRaises(OSError):
            verify_native_files(guards)

    def test_importing_prepare_and_guard_modules_does_not_load_runtime(self):
        code = """
import sys
import experiments.dp_matrix.prepare_launch
import experiments.dp_matrix.service
forbidden = ("torch", "torch_npu", "vllm", "httpx", "requests", "urllib.request")
loaded = [name for name in sys.modules if any(name == item or name.startswith(item + ".") for item in forbidden)]
if loaded:
    raise SystemExit("forbidden modules imported: " + repr(loaded))
"""
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.getcwd(),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
