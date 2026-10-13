"""CPU-only contracts for native-DP workload dispatch and measured wave shape."""
from __future__ import annotations

import json
import hashlib
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from experiments.dp_matrix import benchmark
from experiments.dp_matrix.code8_replay import MANIFEST_SHA256 as CODE8_MANIFEST_SHA256
from experiments.dp_matrix.topology import LOW_CHIPS, Topology, topology_from_args


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = PROJECT_ROOT / "experiments/delivery_matrix/evidence/a2sim_tp8_v4"
CODE8_ROOT = EVIDENCE / "client/code_formal8"
DIHUO_MANIFESTS = {
    2048: (EVIDENCE / "client/dihuo_2k_a2sim_v4_20261013/manifest.json",
           "afb141d345f61c84e9ab7a0b5f0682f4bb2ea918dee029636e2fbbc65cbb94ec"),
    32768: (EVIDENCE / "client/dihuo_32k_a2sim_v4_20261013/manifest.json",
            "20a870425389f8f78d9141b8f5c3242c3b07975a8318688bf7bd24c3124ad402"),
    147456: (EVIDENCE / "client/dihuo_147456_a2sim_v4_20261013/manifest.json",
             "365880e5cd26f14bea085bd8fa67c596683c12c169e215d190b0f6bef0128995"),
}
TASK_TMP_ROOT = Path.home() / "tmp" / "dsv41-dp-benchmark-tests"
TASK_TMP_ROOT.mkdir(parents=True, exist_ok=True)


class NativeDPBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="run-", dir=TASK_TMP_ROOT)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def args(self, *, workload, layout="dp2-tp4", mode="serial", input_tokens=2048,
             output_tokens=None, manifest=None, manifest_sha256=None, requests=None,
             output="result", physical_chips="8,9,10,11,12,13,14,15"):
        if workload == "code":
            manifest = manifest or CODE8_ROOT / "manifest.json"
            manifest_sha256 = manifest_sha256 or CODE8_MANIFEST_SHA256
            requests = requests or CODE8_ROOT / "requests"
            output_tokens = 1024 if output_tokens is None else output_tokens
        else:
            manifest, digest = DIHUO_MANIFESTS[input_tokens]
            manifest_sha256 = digest if manifest_sha256 is None else manifest_sha256
            output_tokens = 256 if output_tokens is None else output_tokens
        return SimpleNamespace(
            base_url="http://127.0.0.1:18960", model="dsv41-dp-test", layout=layout,
            sid="a" * 32, fingerprint="b" * 64, manifest=str(manifest),
            manifest_sha256=manifest_sha256, requests=None if requests is None else str(requests),
            input_tokens=input_tokens, output_tokens=output_tokens, workload=workload,
            output=str(self.root / output), mode=mode, rounds=1, timeout=5.0,
            physical_chips=physical_chips,
        )

    @staticmethod
    def identity(args, *, physical_chips=None):
        chips = args.physical_chips if physical_chips is None else physical_chips
        topology = Topology.named(args.layout, chips=chips)
        rows = []
        for index, row in enumerate(topology.workers()):
            tp_start = row["dp_rank"] * topology.tp
            dcp_start = tp_start + (row["tp_rank"] // topology.dcp) * topology.dcp
            rows.append({
                **row, "dp": topology.dp, "tp": topology.tp, "dcp": topology.dcp,
                "ep_world_size": 8, "enable_expert_parallel": True,
                "worker_pid": 100 + index,
                "tp_group_ranks": list(range(tp_start, tp_start + topology.tp)),
                "dcp_group_ranks": list(range(dcp_start, dcp_start + topology.dcp)),
                "ep_group_ranks": list(range(8)),
            })
        return {
            "served_model": args.model, "service_instance_id": args.sid,
            "runtime_fingerprint": args.fingerprint, "layout": args.layout,
            "engine_initialized": True, "ASGI_startup_completed": True,
            "rank_descriptions": rows,
        }

    def run_with_fake_service(self, args, *, identity_chips=None):
        http_calls = []
        arm_payloads = []
        streams = []

        def http_json(base, path, *, payload=None, timeout=120):
            http_calls.append((path, payload))
            if path == "/dp/identity":
                return self.identity(args, physical_chips=identity_chips)
            if path == "/v1/models":
                return {"data": [{"id": args.model}]}
            if path == "/dp/arm":
                arm_payloads.append(dict(payload))
                return {"dp_rank": payload["dp_rank"], "tag": payload["tag"], "ranks": []}
            if path.startswith("/dp/metrics?"):
                query = parse_qs(urlsplit(path).query)
                return {"dp_rank": int(query["dp_rank"][0]), "tag": query["tag"][0]}
            raise AssertionError("Unexpected fake service request: " + path)

        def stream(base, model, messages, *, dp_rank, tag, max_tokens, seed,
                   ignore_eos, timeout, barrier=None):
            if barrier is not None:
                barrier.wait(timeout=5)
            now = time.monotonic()
            completion = max_tokens if ignore_eos else 320
            output_text = "def generated_function():\n    return 1\n"
            row = {
                "request_tag": tag, "payload": {"messages": messages},
                "http_status": 200, "dp_rank": dp_rank, "done": True, "error": None,
                "usage": {"prompt_tokens": args.input_tokens,
                          "completion_tokens": completion,
                          "total_tokens": args.input_tokens + completion},
                "finish_reason": "length" if ignore_eos else "stop",
                "ttft_s": 0.1, "decode_s": 0.2,
                "first_content_monotonic": now,
                "last_content_monotonic": now + 0.01,
                "output_text": output_text,
                "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
                "clean": {"clean": True},
                "ignore_eos_seen_by_stream": ignore_eos,
                "max_tokens_seen_by_stream": max_tokens,
            }
            streams.append(row)
            return row

        def request_metrics(arm, metrics, stream_row, topology, **kwargs):
            return {"dp_rank": kwargs["dp_rank"], "ignore_eos": kwargs["ignore_eos"],
                    "max_tokens": kwargs["max_tokens"]}

        with (patch.object(benchmark, "http_json", side_effect=http_json),
              patch.object(benchmark, "stream", side_effect=stream),
              patch.object(benchmark, "request_metrics", side_effect=request_metrics)):
            with redirect_stdout(io.StringIO()):
                benchmark.run(args)
        return http_calls, arm_payloads, streams

    def test_code_and_dihuo_keep_distinct_output_budgets_and_eos_policy(self):
        code_args = self.args(workload="code", input_tokens=2048, output="code")
        _, code_arms, code_streams = self.run_with_fake_service(code_args)
        self.assertEqual([row["max_tokens"] for row in code_arms], [1024] * 8)
        self.assertEqual([row["ignore_eos_seen_by_stream"] for row in code_streams], [False] * 8)
        self.assertEqual([row["finish_reason"] for row in code_streams], ["stop"] * 8)
        code_proof = json.loads((Path(code_args.output) / "input_proof.json").read_text())
        self.assertEqual(code_proof["workload"], "code")
        code_response = json.loads(
            (Path(code_args.output) / "code-outputs-r00/request-00.json").read_text())
        self.assertEqual(code_response["scope"],
                         "separate armed performance request; not ordinary Code8 quality gate")
        self.assertEqual(code_response["http_status"], 200)

        dihuo_args = self.args(workload="dihuo", input_tokens=2048, output="dihuo")
        _, dihuo_arms, dihuo_streams = self.run_with_fake_service(dihuo_args)
        self.assertEqual([row["max_tokens"] for row in dihuo_arms], [256] * 8)
        self.assertEqual([row["ignore_eos_seen_by_stream"] for row in dihuo_streams], [True] * 8)
        self.assertEqual([row["finish_reason"] for row in dihuo_streams], ["length"] * 8)

    def test_replica_mode_arms_one_distinct_tag_per_dp_per_wave(self):
        args = self.args(workload="dihuo", layout="dp4-tp2", mode="replicas",
                         input_tokens=32768, output="replicas")
        _, arms, streams = self.run_with_fake_service(args)
        self.assertEqual(len(arms), 8)
        self.assertEqual([row["dp_rank"] for row in arms], [0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(len({row["tag"] for row in arms}), 8)
        self.assertEqual([sum(row["dp_rank"] == rank for row in streams)
                          for rank in range(4)], [2, 2, 2, 2])
        waves = sorted(Path(args.output).glob("wave-r00-i*.json"))
        self.assertEqual(len(waves), 2)
        self.assertTrue(all(json.loads(path.read_text())["maximum_saturation_proven"] is False
                            for path in waves))

    def test_invalid_code_or_dihuo_input_is_rejected_before_any_service_get(self):
        code_root = self.root / "bad-code"
        code_requests = code_root / "requests"
        code_requests.mkdir(parents=True)
        bad_manifest = code_root / "manifest.json"
        bad_manifest.write_bytes((CODE8_ROOT / "manifest.json").read_bytes() + b" ")
        cases = [self.args(workload="code", manifest=bad_manifest, requests=code_requests,
                           output="bad-code-output"),
                 self.args(workload="dihuo", input_tokens=2048, manifest_sha256="0" * 64,
                           output="bad-dihuo-output")]
        for args in cases:
            with self.subTest(workload=args.workload), \
                    patch.object(benchmark, "http_json", side_effect=AssertionError("HTTP before input gate")):
                with self.assertRaises(ValueError):
                    benchmark.run(args)
                self.assertFalse(Path(args.output).exists())

    def test_explicit_low_chip_group_binds_benchmark_identity(self):
        args = self.args(workload="dihuo", physical_chips=",".join(map(str, LOW_CHIPS)),
                         output="low-chips")
        self.assertEqual(topology_from_args(args).chips, LOW_CHIPS)
        self.run_with_fake_service(args)
        identity_doc = json.loads((Path(args.output) / "identity.json").read_text())
        self.assertEqual(
            [row["physical_chip"] for row in identity_doc["rank_descriptions"]], list(LOW_CHIPS)
        )

    def test_low_chip_benchmark_refuses_legacy_eight_through_fifteen_identity(self):
        args = self.args(workload="dihuo", physical_chips=",".join(map(str, LOW_CHIPS)),
                         output="wrong-chip-identity")
        with self.assertRaisesRegex(ValueError, "Unexpected worker physical_chip"):
            self.run_with_fake_service(args, identity_chips=tuple(range(8, 16)))


if __name__ == "__main__":
    unittest.main()
