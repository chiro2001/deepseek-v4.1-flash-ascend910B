"""CPU-only positive/negative controls for ordinary Code8 scoring gates.

The scorer is replaced with a deterministic stub. These tests never execute
generated code, contact a service, or load a device runtime.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from experiments.dp_matrix import code8_score
from experiments.dp_matrix.code8_replay import MANIFEST_SHA256, load_inputs
from experiments.dp_matrix.topology import LOW_CHIPS, Topology


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = PROJECT_ROOT / "experiments/delivery_matrix/evidence/a2sim_tp8_v4/client/code_formal8"
WORKLOADS = PROJECT_ROOT / "experiments/delivery_matrix/workloads.py"
MODEL = "dsv41-dp-code8-test"
SID = "a" * 32
FINGERPRINT = "b" * 64
LAYOUT = "dp2-tp4"
TASK_TMP_ROOT = Path.home() / "tmp" / "dsv41-dp-code8-score-tests"
TASK_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(topology):
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
    return {"served_model": MODEL, "service_instance_id": SID,
            "runtime_fingerprint": FINGERPRINT, "layout": topology.name,
            "engine_initialized": True, "ASGI_startup_completed": True,
            "rank_descriptions": rows}


class FakeWorkloads:
    def __init__(self, cases_by_task, calls):
        self.cases_by_task = cases_by_task
        self.calls = calls

    def fixed_unit_cases(self, task_id):
        return self.cases_by_task[task_id]

    def score_code_output(self, task_id, output, **limits):
        self.calls.append((task_id, output, limits))
        total = len(self.cases_by_task[task_id])
        return {"status": "pass", "passed": total, "total": total,
                "failed_case_ids": [], "isolation": "mocked; no generated code executed"}


class Code8ScoreGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="score-", dir=TASK_TMP_ROOT)
        self.root = Path(self.temp.name)
        self.inputs = load_inputs(EVIDENCE / "manifest.json", EVIDENCE / "requests")
        self.topology = Topology.named(LAYOUT)
        self.scorer_calls = []
        self.workloads = FakeWorkloads(
            {sample["task_id"]: sample["unit_cases"] for sample in self.inputs["samples"]},
            self.scorer_calls)
        self.responses = self.root / "responses"
        self.responses.mkdir()
        self.proxy = self.root / "proxy"
        self.proxy.mkdir()
        self.write_identity()
        self.write_valid_wire_and_responses()

    def tearDown(self):
        self.temp.cleanup()

    def write_identity(self, topology=None):
        doc = identity(self.topology if topology is None else topology)
        (self.responses / "identity_before.json").write_text(json.dumps(doc) + "\n")
        (self.responses / "identity_after.json").write_text(json.dumps(doc) + "\n")

    def write_valid_wire_and_responses(self):
        journal = []
        for sample in self.inputs["samples"]:
            index = sample["index"]
            request_id = f"code8-test-{index}"
            prompt = sample["messages"]
            payload = {"model": MODEL, "messages": prompt, "temperature": 0,
                       "seed": sample["seed"], "request_id": request_id,
                       "max_tokens": 1024, "stream": True,
                       "stream_options": {"include_usage": True}}
            request_raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request_gzip = gzip.compress(request_raw, mtime=0)
            request_name = f"case-{index}.request.raw.gz"
            (self.proxy / request_name).write_bytes(request_gzip)

            text = f"output-for-case-{index}"
            response_raw = (
                b"data: " + json.dumps({
                    "choices": [{"index": 0, "delta": {"content": text},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2048, "completion_tokens": 4,
                              "total_tokens": 2052},
                }).encode("utf-8") + b"\n" + b"data: [DONE]\n"
            )
            response_name = f"case-{index}.response.raw"
            (self.proxy / response_name).write_bytes(response_raw)
            rank = index % self.topology.dp
            body_hash = digest(request_raw)
            journal.append({
                "schema": "dp_quality_proxy_attempt_v1", "generation": True,
                "request_body_sha256": body_hash, "request_body_bytes": len(request_raw),
                "request_body_file": request_name,
                "request_gzip_sha256": digest(request_gzip),
                "selected_dp_rank": rank, "response_dp_rank": rank,
                "route_matches_response": True, "http_status": 200, "error": None,
                "response_closed": True, "response_is_sse": True, "done_consumed": True,
                "unparsed_sse_data_lines": 0,
                "response_body_sha256": digest(response_raw),
                "response_body_bytes": len(response_raw),
                "response_body_file": response_name,
                "usage": {"prompt_tokens": 2048, "completion_tokens": 4,
                          "total_tokens": 2052},
                "finish_reasons": [{"index": 0, "finish_reason": "stop"}],
            })
            (self.responses / f"request-{index:02d}.json").write_text(json.dumps({
                "schema": "native_dp_code8_response_v1", "request_scope": "ordinary_quality",
                "service_instance_id": SID, "runtime_fingerprint": FINGERPRINT,
                "served_model": MODEL, "layout": LAYOUT,
                "sample_id": sample["sample_id"], "request_id": request_id,
                "request_body_sha256": body_hash, "task_id": sample["task_id"],
                "split_artifact_sha256": sample["artifact_sha256"],
                "prompt_sha256": sample["prompt_sha256"],
                "unit_cases_sha256": sample["unit_cases_sha256"],
                "http_status": 200, "finish_reason": "stop", "error": None,
                "usage": {"prompt_tokens": 2048, "completion_tokens": 4,
                          "total_tokens": 2052},
                "output_text": text, "output_sha256": digest(text.encode("utf-8")),
                "response_complete_by_frozen_client": True,
            }, ensure_ascii=False) + "\n")
        (self.proxy / "attempts.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in journal))

    def args(self, *, scope="ordinary_quality", sid=SID, output="score.json",
             physical_chips="8,9,10,11,12,13,14,15"):
        return SimpleNamespace(
            manifest=EVIDENCE / "manifest.json", manifest_sha256=MANIFEST_SHA256,
            requests=EVIDENCE / "requests", workloads=WORKLOADS,
            responses=self.responses, output=self.root / output,
            sid=sid, fingerprint=FINGERPRINT, model=MODEL, layout=LAYOUT,
            scope=scope, proxy_root=self.proxy,
            timeout=3.0, memory_mb=512, cpu_seconds=2,
            physical_chips=physical_chips,
        )

    def run_with_fake_scorer(self, args):
        with patch.object(code8_score, "load_frozen_workloads", return_value=self.workloads):
            with redirect_stdout(io.StringIO()):
                code8_score.score(args)

    def test_valid_ordinary_receipt_binds_identity_wire_and_scorer(self):
        self.run_with_fake_scorer(self.args())
        result = json.loads((self.root / "score.json").read_text())
        self.assertEqual(len(self.scorer_calls), 8)
        self.assertTrue(result["summary"]["code8_quality_gate_passed"])
        self.assertTrue(result["wire_audit"]["wire_audit_passed"])
        self.assertEqual(result["summary"]["request_scope"], "ordinary_quality")

    def test_scorer_accepts_only_identity_for_explicit_low_chip_group(self):
        low_topology = Topology.named(LAYOUT, chips=LOW_CHIPS)
        self.write_identity(low_topology)
        args = self.args(output="score-low-chips.json",
                         physical_chips=",".join(map(str, LOW_CHIPS)))
        self.run_with_fake_scorer(args)
        result = json.loads((self.root / "score-low-chips.json").read_text())
        self.assertTrue(result["summary"]["code8_quality_gate_passed"])

        self.write_identity(self.topology)
        self.scorer_calls.clear()
        with patch.object(code8_score, "load_frozen_workloads", return_value=self.workloads):
            with self.assertRaisesRegex(ValueError, "Unexpected worker physical_chip"):
                code8_score.score(args)
        self.assertEqual(self.scorer_calls, [])

    def test_bad_identity_and_scope_are_rejected_before_scorer(self):
        after = json.loads((self.responses / "identity_after.json").read_text())
        after["service_instance_id"] = "c" * 32
        (self.responses / "identity_after.json").write_text(json.dumps(after))
        with patch.object(code8_score, "load_frozen_workloads", return_value=self.workloads):
            with self.assertRaisesRegex(ValueError, "service_instance_id"):
                code8_score.score(self.args(output="bad-identity.json"))
        self.assertEqual(self.scorer_calls, [])

        self.write_identity()
        first = self.responses / "request-00.json"
        row = json.loads(first.read_text())
        row["request_scope"] = "armed_performance"
        first.write_text(json.dumps(row))
        with patch.object(code8_score, "load_frozen_workloads", return_value=self.workloads):
            with self.assertRaisesRegex(ValueError, "own scope"):
                code8_score.score(self.args(output="bad-scope.json"))
        self.assertEqual(self.scorer_calls, [])

        strict_args = self.args(output="bad-limits.json", scope="ordinary_quality")
        strict_args.timeout = 3.001
        with patch.object(code8_score, "load_frozen_workloads", return_value=self.workloads):
            with self.assertRaisesRegex(ValueError, "3s/512MiB/2CPU"):
                code8_score.score(strict_args)
        self.assertEqual(self.scorer_calls, [])

    def test_armed_performance_output_is_scoreable_but_never_a_quality_gate(self):
        for index in range(8):
            path = self.responses / f"request-{index:02d}.json"
            row = json.loads(path.read_text())
            row["request_scope"] = "armed_performance"
            path.write_text(json.dumps(row))
        args = self.args(scope="armed_performance", output="performance-score.json")
        args.proxy_root = None
        self.run_with_fake_scorer(args)
        result = json.loads((self.root / "performance-score.json").read_text())
        self.assertEqual(len(self.scorer_calls), 8)
        self.assertTrue(result["summary"]["all_unit_cases_passed"])
        self.assertFalse(result["summary"]["code8_quality_gate_passed"])
        self.assertIsNone(result["wire_audit"])

    def test_wire_route_and_done_failures_are_checked_before_scorer(self):
        cases = ("route", "done")
        for failure in cases:
            with self.subTest(failure=failure):
                self.scorer_calls.clear()
                self.write_identity()
                if failure == "route":
                    row = json.loads((self.proxy / "attempts.jsonl").read_text().splitlines()[0])
                    row["route_matches_response"] = False
                    row["response_dp_rank"] = 1
                    rows = [json.loads(line) for line in
                            (self.proxy / "attempts.jsonl").read_text().splitlines()]
                    rows[0] = row
                    (self.proxy / "attempts.jsonl").write_text(
                        "".join(json.dumps(item) + "\n" for item in rows))
                else:
                    rows = [json.loads(line) for line in
                            (self.proxy / "attempts.jsonl").read_text().splitlines()]
                    rows[0]["done_consumed"] = False
                    response_path = self.proxy / rows[0]["response_body_file"]
                    raw = response_path.read_bytes().removesuffix(b"data: [DONE]\n")
                    response_path.write_bytes(raw)
                    rows[0]["response_body_sha256"] = digest(raw)
                    rows[0]["response_body_bytes"] = len(raw)
                    (self.proxy / "attempts.jsonl").write_text(
                        "".join(json.dumps(item) + "\n" for item in rows))
                with patch.object(code8_score, "load_frozen_workloads", return_value=self.workloads):
                    with redirect_stdout(io.StringIO()):
                        with self.assertRaises(ValueError):
                            code8_score.score(self.args(output=f"bad-wire-{failure}.json"))
                self.assertEqual(self.scorer_calls, [],
                                 "wire evidence must be fully audited before bwrap scoring")

    def test_wrong_actual_prompt_usage_is_rejected_before_scorer(self):
        rows = [json.loads(line) for line in (self.proxy / "attempts.jsonl").read_text().splitlines()]
        wire = rows[0]
        response_path = self.proxy / wire["response_body_file"]
        raw = response_path.read_bytes()
        event_line, done_line = raw.splitlines()
        event = json.loads(event_line[6:])
        event["usage"]["prompt_tokens"] = 2047
        changed = b"data: " + json.dumps(event).encode("utf-8") + b"\n" + done_line + b"\n"
        response_path.write_bytes(changed)
        wire["response_body_sha256"] = digest(changed)
        wire["response_body_bytes"] = len(changed)
        wire["usage"]["prompt_tokens"] = 2047
        (self.proxy / "attempts.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n")
        with patch.object(code8_score, "load_frozen_workloads", return_value=self.workloads):
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "2048-token|usage|prompt"):
                    code8_score.score(self.args(output="bad-prompt-usage.json"))
        self.assertEqual(self.scorer_calls, [],
                         "wrong actual prompt usage must be rejected before bwrap scoring")

    def test_sse_error_event_is_rejected_before_scorer(self):
        rows = [json.loads(line) for line in (self.proxy / "attempts.jsonl").read_text().splitlines()]
        wire = rows[0]
        response_path = self.proxy / wire["response_body_file"]
        raw = response_path.read_bytes()
        content_event, _done = raw.splitlines()
        error_event = b'data: {"error":{"message":"synthetic server SSE error"}}'
        changed = content_event + b"\n" + error_event + b"\n" + b"data: [DONE]\n"
        response_path.write_bytes(changed)
        wire["response_body_sha256"] = digest(changed)
        wire["response_body_bytes"] = len(changed)
        (self.proxy / "attempts.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n")
        replay = self.responses / "request-00.json"
        response = json.loads(replay.read_text())
        response["error"] = "server SSE error event: synthetic server SSE error"
        response["response_complete_by_frozen_client"] = False
        replay.write_text(json.dumps(response))
        with patch.object(code8_score, "load_frozen_workloads", return_value=self.workloads):
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "SSE error|error event|response error"):
                    code8_score.score(self.args(output="bad-sse-error.json"))
        self.assertEqual(self.scorer_calls, [],
                         "SSE error events must fail before bwrap scoring")


if __name__ == "__main__":
    unittest.main()
