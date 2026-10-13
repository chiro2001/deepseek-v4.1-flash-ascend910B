"""CPU-only positive/negative contracts for single-request and wave reducers."""
from __future__ import annotations

import copy
import math
import unittest

from experiments.dp_matrix.measurement import request_metrics, wave_metrics
from experiments.dp_matrix.topology import Topology


TOPOLOGY = Topology.named("dp4-tp2")
DP_RANK = 2
TAG = "dp2-measurement-α"
SID = "a" * 32
FINGERPRINT = "b" * 64
REQUEST_ID = "request-dp2-001"
COVERAGE = "c" * 64
PROMPT_TOKENS = 2048
MAX_TOKENS = 256
ROUNDS = 50


def _valid_receipts():
    arm_rows = []
    metric_rows = []
    for rank in range(TOPOLOGY.tp):
        before = {
            "prefill_seconds": 1.0 + rank,
            "decode_seconds": 2.0 + rank,
            "decode_rounds": 100,
            "draft_graph_calls": 200,
            "draft_eager_calls": 0,
        }
        arm_rows.append({
            "rank": rank,
            "dp_rank": DP_RANK,
            "armed": True,
            "async_mode": True,
            "cumulative_before": before,
        })
        cumulative = {
            "prefill_seconds": before["prefill_seconds"] + 0.2,
            "decode_seconds": before["decode_seconds"] + 0.5,
            "decode_rounds": before["decode_rounds"] + ROUNDS,
            "draft_graph_calls": before["draft_graph_calls"] + ROUNDS,
            "draft_eager_calls": before["draft_eager_calls"],
        }
        request = {
            "tag": TAG,
            "request_id": REQUEST_ID,
            "terminal_fence_observed": True,
            "terminal_tp_join_observed": True,
            "profile_excluded": True,
            "window_fingerprint": COVERAGE,
            "decode_rounds": ROUNDS,
            "prompt_tokens": PROMPT_TOKENS,
            "seed_tokens": 1,
            "async_mode": True,
            "decode_start_ns": 1_000_000_000,
            "terminal_sample_end_ns": 1_400_000_000,
            "fenced_end_ns": 1_500_000_000,
            "draft_phase_counts": {
                "schema": "actual_draft_phase_counts_v1",
                "initial_prefill": {"graph_calls": 0, "eager_calls": 0, "proposal_calls": 0},
                "decode_rounds": [{"round": i, "graph_calls": 1, "eager_calls": 0,
                                   "proposal_calls": 1} for i in range(1, ROUNDS + 1)],
            },
        }
        metric_rows.append({
            "rank": rank,
            "dp_rank": DP_RANK,
            "cumulative": cumulative,
            "last_window": {"tag": TAG, "window_fingerprint": COVERAGE},
            "requests": [request],
        })

    binding = {
        "request_id": REQUEST_ID,
        "window_fingerprint": COVERAGE,
        "decode_rounds": ROUNDS,
        "prompt_tokens": PROMPT_TOKENS,
        "seed_tokens": 1,
    }
    arm = {
        "service_instance_id": SID,
        "dp_rank": DP_RANK,
        "tag": TAG,
        "ranks": arm_rows,
    }
    metrics = {
        "schema": "dp_request_metrics_v1",
        "trusted": True,
        "service_instance_id": SID,
        "runtime_fingerprint": FINGERPRINT,
        "dp_rank": DP_RANK,
        "request_binding": binding,
        "ranks": metric_rows,
    }
    stream = {
        "http_status": 200,
        "dp_rank": DP_RANK,
        "done": True,
        "error": None,
        "usage": {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": MAX_TOKENS,
            "total_tokens": PROMPT_TOKENS + MAX_TOKENS,
        },
        "finish_reason": "length",
        "decode_s": 0.4,
        "ttft_s": 0.3,
    }
    return arm, metrics, stream


def _reduce(arm=None, metrics=None, stream=None, *, ignore_eos=True,
            prompt_tokens=PROMPT_TOKENS, max_tokens=MAX_TOKENS):
    actual_arm, actual_metrics, actual_stream = _valid_receipts()
    return request_metrics(
        actual_arm if arm is None else arm,
        actual_metrics if metrics is None else metrics,
        actual_stream if stream is None else stream,
        TOPOLOGY,
        dp_rank=DP_RANK,
        tag=TAG,
        sid=SID,
        fingerprint=FINGERPRINT,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
    )


def _rebind_dp_rank(arm, metrics, stream, dp_rank):
    arm["dp_rank"] = dp_rank
    metrics["dp_rank"] = dp_rank
    stream["dp_rank"] = dp_rank
    for row in arm["ranks"] + metrics["ranks"]:
        row["dp_rank"] = dp_rank


class RequestMeasurementTests(unittest.TestCase):
    def test_complete_http_and_worker_receipts_reduce_to_one_logical_tp_request(self):
        result = _reduce()

        self.assertEqual(result["dp_rank"], DP_RANK)
        self.assertEqual(result["decode_tokens"], MAX_TOKENS - 1)
        self.assertEqual(result["verifier_rounds"], ROUNDS)
        self.assertAlmostEqual(result["A"], (MAX_TOKENS - 1) / ROUNDS)
        self.assertAlmostEqual(result["http"]["decode_s"], 0.4)
        self.assertAlmostEqual(result["http"]["ms_step"], 8.0)
        self.assertAlmostEqual(result["http"]["tok_s"], (MAX_TOKENS - 1) / 0.4)
        self.assertAlmostEqual(result["server_fenced"]["decode_s"], 0.5)
        self.assertAlmostEqual(result["server_fenced"]["ms_step"], 10.0)
        self.assertAlmostEqual(result["ttft_s"], 0.3)
        self.assertAlmostEqual(result["worker_prefill_s"], 0.2)
        self.assertEqual(result["draft_graph_calls_by_tp"], [ROUNDS] * TOPOLOGY.tp)
        self.assertEqual(result["draft_eager_calls_by_tp"], [0] * TOPOLOGY.tp)
        self.assertTrue(result["steps_deduplicated_across_tp_dcp"])

    def test_rejects_wrong_service_identity_fingerprint_or_request_binding(self):
        arm, metrics, stream = _valid_receipts()
        bad_arm = copy.deepcopy(arm)
        bad_arm["service_instance_id"] = "z" * 32
        with self.assertRaisesRegex(ValueError, "service instance changed"):
            _reduce(arm=bad_arm, metrics=metrics, stream=stream)

        bad_metrics = copy.deepcopy(metrics)
        bad_metrics["service_instance_id"] = "z" * 32
        with self.assertRaisesRegex(ValueError, "service instance changed"):
            _reduce(arm=arm, metrics=bad_metrics, stream=stream)

        bad_metrics = copy.deepcopy(metrics)
        bad_metrics["runtime_fingerprint"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "Unbound DP counter receipt"):
            _reduce(arm=arm, metrics=bad_metrics, stream=stream)

        bad_metrics = copy.deepcopy(metrics)
        bad_metrics["request_binding"]["request_id"] = "other-request"
        with self.assertRaises(ValueError):
            _reduce(arm=arm, metrics=bad_metrics, stream=stream)

        changed_arm = copy.deepcopy(arm)
        changed_arm["tag"] = "wrong-tag"
        with self.assertRaisesRegex(ValueError, "Arm disagrees"):
            _reduce(arm=changed_arm, metrics=metrics, stream=stream)

    def test_rejects_untrusted_wrong_schema_and_wrong_dp_or_arm_tag(self):
        arm, metrics, stream = _valid_receipts()
        for key, value in (("trusted", False), ("schema", "old-schema"),
                           ("dp_rank", 1)):
            changed = copy.deepcopy(metrics)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                _reduce(arm=arm, metrics=changed, stream=stream)

        changed_arm = copy.deepcopy(arm)
        changed_arm["tag"] = "wrong-tag"
        with self.assertRaisesRegex(ValueError, "Arm disagrees"):
            _reduce(arm=changed_arm, metrics=metrics, stream=stream)

    def test_dp_rank_must_be_strict_integer_at_every_receipt_boundary(self):
        cases = []
        arm, metrics, stream = _valid_receipts()
        _rebind_dp_rank(arm, metrics, stream, 1)
        cases.append(("argument", arm, metrics, stream, True))

        arm, metrics, stream = _valid_receipts()
        _rebind_dp_rank(arm, metrics, stream, 1)
        arm["dp_rank"] = True
        cases.append(("arm", arm, metrics, stream, 1))

        arm, metrics, stream = _valid_receipts()
        _rebind_dp_rank(arm, metrics, stream, 1)
        metrics["dp_rank"] = True
        cases.append(("metrics", arm, metrics, stream, 1))

        arm, metrics, stream = _valid_receipts()
        _rebind_dp_rank(arm, metrics, stream, 1)
        stream["dp_rank"] = True
        cases.append(("stream", arm, metrics, stream, 1))

        for label, arm, metrics, stream, dp_rank in cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                request_metrics(
                    arm, metrics, stream, TOPOLOGY,
                    dp_rank=dp_rank, tag=TAG, sid=SID, fingerprint=FINGERPRINT,
                    prompt_tokens=PROMPT_TOKENS, max_tokens=MAX_TOKENS, ignore_eos=True,
                )

    def test_rejects_http_route_status_done_or_sse_error_mismatch(self):
        arm, metrics, stream = _valid_receipts()
        cases = []
        for key, value, message in (
            ("http_status", 500, "HTTP/SSE generation"),
            ("dp_rank", 1, "Actual DP rank"),
            ("done", False, "HTTP/SSE generation"),
            ("error", "partial stream", "HTTP/SSE generation"),
        ):
            changed = copy.deepcopy(stream)
            changed[key] = value
            cases.append((key, changed, message))
        for key, changed, message in cases:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                _reduce(arm=arm, metrics=metrics, stream=changed)

    def test_rejects_invalid_usage_finish_reason_and_fixed_budget(self):
        arm, metrics, stream = _valid_receipts()
        cases = []
        for key, value in (("prompt_tokens", PROMPT_TOKENS + 1),
                           ("completion_tokens", MAX_TOKENS + 1),
                           ("total_tokens", 1)):
            changed = copy.deepcopy(stream)
            changed["usage"][key] = value
            cases.append((key, changed))
        bad_finish = copy.deepcopy(stream)
        bad_finish["finish_reason"] = "content_filter"
        cases.append(("finish_reason", bad_finish))
        short_length = copy.deepcopy(stream)
        short_length["usage"]["completion_tokens"] -= 1
        short_length["usage"]["total_tokens"] -= 1

        for key, changed in cases:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "usage/finish"):
                _reduce(arm=arm, metrics=metrics, stream=changed)

        with self.assertRaisesRegex(ValueError, "usage/finish"):
            _reduce(arm=arm, metrics=metrics, stream=stream, prompt_tokens=PROMPT_TOKENS - 1)

        with self.assertRaisesRegex(ValueError, "Fixed-budget"):
            _reduce(arm=arm, metrics=metrics, stream=short_length, ignore_eos=True,
                    max_tokens=MAX_TOKENS)

    def test_rejects_missing_duplicate_foreign_or_unarmed_baselines(self):
        arm, metrics, stream = _valid_receipts()
        cases = []
        missing = copy.deepcopy(arm)
        missing["ranks"] = missing["ranks"][:1]
        cases.append(("missing", missing))
        duplicate = copy.deepcopy(arm)
        duplicate["ranks"][1]["rank"] = duplicate["ranks"][0]["rank"]
        cases.append(("duplicate", duplicate))
        foreign = copy.deepcopy(arm)
        foreign["ranks"][0]["dp_rank"] = 1
        cases.append(("foreign DP", foreign))
        bool_rank = copy.deepcopy(arm)
        bool_rank["ranks"][0]["rank"] = False
        cases.append(("boolean rank", bool_rank))
        unarmed = copy.deepcopy(arm)
        unarmed["ranks"][0]["armed"] = False
        cases.append(("unarmed", unarmed))
        synchronous = copy.deepcopy(arm)
        synchronous["ranks"][0]["async_mode"] = False
        cases.append(("non-async", synchronous))

        for label, changed in cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                _reduce(arm=changed, metrics=metrics, stream=stream)

    def test_rejects_cumulative_round_delta_not_equal_to_bound_request(self):
        arm, metrics, stream = _valid_receipts()
        changed = copy.deepcopy(metrics)
        changed["ranks"][0]["cumulative"]["decode_rounds"] -= 1
        with self.assertRaisesRegex(ValueError, "verifier rounds"):
            _reduce(arm=arm, metrics=changed, stream=stream)

    def test_rejects_fence_timestamp_order_or_wall_disagreement(self):
        arm, metrics, stream = _valid_receipts()
        cases = []
        reversed_order = copy.deepcopy(metrics)
        reversed_order["ranks"][0]["requests"][0]["terminal_sample_end_ns"] = 999_000_000
        cases.append(reversed_order)
        fence_before_sample = copy.deepcopy(metrics)
        fence_before_sample["ranks"][0]["requests"][0]["fenced_end_ns"] = 1_300_000_000
        cases.append(fence_before_sample)
        timestamp_delta = copy.deepcopy(metrics)
        timestamp_delta["ranks"][0]["requests"][0]["fenced_end_ns"] = 1_700_000_000
        cases.append(timestamp_delta)

        for changed in cases:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "fence timestamps"):
                _reduce(arm=arm, metrics=changed, stream=stream)

    def test_requires_real_draft_graph_on_every_tp_rank_and_no_eager_fallback(self):
        arm, metrics, stream = _valid_receipts()
        no_graph = copy.deepcopy(metrics)
        no_graph["ranks"][0]["cumulative"]["draft_graph_calls"] = 200
        with self.assertRaisesRegex(ValueError, "draft phase counts"):
            _reduce(arm=arm, metrics=no_graph, stream=stream)

        eager = copy.deepcopy(metrics)
        eager["ranks"][1]["cumulative"]["draft_eager_calls"] = 1
        with self.assertRaisesRegex(ValueError, "draft phase counts"):
            _reduce(arm=arm, metrics=eager, stream=stream)

    def test_prefill_eager_is_allowed_only_with_actual_phase_proof(self):
        arm, metrics, stream = _valid_receipts()
        for row in metrics["ranks"]:
            row["cumulative"]["draft_eager_calls"] += 1
            row["requests"][0]["draft_phase_counts"]["initial_prefill"] = {
                "graph_calls": 0, "eager_calls": 1, "proposal_calls": 1}
        result = _reduce(arm=arm, metrics=metrics, stream=stream)
        self.assertEqual(result["draft_initial_eager_calls_by_tp"], [1] * TOPOLOGY.tp)
        self.assertEqual(result["draft_decode_eager_calls_by_tp"], [0] * TOPOLOGY.tp)
        proof = metrics["ranks"][0]["requests"][0]["draft_phase_counts"]
        proof["initial_prefill"] = {"graph_calls": 1, "eager_calls": 0, "proposal_calls": 1}
        proof["decode_rounds"][0].update(graph_calls=0, eager_calls=1)
        with self.assertRaisesRegex(ValueError, "decode round.*eager fallback"):
            _reduce(arm=arm, metrics=metrics, stream=stream)

    def test_rejects_nonfinite_or_nonpositive_timing_and_counter_values(self):
        arm, metrics, stream = _valid_receipts()
        cases = []
        nan_stream = copy.deepcopy(stream)
        nan_stream["decode_s"] = math.nan
        cases.append(("HTTP NaN", arm, metrics, nan_stream))
        inf_ttft = copy.deepcopy(stream)
        inf_ttft["ttft_s"] = math.inf
        cases.append(("TTFT infinity", arm, metrics, inf_ttft))
        nan_baseline = copy.deepcopy(arm)
        nan_baseline["ranks"][0]["cumulative_before"]["prefill_seconds"] = math.nan
        cases.append(("baseline NaN", nan_baseline, metrics, stream))
        inf_current = copy.deepcopy(metrics)
        inf_current["ranks"][0]["cumulative"]["decode_seconds"] = math.inf
        cases.append(("worker infinity", arm, inf_current, stream))
        zero_http = copy.deepcopy(stream)
        zero_http["decode_s"] = 0
        cases.append(("zero decode", arm, metrics, zero_http))

        for label, changed_arm, changed_metrics, changed_stream in cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                _reduce(arm=changed_arm, metrics=changed_metrics, stream=changed_stream)


class WaveMeasurementTests(unittest.TestCase):
    def test_wave_counts_one_logical_http_stream_not_tp_counter_replicas(self):
        streams = [
            {"usage": {"completion_tokens": 11},
             "first_content_monotonic": 10.0, "last_content_monotonic": 11.0},
            {"usage": {"completion_tokens": 21},
             "first_content_monotonic": 10.2, "last_content_monotonic": 12.0},
        ]
        result = wave_metrics(streams)
        self.assertEqual(result["concurrency"], 2)
        self.assertEqual(result["decode_tokens"], 30)
        self.assertEqual(result["decode_wall_s"], 2.0)
        self.assertEqual(result["aggregate_decode_tok_s"], 15.0)
        self.assertFalse(result["maximum_saturation_proven"])
        self.assertIn("not single-stream tok/s", result["scope"])

    def test_wave_rejects_empty_invalid_or_zero_interval(self):
        for streams in (
            [],
            [{"usage": {"completion_tokens": 11},
              "first_content_monotonic": 10.0, "last_content_monotonic": 10.0}],
            [{"usage": {"completion_tokens": 11},
              "first_content_monotonic": math.nan, "last_content_monotonic": 11.0}],
            [{"usage": {"completion_tokens": True},
              "first_content_monotonic": 10.0, "last_content_monotonic": 11.0}],
        ):
            with self.subTest(streams=streams), self.assertRaises(ValueError):
                wave_metrics(streams)


if __name__ == "__main__":
    unittest.main()
