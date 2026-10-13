from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark
import workloads


def _metric_snapshot(
    rounds: int,
    raw_tokens: int | None,
    decode_s: float,
    prefill_s: float,
    rows: list[dict],
    last_window: dict | None = None,
):
    ranks = []
    for rank in range(8):
        ranks.append(
            {
                "rank": rank,
                "cumulative": {
                    "decode_rounds": rounds,
                    "decode_tokens": raw_tokens,
                    "decode_seconds": decode_s,
                    "prefill_seconds": prefill_s,
                },
                "requests": rows,
                "last_window": last_window,
                "pending": False,
                "errors": [],
                "counter_scope": benchmark.COUNTER_SCOPE,
                "decode_tokens_scope": "HTTP final usage required for async; raw worker tokens before terminal trimming for sync",
            }
        )
    return {
        "ok": True,
        "data": {
            "schema": benchmark.COUNTER_SCHEMA,
            "trusted": True,
            "counter_scope": benchmark.COUNTER_SCOPE,
            "service_instance_id": "instance-a",
            "form": "a2-tp8-a3sim",
            "served_model": "test-model",
            "ranks": ranks,
        },
    }


def _request_row(request_id: str, prompt_tokens: int = 2048):
    return {
        "request_id": f"chatcmpl-{request_id}-engine-suffix",
        "tag": request_id,
        "window_fingerprint": "a" * 64,
        "async_mode": False,
        "prompt_tokens": prompt_tokens,
        "seed_tokens": 1,
        "decode_rounds": 4,
        "raw_decode_tokens": 9,
        "last_round_tokens": 3,
        "terminal_fence_observed": True,
        "terminal_tp_join_observed": True,
        "profile_excluded": True,
        "decode_start_ns": 1_000_000_000,
        "terminal_sample_end_ns": 1_700_000_000,
        "fenced_end_ns": 1_800_000_000,
        "prefill_wall_scope": "first scheduled prefill -> terminal prefill CPU bookkeeping",
        "decode_wall_scope": "terminal prefill CPU bookkeeping -> worker completion fence; may include metrics RPC delay",
    }


def _arm_receipt(
    request_id: str,
    *,
    max_tokens: int = 256,
    async_mode: bool = False,
    service_instance_id: str = "instance-a",
    prefill_ack: dict | None = None,
):
    receipt = {
        "ok": True,
        "data": {
            "schema": "delivery_counter_arm_v1",
            "service_instance_id": service_instance_id,
            "tag": request_id,
            "max_tokens": max_tokens,
            "ranks": [
                {"rank": rank, "tag": request_id, "max_tokens": max_tokens, "async_mode": async_mode, "armed": True}
                for rank in range(8)
            ],
        },
    }
    if prefill_ack is not None:
        receipt["data"]["prefill_ack"] = prefill_ack
    return receipt


def _ced_identity():
    manifest = "a" * 64
    opp = "b" * 64

    def ranks():
        return [
            {"rank": rank, "form": benchmark.CED_FORM, "tp": 8, "dcp": 1}
            for rank in range(8)
        ]

    prefill = {
        "service_instance_id": "p-instance",
        "runtime_fingerprint": hashlib.sha256(b"p-runtime").hexdigest(),
        "served_model": "ced-test-model",
        "form": benchmark.CED_FORM,
        "role": "prefill",
        "engine_initialized": True,
        "ASGI_startup_completed": True,
        "model_kwargs": {"enable_prefix_caching": False, "decode_context_parallel_size": 1},
        "rank_descriptions": ranks(),
        "ced_overlay_manifest_sha256": manifest,
        "native_opp_receipt": {"artifact_fingerprint": opp},
    }
    decode = {
        "service_instance_id": "d-instance",
        "runtime_fingerprint": hashlib.sha256(b"d-runtime").hexdigest(),
        "served_model": "ced-test-model",
        "form": benchmark.CED_FORM,
        "role": "decode",
        "engine_initialized": True,
        "ASGI_startup_completed": True,
        "model_kwargs": {"enable_prefix_caching": False, "decode_context_parallel_size": 1},
        "rank_descriptions": ranks(),
        "ced_overlay_manifest_sha256": manifest,
        "native_opp_receipt": {"artifact_fingerprint": opp},
    }
    return {
        **decode,
        "service_instance_id": hashlib.sha256(b"p-instance:d-instance").hexdigest(),
        "runtime_fingerprint": hashlib.sha256(
            (prefill["runtime_fingerprint"] + decode["runtime_fingerprint"]).encode()
        ).hexdigest(),
        "prefill_identity": prefill,
        "decode_identity": decode,
        "topology": "P8+D8; DCP=1",
    }


def _ced_arm_receipt(root_sid, prefill_sid, request_id, *, async_mode=True, prefill_tag=None):
    prefill_ack = {
        "schema": "delivery_counter_arm_v1",
        "service_instance_id": prefill_sid,
        "tag": prefill_tag or request_id,
        "max_tokens": 256,
        "ranks": [
            {
                "rank": rank,
                "tag": prefill_tag or request_id,
                "max_tokens": 256,
                "async_mode": async_mode,
                "armed": True,
            }
            for rank in range(8)
        ],
    }
    return _arm_receipt(
        request_id,
        max_tokens=256,
        async_mode=async_mode,
        service_instance_id=root_sid,
        prefill_ack=prefill_ack,
    )


def _ced_request_row(*, request_id="d-engine-request", prefill_scope=None):
    return {
        "request_id": request_id,
        "tag": "dm-ced-window",
        "window_fingerprint": "f" * 64,
        "async_mode": True,
        "prompt_tokens": 2048,
        "seed_tokens": 1,
        "decode_rounds": 4,
        "raw_decode_tokens": None,
        "last_round_tokens": None,
        "terminal_fence_observed": True,
        "terminal_tp_join_observed": True,
        "profile_excluded": True,
        "decode_start_ns": 1_000_000_000,
        "terminal_sample_end_ns": 1_700_000_000,
        "fenced_end_ns": 1_800_000_000,
        "prefill_wall_scope": prefill_scope or "P scheduled prefill + D bounded replay/tail CPU bookkeeping; transfer wait excluded",
        "decode_wall_scope": "terminal prefill CPU bookkeeping -> worker completion fence; may include metrics RPC delay",
    }


def _ced_prefill_row(seconds, *, tag="dm-ced-window"):
    start_ns = 1_000_000_000
    return {
        "tag": tag,
        "request_id": "p-engine-request",
        "start_ns": start_ns,
        "end_ns": start_ns + round(seconds * 1e9),
        "prompt_tokens": 2048,
        "calls": 3,
        "profile_excluded": True,
        "terminal_fence_observed": True,
        "prefill_wall_scope": "P first scheduled prompt -> terminal P CPU bookkeeping",
    }


def _ced_metrics_snapshot(
    service_instance_id,
    p_base,
    p_delta=None,
    d_base=None,
    d_delta=None,
    *,
    after=False,
    prefill_tag="dm-ced-window",
    prefill_reporting=benchmark.CED_PREFILL_REPORTING,
    negative_d_rank=None,
):
    p_delta = p_delta or [0.0] * 8
    d_base = d_base or [0.5] * 8
    d_delta = d_delta or [0.0] * 8
    d_rows = []
    p_rows = []
    for rank in range(8):
        d_cumulative_prefill = p_base[rank] + d_base[rank]
        p_cumulative_prefill = p_base[rank]
        d_requests = []
        p_requests = []
        if after:
            d_requests = [_ced_request_row()]
            p_requests = [_ced_prefill_row(p_delta[rank], tag=prefill_tag)]
            p_cumulative_prefill += p_delta[rank]
            d_cumulative_prefill += p_delta[rank] + d_delta[rank]
            if negative_d_rank == rank:
                d_cumulative_prefill = p_base[rank] + d_base[rank] + p_delta[rank] - 0.01
        d_rows.append(
            {
                "rank": rank,
                "cumulative": {
                    "decode_rounds": 14 if after else 10,
                    "decode_tokens": None,
                    "decode_seconds": 20.8 if after else 20.0,
                    "prefill_seconds": d_cumulative_prefill,
                },
                "requests": d_requests,
                "last_window": {"tag": "dm-ced-window", "window_fingerprint": "f" * 64} if after else None,
                "pending": False,
                "errors": [],
                "counter_scope": benchmark.COUNTER_SCOPE,
                "decode_tokens_scope": "HTTP final usage required for async; raw worker tokens before terminal trimming for sync",
            }
        )
        p_rows.append(
            {
                "rank": rank,
                "role": "prefill",
                "cumulative": {"prefill_seconds": p_cumulative_prefill},
                "requests": p_requests,
                "pending": False,
                "errors": [],
            }
        )
    return {
        "ok": True,
        "data": {
            "schema": benchmark.COUNTER_SCHEMA,
            "trusted": True,
            "counter_scope": benchmark.COUNTER_SCOPE,
            "service_instance_id": service_instance_id,
            "form": benchmark.CED_FORM,
            "served_model": "ced-test-model",
            "ranks": d_rows,
            "prefill_ranks": p_rows,
            "prefill_reporting": prefill_reporting,
        },
    }


class WorkloadDefinitionsTests(unittest.TestCase):
    class CharacterCounter:
        def count(self, messages):
            return len(messages[0]["content"])

        def estimate(self, messages):
            return self.count(messages)

    def test_eight_fixed_code_tasks_and_deterministic_boundary_random_cases(self):
        self.assertEqual(len(workloads.CODE_TASKS), 8)
        for task_id in workloads.CODE_TASK_IDS:
            first = workloads.fixed_unit_cases(task_id)
            second = workloads.fixed_unit_cases(task_id)
            self.assertGreaterEqual(len(first), 10)
            self.assertEqual(first, second)
            self.assertTrue(any(case["id"].startswith("random-") for case in first))
            self.assertTrue(any(not case["id"].startswith("random-") for case in first))

    def test_prompt_calibration_hits_exact_length_for_both_workload_families(self):
        counter = self.CharacterCounter()
        suffixes = [{"name": f"suffix-{i}.txt", "text": f"问题 {i}"} for i in range(5)]
        dih = benchmark.prepare_dihuo_prompts("甲乙丙丁戊己", suffixes, counter, 256)
        code = benchmark.prepare_code_prompts(counter, 2048)
        self.assertEqual(len(dih), 8)
        self.assertEqual(len(code), 8)
        self.assertTrue(all(len(row["prompt_text"]) == 256 for row in dih))
        self.assertTrue(all(len(row["prompt_text"]) == 2048 for row in code))
        self.assertEqual(len({row["prompt_text"] for row in dih}), 8)

    def test_remote_tokenizer_fallback_uses_usage_for_final_exact_count(self):
        remote = benchmark.RemoteTokenizer("http://example.invalid", "test-model", 1.0, None)
        remote._message_tokenize_supported = False
        remote._raw_tokenize_supported = True
        remote.frame_overhead = 3
        exact_calls = []
        remote._tokenize_request = lambda payload: len(payload["prompt"])
        remote._completion_usage_count = lambda messages: exact_calls.append(messages) or len(messages[0]["content"]) + 3
        counter = benchmark.TokenCounter(None, remote)
        calibrated = benchmark.exact_prompt_text(lambda n: "prefix" + ("x" * n), counter, 256)
        self.assertTrue(calibrated["exact"])
        self.assertEqual(calibrated["tokens"], 256)
        self.assertTrue(exact_calls)

    def test_repeated_corpus_slice_is_stable_and_wraps(self):
        self.assertEqual(workloads.repeated_corpus_slice("abc", 2, 8), "cabcabca")
        self.assertEqual(workloads.repeated_corpus_slice("abc", 2, 0), "")

    def test_cleanliness_screen_reports_empty_corruption_and_repeated_blocks(self):
        self.assertFalse(workloads.output_cleanliness("") ["clean"])
        self.assertIn("replacement_character", workloads.output_cleanliness("bad\ufffd") ["reasons"])
        block = "x" * 80
        self.assertIn("repeated_block", workloads.output_cleanliness(block * 3)["reasons"])


class TrustedMetricDeltaTests(unittest.TestCase):
    def test_missing_counter_data_is_null_not_zero(self):
        delta = benchmark.counter_metrics_delta(
            None,
            None,
            "test-model",
            2048,
            {"prompt_tokens": 2048, "completion_tokens": 10, "total_tokens": 2058},
            "instance-a",
            "request-a",
            256,
            _arm_receipt("request-a"),
            "a2-tp8-a3sim",
        )
        self.assertFalse(delta["available"])
        self.assertIsNone(delta["steps"])
        self.assertIsNone(delta["A"])
        self.assertIsNone(delta["ms_step"])

    def test_trusted_sync_counter_delta_checks_raw_output_trimming(self):
        request_id = "request-a"
        row = _request_row(request_id)
        before = _metric_snapshot(0, 0, 0.0, 0.0, [])
        after = _metric_snapshot(4, 9, 0.8, 0.3, [row], {"tag": request_id, "window_fingerprint": row["window_fingerprint"]})
        usage = {"prompt_tokens": 2048, "completion_tokens": 10, "total_tokens": 2058}
        delta = benchmark.counter_metrics_delta(
            before,
            after,
            "test-model",
            2048,
            usage,
            "instance-a",
            request_id,
            256,
            _arm_receipt(request_id),
            "a2-tp8-a3sim",
        )
        self.assertTrue(delta["available"], delta["reason"])
        self.assertEqual(delta["steps"], 4)
        self.assertEqual(delta["decode_tokens"], 9)
        self.assertAlmostEqual(delta["A"], 2.25)
        self.assertAlmostEqual(delta["ms_step"], 200.0)
        self.assertAlmostEqual(delta["server_fenced_ms_per_step"], 200.0)
        self.assertIn("GET /delivery/metrics", delta["server_fenced_ms_per_step_scope"])
        self.assertEqual(delta["ms_step_legacy_alias_of"], "server_fenced_ms_per_step")
        self.assertEqual(delta["server_prefill_s"], 0.3)

    def test_rank_disagreement_is_unavailable(self):
        request_id = "request-a"
        row = _request_row(request_id)
        before = _metric_snapshot(0, 0, 0.0, 0.0, [])
        after = _metric_snapshot(4, 9, 0.8, 0.3, [row], {"tag": request_id, "window_fingerprint": row["window_fingerprint"]})
        after["data"]["ranks"][7]["cumulative"]["decode_rounds"] = 3
        delta = benchmark.counter_metrics_delta(
            before,
            after,
            "test-model",
            2048,
            {"prompt_tokens": 2048, "completion_tokens": 10, "total_tokens": 2058},
            "instance-a",
            request_id,
            256,
            _arm_receipt(request_id),
            "a2-tp8-a3sim",
        )
        self.assertFalse(delta["available"])
        self.assertIsNone(delta["steps"])
        self.assertIsNone(delta["A"])

    def test_http_usage_that_cannot_be_explained_by_last_round_trimming_is_rejected(self):
        request_id = "request-a"
        before = _metric_snapshot(0, 0, 0.0, 0.0, [])
        row = _request_row(request_id)
        after = _metric_snapshot(4, 9, 0.8, 0.3, [row], {"tag": request_id, "window_fingerprint": row["window_fingerprint"]})
        usage = {"prompt_tokens": 2048, "completion_tokens": 5, "total_tokens": 2053}
        delta = benchmark.counter_metrics_delta(
            before,
            after,
            "test-model",
            2048,
            usage,
            "instance-a",
            request_id,
            256,
            _arm_receipt(request_id),
            "a2-tp8-a3sim",
        )
        self.assertFalse(delta["available"])
        self.assertEqual(delta["reason"], "http_completion_usage_not_explainable_by_terminal_round_trimming")

    def test_async_mode_uses_actual_http_usage_and_keeps_raw_tokens_null(self):
        request_id = "request-async"
        row = _request_row(request_id)
        row.update({"async_mode": True, "raw_decode_tokens": None, "last_round_tokens": None})
        before = _metric_snapshot(0, None, 0.0, 0.0, [])
        after = _metric_snapshot(
            4,
            None,
            0.8,
            0.3,
            [row],
            {"tag": request_id, "window_fingerprint": row["window_fingerprint"]},
        )
        delta = benchmark.counter_metrics_delta(
            before,
            after,
            "test-model",
            2048,
            {"prompt_tokens": 2048, "completion_tokens": 10, "total_tokens": 2058},
            "instance-a",
            request_id,
            256,
            _arm_receipt(request_id, async_mode=True),
            "a2-tp8-a3sim",
        )
        self.assertTrue(delta["available"], delta["reason"])
        self.assertTrue(delta["async_mode"])
        self.assertEqual(delta["decode_tokens"], 9)
        self.assertIsNone(delta["decode_tokens_raw"])
        self.assertAlmostEqual(delta["A"], 2.25)


class CEDClientContractTests(unittest.TestCase):
    def test_identity_binds_ready_tp8_dcp1_pair_and_root_sid_hash(self):
        identity = _ced_identity()
        with patch.object(benchmark, "_get_json", return_value=identity):
            result = benchmark.verify_service_identity(
                "http://ced-proxy", "ced-test-model", benchmark.CED_FORM, 1.0, None
            )
        self.assertTrue(result["verified"])
        self.assertEqual(result["prefill_service_instance_id"], "p-instance")
        self.assertEqual(result["decode_service_instance_id"], "d-instance")
        self.assertEqual(
            result["service_instance_id"],
            hashlib.sha256(b"p-instance:d-instance").hexdigest(),
        )
        self.assertEqual(result["raw_identity"]["topology"], "P8+D8; DCP=1")

    def test_identity_rejects_wrong_pair_topology_manifest_opp_and_dcp(self):
        mutations = (
            lambda value: value.update(service_instance_id="wrong-root-pair"),
            lambda value: value.update(topology="P8+D8; DCP=8"),
            lambda value: value["prefill_identity"].update(
                ced_overlay_manifest_sha256="c" * 64
            ),
            lambda value: value["decode_identity"]["native_opp_receipt"].update(
                artifact_fingerprint="d" * 64
            ),
            lambda value: value["prefill_identity"]["rank_descriptions"][3].update(dcp=8),
        )
        for mutate in mutations:
            identity = _ced_identity()
            mutate(identity)
            with self.subTest(mutation=mutate), patch.object(
                benchmark, "_get_json", return_value=identity
            ), self.assertRaises(ValueError):
                benchmark.verify_service_identity(
                    "http://ced-proxy", "ced-test-model", benchmark.CED_FORM, 1.0, None
                )

    def test_metrics_use_independent_prefill_ranks_and_sum_per_request_p_plus_d_wall(self):
        identity = _ced_identity()
        p_sid = identity["prefill_identity"]["service_instance_id"]
        d_sid = identity["decode_identity"]["service_instance_id"]
        root_sid = identity["service_instance_id"]
        p_base = [10.0 + rank for rank in range(8)]
        d_base = [20.0 + rank for rank in range(8)]
        p_delta = [0.20, 0.10, 0.15, 0.16, 0.18, 0.12, 0.11, 0.10]
        d_delta = [0.05, 0.07, 0.08, 0.10, 0.04, 0.03, 0.02, 0.01]
        before = _ced_metrics_snapshot(root_sid, p_base, d_base=d_base)
        after = _ced_metrics_snapshot(
            root_sid, p_base, p_delta, d_base, d_delta, after=True
        )
        self.assertIsNot(before["data"]["ranks"], before["data"]["prefill_ranks"])
        self.assertIsNot(after["data"]["ranks"], after["data"]["prefill_ranks"])

        delta = benchmark.counter_metrics_delta(
            before,
            after,
            "ced-test-model",
            2048,
            {"prompt_tokens": 2048, "completion_tokens": 10, "total_tokens": 2058},
            root_sid,
            "dm-ced-window",
            256,
            _ced_arm_receipt(root_sid, p_sid, "dm-ced-window"),
            benchmark.CED_FORM,
            p_sid,
            d_sid,
        )
        self.assertTrue(delta["available"], delta["reason"])
        self.assertEqual(delta["request_tag"], "dm-ced-window")
        self.assertEqual(delta["trusted_source"]["service_instance_id"], root_sid)
        self.assertEqual(delta["trusted_source"]["prefill_service_instance_id"], p_sid)
        self.assertEqual(delta["trusted_source"]["decode_service_instance_id"], d_sid)
        # The slow P rank and slow D replay/tail rank differ; sum the two
        # serial stage wall times instead of max(P_rank + D_rank).
        self.assertAlmostEqual(delta["server_prefill_components"]["prefill_encoder_s"], 0.20)
        self.assertAlmostEqual(delta["server_prefill_components"]["decode_replay_tail_s"], 0.10)
        self.assertAlmostEqual(delta["server_prefill_s"], 0.30)
        self.assertGreater(delta["server_prefill_s"], max(p + d for p, d in zip(p_delta, d_delta)))
        self.assertEqual(delta["prefill_reporting"], benchmark.CED_PREFILL_REPORTING)

        group_request = {
            "output_text": "answer",
            "error": None,
            "finish_reason": "length",
            "usage_tokens": {"prompt_tokens": 2048, "completion_tokens": 10, "total_tokens": 2058},
            "ttft_s": 1.0,
            "decode_s": 2.0,
            "decode_rate_completion_tokens": 9,
            "clean": {"clean": True},
            "delivery_counter_delta": delta,
        }
        group_summary = benchmark.summarize_group(
            {"task": "dihuo", "requests": [group_request] * 8}, 8, 2048
        )
        self.assertAlmostEqual(group_summary["server_prefill_s"], 2.4)
        self.assertAlmostEqual(group_summary["server_prefill_components"]["prefill_encoder_s"], 1.6)
        self.assertAlmostEqual(group_summary["server_prefill_components"]["decode_replay_tail_s"], 0.8)
        round_summary = benchmark.aggregate_rounds(
            [{"round_index": 0, "aggregate": group_summary}], 1
        )
        self.assertAlmostEqual(round_summary["server_prefill_s"], 2.4)
        self.assertAlmostEqual(round_summary["server_prefill_components"]["server_prefill_s"], 2.4)

    def test_metrics_reject_cross_window_prefill_and_negative_decode_delta(self):
        identity = _ced_identity()
        root_sid = identity["service_instance_id"]
        p_sid = identity["prefill_identity"]["service_instance_id"]
        d_sid = identity["decode_identity"]["service_instance_id"]
        p_base = [10.0 + rank for rank in range(8)]
        d_base = [20.0 + rank for rank in range(8)]
        p_delta = [0.2] * 8
        d_delta = [0.1] * 8
        arm = _ced_arm_receipt(root_sid, p_sid, "dm-ced-window")
        usage = {"prompt_tokens": 2048, "completion_tokens": 10, "total_tokens": 2058}
        good_before = _ced_metrics_snapshot(root_sid, p_base, d_base=d_base)
        good_after = _ced_metrics_snapshot(
            root_sid, p_base, p_delta, d_base, d_delta, after=True
        )

        bad_arm = copy.deepcopy(arm)
        bad_arm["data"]["prefill_ack"]["service_instance_id"] = "foreign-prefill-sid"
        result = benchmark.counter_metrics_delta(
            good_before, good_after, "ced-test-model", 2048, usage, root_sid,
            "dm-ced-window", 256, bad_arm, benchmark.CED_FORM, p_sid, d_sid,
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "ced_prefill_arm_identity_mismatch")

        bad_arm = copy.deepcopy(arm)
        bad_arm["data"]["prefill_ack"]["ranks"][5]["async_mode"] = False
        result = benchmark.counter_metrics_delta(
            good_before, good_after, "ced-test-model", 2048, usage, root_sid,
            "dm-ced-window", 256, bad_arm, benchmark.CED_FORM, p_sid, d_sid,
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "ced_prefill_arm_rank_ack_mismatch")

        bad_tag = _ced_metrics_snapshot(
            root_sid, p_base, p_delta, d_base, d_delta, after=True, prefill_tag="other-window"
        )
        result = benchmark.counter_metrics_delta(
            good_before,
            bad_tag,
            "ced-test-model", 2048, usage, root_sid, "dm-ced-window", 256, arm,
            benchmark.CED_FORM, p_sid, d_sid,
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "ced_prefill_request_row_tag_or_timing_invalid")

        negative_d = _ced_metrics_snapshot(
            root_sid, p_base, p_delta, d_base, d_delta, after=True, negative_d_rank=2
        )
        result = benchmark.counter_metrics_delta(
            good_before,
            negative_d,
            "ced-test-model", 2048, usage, root_sid, "dm-ced-window", 256, arm,
            benchmark.CED_FORM, p_sid, d_sid,
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "ced_decode_prefill_delta_is_negative")


class CodeScoringTests(unittest.TestCase):
    def test_correct_code_passes_reference_cases_in_sandbox(self):
        source = """def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result
"""
        score = workloads.score_code_output("merge_intervals", source, timeout_seconds=3.0)
        self.assertEqual(score["status"], "pass", score)
        self.assertEqual(score["passed"], score["total"])
        self.assertTrue(score["extracted_code_sha256"])

    def test_wrong_answer_is_negative_control(self):
        score = workloads.score_code_output(
            "merge_intervals",
            "def merge_intervals(intervals):\n    return []\n",
            timeout_seconds=2.0,
        )
        self.assertEqual(score["status"], "fail")
        self.assertLess(score["passed"], score["total"])
        self.assertTrue(score["failed_case_ids"])

    def test_infinite_loop_is_stopped_by_wall_timeout(self):
        score = workloads.score_code_output(
            "longest_unique_substring",
            "def longest_unique_substring(text):\n    while True:\n        pass\n",
            timeout_seconds=0.25,
            cpu_seconds=2,
        )
        self.assertEqual(score["status"], "timeout")
        self.assertEqual(score["passed"], 0)
        self.assertEqual(score["total"], len(workloads.fixed_unit_cases("longest_unique_substring")))

    def test_network_socket_creation_is_denied(self):
        score = workloads.score_code_output(
            "flatten_dict",
            "import socket\ndef flatten_dict(data, sep='.'): \n    socket.socket()\n    return {}\n",
            timeout_seconds=2.0,
        )
        self.assertEqual(score["status"], "fail")
        self.assertEqual(score["passed"], 0)
        self.assertTrue(any("PermissionError" in row.get("error", "") for row in score["case_results"]))

    def test_sandbox_does_not_bind_home_or_workspace(self):
        with tempfile.TemporaryDirectory(prefix="sandbox-build-", dir=workloads._temp_root()) as temp_dir:
            command, description = workloads._sandbox_command(Path(temp_dir))
            self.assertIn("--unshare-net", command)
            repo_root = str(Path(__file__).resolve().parents[3])
            ro_bind_sources = [command[index + 1] for index, arg in enumerate(command[:-2]) if arg == "--ro-bind"]
            task_tmp_source = command[command.index("--bind") + 1]
            self.assertTrue(task_tmp_source.startswith(str(workloads._temp_root())))
            self.assertTrue(all(not source.startswith(("/home/", repo_root)) for source in ro_bind_sources))
            self.assertIn("minimal Python/stdlib/libs", description)


class AggregationTests(unittest.TestCase):
    def test_step_timing_aliases_and_http_rate_equation_controls(self):
        sample = {
            "sample_id": "timing-control",
            "prompt_text": "p",
            "messages": [{"role": "user", "content": "p"}],
        }
        response = {
            "output_text": "ok",
            "output_sha256": benchmark.sha256_text("ok"),
            "output_utf8_bytes": 2,
            "usage_tokens": {"prompt_tokens": 2048, "completion_tokens": 20, "total_tokens": 2068},
            "finish_reason": "length",
            "http_status": 200,
            "error": None,
            "sse_error_event": None,
            "sse_json_events": 2,
            "request_wall_s": 7.0,
            "ttft_s": 2.0,
            "decode_s": 5.0,
            "decode_rate_completion_tokens": 19,
            "decode_tok_s": 3.8,
            "tpot_s": 5.0 / 19.0,
        }
        metrics = {
            "available": True,
            "steps": 4,
            "A": 19.0 / 4.0,
            "ms_step": 100.0,
            "server_fenced_ms_per_step": 100.0,
            "server_fenced_ms_per_step_scope": "GET /delivery/metrics decode_seconds / decode_rounds",
            "decode_tokens": 19,
            "decode_seconds_s": 0.4,
            "server_prefill_s": 0.05,
        }
        positive = benchmark._record_response(
            sample, response, metrics, "dihuo", 0, 0, 3.0, 512, 2
        )
        self.assertAlmostEqual(positive["client_decode_ms_per_step"], 1250.0)
        self.assertAlmostEqual(positive["client_e2e_ms_per_round"], positive["client_decode_ms_per_step"])
        self.assertAlmostEqual(positive["server_fenced_ms_per_step"], 100.0)
        self.assertEqual(positive["ms_step_legacy_alias_of"], "server_fenced_ms_per_step")
        self.assertIn("excludes TTFT", positive["client_decode_ms_per_step_scope"])
        self.assertAlmostEqual(
            1000.0 * metrics["A"] / positive["client_decode_ms_per_step"],
            response["decode_tok_s"],
        )
        self.assertAlmostEqual(positive["http_rate_consistency_abs_delta_tok_s"], 0.0)

        negative_metrics = {**metrics, "A": metrics["A"] + 0.125}
        negative = benchmark._record_response(
            sample, response, negative_metrics, "dihuo", 0, 0, 3.0, 512, 2
        )
        self.assertAlmostEqual(negative["http_rate_from_A_tok_s"], 3.9)
        self.assertAlmostEqual(negative["http_rate_consistency_abs_delta_tok_s"], 0.1)

        group = {"task": "dihuo", "requests": [positive] * 8}
        group_summary = benchmark.summarize_group(group, 8, 2048)
        self.assertAlmostEqual(group_summary["client_decode_ms_per_step"], 1250.0)
        self.assertAlmostEqual(group_summary["client_e2e_ms_per_round"], group_summary["client_decode_ms_per_step"])
        self.assertAlmostEqual(group_summary["server_fenced_ms_per_step"], 100.0)
        self.assertAlmostEqual(group_summary["http_rate_from_A_tok_s"], 3.8)
        self.assertAlmostEqual(group_summary["http_rate_consistency_abs_delta_tok_s"], 0.0)

        rounds_summary = benchmark.aggregate_rounds(
            [{"round_index": 0, "aggregate": group_summary}], 1
        )
        self.assertAlmostEqual(rounds_summary["client_decode_ms_per_step"], 1250.0)
        self.assertAlmostEqual(rounds_summary["server_fenced_ms_per_step"], 100.0)
        self.assertAlmostEqual(rounds_summary["http_rate_from_A_tok_s"], 3.8)
        self.assertAlmostEqual(rounds_summary["http_rate_consistency_abs_delta_tok_s"], 0.0)

    def test_group_aggregate_requires_eight_exact_usage_outputs(self):
        group = {
            "task": "dihuo",
            "requests": [
                {
                    "output_text": "x",
                    "error": None,
                    "finish_reason": "length",
                    "usage_tokens": {"prompt_tokens": 2048, "completion_tokens": 2, "total_tokens": 2050},
                    "ttft_s": 0.1,
                    "decode_s": 0.2,
                    "decode_rate_completion_tokens": 1,
                    "clean": {"clean": True},
                    "delivery_counter_delta": {
                        "available": True,
                        "steps": 1,
                        "decode_tokens": 1,
                        "decode_seconds_s": 0.1,
                        "server_prefill_s": 0.05,
                    },
                }
                for _ in range(8)
            ],
        }
        aggregate = benchmark.summarize_group(group, 8, 2048)
        self.assertTrue(aggregate["valid_for_aggregation"])
        self.assertEqual(aggregate["actual_outputs"], 8)
        self.assertEqual(aggregate["actual_output_tokens"], 16)
        self.assertEqual(aggregate["clean_rate"], 1.0)

        group["requests"].pop()
        aggregate = benchmark.summarize_group(group, 8, 2048)
        self.assertFalse(aggregate["valid_for_aggregation"])
        self.assertIsNone(aggregate["actual_output_tokens"])


if __name__ == "__main__":
    unittest.main()

