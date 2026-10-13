"""CPU-only validation of complete active-DP single-request receipts."""
from __future__ import annotations

import unittest

from experiments.dp_matrix.metrics_contract import validate_metrics, validate_status
from experiments.dp_matrix.topology import Topology


TOPOLOGY = Topology.named("dp4-tp2")
DP_RANK = 1
TAG = "dp1-window-测试"
REQUEST_ID = "engine-request-abc123"
FINGERPRINT = "a" * 64


def _status_rows():
    return [
        {
            "rank": rank,
            "dp_rank": DP_RANK,
            "counter_tag": TAG,
            "counter_pending": False,
            "sample_pending": False,
            "counter_request_ids": [REQUEST_ID],
        }
        for rank in range(TOPOLOGY.tp)
    ]


def _metrics_rows():
    result = []
    for rank in range(TOPOLOGY.tp):
        request = {
            "tag": TAG,
            "request_id": REQUEST_ID,
            "terminal_fence_observed": True,
            "terminal_tp_join_observed": True,
            "profile_excluded": True,
            "window_fingerprint": FINGERPRINT,
            "decode_rounds": 42,
            "prompt_tokens": 2048,
            "seed_tokens": 1,
        }
        result.append({
            "rank": rank,
            "dp_rank": DP_RANK,
            "last_window": {"tag": TAG, "window_fingerprint": FINGERPRINT},
            "requests": [request],
        })
    return result


class MetricsContractTests(unittest.TestCase):
    def test_status_binds_one_actual_request_on_each_active_tp_rank(self):
        self.assertEqual(
            validate_status(_status_rows(), TOPOLOGY, dp_rank=DP_RANK, tag=TAG),
            REQUEST_ID,
        )

    def test_status_rejects_partial_or_extra_tp_replies(self):
        for replies in (_status_rows()[:1], _status_rows() + [_status_rows()[0]], None, ()):
            with self.subTest(replies=replies), self.assertRaises(ValueError):
                validate_status(replies, TOPOLOGY, dp_rank=DP_RANK, tag=TAG)

    def test_status_rejects_duplicate_foreign_and_bool_ranks(self):
        cases = []

        duplicate = _status_rows()
        duplicate[1]["rank"] = duplicate[0]["rank"]
        cases.append(duplicate)

        foreign = _status_rows()
        foreign[1]["rank"] = TOPOLOGY.tp
        cases.append(foreign)

        bool_rank = _status_rows()
        bool_rank[1]["rank"] = True
        cases.append(bool_rank)

        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_status(rows, TOPOLOGY, dp_rank=DP_RANK, tag=TAG)

    def test_status_rejects_foreign_or_bool_dp_rank(self):
        foreign = _status_rows()
        foreign[0]["dp_rank"] = 2
        bool_rank = _status_rows()
        bool_rank[0]["dp_rank"] = True
        for rows in (foreign, bool_rank):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_status(rows, TOPOLOGY, dp_rank=DP_RANK, tag=TAG)

    def test_status_rejects_wrong_tag_pending_state_or_ambiguous_request(self):
        cases = []
        wrong_tag = _status_rows()
        wrong_tag[0]["counter_tag"] = "other"
        cases.append(wrong_tag)
        counter_pending = _status_rows()
        counter_pending[0]["counter_pending"] = True
        cases.append(counter_pending)
        sample_pending = _status_rows()
        sample_pending[0]["sample_pending"] = True
        cases.append(sample_pending)
        no_request = _status_rows()
        no_request[0]["counter_request_ids"] = []
        cases.append(no_request)
        multiple_requests = _status_rows()
        multiple_requests[0]["counter_request_ids"] = [REQUEST_ID, "other"]
        cases.append(multiple_requests)
        different_request = _status_rows()
        different_request[1]["counter_request_ids"] = ["other-request"]
        cases.append(different_request)

        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_status(rows, TOPOLOGY, dp_rank=DP_RANK, tag=TAG)

    def test_metrics_accepts_complete_fenced_profile_excluded_window(self):
        self.assertEqual(
            validate_metrics(
                _metrics_rows(), TOPOLOGY, dp_rank=DP_RANK,
                tag=TAG, request_id=REQUEST_ID,
            ),
            {
                "request_id": REQUEST_ID,
                "window_fingerprint": FINGERPRINT,
                "decode_rounds": 42,
                "prompt_tokens": 2048,
                "seed_tokens": 1,
            },
        )

    def test_metrics_rejects_partial_or_extra_tp_replies(self):
        for replies in (_metrics_rows()[:1], _metrics_rows() + [_metrics_rows()[0]], None, ()):
            with self.subTest(replies=replies), self.assertRaises(ValueError):
                validate_metrics(
                    replies, TOPOLOGY, dp_rank=DP_RANK, tag=TAG, request_id=REQUEST_ID
                )

    def test_metrics_rejects_duplicate_foreign_and_bool_ranks(self):
        cases = []

        duplicate = _metrics_rows()
        duplicate[1]["rank"] = duplicate[0]["rank"]
        cases.append(duplicate)

        foreign = _metrics_rows()
        foreign[1]["rank"] = TOPOLOGY.tp
        cases.append(foreign)

        bool_rank = _metrics_rows()
        bool_rank[1]["rank"] = True
        cases.append(bool_rank)

        foreign_dp = _metrics_rows()
        foreign_dp[0]["dp_rank"] = 2
        cases.append(foreign_dp)

        bool_dp = _metrics_rows()
        bool_dp[0]["dp_rank"] = True
        cases.append(bool_dp)

        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_metrics(rows, TOPOLOGY, dp_rank=DP_RANK, tag=TAG, request_id=REQUEST_ID)

    def test_metrics_rejects_different_request_tag_and_fingerprint(self):
        different_request = _metrics_rows()
        different_request[1]["requests"][0]["request_id"] = "another-request"
        different_tag = _metrics_rows()
        different_tag[1]["requests"][0]["tag"] = "another-tag"
        different_last_window_tag = _metrics_rows()
        different_last_window_tag[1]["last_window"]["tag"] = "another-tag"
        different_fingerprint = _metrics_rows()
        different_fingerprint[1]["requests"][0]["window_fingerprint"] = "b" * 64
        different_last_fingerprint = _metrics_rows()
        different_last_fingerprint[1]["last_window"]["window_fingerprint"] = "b" * 64

        for rows in (
            different_request,
            different_tag,
            different_last_window_tag,
            different_fingerprint,
            different_last_fingerprint,
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_metrics(rows, TOPOLOGY, dp_rank=DP_RANK, tag=TAG, request_id=REQUEST_ID)

    def test_metrics_rejects_missing_terminal_fence_join_or_profile_exclusion(self):
        cases = []
        for key in ("terminal_fence_observed", "terminal_tp_join_observed", "profile_excluded"):
            rows = _metrics_rows()
            rows[0]["requests"][0][key] = False
            cases.append((key, rows))
        for key, rows in cases:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_metrics(rows, TOPOLOGY, dp_rank=DP_RANK, tag=TAG, request_id=REQUEST_ID)

    def test_metrics_rejects_inconsistent_round_prompt_or_seed_counts(self):
        cases = []
        for key, value in (("decode_rounds", 41), ("prompt_tokens", 2049), ("seed_tokens", 2)):
            rows = _metrics_rows()
            rows[1]["requests"][0][key] = value
            cases.append((key, rows))
        for key, rows in cases:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_metrics(rows, TOPOLOGY, dp_rank=DP_RANK, tag=TAG, request_id=REQUEST_ID)

    def test_metrics_rejects_invalid_rounds_prompt_seed_and_coverage(self):
        cases = []
        for key, value in (("decode_rounds", 0), ("decode_rounds", True),
                           ("prompt_tokens", 0), ("seed_tokens", 0), ("seed_tokens", 2)):
            rows = _metrics_rows()
            rows[0]["requests"][0][key] = value
            cases.append((key, value, rows))
        for value in (None, "bad", "A" * 64, "g" * 64):
            rows = _metrics_rows()
            rows[0]["requests"][0]["window_fingerprint"] = value
            cases.append(("window_fingerprint", value, rows))
        for key, value, rows in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_metrics(rows, TOPOLOGY, dp_rank=DP_RANK, tag=TAG, request_id=REQUEST_ID)


if __name__ == "__main__":
    unittest.main()
