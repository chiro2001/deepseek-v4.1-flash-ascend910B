"""CPU-only 2048/256 geometry checks, reference gate and guards for the timing diagnostic."""
import json
import math
from pathlib import Path

PROMPT_TOKENS = 2048
OUTPUT_TOKENS = 256
K = 5
VERIFY_TOKENS = K + 1
PREFILL_CHUNKS = [2044, 4]
VOCAB_SIZE = 129280
REFERENCE_JOB = "dspark_lane_20261011_c4_graph_geometry2k256_v3"


class WindowGuard:
    def __init__(self):
        self.active = False
        self.terminal_fence_expected = False
        self.outside_calls = {}
        self.rejected_diagnostic_calls = 0
        self.explicit_terminal_fences = 0
        self.other_native_explicit_fences = 0

    def require_outside(self, label):
        if self.active:
            self.rejected_diagnostic_calls += 1
            raise AssertionError("Diagnostic work is forbidden in the timing window: " + label)
        self.outside_calls[label] = self.outside_calls.get(label, 0) + 1

    def protect(self, label, original):
        def protected(*args, **kwargs):
            self.require_outside(label)
            return original(*args, **kwargs)
        return protected

    def observe_synchronize(self, original):
        def synchronized(*args, **kwargs):
            if self.active:
                if self.terminal_fence_expected:
                    self.explicit_terminal_fences += 1
                else:
                    self.other_native_explicit_fences += 1
            return original(*args, **kwargs)
        return synchronized

    def begin(self):
        assert not self.active and not self.terminal_fence_expected
        self.rejected_diagnostic_calls = 0
        self.explicit_terminal_fences = 0
        self.other_native_explicit_fences = 0
        self.active = True

    def end(self):
        assert self.active and not self.terminal_fence_expected
        self.active = False

    def receipt(self):
        assert not self.active
        return {"diagnostic_device_reads_or_hash_calls_in_window": self.rejected_diagnostic_calls,
                "explicit_terminal_device_fences": self.explicit_terminal_fences,
                "other_native_explicit_fences": self.other_native_explicit_fences,
                "outside_diagnostic_call_counts": dict(self.outside_calls),
                "guard_scope": "Diagnostic helpers only; native CPU bookkeeping remains unchanged",
                "route_logprob_audit_and_profiler": False}


def period2_after_prefill(tokens):
    tail = tokens[1:]
    return (len(tail) >= 16 and len(set(tail)) == 2
            and all(value == tail[index % 2] for index, value in enumerate(tail)))


def load_reference(path):
    """Load the frozen 2K/256 token ids extracted from the v3 terminal geometry evidence."""
    path = Path(path)
    assert path.is_file(), ("2K/256 reference token ids are pending v3 terminal evidence; "
                            "run build_reference_2k256.py after the v3 fetch materializes: " + str(path))
    data = json.loads(path.read_text())
    assert data.get("pending") is False, "Reference is still marked pending v3 terminal evidence"
    assert data.get("source_job") == REFERENCE_JOB, data.get("source_job")
    source_sha = data.get("source_requests_sha256")
    assert isinstance(source_sha, str) and len(source_sha) == 64, source_sha
    assert isinstance(data.get("source_requests_path"), str) and REFERENCE_JOB in data["source_requests_path"]
    seeds = data.get("seeds")
    assert isinstance(seeds, dict) and set(seeds) == {"0", "1", "2"}, seeds
    for seed, row in seeds.items():
        assert isinstance(row, dict) and isinstance(row.get("tags"), list) and row["tags"], seed
        assert all(isinstance(tag, str) and tag for tag in row["tags"]), seed
        tokens = row.get("token_ids")
        assert isinstance(tokens, list) and len(tokens) == OUTPUT_TOKENS, (seed, type(tokens))
        assert all(isinstance(token, int) and 0 <= token < VOCAB_SIZE for token in tokens), seed
    assert len({tuple(seeds[seed]["token_ids"]) for seed in seeds}) == 3, "Seed references must not collide"
    return data


def validate_geometry(raw, tokens, arm, core_id, reference_tokens):
    """Validate this fresh 2048/256 run from actual counters; decode rounds are derived, never assumed."""
    assert arm in ("off", "on") and len(tokens) == OUTPUT_TOKENS
    assert tokens == reference_tokens, "Fresh output differs from the frozen 2K/256 reference"
    rows = raw["counter"]["rounds"]
    assert rows and all(row["req_id"] == core_id for row in rows)
    prefill = [row for row in rows if not row["is_decode"]]
    decode = [row for row in rows if row["is_decode"]]
    assert rows == prefill + decode, "Illegal prefill/decode order"
    assert len(prefill) == len(PREFILL_CHUNKS)
    computed = 0
    for index, row in enumerate(prefill):
        assert row["computed_before"] == computed and row["prompt_tokens"] == PROMPT_TOKENS
        assert row["scheduled_target_tokens"] == PREFILL_CHUNKS[index]
        assert row["verified_draft_tokens"] == 0 and row["draft_calls"] == 1
        computed += row["scheduled_target_tokens"]
        if index == 0:
            assert row["sampled_token_ids"] == [], "Partial prefill must not create a sampling seed"
        else:
            assert row["sampled_token_ids"] == [tokens[0]], "Exactly one sampling seed is required"
    assert computed == PROMPT_TOKENS
    assert decode and all(row["prompt_tokens"] == PROMPT_TOKENS
                          and row["scheduled_target_tokens"] == VERIFY_TOKENS
                          and row["verified_draft_tokens"] == K and row["draft_calls"] == 1
                          for row in decode)
    assert all(row["computed_before"] >= PROMPT_TOKENS
               and 1 <= len(row["sampled_token_ids"]) <= VERIFY_TOKENS for row in decode)
    assert decode[0]["computed_before"] == PROMPT_TOKENS
    assert all(left["computed_before"] + len(left["sampled_token_ids"]) == right["computed_before"]
               for left, right in zip(decode, decode[1:])), "Illegal native decode progress"
    assert all(row["draft_calls"] == 1 for row in rows)
    assert [token for row in rows for token in row["sampled_token_ids"]][:OUTPUT_TOKENS] == tokens
    assert sum(len(row["sampled_token_ids"]) for row in rows) == OUTPUT_TOKENS
    decode_tokens = sum(len(row["sampled_token_ids"]) for row in decode)
    assert decode_tokens == OUTPUT_TOKENS - 1
    assert raw["counter"]["terminal_sync_count"] == 1
    assert raw["counter"]["steady_extra_sync_count"] == 0
    assert raw["counter"]["terminal_device_done_s"] >= max(row["sample_done_s"] for row in rows)
    events = raw["graph"]["events"]
    assert all(not event["capturing"] and event["call_returned"] for event in events)

    def replays_in(role, row):
        return sum(event["role"] == role and event["native_replay_returned"]
                   and row["execute_begin_s"] <= event["begin_s"] and event["end_s"] <= row["sample_done_s"]
                   for event in events)

    replayed = [event for event in events if event["native_replay_returned"]]
    assert all(event["existing_entry"] and event["positional_address_match"] and event["mode"] == "FULL"
               for event in replayed)
    replay_counts = {"target": sum(event["role"] == "target" and event["native_replay_returned"]
                                   for event in events),
                     "draft": sum(event["role"] == "draft" and event["native_replay_returned"]
                                  for event in events)}
    assert replay_counts == {"target": len(decode), "draft": len(decode) if arm == "on" else 0}
    assert all(replays_in("target", row) == 1 for row in decode)
    assert all(replays_in("target", row) == 0 for row in prefill)
    assert all(replays_in("draft", row) == 0 for row in prefill)
    if arm == "on":
        assert all(replays_in("draft", row) == 1 for row in decode)
        assert all(sum(event["role"] == "draft" and event["native_replay_returned"]
                       and row["draft_propose_begin_s"] <= event["begin_s"]
                       and event["end_s"] <= row["draft_propose_done_s"] for event in events) == 1
                   for row in decode)
    else:
        assert all(replays_in("draft", row) == 0 for row in decode)
    phase_replays = {}
    for phase, row, expected in (("partial_prefill", prefill[0], 0), ("seed", prefill[1], 0),
                                 ("terminal", decode[-1], 1 if arm == "on" else 0)):
        phase_replays[phase] = replays_in("draft", row)
        assert phase_replays[phase] == expected, (phase, phase_replays[phase], expected)
    helper = raw["helper"]
    expected_helpers = len(rows) if arm == "off" else len(prefill)
    assert helper["native_helper_calls_this_request"] == expected_helpers
    assert helper["native_helper_successful_returns_this_request"] == expected_helpers
    assert helper["cold_capture_helper_calls_global"] == 1
    assert helper["cold_capture_helper_successful_returns_global"] == 1
    assert len(helper["history"]) == expected_helpers
    assert all(not row["capturing"] and row["runtime_mode"] == "NONE" and row["returned"]
               and row["actual_round_at_entry"] is not None
               and row["actual_round_at_entry"]["req_id"] == core_id for row in helper["history"])
    expected_rows = rows if arm == "off" else prefill
    assert [(row["actual_round_at_entry"]["req_id"], row["actual_round_at_entry"]["execute_begin_s"])
            for row in helper["history"]] == [(row["req_id"], row["execute_begin_s"])
                                              for row in expected_rows]
    guard = raw["window_guard"]
    assert guard["diagnostic_device_reads_or_hash_calls_in_window"] == 0
    assert guard["explicit_terminal_device_fences"] == 1
    assert guard["route_logprob_audit_and_profiler"] is False
    assert period2_after_prefill(tokens), "Dummy repetition classification changed"
    return {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": OUTPUT_TOKENS,
            "decode_tokens": decode_tokens,
            "target_decode_rounds": len(decode), "actual_draft_propose_calls": len(rows),
            "prefill_chunks": list(PREFILL_CHUNKS), "prefill_rounds": len(prefill),
            "initial_actual_draft_calls": prefill[0]["draft_calls"],
            "seed_actual_draft_calls": prefill[1]["draft_calls"],
            "terminal_actual_draft_calls": decode[-1]["draft_calls"],
            "target_native_replay_returns": replay_counts["target"],
            "draft_native_replay_returns": replay_counts["draft"],
            "partial_prefill_native_draft_replay_returns": phase_replays["partial_prefill"],
            "seed_native_draft_replay_returns": phase_replays["seed"],
            "terminal_native_draft_replay_returns": phase_replays["terminal"],
            "native_helper_calls": expected_helpers, "native_helper_successful_returns": expected_helpers,
            "terminal_device_fences": 1, "reference_tokens_equal": True,
            "period2_repetition": True, "dummy_clean": False,
            "round_count_derived_from_fresh_receipts": True}


def triplet(seconds, steps, tokens):
    assert math.isfinite(seconds) and seconds > 0 and steps > 0 and tokens > 0
    result = {"ms_per_step": seconds * 1000 / steps, "A": tokens / steps,
              "single_stream_tok_s": tokens / seconds}
    assert math.isclose(result["ms_per_step"], 1000 * result["A"] / result["single_stream_tok_s"], rel_tol=1e-12)
    return result


def pooled_triplet(records):
    assert records
    seconds = sum(row["decode_s"] for row in records)
    steps = sum(row["geometry"]["target_decode_rounds"] for row in records)
    tokens = sum(row["geometry"]["decode_tokens"] for row in records)
    return {"requests": len(records), "decode_s": seconds, "steps": steps, "decode_tokens": tokens,
            **triplet(seconds, steps, tokens), "aggregation": "Sum of matching request decode windows"}
