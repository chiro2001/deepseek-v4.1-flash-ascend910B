"""CPU-only geometry checks and guards for the tiny timing diagnostic."""
import math


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


def validate_geometry(raw, tokens, arm, core_id, reference_tokens):
    """Validate this fresh 128/32 run from actual counters, including both end calls."""
    assert arm in ("off", "on") and len(tokens) == 32
    assert tokens == reference_tokens, "Fresh output differs from the frozen 128/32 reference"
    rows = raw["counter"]["rounds"]
    assert rows and all(row["req_id"] == core_id for row in rows)
    seed = [row for row in rows if not row["is_decode"]]
    decode = [row for row in rows if row["is_decode"]]
    assert len(seed) == 1 and len(decode) == 16 and len(rows) == 17
    assert seed[0]["prompt_tokens"] == seed[0]["scheduled_target_tokens"] == 128
    assert seed[0]["computed_before"] == 0 and seed[0]["verified_draft_tokens"] == 0
    assert all(row["scheduled_target_tokens"] == 6 and row["verified_draft_tokens"] == 5
               and row["prompt_tokens"] == 128 for row in decode)
    assert all(row["draft_calls"] == 1 for row in rows)
    assert [token for row in rows for token in row["sampled_token_ids"]][:32] == tokens
    assert raw["counter"]["terminal_sync_count"] == 1
    assert raw["counter"]["steady_extra_sync_count"] == 0
    assert raw["counter"]["terminal_device_done_s"] >= max(row["sample_done_s"] for row in rows)
    events = raw["graph"]["events"]
    assert all(not event["capturing"] and event["call_returned"] for event in events)
    replay_counts = {role: sum(event["role"] == role and event["native_replay_returned"]
                              for event in events) for role in ("target", "draft")}
    assert replay_counts == {"target": 16, "draft": 16 if arm == "on" else 0}
    assert all(event["existing_entry"] and event["positional_address_match"] and event["mode"] == "FULL"
               for event in events if event["native_replay_returned"])
    phase_replays = {}
    for phase, row in (("initial", seed[0]), ("terminal", decode[-1])):
        matching = [event for event in events if event["role"] == "draft"
                    and row["execute_begin_s"] <= event["begin_s"]
                    and event["end_s"] <= row["sample_done_s"]]
        phase_replays[phase] = sum(event["native_replay_returned"] for event in matching)
    assert phase_replays == {"initial": 0, "terminal": 1 if arm == "on" else 0}
    helper = raw["helper"]
    expected_helpers = 1 if arm == "on" else 17
    assert helper["native_helper_calls_this_request"] == expected_helpers
    assert helper["native_helper_successful_returns_this_request"] == expected_helpers
    assert helper["cold_capture_helper_calls_global"] == 1
    assert helper["cold_capture_helper_successful_returns_global"] == 1
    assert len(helper["history"]) == expected_helpers
    assert all(not row["capturing"] and row["runtime_mode"] == "NONE" and row["returned"]
               and row["actual_round_at_entry"]["req_id"] == core_id for row in helper["history"])
    assert sum(not row["actual_round_at_entry"]["is_decode"] for row in helper["history"]) == 1
    guard = raw["window_guard"]
    assert guard["diagnostic_device_reads_or_hash_calls_in_window"] == 0
    assert guard["explicit_terminal_device_fences"] == 1
    assert guard["route_logprob_audit_and_profiler"] is False
    assert period2_after_prefill(tokens), "Dummy repetition classification changed"
    return {"prompt_tokens": 128, "completion_tokens": 32, "decode_tokens": 31,
            "target_decode_rounds": 16, "actual_draft_propose_calls": 17,
            "initial_actual_draft_calls": seed[0]["draft_calls"],
            "terminal_actual_draft_calls": decode[-1]["draft_calls"],
            "target_native_replay_returns": replay_counts["target"],
            "draft_native_replay_returns": replay_counts["draft"],
            "initial_native_draft_replay_returns": phase_replays["initial"],
            "terminal_native_draft_replay_returns": phase_replays["terminal"],
            "native_helper_calls": expected_helpers, "native_helper_successful_returns": expected_helpers,
            "terminal_device_fences": 1, "reference_tokens_equal": True,
            "period2_repetition": True, "dummy_clean": False}


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
