"""Pure CPU reduction of one logical DP request, not TP rank sums."""
import math

from .metrics_contract import validate_metrics
from .topology import integer
from .draft_proof import validate_phase_counts


def finite(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError("Invalid actual timing/count: " + name)
    return value


def request_metrics(arm, metrics, stream, topology, *, dp_rank, tag, sid,
                    fingerprint, prompt_tokens, max_tokens, ignore_eos):
    integer(dp_rank, "dp_rank")
    if dp_rank >= topology.dp:
        raise ValueError("Foreign DP rank")
    for receipt in (arm, metrics, stream):
        if integer(receipt.get("dp_rank"), "dp_rank") != dp_rank:
            raise ValueError("Actual DP rank differs from the measured request")
    for receipt in (arm, metrics):
        if receipt.get("service_instance_id") != sid:
            raise ValueError("Measurement service instance changed")
    if (metrics.get("runtime_fingerprint") != fingerprint
            or metrics.get("dp_rank") != dp_rank
            or metrics.get("schema") != "dp_request_metrics_v1"
            or metrics.get("trusted") is not True):
        raise ValueError("Unbound DP counter receipt")
    if arm.get("dp_rank") != dp_rank or arm.get("tag") != tag:
        raise ValueError("Arm disagrees with actual DP request")
    binding = metrics.get("request_binding", {})
    validated = validate_metrics(metrics.get("ranks"), topology, dp_rank=dp_rank,
                                 tag=tag, request_id=binding.get("request_id"))
    if binding != validated:
        raise ValueError("Counter request binding differs from rank receipts")
    if (stream.get("http_status") != 200 or stream.get("dp_rank") != dp_rank
            or stream.get("done") is not True or stream.get("error") is not None):
        raise ValueError("Actual routed HTTP/SSE generation did not complete")
    usage = stream.get("usage", {})
    prompt = integer(usage.get("prompt_tokens"), "prompt_tokens", minimum=1)
    completion = integer(usage.get("completion_tokens"), "completion_tokens", minimum=2)
    if (prompt != prompt_tokens or prompt != binding["prompt_tokens"]
            or usage.get("total_tokens") != prompt + completion
            or completion > max_tokens or stream.get("finish_reason") not in ("stop", "length")):
        raise ValueError("Actual usage/finish contract differs")
    if ignore_eos and (completion != max_tokens or stream["finish_reason"] != "length"):
        raise ValueError("Fixed-budget generation did not reach its exact budget")
    before_rows = arm.get("ranks")
    if (not isinstance(before_rows, list) or len(before_rows) != topology.tp
            or any(type(r.get("rank")) is not int for r in before_rows)
            or {r["rank"] for r in before_rows} != set(range(topology.tp))):
        raise ValueError("Incomplete TP arm baselines")
    before = {r["rank"]: r for r in before_rows}
    prefills, decodes, graph_calls, eager_calls, phases = [], [], [], [], []
    rounds = binding["decode_rounds"]
    for row in metrics["ranks"]:
        baseline = before[row["rank"]]
        if (baseline.get("dp_rank") != dp_rank or baseline.get("armed") is not True
                or baseline.get("async_mode") is not True):
            raise ValueError("Wrong DP/async arm baseline")
        delta = {}
        for key in ("prefill_seconds", "decode_seconds", "decode_rounds",
                    "draft_graph_calls", "draft_eager_calls"):
            previous = finite(baseline["cumulative_before"].get(key), key)
            current = finite(row["cumulative"].get(key), key)
            delta[key] = finite(current - previous, key)
        detail = next(r for r in row["requests"] if r.get("tag") == tag)
        if detail.get("async_mode") is not True or delta["decode_rounds"] != rounds:
            raise ValueError("Actual async verifier rounds differ from cumulative delta")
        start = integer(detail.get("decode_start_ns"), "decode_start_ns", minimum=1)
        sample = integer(detail.get("terminal_sample_end_ns"), "terminal_sample_end_ns", minimum=1)
        end = integer(detail.get("fenced_end_ns"), "fenced_end_ns", minimum=1)
        if not start <= sample <= end or abs(delta["decode_seconds"] - (end-start)/1e9) > 0.005:
            raise ValueError("Worker fence timestamps differ from decode wall")
        prefills.append(delta["prefill_seconds"])
        decodes.append(delta["decode_seconds"])
        graph_calls.append(delta["draft_graph_calls"])
        eager_calls.append(delta["draft_eager_calls"])
        phases.append(validate_phase_counts(detail.get("draft_phase_counts"), rounds=rounds,
                                            graph_calls=delta["draft_graph_calls"],
                                            eager_calls=delta["draft_eager_calls"]))
    http_s = finite(stream.get("decode_s"), "HTTP decode_s", positive=True)
    fenced_s = finite(max(decodes), "worker-fenced decode_s", positive=True)
    tokens = completion - binding["seed_tokens"]
    acceptance = tokens / rounds
    return {"dp_rank": dp_rank, "request_binding": binding,
            "decode_tokens": tokens, "verifier_rounds": rounds, "A": acceptance,
            "http": {"decode_s": http_s, "ms_step": 1000*http_s/rounds,
                     "tok_s": tokens/http_s,
                     "scope": "first-content -> last-content SSE; excludes TTFT"},
            "server_fenced": {"decode_s": fenced_s, "ms_step": 1000*fenced_s/rounds,
                              "tok_s": tokens/fenced_s,
                              "scope": "max active-TP terminal-prefill -> device fence; includes metrics RPC delay"},
            "ttft_s": finite(stream.get("ttft_s"), "TTFT", positive=True),
            "worker_prefill_s": finite(max(prefills), "worker prefill_s", positive=True),
            "draft_graph_calls_by_tp": graph_calls,
            "draft_eager_calls_by_tp": eager_calls,
            "draft_initial_eager_calls_by_tp": [p["initial_eager_calls"] for p in phases],
            "draft_decode_graph_calls_by_tp": [p["decode_graph_calls"] for p in phases],
            "draft_decode_eager_calls_by_tp": [p["decode_eager_calls"] for p in phases],
            "steps_deduplicated_across_tp_dcp": True}


def wave_metrics(streams):
    """Same-client-clock global decode interval; never a per-stream rate."""
    if not streams:
        raise ValueError("Actual streams are required")
    start = min(finite(r.get("first_content_monotonic"), "first content", positive=True) for r in streams)
    end = max(finite(r.get("last_content_monotonic"), "last content", positive=True) for r in streams)
    seconds = finite(end-start, "global decode interval", positive=True)
    tokens = sum(integer(r["usage"]["completion_tokens"], "completion tokens", minimum=2)-1 for r in streams)
    return {"concurrency": len(streams), "decode_tokens": tokens,
            "decode_wall_s": seconds, "aggregate_decode_tok_s": tokens/seconds,
            "scope": "sum logical HTTP completion-minus-seed tokens / earliest first-content -> latest last-content; not single-stream tok/s",
            "maximum_saturation_proven": False}
