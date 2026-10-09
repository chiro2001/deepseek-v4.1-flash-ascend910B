#!/usr/bin/env python3
"""Real CED P + mock consumption: cold, local, eviction and DRAM return.

The timings are P response / mock consumption, never real-decode TTFT.
H2D byte deltas and server-side cache counters identify actual DRAM loads.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


KEEP = ("vllm:kv_offload_", "vllm:external_prefix_cache_", "vllm:prefix_cache_",
        "vllm:prompt_tokens_by_source_", "vllm:num_requests_")


def prompt(seed, count):
    return [1000 + ((seed * 100003 + i * 7919) % 100000) for i in range(count)]


def metrics(base):
    response = requests.get(f"{base}/metrics", timeout=30)
    response.raise_for_status()
    return [line for line in response.text.splitlines()
            if not line.startswith("#") and line.startswith(KEEP)]


def metric_sum(lines, name_prefix, required_label=None):
    result = 0.0
    for line in lines:
        if line.startswith(name_prefix) and (required_label is None or required_label in line):
            result += float(line.rsplit(" ", 1)[-1])
    return result


def metric_deltas(before, after):
    queries = {
        "h2d_bytes": ("vllm:kv_offload_total_bytes_total", 'transfer_type="CPU_to_GPU"'),
        "d2h_bytes": ("vllm:kv_offload_total_bytes_total", 'transfer_type="GPU_to_CPU"'),
        "local_hit_tokens": ("vllm:prompt_tokens_by_source_total", 'source="local_cache_hit"'),
        "external_hit_tokens": ("vllm:prompt_tokens_by_source_total", 'source="external_kv_transfer"'),
        "external_cache_hits": ("vllm:external_prefix_cache_hits_total", None),
        "computed_tokens": ("vllm:prompt_tokens_by_source_total", 'source="local_compute"'),
    }
    return {key: metric_sum(after, *query) - metric_sum(before, *query)
            for key, query in queries.items()}


def fingerprints(result):
    return {f"r{row['rank']}/{key}": value
            for row in result["ranks"] for key, value in row["fingerprints"].items()}


def one_request(args, seed, count, tag, checkpoint=None):
    before = metrics(args.prefill_url)
    started = time.perf_counter()
    response = requests.post(
        f"{args.prefill_url}/v1/completions",
        json={"model": args.model, "prompt": prompt(seed, count), "max_tokens": 1,
              "temperature": 0, "ignore_eos": True, "stream": False,
              "kv_transfer_params": {"do_remote_decode": True, "do_remote_prefill": False}},
        timeout=args.timeout,
    )
    response.raise_for_status()
    document = response.json()
    p_response_s = time.perf_counter() - started
    params = document.get("kv_transfer_params")
    if not params or not params.get("remote_block_ids"):
        raise RuntimeError(f"P did not return a CED handoff: {document}")
    result = {
        "tag": tag, "seed": seed, "prompt_tokens": count,
        "ced_prefix_tokens": params["ced_prefix_tokens"], "request_id": params["remote_request_id"],
        "kv_transfer_params": params, "prefill_response_s": p_response_s,
        "metrics_before": before, "phase": "prefill_complete",
    }
    if checkpoint is not None:
        checkpoint(result)
    consumed = requests.post(
        f"{args.mock_url}/consume",
        json={"kv_transfer_params": params, "hold_ms": args.hold_ms,
              "page_fingerprints": args.page_fingerprints}, timeout=args.timeout,
    )
    consumed.raise_for_status()
    mock = consumed.json()
    if not mock.get("ok") or len(mock["ranks"]) != 8:
        raise RuntimeError(f"Mock did not consume eight P ranks: {mock}")
    # The native engine keeps stepping to publish async completion metadata.
    # Give its counters a short collection window after consumption ACK.
    deadline = time.monotonic() + 10
    while True:
        after = metrics(args.prefill_url)
        writing = metric_sum(after, "vllm:kv_offload_cpu_cache_write_usage_perc")
        if writing == 0 or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    result.update({
        "mock": mock, "phase": "consumption_complete", "metrics_after": after,
        "delta": metric_deltas(before, after), "fingerprints": fingerprints(mock),
    })
    print(json.dumps({k: result[k] for k in ("tag", "prompt_tokens", "prefill_response_s", "delta")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--prefill-url", default="http://127.0.0.1:19190")
    parser.add_argument("--mock-url", default="http://127.0.0.1:19191")
    parser.add_argument("--model", default="deepseek-v41-ced-dram-p")
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--interleave", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--hold-ms", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--boundaries", action="store_true")
    parser.add_argument("--page-fingerprints", action="store_true",
                        help="Record per-page hashes to distinguish loaded prefix from recomputed tail")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 3 or args.interleave < 4:
        parser.error("Use at least three rounds and four independent prefixes")
    models = requests.get(f"{args.prefill_url}/v1/models", timeout=30).json()
    if args.model not in {row["id"] for row in models.get("data", [])}:
        raise RuntimeError("P model identity differs from the experiment")
    health = requests.get(f"{args.mock_url}/health", timeout=30)
    if health.text != "ced-mock-decode":
        raise RuntimeError("Mock endpoint identity differs from the experiment")
    report = {"settings": vars(args).copy(), "results": [], "errors": []}
    report["settings"]["out"] = str(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Check result-file access before sending a request that pins producer KV.
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def save():
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def run(seed, count, tag):
        def checkpoint(partial):
            report["inflight"] = partial
            save()

        result = one_request(args, seed, count, tag, checkpoint)
        report.pop("inflight", None)
        report["results"].append(result)
        save()
        return result

    try:
        run(99, 1025, "warmup")
        reference = run(0, args.prompt_tokens, "target_cold")
        hbm = run(0, args.prompt_tokens, "target_hbm")
        reference_fp = reference["fingerprints"]
        comparisons = [hbm["fingerprints"] == reference_fp]
        h2d = []
        local_at_return = []
        for iteration in range(args.rounds):
            # Restored requests allocate fewer unused CED pages than cold
            # requests. Reusing the same evictors can make all four fit in
            # HBM after round one. Use fresh, independent cold prefixes to
            # maintain pressure, and still require observed H2D each return.
            for offset in range(args.interleave - 1):
                seed = 1 + iteration * (args.interleave - 1) + offset
                run(seed, args.prompt_tokens, f"evict_{iteration}_{seed}")
            returned = run(0, args.prompt_tokens, f"target_return_{iteration}")
            comparisons.append(returned["fingerprints"] == reference_fp)
            h2d.append(returned["delta"]["h2d_bytes"])
            local_at_return.append(returned["delta"]["local_hit_tokens"])
        if args.boundaries:
            for count in (1, 127, 128, 129, 1023, 1024, 1025):
                for iteration in range(3):
                    run(100 + count, count, f"boundary_{count}_{iteration}")
            run(0, args.prompt_tokens // 2, "partial_prefix")
            run(0, args.prompt_tokens + 1024, "append_prefix")
        report["verdict"] = {
            "nonempty_fingerprints": bool(reference_fp),
            "all_three_returns_loaded_dram": all(value > 0 for value in h2d),
            "all_return_payloads_equal_cold": all(comparisons),
            "h2d_bytes_each_return": h2d, "local_hit_tokens_each_return": local_at_return,
            "real_decode_text_and_ttft_verified": False,
        }
        report["ok"] = all(report["verdict"][key] for key in (
            "nonempty_fingerprints", "all_three_returns_loaded_dram", "all_return_payloads_equal_cold",
        ))
    except Exception as exc:
        report["ok"] = False
        report["errors"].append(repr(exc))
        raise
    finally:
        save()
    print(json.dumps(report["verdict"], ensure_ascii=False), flush=True)
    return 0 if report["ok"] else 9


if __name__ == "__main__":
    raise SystemExit(main())
