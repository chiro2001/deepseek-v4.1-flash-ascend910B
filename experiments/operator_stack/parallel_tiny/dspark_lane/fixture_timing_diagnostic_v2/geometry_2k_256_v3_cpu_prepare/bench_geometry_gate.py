"""Same-instance real eager/graph path, strict actual consumers, then paired timing."""
import argparse
import hashlib
from geometry_contract import validate_rounds, validate_native_helpers
import json
import math
import os
from pathlib import Path
import statistics
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def latest_round(worker):
    rows = worker.model_runner._tiny_dspark_receipts
    assert rows and worker.model_runner._tiny_dspark_pending is None
    return rows[-1]


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--prompt-tokens", type=int, default=2048)
    args = p.parse_args()
    assert args.pairs >= 8 and args.tokens == 256 and args.prompt_tokens == 2048
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "4" and os.environ["ASCEND_VISIBLE_DEVICES"] == "4"
    assert os.environ["HCCL_DETERMINISTIC"] == "strict"
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(out / "vllm_cache")
    import torch
    import torch_npu
    from vllm import LLM, SamplingParams
    assert torch.npu.device_count() == 1
    kwargs = dict(model=args.model, tokenizer=args.model, load_format="dummy", dtype="bfloat16",
                  worker_cls="geometry_quality_worker.GeometryQualityWorker", tensor_parallel_size=1,
                  enable_expert_parallel=True, seed=0, trust_remote_code=True, async_scheduling=False,
                  speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": False},
                  limit_mm_per_prompt={"image": 0}, max_model_len=4096, max_num_seqs=1,
                  max_num_batched_tokens=2048, gpu_memory_utilization=.70,
                  kv_cache_memory_bytes=2 * 1024**3, block_size=128, enable_prefix_caching=False,
                  enable_return_routed_experts=False,
                  compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [6]},
                  additional_config={"enable_engram": False, "enable_cpu_binding": False,
                                     "ascend_compilation_config": {"enable_npugraph_ex": True,
                                                                   "enable_static_kernel": False},
                                     "multistream_dsv4_dsa_overlap": False})
    dump(out / "config.json", {"llm": kwargs, "latest_formal_target_ms": 20, "physical_device": 4,
                               "same_model_instance": True, "dummy_weights": True,
                               "target_graph": True, "draft_graph_candidate": True,
                               "actual_entry_replay_required": True, "formal_tp8_claim": False})
    llm = LLM(**kwargs)
    # Let the normal platform/worker import order initialize the native factory.
    import dspark_worker
    import draft_graph_observer as observer
    import graph_precision_audit as audit
    import geometry_cache_probe
    dump(out / "topology.json", llm.collective_rpc(dspark_worker.topology)[0])
    engine, records, sequence = llm.llm_engine, [], 0

    def request(arm, seed, tag, auditing=True):
        nonlocal sequence
        sequence += 1
        rid = f"draft-graph-{sequence:03d}"
        import o_proj_cache_fixture
        fixture_before = llm.collective_rpc(o_proj_cache_fixture.validate_all, args=(tag + ".before",))[0]
        dump(out / f"fixture_before_{sequence:03d}.json", fixture_before)
        llm.collective_rpc(audit.switch, args=(arm, auditing, tag))
        llm.collective_rpc(dspark_worker.reset_receipts)
        llm.collective_rpc(observer.begin)
        if os.environ.get("DSPARK_FIRST_VALUE_TRACE") == "1":
            import native_first_value_trace as trace
            llm.collective_rpc(trace.begin_request)
        prompt = [100 + (index * 17 + seed * 19) % 97 for index in range(args.prompt_tokens)]
        core_id = engine.add_request(rid, {"prompt_token_ids": prompt}, SamplingParams(
            temperature=0, max_tokens=args.tokens, ignore_eos=True, detokenize=False,
            logprobs=5 if auditing else None))
        assert engine.output_processor.external_req_ids[rid] == [core_id]
        final, graph_events_seen, step_audit = None, 0, []
        geometry_points, geometry_steps = {}, 0
        geometry_digest = hashlib.sha256()
        while engine.has_unfinished_requests():
            outputs = engine.step()
            if auditing:
                row = llm.collective_rpc(latest_round)[0]
                events = llm.collective_rpc(observer.read)[0]["events"]
                fresh = events[graph_events_seen:]
                graph_events_seen = len(events)
                replayed = any(e["role"] == "draft" and e["native_replay_returned"] for e in fresh)
                if row["is_decode"]:
                    assert replayed == (arm == "on"), "Requested arm is not the actual steady submission path"
                step_audit.append(llm.collective_rpc(audit.snapshot_step, args=(row["is_decode"], replayed))[0])
                if row["is_decode"]:
                    point = llm.collective_rpc("snapshot_geometry_step", args=(True, replayed))[0]
                    geometry_digest.update(json.dumps(point, sort_keys=True).encode())
                    geometry_steps += 1
                    if "first_decode" not in geometry_points:
                        geometry_points["first_decode"] = point
                    geometry_points["terminal"] = point
            for item in outputs:
                if item.finished:
                    final = item
        client_done = time.perf_counter()
        raw = llm.collective_rpc(dspark_worker.read_receipts)[0]
        if os.environ.get("DSPARK_O_PROJ_CACHE_FIXTURE") == "1":
            import o_proj_cache_fixture
            fixture = llm.collective_rpc(o_proj_cache_fixture.validate_all, args=(tag,))[0]
            dump(out / f"fixture_{sequence:03d}.json", fixture)
        observation = llm.collective_rpc(observer.read)[0]
        helper_receipt = (llm.collective_rpc(trace.read_helper_receipt)[0]
                          if os.environ.get("DSPARK_FIRST_VALUE_TRACE") == "1" else None)
        assert raw["pid"] == os.getpid() and final is not None and final.request_id == rid
        tokens = list(final.outputs[0].token_ids)
        rows = raw["rounds"]
        dump(out / f"raw_{sequence:03d}.json", {"external_id": rid, "core_id": core_id,
             "counter": raw, "graph": observation, "helper": helper_receipt,
             "token_ids": tokens, "arm": arm})
        assert all(row["req_id"] == core_id for row in rows)
        prefill_rows = [row for row in rows if not row["is_decode"]]
        seed_rows = [row for row in prefill_rows if row["sampled_token_ids"]]
        decode = [row for row in rows if row["is_decode"]]
        assert len(seed_rows) == 1 and decode and len(tokens) == args.tokens
        independent_geometry = validate_rounds(rows, tokens, core_id)
        independent_geometry.update(validate_native_helpers({"counter": raw, "helper": helper_receipt}, arm))
        if auditing:
            assert geometry_steps == len(decode)
            dump(out / f"geometry_{sequence:03d}.json", {"tag": tag, "steps": geometry_steps,
                 "all_step_receipt_sha256": geometry_digest.hexdigest(), "points": geometry_points})
        assert [t for row in rows for t in row["sampled_token_ids"]][:len(tokens)] == tokens
        draft_replays = sum(e["role"] == "draft" and e["native_replay_returned"] for e in observation["events"])
        target_replays = sum(e["role"] == "target" and e["native_replay_returned"] for e in observation["events"])
        initial_events = [e for e in observation["events"] if e["role"] == "draft"
                          and seed_rows[0]["execute_begin_s"] <= e["begin_s"]
                          and e["end_s"] <= seed_rows[0]["sample_done_s"]]
        terminal_events = [e for e in observation["events"] if e["role"] == "draft"
                           and decode[-1]["execute_begin_s"] <= e["begin_s"]
                           and e["end_s"] <= decode[-1]["sample_done_s"]]
        assert target_replays == len(decode)
        assert draft_replays == len(decode) if arm == "on" else draft_replays == 0
        assert all(row["draft_calls"] == 1 for row in rows)
        assert raw["terminal_sync_count"] == 1 and raw["steady_extra_sync_count"] == 0
        assert raw["terminal_device_done_s"] >= max(row["sample_done_s"] for row in rows)
        expected_helpers = len(rows) if arm == "off" else len(prefill_rows)
        assert helper_receipt["native_helper_calls_this_request"] == expected_helpers
        assert helper_receipt["native_helper_successful_returns_this_request"] == expected_helpers
        assert helper_receipt["cold_capture_helper_calls_global"] == 1
        assert helper_receipt["cold_capture_helper_successful_returns_global"] == 1
        assert all(not row["capturing"] and row["runtime_mode"] == "NONE" and row["native_helper_returned"]
                   for row in helper_receipt["helper_history_this_request"])
        assert sum(e["native_replay_returned"] for e in initial_events) == 0
        assert sum(e["native_replay_returned"] for e in terminal_events) == (1 if arm == "on" else 0)
        decode_tokens = len(tokens) - len(seed_rows[0]["sampled_token_ids"])
        decode_s = max(client_done, raw["terminal_device_done_s"]) - seed_rows[0]["bookkeeping_done_s"]
        rounds = len(decode)
        record = {"tag": tag, "arm": arm, "seed": seed, "audit": auditing,
                  "external_id": rid, "core_id": core_id, "token_ids": tokens,
                  "target_decode_rounds": rounds, "decode_tokens": decode_tokens,
                  "draft_calls": sum(row["draft_calls"] for row in rows),
                  "verified_proposals": sum(row["verified_draft_tokens"] for row in decode),
                  "raw_accepts": sum(row["accepted_draft_tokens"] for row in decode),
                  "target_native_replay_returns": target_replays, "draft_native_replay_returns": draft_replays,
                  "initial_draft_actual_eager_or_graph": step_audit[0]["draft_replayed"] if auditing else None,
                  "initial_actual_draft_calls": prefill_rows[0]["draft_calls"],
                  "sampling_seed_actual_draft_calls": seed_rows[0]["draft_calls"],
                  "partial_prefill_actual_draft_calls": len(prefill_rows) - 1,
                  "all_prefill_actual_draft_calls": len(prefill_rows),
                  "terminal_actual_draft_calls": decode[-1]["draft_calls"],
                  "initial_native_draft_replay_returns": sum(e["native_replay_returned"] for e in initial_events),
                  "terminal_native_draft_replay_returns": sum(e["native_replay_returned"] for e in terminal_events),
                  "initial_and_terminal_device_complete_at_request_fence": raw["terminal_sync_count"] == 1,
                  "native_helper_receipt": {key: value for key, value in helper_receipt.items() if key != "helper_history_this_request"},
                  "independent_2048_256_geometry": independent_geometry,
                  "all_step_cache_geometry_receipts": geometry_steps,
                  "performance_evaluated": False,
                  "invasive_diagnostic_reads_add_sync": True,
                  "dummy_clean_rate": 0.0,
                  "profiler": False, "formal_tp8_claim": False,
                  "target_full_graph": True, "target_capture_sizes": [6],
                  "draft_constructor_configured": True, "draft_actual_steady_graph": arm == "on"}
        if auditing:
            record["consumer_receipt"] = llm.collective_rpc(audit.finish)[0]
            record["logprobs"] = [{str(key): float(value.logprob) for key, value in row.items()}
                                  for row in final.outputs[0].logprobs]
        records.append(record)
        dump(out / "requests.json", records)
        print("DRAFT_GRAPH_REQUEST", json.dumps({k: v for k, v in record.items()
                                                if k not in ("token_ids", "logprobs")}), flush=True)
        return record

    def compare(a, b, kind):
        assert a["token_ids"] == b["token_ids"]
        assert len(a["logprobs"]) == len(b["logprobs"])
        delta = 0.0
        for left, right in zip(a["logprobs"], b["logprobs"]):
            assert set(left) == set(right)
            assert all(math.isfinite(value) for value in (*left.values(), *right.values()))
            delta = max(delta, *(abs(left[key] - right[key]) for key in left))
        assert delta < 1e-3
        consumers = llm.collective_rpc(audit.compare, args=(a["tag"], b["tag"]))[0]
        return {"kind": kind, "token_equal": True, "top5_keys_equal": True,
                "max_logprob_delta": delta, "actual_consumers": consumers, "passed": True}

    request("off", 0, "warmup-off", auditing=False)
    request("on", 1, "warmup-on", auditing=False)
    checks = []
    for seed in range(3):
        a = request("off", seed, f"native-off-{seed}-A")
        b = request("off", seed, f"native-off-{seed}-B")
        checks.append(compare(a, b, "native_off_AA"))
        dump(out / "checks.json", checks)
    for seed in range(3):
        arms = ("off", "on") if seed % 2 == 0 else ("on", "off")
        values = {arm: request(arm, seed, f"candidate-{seed}-{arm}") for arm in arms}
        checks.append(compare(values["off"], values["on"], "off_on"))
        a = request("on", seed, f"native-on-{seed}-A")
        b = request("on", seed, f"native-on-{seed}-B")
        checks.append(compare(a, b, "native_on_AA"))
        dump(out / "checks.json", checks)
    dump(out / "accuracy_gate.json", {"passed": True, "checks": checks,
                                       "actual_route_attention_cache_gate": True, "independent_2048_256_geometry_cache_gate": True,
                                       "dummy_clean_rate": 0.0, "formal_quality_claim": False})
    if (os.environ.get("DSPARK_ROUTE_IMMEDIATE_DIAGNOSTIC") == "1"
            or os.environ.get("DSPARK_FIRST_VALUE_TRACE") == "1"
            or os.environ.get("DSPARK_O_PROJ_CACHE_FIXTURE") == "1"):
        dump(out / "diagnostic_completion.json", {"consumer_comparator_passed_in_diagnostic_mode": True,
              "invasive_diagnostic_reads_add_sync": True, "performance_evaluation_disabled": True,
              "tiny_mechanism_candidate_admitted": False, "formal_candidate_admitted": False, "independent_geometry": {"prompt_tokens": 2048, "completion_tokens": 256},
              "dummy_clean_rate": 0.0, "complete_actual_per_group_SWA_cache_gate": True})
        return
    raise AssertionError("This independent geometry/cache gate requires diagnostics and never enters timing")


if __name__ == "__main__":
    main()
