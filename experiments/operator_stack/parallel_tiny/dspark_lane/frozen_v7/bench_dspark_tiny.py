"""Actual target/MTP/verifier diagnostic baseline; tiny is never formal TP8."""
import argparse
import faulthandler
import json
import math
import os
from pathlib import Path
import statistics
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
faulthandler.enable()
faulthandler.dump_traceback_later(180, repeat=True)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--target-ms", type=float, default=20)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "4"
    assert os.environ["ASCEND_VISIBLE_DEVICES"] == "4"
    assert os.environ["HCCL_DETERMINISTIC"] == "strict"
    assert args.target_ms == 20
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "runner.pid").write_text(str(os.getpid()) + "\n")
    os.environ["VLLM_CACHE_ROOT"] = str(out / "vllm_cache")
    import torch
    import torch_npu
    from vllm import LLM, SamplingParams
    assert torch.npu.device_count() == 1
    kwargs = dict(model=str(args.model), tokenizer=str(args.model), load_format="dummy",
                  dtype="bfloat16", worker_cls="dspark_worker.DSparkTinyWorker",
                  tensor_parallel_size=1, enable_expert_parallel=True, seed=0,
                  trust_remote_code=True, async_scheduling=False,
                  speculative_config={"method": "dspark", "num_speculative_tokens": 5,
                                      "enforce_eager": True},
                  limit_mm_per_prompt={"image": 0}, max_model_len=2048, max_num_seqs=1,
                  max_num_batched_tokens=256, gpu_memory_utilization=.70,
                  kv_cache_memory_bytes=2 * 1024 ** 3, block_size=128,
                  enable_prefix_caching=False, enable_return_routed_experts=False,
                  compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                                      "cudagraph_capture_sizes": [6]},
                  additional_config={"enable_engram": False, "enable_cpu_binding": False,
                                     "ascend_compilation_config": {"enable_npugraph_ex": True,
                                                                   "enable_static_kernel": False},
                                     "multistream_dsv4_dsa_overlap": False})
    dump(out / "config.json", {"llm": kwargs, "physical_device": 4,
                               "latest_formal_target_ms": args.target_ms,
                               "formal_tp8_claim": False, "dummy_weights": True,
                               "engram_enabled": False, "target_graph": True,
                               "draft_graph": False, "torch": torch.__version__,
                               "torch_npu": torch_npu.__version__,
                               "clock": time.get_clock_info("perf_counter").implementation,
                               "counter_probe_extra_d2h": False,
                               "terminal_device_sync_per_request": 1,
                               "steady_extra_sync_per_round": 0,
                               "startup_profile_capture_excluded": True})
    print("DSPARK_TINY_INITIALIZING", flush=True)
    begin = time.perf_counter()
    llm = LLM(**kwargs)
    startup_s = time.perf_counter() - begin
    # Importing NPUWorker before LLM initializes the platform binds a model's
    # FusedMoEFactory to the upstream function before the Ascend patch runs.
    # Let the engine resolve worker_cls in its normal initialization order.
    import dspark_worker
    dump(out / "topology.json", llm.collective_rpc(dspark_worker.topology)[0])
    engine, counter, records = llm.llm_engine, 0, []

    def request(seed, warmup=False, audit=False, profile=False):
        nonlocal counter
        counter += 1
        rid = f"dspark-tiny-{counter:03d}"
        llm.collective_rpc(dspark_worker.reset_receipts)
        prompt = [100 + (i * 17 + seed * 19) % 97 for i in range(args.prompt_tokens)]
        max_tokens = max(args.tokens, 96) if profile else args.tokens
        core_rid = engine.add_request(rid, {"prompt_token_ids": prompt},
                                      SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True,
                                                     detokenize=False, logprobs=5 if audit else None))
        assert isinstance(core_rid, str)
        assert engine.output_processor.external_req_ids[rid] == [core_rid]
        final, calls, profiler, profile_steps = None, 0, None, 0
        while engine.has_unfinished_requests():
            # The first engine call is the single unchunked prefill. Start the
            # profiler only afterwards; five actual decode calls warm it up.
            if profile and calls == 1:
                profiler = torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=10, repeat=1),
                    record_shapes=True, with_stack=False, profile_memory=False,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                        export_type=torch_npu.profiler.ExportType.Text, data_simplification=False),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(out / "profile"), worker_name="dspark_physical4"))
                profiler.start()
            outputs = engine.step()
            calls += 1
            if profiler is not None:
                profiler.step()
                profile_steps += 1
                if profile_steps == 15:
                    profiler.stop()
                    profiler = None
            for item in outputs:
                if item.finished:
                    final = item
        client_done = time.perf_counter()
        assert profiler is None and (not profile or profile_steps == 15)
        receipt = llm.collective_rpc(dspark_worker.read_receipts)[0]
        dump(out / f"raw_request_{counter:03d}.json", {"external_request_id": rid,
             "core_request_id": core_rid, "client_done_s": client_done, "receipt": receipt,
             "output_request_id": final.request_id if final is not None else None,
             "token_ids": list(final.outputs[0].token_ids) if final is not None else None})
        assert receipt["pid"] == os.getpid(), "Counter clock/process must match synchronous client"
        assert final is not None
        token_ids = list(final.outputs[0].token_ids)
        rows = receipt["rounds"]
        prefill = [r for r in rows if not r["is_decode"]]
        decode = [r for r in rows if r["is_decode"]]
        assert len(prefill) == 1 and len(decode) > 0
        assert final.request_id == rid
        assert all(r["req_id"] == core_rid for r in rows)
        assert len(token_ids) == max_tokens
        seed_tokens = len(prefill[0]["sampled_token_ids"])
        decode_tokens = len(token_ids) - seed_tokens
        end_s = max(client_done, receipt["terminal_device_done_s"])
        decode_s = end_s - prefill[0]["bookkeeping_done_s"]
        rounds = len(decode)
        acceptance = decode_tokens / rounds
        ms = 1000 * decode_s / rounds
        tps = decode_tokens / decode_s
        assert abs(ms - 1000 * acceptance / tps) < 1e-10
        raw_tokens = sum(r["raw_generated_tokens"] for r in decode)
        accepted = sum(r["accepted_draft_tokens"] for r in decode)
        proposed = sum(r["verified_draft_tokens"] for r in decode)
        repeat_period = None
        for period in range(1, min(16, len(token_ids) // 3) + 1):
            tail = token_ids[-period:]
            if token_ids[-3 * period:] == tail * 3:
                repeat_period = period
                break
        # The final verifier may produce beyond the request budget; only client
        # output tokens enter A/tok/s. Raw accepts remain separately observable.
        record = {"request_id": rid, "core_request_id": core_rid, "seed": seed, "warmup": warmup, "audit": audit,
                  "profiler_enabled": profile, "profile_schedule_decode_steps": profile_steps,
                  "profile_warmup_decode_steps": 5 if profile else 0,
                  "profile_active_decode_steps": 10 if profile else 0,
                  "frontend_step_calls": calls, "target_decode_rounds": rounds,
                  "verifier_rounds": sum(r["verified_draft_tokens"] > 0 for r in decode),
                  "draft_calls": sum(r["draft_calls"] for r in rows),
                  "initial_draft_calls": prefill[0]["draft_calls"],
                  "proposed_tokens_verified": proposed, "raw_accepted_draft_tokens": accepted,
                  "raw_decode_generated_tokens": raw_tokens,
                  "completion_tokens": len(token_ids), "prefill_seed_tokens": seed_tokens,
                  "decode_tokens": decode_tokens, "final_budget_clipped_tokens": raw_tokens - decode_tokens,
                  "decode_s": decode_s, "ms_per_step": ms, "A": acceptance, "single_stream_tok_s": tps,
                  "decode_window_start": "prefill/seed bookkeeping done, before initial MTP draft",
                  "decode_window_end": "final RequestOutput received plus one terminal device completion fence",
                  "terminal_device_sync_count": receipt["terminal_sync_count"],
                  "steady_extra_sync_count": receipt["steady_extra_sync_count"],
                  "token_ids": token_ids, "receipts": rows,
                  "dummy_output_quality_claim": False,
                  "empty_output": not token_ids,
                  "token_suffix_repetition_period": repeat_period,
                  "token_unique_count": len(set(token_ids)),
                  "text_exported": False,
                  "draft_ineffective": accepted == 0 and acceptance <= 1.01}
        if audit:
            record["logprobs"] = [{str(k): float(v.logprob) for k, v in row.items()}
                                  for row in final.outputs[0].logprobs]
        records.append(record)
        dump(out / "requests.json", records)
        print("DSPARK_TINY_REQUEST", json.dumps({k: v for k, v in record.items()
                                                if k not in ("token_ids", "receipts", "logprobs")}), flush=True)
        return record

    request(0, warmup=True)
    request(1, warmup=True)
    controls = []
    for seed in range(3):
        a, b = request(seed, audit=True), request(seed, audit=True)
        assert a["token_ids"] == b["token_ids"], "Native DSpark A/A token instability"
        assert len(a["logprobs"]) == len(b["logprobs"])
        max_delta = 0.0
        for left, right in zip(a["logprobs"], b["logprobs"]):
            assert set(left) == set(right), "Native top5 instability"
            assert all(math.isfinite(v) for v in (*left.values(), *right.values())), "Nonfinite native logprob"
            max_delta = max(max_delta, *(abs(left[k] - right[k]) for k in left))
        assert max_delta < 1e-3, max_delta
        controls.append({"seed": seed, "tokens_equal": True, "top5_sets_equal": True,
                         "max_logprob_delta": max_delta})
    dump(out / "native_AA.json", {"passed": True, "controls": controls,
                                  "complete_consumer_gate": False,
                                  "route_and_cache_value_audit_pending": True})
    perf = [request(seed % 3) for seed in range(args.requests)]
    total_decode_s = sum(r["decode_s"] for r in perf)
    total_decode_tokens = sum(r["decode_tokens"] for r in perf)
    total_rounds = sum(r["target_decode_rounds"] for r in perf)
    summary = {"formal_tp8_claim": False, "latest_formal_target_ms": args.target_ms,
               "dummy_weights": True, "includes_main_mtp_verify_scheduler_cache": True,
               "startup_s": startup_s,
               "requests": len(perf), "serial": True, "profiler": False,
               "native_AA_passed": True, "complete_accuracy_gate": False,
               "median_ms_per_step": statistics.median(r["ms_per_step"] for r in perf),
               "median_A": statistics.median(r["A"] for r in perf),
               "median_single_stream_tok_s": statistics.median(r["single_stream_tok_s"] for r in perf),
               "serial_pooled_actual_rounds": {"decode_s": total_decode_s, "decode_tokens": total_decode_tokens,
                                               "target_decode_rounds": total_rounds,
                                               "ms_per_step": 1000 * total_decode_s / total_rounds,
                                               "A": total_decode_tokens / total_rounds,
                                               "single_stream_tok_s": total_decode_tokens / total_decode_s},
               "ineffective_draft_requests": sum(r["draft_ineffective"] for r in perf),
               "empty_output_requests": sum(r["empty_output"] for r in perf),
               "token_suffix_repetition_requests": sum(r["token_suffix_repetition_period"] is not None for r in perf),
               "quality_clean_rate": None,
               "quality_reason": "Dummy weights; output quality/acceptance cannot qualify formal performance",
               "candidate_admitted": False}
    dump(out / "summary.json", summary)
    if args.profile:
        prof = request(0, profile=True)
        dump(out / "profile_worker_receipt.json", {"req_id": prof["core_request_id"],
             "manual_step_calls": 0, "actual_execute_driven": True,
             "warmup_actual_decode_rounds": 5, "active_actual_decode_rounds": 10,
             "schedule_actual_decode_rounds": prof["profile_schedule_decode_steps"],
             "target_decode_rounds": prof["target_decode_rounds"],
             "draft_calls": prof["draft_calls"], "verifier_rounds": prof["verifier_rounds"],
             "profiler_used_in_performance": False, "physical_device": 4,
             "formal_tp8_claim": False})
    print("DSPARK_TINY_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
