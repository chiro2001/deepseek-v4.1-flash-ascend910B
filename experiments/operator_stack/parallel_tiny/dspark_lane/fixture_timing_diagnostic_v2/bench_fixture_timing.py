"""128/32 same-instance OFF/ON timing diagnostic; dummy results are never admitted."""
import argparse
import json
import os
from pathlib import Path
import time

from timing_controls import pooled_triplet, triplet, validate_geometry

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=8)
    args = parser.parse_args()
    assert args.pairs >= 8
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == os.environ["ASCEND_VISIBLE_DEVICES"] == "4"
    assert os.environ["HCCL_DETERMINISTIC"] == "strict"
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(out / "vllm_cache")
    reference = json.loads((Path(__file__).parent / "reference_token_ids.json").read_text())
    import torch
    import torch_npu
    from vllm import LLM, SamplingParams
    assert torch.npu.device_count() == 1
    kwargs = dict(model=args.model, tokenizer=args.model, load_format="dummy", dtype="bfloat16",
                  worker_cls="fixture_timing_worker.FixtureTimingWorker", tensor_parallel_size=1,
                  enable_expert_parallel=True, seed=0, trust_remote_code=True, async_scheduling=False,
                  speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": False},
                  limit_mm_per_prompt={"image": 0}, max_model_len=2048, max_num_seqs=1,
                  max_num_batched_tokens=256, gpu_memory_utilization=.70,
                  kv_cache_memory_bytes=2 * 1024**3, block_size=128, enable_prefix_caching=False,
                  enable_return_routed_experts=False,
                  compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [6]},
                  additional_config={"enable_engram": False, "enable_cpu_binding": False,
                                     "ascend_compilation_config": {"enable_npugraph_ex": True,
                                                                   "enable_static_kernel": False},
                                     "multistream_dsv4_dsa_overlap": False})
    dump(out / "config.json", {"llm": kwargs, "input_tokens": 128, "output_tokens": 32,
                               "pairs": args.pairs, "same_instance": True, "dummy_weights": True,
                               "route_logprob_audit": False, "profiler": False,
                               "diagnostic_only": True, "tiny_candidate_admitted": False,
                               "formal_candidate_admitted": False, "formal_tp8_claim": False})
    llm = LLM(**kwargs)
    import fixture_timing_worker as worker
    import dspark_worker
    dump(out / "topology.json", llm.collective_rpc(dspark_worker.topology)[0])
    dump(out / "startup.json", llm.collective_rpc(worker.metadata)[0])
    engine, records, sequence = llm.llm_engine, [], 0

    def request(arm, seed, tag, measured):
        nonlocal sequence
        sequence += 1
        rid = f"fixture-timing-{sequence:03d}"
        before = llm.collective_rpc(worker.begin_request, args=(arm, tag))[0]
        dump(out / f"fixture_before_{sequence:03d}.json", before)
        prompt = [100 + (index * 17 + seed * 19) % 97 for index in range(128)]
        core_id = engine.add_request(rid, {"prompt_token_ids": prompt}, SamplingParams(
            temperature=0, max_tokens=32, ignore_eos=True, detokenize=False, logprobs=None))
        assert engine.output_processor.external_req_ids[rid] == [core_id]
        final = None
        while engine.has_unfinished_requests():
            for item in engine.step():
                if item.finished:
                    final = item
        client_done = time.perf_counter()
        raw = llm.collective_rpc(worker.finish_request)[0]
        assert raw["counter"]["pid"] == os.getpid() and final is not None and final.request_id == rid
        assert final.outputs[0].logprobs is None
        tokens = list(final.outputs[0].token_ids)
        geometry = validate_geometry(raw, tokens, arm, core_id, reference[str(seed)])
        first = raw["counter"]["rounds"][0]
        begin, end = first["bookkeeping_done_s"], max(client_done, raw["counter"]["terminal_device_done_s"])
        assert begin <= first["draft_propose_begin_s"] < first["draft_propose_done_s"] <= end
        assert raw["counter"]["rounds"][-1]["draft_propose_done_s"] <= end
        decode_s = end - begin
        record = {"tag": tag, "arm": arm, "seed": seed, "measured": measured,
                  "external_id": rid, "core_id": core_id, "token_ids": tokens, "geometry": geometry,
                  "decode_window_begin_s": begin, "decode_window_end_s": end, "decode_s": decode_s,
                  **triplet(decode_s, geometry["target_decode_rounds"], geometry["decode_tokens"]),
                  "initial_and_terminal_draft_included_in_window": True,
                  "before_and_after_full43_fixture_seals_passed": before["passed"] and raw["fixture_after"]["passed"],
                  "diagnostic_device_reads_or_hash_calls_in_window": raw["window_guard"]["diagnostic_device_reads_or_hash_calls_in_window"],
                  "other_native_explicit_fences": raw["window_guard"]["other_native_explicit_fences"],
                  "route_logprob_audit": False, "profiler": False, "diagnostic_only": True,
                  "tiny_candidate_admitted": False, "formal_candidate_admitted": False, "formal_tp8_claim": False}
        dump(out / f"raw_{sequence:03d}.json", {"external_id": rid, "core_id": core_id,
             "arm": arm, "token_ids": tokens, **raw})
        dump(out / f"fixture_after_{sequence:03d}.json", raw["fixture_after"])
        records.append(record)
        dump(out / "requests.json", records)
        print("FIXTURE_TIMING_REQUEST", json.dumps({key: value for key, value in record.items()
                                                    if key != "token_ids"}), flush=True)
        return record

    request("off", 0, "warmup-off", measured=False)
    request("on", 1, "warmup-on", measured=False)
    pairs = []
    for pair in range(args.pairs):
        order = ("off", "on") if pair % 2 == 0 else ("on", "off")
        values = {arm: request(arm, pair % 3, f"pair-{pair}-{arm}", measured=True) for arm in order}
        assert values["off"]["token_ids"] == values["on"]["token_ids"]
        pairs.append({"pair": pair, "order": list(order), "off_tag": values["off"]["tag"],
                      "on_tag": values["on"]["tag"], "token_ids_equal": True,
                      "off_triplet": {key: values["off"][key] for key in ("ms_per_step", "A", "single_stream_tok_s")},
                      "on_triplet": {key: values["on"][key] for key in ("ms_per_step", "A", "single_stream_tok_s")}})
        dump(out / "pairs.json", pairs)
    measured = [row for row in records if row["measured"]]
    summary = {"diagnostic_only": True, "dummy_weights": True, "requests": len(records),
               "measured_requests": len(measured), "pairs": len(pairs), "geometry": {"input": 128, "output": 32},
               "off": pooled_triplet([row for row in measured if row["arm"] == "off"]),
               "on": pooled_triplet([row for row in measured if row["arm"] == "on"]),
               "all_pair_token_ids_equal": True, "before_after_full43_fixture_seals_passed": True,
               "diagnostic_device_reads_or_hash_calls_in_window": 0,
               "period2_repetition_requests": sum(row["geometry"]["period2_repetition"] for row in records),
               "dummy_clean_rate": 0.0, "tiny_candidate_admitted": False,
               "formal_candidate_admitted": False, "formal_tp8_claim": False,
               "formal_full_main_mtp_verify_scheduler_cache_20ms_goal_achieved": False,
               "acceptance_from_dummy_repetition_must_not_be_used_as_benefit": True,
               "expanded_2K_256_geometry_cache_gate_completed": False}
    dump(out / "summary.json", summary)
    print("FIXTURE_TIMING_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
