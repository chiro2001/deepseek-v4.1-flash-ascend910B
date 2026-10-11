"""Same-instance native A/A, ID-hoist accuracy, gated timing and real Cast profiles."""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["V41_IDS64_HOIST"] = "0"
faulthandler.enable()
faulthandler.dump_traceback_later(180, repeat=True)

import ids_patches as patches


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "4"
    assert os.environ["HCCL_DETERMINISTIC"] == "strict"
    assert args.pairs >= 6
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "runner.pid").write_text(str(os.getpid()) + "\n")
    os.environ["VLLM_CACHE_ROOT"] = str(out / "vllm_cache")
    import numpy as np
    import torch
    import torch_npu
    import vllm
    import vllm_ascend
    from vllm import LLM, SamplingParams
    kwargs = dict(model="/model", tokenizer="/model", load_format="dummy", dtype="bfloat16",
                  worker_cls="ids_worker.IdsWorker", tensor_parallel_size=1, enable_expert_parallel=True,
                  seed=0, trust_remote_code=True, async_scheduling=False,
                  limit_mm_per_prompt={"image": 0}, max_model_len=8192, max_num_seqs=1,
                  max_num_batched_tokens=2048, gpu_memory_utilization=.70,
                  kv_cache_memory_bytes=4 * 1024 ** 3, block_size=128, enable_prefix_caching=False,
                  enable_return_routed_experts=True,
                  compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1]},
                  additional_config={"enable_engram": False, "enable_cpu_binding": True,
                                     "ascend_compilation_config": {"enable_npugraph_ex": True,
                                                                   "enable_static_kernel": False},
                                     "multistream_dsv4_dsa_overlap": False})
    dump(out / "config.json", {"llm": kwargs, "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                               "vllm": vllm.__version__, "vllm_ascend": vllm_ascend.__file__,
                               "physical_chip": 4, "hccl_deterministic": os.getenv("HCCL_DETERMINISTIC"),
                               "engine_timing_scope": "synchronous engine.step wall-clock; profiler/audits OFF",
                               "baseline_source": "private copy from dsv41-tiny-prof-20261009-c4",
                               "same_model_instance": True, "formal_17ms_claim": False})
    print("IDS_INITIALIZING", flush=True)
    start = time.monotonic()
    llm = LLM(**kwargs)
    startup_s = time.monotonic() - start
    topology = llm.collective_rpc(patches.topology)[0]
    dump(out / "topology.json", topology)
    banks = [llm.collective_rpc(patches.save, args=("off",))[0]]
    dump(out / "banks.json", banks)
    engine, counter, records = llm.llm_engine, 0, []

    def request(arm, tag, seed=0, audit=True, profile=False):
        nonlocal counter
        llm.collective_rpc(patches.switch, args=(arm,))
        counter += 1
        prompt = [100 + (i * 17 + seed * 19) % 97 for i in range(2048)]
        max_tokens = 47 + seed % 2 if audit else 48
        engine.add_request(f"ids-{counter}", {"prompt_token_ids": prompt},
                           SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True,
                                          detokenize=False, logprobs=5 if audit else None))
        times, final, prof, profile_steps = [], None, None, 0
        while engine.has_unfinished_requests():
            if profile and len(times) == 9:
                profile_dir = out / "prof" / arm
                prof = torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=10, repeat=1),
                    record_shapes=True, with_stack=False, profile_memory=False,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                        export_type=torch_npu.profiler.ExportType.Text, data_simplification=False),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir), worker_name="ids_chip4"))
                prof.start()
            begin = time.perf_counter()
            outputs = engine.step()
            times.append((time.perf_counter() - begin) * 1000)
            if prof is not None:
                prof.step()
                profile_steps += 1
                if profile_steps == 10:
                    prof.stop()
                    prof = None
            for item in outputs:
                if item.finished:
                    final = item
        assert final is not None and prof is None
        output = final.outputs[0]
        assert len(output.token_ids) == max_tokens
        record = {"arm": arm, "tag": tag, "seed": seed, "step_ms": times,
                  "decode_median_ms": statistics.median(times[9:]), "prefill_ms": times[0],
                  "output_tokens": max_tokens, "engine_steps": len(times), "token_ids": output.token_ids,
                  "audit": audit, "profiler_enabled": profile, "profile_actual_decode_steps": profile_steps}
        if audit:
            routes = np.asarray(output.routed_experts)
            assert routes.shape == (2048 + max_tokens - 1, 40, 2), routes.shape
            assert routes.min() >= 0 and routes.max() < 8
            assert np.all(routes[..., 0] != routes[..., 1]), "Degenerate native routing"
            route_file = out / f"routes_{counter:03d}.npy"
            np.save(route_file, routes)
            record.update({"route_file": str(route_file), "route_shape": list(routes.shape),
                           "route_sha256": hashlib.sha256(routes.tobytes()).hexdigest(),
                           "route_unique_experts": np.unique(routes).tolist(),
                           "route_token_layer_positions": int(routes.shape[0] * routes.shape[1]),
                           "logprobs": [{str(k): float(v.logprob) for k, v in row.items()} for row in output.logprobs]})
        else:
            assert output.logprobs is None
            assert output.routed_experts is None, "Route export remains enabled during performance"
        records.append(record)
        dump(out / "requests.json", records)
        print("IDS_REQUEST", json.dumps({k: v for k, v in record.items()
                                          if k not in ("step_ms", "token_ids", "logprobs")}), flush=True)
        return record

    def compare(left, right, kind, pair):
        assert left["token_ids"] == right["token_ids"], (kind, pair, "tokens")
        assert left["route_sha256"] == right["route_sha256"], (kind, pair, "routes")
        delta = 0.
        for i, (a, b) in enumerate(zip(left["logprobs"], right["logprobs"])):
            assert set(a) == set(b), (kind, pair, i, "top5 keys")
            delta = max(delta, max(abs(a[k] - b[k]) for k in a))
        assert delta < 1e-3, (kind, pair, delta)
        return {"kind": kind, "pair": pair, "seed": left["seed"], "passed": True,
                "tokens_equal": True, "ordered_routes_equal": True, "top5_key_sets_equal": True,
                "max_logprob_delta": delta, "route_token_layer_positions": left["route_token_layer_positions"],
                "left": left["arm"], "right": right["arm"]}

    request("off", "warmup-0", 0)
    request("off", "warmup-1", 1)
    native = []
    for pair in range(3):
        a = request("off", f"native-{pair}-A", pair)
        b = request("off", f"native-{pair}-B", pair)
        native.append(compare(a, b, "native_AA", pair))
        dump(out / "native_AA.json", native)
    dump(out / "native_gate.json", {"passed": True, "comparisons": native})
    print("IDS_NATIVE_GATE_PASS", flush=True)
    for name in ("off_repeat", "on"):
        banks.append(llm.collective_rpc(patches.create, args=(name,))[0])
        dump(out / "banks.json", banks)
    capture_aa, candidates, witnesses = [], [], []
    for seed in range(3):
        a = request("off", f"capture-AA-{seed}-A", seed)
        b = request("off_repeat", f"capture-AA-{seed}-B", seed)
        capture_aa.append(compare(a, b, "postcapture_native_AA", seed))
        dump(out / "capture_AA.json", capture_aa)
    for pair in range(6):
        arms = ("off_repeat", "on") if pair % 2 == 0 else ("on", "off_repeat")
        results = {arm: request(arm, f"candidate-{pair}", pair % 3) for arm in arms}
        candidates.append(compare(results["off_repeat"], results["on"], "off_on", pair))
        dump(out / "candidate_comparisons.json", candidates)
    for arm in ("off", "off_repeat", "on"):
        request(arm, "integer-consumer-audit", 0)
        witnesses.append(llm.collective_rpc(patches.audit, args=(arm,))[0])
        dump(out / "consumer_witnesses.json", witnesses)
    gate = {"passed": True, "native_AA": native, "postcapture_native_AA": capture_aa,
            "off_on": candidates, "consumer_witnesses": witnesses,
            "same_model_instance": True, "formal_quality_claim": False}
    dump(out / "accuracy_gate.json", gate)
    print("IDS_ACCURACY_GATE_PASS", flush=True)
    disable = llm.collective_rpc(patches.disable_route_export)[0]
    # Single-process engine and worker share this ModelConfig; set both handles
    # explicitly and verify requests no longer export routed experts.
    engine.model_config.enable_return_routed_experts = False
    if hasattr(llm, "llm_engine") and hasattr(engine, "vllm_config"):
        engine.vllm_config.model_config.enable_return_routed_experts = False
    core = engine.engine_core.engine_core
    core.scheduler.enable_return_routed_experts = False
    disable["scheduler_route_export_enabled"] = core.scheduler.enable_return_routed_experts
    dump(out / "performance_audit_disabled.json", disable)
    for name in ("off_perf", "on_perf"):
        banks.append(llm.collective_rpc(patches.create, args=(name,))[0])
        dump(out / "banks.json", banks)
        request(name, "perf-warmup-0", 0, audit=False)
        request(name, "perf-warmup-1", 1, audit=False)
    timings = []
    for pair in range(args.pairs):
        arms = ("off_perf", "on_perf") if pair % 2 == 0 else ("on_perf", "off_perf")
        results = {arm: request(arm, f"perf-{pair}", pair % 3, audit=False) for arm in arms}
        a, b = results["off_perf"], results["on_perf"]
        assert a["token_ids"] == b["token_ids"]
        timings.append({"pair": pair, "seed": pair % 3, "order": list(arms),
                        "off_ms": a["decode_median_ms"], "on_ms": b["decode_median_ms"],
                        "delta_ms": a["decode_median_ms"] - b["decode_median_ms"],
                        "ratio": a["decode_median_ms"] / b["decode_median_ms"]})
        dump(out / "performance_pairs.json", timings)
    off_ms = statistics.median(row["off_ms"] for row in timings)
    on_ms = statistics.median(row["on_ms"] for row in timings)
    result = {"passed": True, "diagnostic_only": True, "same_model_instance": True,
              "physical_chip": 4, "topology": topology, "startup_s": startup_s,
              "accuracy_gate": "accuracy_gate.json", "timing_pairs": len(timings),
              "timing_profiler": "OFF", "timing_route_export": False, "timing_logprobs": False,
              "off": {"ms_per_step": off_ms, "A": 1., "tok_per_s": 1000 / off_ms},
              "on": {"ms_per_step": on_ms, "A": 1., "tok_per_s": 1000 / on_ms},
              "paired_delta_median_ms": statistics.median(row["delta_ms"] for row in timings),
              "paired_ratio_median": statistics.median(row["ratio"] for row in timings),
              "winning_pairs": sum(row["delta_ms"] > 0 for row in timings),
              "formal_17ms_claim": False, "formal_tp8_speedup_claim": False}
    dump(out / "result.json", result)
    if args.profile:
        profiles = []
        for arm in ("off_perf", "on_perf"):
            row = request(arm, "profile", 0, audit=False, profile=True)
            assert row["profile_actual_decode_steps"] == 10
            profiles.append({"arm": arm, "actual_decode_steps": 10, "physical_chip": 4,
                             "performance_quote_uses_this": False})
        dump(out / "profile_windows.json", profiles)
    print("IDS_COMPLETE", json.dumps(result), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
