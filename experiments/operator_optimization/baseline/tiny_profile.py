"""Isolated TP1 tiny replay, phase-separated hardware metrics, and wall baseline."""
import argparse
import faulthandler
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
os.environ.setdefault("V41_DUMMY_WO_A_FIX", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # trusted local collective_rpc callable
faulthandler.enable()
faulthandler.dump_traceback_later(180, repeat=True)


def configure_profiler(worker, metric, output, active):
    worker = getattr(worker, "worker", worker)
    import torch_npu
    from vllm.config import ProfilerConfig
    from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

    config = ProfilerConfig(profiler="torch", torch_profiler_dir=output,
                            torch_profiler_with_stack=False,
                            torch_profiler_record_shapes=True)
    class CounterProfiler(TorchNPUProfilerWrapper):
        @staticmethod
        def _create_profiler(config, trace_name):
            return torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=active, repeat=1),
                record_shapes=True, with_stack=False, profile_memory=False,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=getattr(torch_npu.profiler.AiCMetrics, metric),
                    export_type=torch_npu.profiler.ExportType.Text,
                    data_simplification=False),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output,
                                                                         worker_name="tiny_chip4"))
    wrapper = CounterProfiler(config, "tiny_chip4")
    worker.profiler_config = config
    worker.profiler = wrapper
    return {"metric": metric, "output": output, "worker_type": type(worker).__name__}


def advance_profiler(worker):
    worker = getattr(worker, "worker", worker)
    worker.profiler.profiler.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/work/results/full")
    p.add_argument("--prompt-len", type=int, default=2048)
    p.add_argument("--decode-steps", type=int, default=20)
    p.add_argument("--metrics", default="PipeUtilization,ArithmeticUtilization,Memory,MemoryL0,MemoryUB,L2Cache,ResourceConflictRatio")
    p.add_argument("--eager", action="store_true")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    from vllm import LLM, SamplingParams
    import torch
    import torch_npu
    import vllm
    import vllm_ascend

    kwargs = dict(model="/model", tokenizer="/model", load_format="dummy",
                  dtype="bfloat16", tensor_parallel_size=1,
                  enable_expert_parallel=True, seed=0, trust_remote_code=True,
                  async_scheduling=False, limit_mm_per_prompt={"image": 0},
                  max_model_len=8192, max_num_seqs=1, max_num_batched_tokens=2048,
                  gpu_memory_utilization=0.70,
                  kv_cache_memory_bytes=4 * 1024**3, block_size=128,
                  enable_prefix_caching=False, enforce_eager=args.eager,
                  profiler_config={"profiler": "torch", "torch_profiler_dir": str(out / "prof"),
                                   "torch_profiler_with_stack": False},
                  compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                                      "cudagraph_capture_sizes": [1]},
                  additional_config={"enable_engram": False,
                                     "enable_cpu_binding": True,
                                     "ascend_compilation_config": {
                                         "enable_npugraph_ex": True,
                                         "enable_static_kernel": False},
                                     "multistream_dsv4_dsa_overlap": False})
    (out / "config.json").write_text(json.dumps({"llm": kwargs, "args": vars(args),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "vllm": vllm.__version__, "vllm_ascend": vllm_ascend.__file__,
        "physical_chip": int(os.environ.get("ASCEND_VISIBLE_DEVICES", "3")),
        "dummy_weights": True, "speculation": False}, indent=2))
    print("INITIALIZING", flush=True)
    llm = LLM(**kwargs)
    engine = llm.llm_engine
    counter = 0
    records = []

    def run_request(tag, metric=None, phase=None):
        nonlocal counter
        counter += 1
        rid = f"tiny-{counter}"
        params = SamplingParams(temperature=0, max_tokens=args.decode_steps + 14,
                                ignore_eos=True, detokenize=False)
        prompt = {"prompt_token_ids": [100 + (i % 97) for i in range(args.prompt_len)]}
        engine.add_request(rid, prompt, params)
        timings = []
        started = False
        stopped = False
        result = None
        step = 0
        while engine.has_unfinished_requests():
            should_start = metric and ((phase == "prefill" and step == 0)
                                      or (phase == "decode" and step == 9))
            if should_start:
                target = str(out / "prof" / metric / phase)
                active = 1 if phase == "prefill" else args.decode_steps
                llm.collective_rpc(configure_profiler, args=(metric, target, active))
                llm.start_profile()
                started = True
            begin = time.perf_counter()
            outputs = engine.step()
            timings.append((time.perf_counter() - begin) * 1000)
            if started and not stopped:
                llm.collective_rpc(advance_profiler)
            for r in outputs:
                if r.finished:
                    result = r
            step += 1
            should_stop = started and not stopped and (
                (phase == "prefill" and step == 1)
                or (phase == "decode" and step == 9 + args.decode_steps))
            if should_stop:
                llm.stop_profile()
                stopped = True
        if started and not stopped:
            llm.stop_profile()
            raise RuntimeError("Request ended before the requested profile window")
        assert result is not None and len(result.outputs[0].token_ids) == params.max_tokens
        record = {"tag": tag, "metric": metric, "phase": phase, "engine_steps": step,
                  "output_tokens": len(result.outputs[0].token_ids), "step_ms": timings,
                  "prefill_ms": timings[0], "decode_median_ms": statistics.median(timings[9:]),
                  "profile_steps": (1 if phase == "prefill" else args.decode_steps) if metric else 0,
                  "token_ids": result.outputs[0].token_ids}
        records.append(record)
        (out / "requests.json").write_text(json.dumps(records, indent=2))
        print("REQUEST", json.dumps({k: v for k, v in record.items()
                                      if k not in ("step_ms", "token_ids")}), flush=True)

    run_request("warmup-1")
    run_request("warmup-2")
    for i in range(3):
        run_request(f"baseline-{i}")
    baselines = [r for r in records if r["tag"].startswith("baseline")]
    ms = statistics.median([r["decode_median_ms"] for r in baselines])
    baseline = {"decode_ms_per_step": ms, "A": 1.0, "decode_tokens_per_second": 1000/ms,
                "prefill_median_ms": statistics.median([r["prefill_ms"] for r in baselines]),
                "repeats": 3, "dummy_weight_semantic_accuracy": "not applicable"}
    (out / "baseline.json").write_text(json.dumps(baseline, indent=2))
    print("BASELINE", json.dumps(baseline), flush=True)
    for metric in filter(None, args.metrics.split(",")):
        for phase in ("prefill", "decode"):
            run_request(f"{metric}-{phase}", metric, phase)
    print("COMPLETE", flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
