"""Fresh native eager-only draft diagnostic, without any draft graph capture."""
import argparse
import json
import os
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(args.output / "vllm_cache")
    from vllm import LLM, SamplingParams
    kwargs = dict(model=args.model, tokenizer=args.model, load_format="dummy", dtype="bfloat16",
                  worker_cls=("o_proj_cache_fixture_worker.NativeCacheFixtureWorker"
                              if os.environ.get("DSPARK_O_PROJ_CACHE_FIXTURE") == "1"
                              else "native_route_diagnostic_worker.NativeRouteDiagnosticWorker"),
                  tensor_parallel_size=1, enable_expert_parallel=True, seed=0, trust_remote_code=True,
                  async_scheduling=False, speculative_config={"method": "dspark", "num_speculative_tokens": 5,
                                                              "enforce_eager": True},
                  limit_mm_per_prompt={"image": 0}, max_model_len=2048, max_num_seqs=1,
                  max_num_batched_tokens=256, gpu_memory_utilization=.70,
                  kv_cache_memory_bytes=2*1024**3, block_size=128, enable_prefix_caching=False,
                  enable_return_routed_experts=False,
                  compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [6]},
                  additional_config={"enable_engram": False, "enable_cpu_binding": False,
                      "ascend_compilation_config": {"enable_npugraph_ex": True, "enable_static_kernel": False},
                      "multistream_dsv4_dsa_overlap": False})
    (args.output/"config.json").write_text(json.dumps({"llm":kwargs,"fresh_native_eager_only":True,
        "draft_capture":False,"formal_quality_claim":False,"latest_formal_target_ms":20},indent=2)+"\n")
    llm = LLM(**kwargs)
    import dspark_worker
    import native_route_diagnostic_worker
    import graph_precision_audit as audit
    engine = llm.llm_engine
    llm.collective_rpc(audit.switch,args=("off",True,"native-eager-only-diagnostic"))
    llm.collective_rpc(dspark_worker.reset_receipts)
    core = engine.add_request("native-eager-only",{"prompt_token_ids":[100+(i*17)%97 for i in range(128)]},
                             SamplingParams(temperature=0,max_tokens=32,ignore_eos=True,detokenize=False))
    try:
        while engine.has_unfinished_requests():
            outputs = engine.step()
            for item in outputs:
                if item.finished:
                    (args.output/"output.json").write_text(json.dumps({"core_id":core,"token_ids":item.outputs[0].token_ids,
                        "immediate_router_diagnostic_only":True,"full_consumer_quality_gate":False},indent=2)+"\n")
    finally:
        native_coverage = llm.collective_rpc(native_route_diagnostic_worker.coverage)[0]
        (args.output/"native_coverage.json").write_text(json.dumps(native_coverage,indent=2)+"\n")
        if os.environ.get("DSPARK_O_PROJ_CACHE_FIXTURE") == "1":
            import o_proj_cache_fixture
            fixture_receipt = llm.collective_rpc(o_proj_cache_fixture.validate_all,
                                                args=("native_after_real_request",))[0]
            (args.output/"fixed_fixture_validation.json").write_text(json.dumps(fixture_receipt,indent=2)+"\n")
    receipt = llm.collective_rpc(dspark_worker.read_receipts)[0]
    (args.output/"counter.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__ == "__main__":
    main()
