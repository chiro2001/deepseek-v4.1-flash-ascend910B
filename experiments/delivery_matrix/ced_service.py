"""Separate native P/D role servers based on the verified best model stack."""
import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path

from .service import fingerprint, serve


def make_ced_plan(candidate, args):
    if args.role not in ("prefill", "decode"):
        raise ValueError("An explicit CED P or D role is required")
    kw = copy.deepcopy(candidate["model_kwargs"])
    env = copy.deepcopy(candidate["startup_env"])
    prefill = args.role == "prefill"
    physical = "0,1,2,3,4,5,6,7" if prefill else "8,9,10,11,12,13,14,15"
    if kw["tensor_parallel_size"] != 8 or kw.get("load_format") == "dummy":
        raise ValueError("Full real TP8 CED weights are required")
    spec = kw.get("speculative_config") or {}
    if spec.get("method") != "dspark" or spec.get("num_speculative_tokens") != 5:
        raise ValueError("The original best DSpark5 definition must be retained for D")
    env.update(STACK_DSPARK_FORMAL_OUTPUT=str(args.output / "native"),
               VLLM_CACHE_ROOT=str(args.output / "cache"),
               DELIVERY_BASE_WORKER=kw["worker_cls"], DELIVERY_FORM="a3-ced-pd",
               V41_CED_ROLE=args.role, V41_CED_ALLOW_DSPARK="1", VLLM_ADMISSION_GATE="1",
               STACK_PHYSICAL_CHIPS=physical, STACK_EXPECTED_CHIPS=physical,
               ASCEND_RT_VISIBLE_DEVICES=physical, ASCEND_VISIBLE_DEVICES=physical,
               V41_ENGRAM_DEVICE_INDEX=str(args.device_index), V41_ENGRAM_ROUTE_PROBE="",
               DELIVERY_CED_OVERLAY=str(args.overlay),
               DELIVERY_CED_MANIFEST_SHA256=getattr(args, "overlay_manifest_sha256", ""),
               DELIVERY_CED_OPP_FINGERPRINT=(getattr(args, "native_opp_receipt", None) or {}).get("artifact_fingerprint", ""),
               DSPARK_API_FROZEN_COUNTER=str(args.source / "stack/parallel_tiny/dspark_general_quality_launcher_cpu/assets/frozen_v4_counter.py"))
    for key in ("V41_DCP", "DELIVERY_DCP_OVERLAY", "DELIVERY_DCP_DRAFT"):
        env.pop(key, None)
    compressor = getattr(args, "compressor_overlay", None)
    for key in ("DELIVERY_CED_COMPRESSOR", "DELIVERY_CED_COMPRESSOR_MANIFEST_SHA256"):
        env.pop(key, None)
    if compressor is not None:
        if prefill:
            raise ValueError("The compressor safety candidate is decode-only")
        from .build_ced_compressor import verify, sha
        verify(compressor)
        env.update(DELIVERY_CED_COMPRESSOR=str(compressor),
                   DELIVERY_CED_COMPRESSOR_MANIFEST_SHA256=sha(Path(compressor) / "manifest.json"))
    for key in ("DELIVERY_SHARED_ENGRAM", "DELIVERY_SHARED_ENGRAM_MANIFEST_SHA256", "V41_ENGRAM_SHARED_HOST_MMAP"):
        env.pop(key, None)
    shared = getattr(args, "shared_engram_overlay", None)
    if shared is not None:
        from .shared_engram_contract import MANIFEST_SHA, verify_component
        verify_component(shared)
        if (args.device_index != 0 or env.get("V41_ENGRAM_HOST_RESIDENT") != "1"
                or kw.get("additional_config", {}).get("engram_storage") != "int8"):
            raise ValueError("Shared Engram requires explicit host-resident INT8 and device index 0")
        env.update(DELIVERY_SHARED_ENGRAM=str(shared), DELIVERY_SHARED_ENGRAM_MANIFEST_SHA256=MANIFEST_SHA,
                   V41_ENGRAM_SHARED_HOST_MMAP="1")
    kw.update(max_model_len=args.max_model_len, max_num_batched_tokens=8192,
              max_num_seqs=1, enable_prefix_caching=False,
              kv_cache_memory_bytes=args.kv_bytes, decode_context_parallel_size=1)
    if prefill:
        kw.update(worker_cls="delivery_matrix.ced_worker.PrefillDeliveryWorker",
                  speculative_config=None)
        kw["compilation_config"]["cudagraph_capture_sizes"] = [1]
        for key in ("DRAFT_GRAPH", "DSPARK_DRAFT_USE_CUDAGRAPH", "DSPARK_GRAPH_CAPTURE_METADATA"):
            env[key] = "0"
    else:
        if any(env.get(key) != "1" for key in ("DRAFT_GRAPH", "DSPARK_GRAPH_CAPTURE_METADATA")):
            raise ValueError("CED D requires real draft graphs and their attention metadata")
        kw["worker_cls"] = "delivery_matrix.worker.DeliveryWorker"
        env.update(V41_CED_GRAPH_PROMPT_TAIL_EAGER="1", V41_CED_SWA_CLIP="1")
        if getattr(args, "decode_multistream", "best") == "off":
            kw["additional_config"].update(multistream_dsv4_dsa_overlap=False,
                                           multistream_overlap_shared_expert=False)
    # Retain BI, async and both supported best-stream options initially.
    # Any role-specific correctness fallback must become a different receipt.
    kw["kv_transfer_config"] = {
        "kv_connector": "MooncakeHybridConnector", "kv_role": "kv_producer" if prefill else "kv_consumer",
        "kv_port": args.kv_port,
        "kv_connector_extra_config": {"prefill": {"dp_size": 1, "tp_size": 8},
                                      "decode": {"dp_size": 1, "tp_size": 8}},
    }
    if (not kw.get("additional_config", {}).get("enable_engram")
            or env.get("VLLM_BATCH_INVARIANT") != "1" or not env.get("ASCEND_CUSTOM_OPP_PATH")):
        raise ValueError("Both roles must keep real Engram and the quality-critical BI operators")
    middleware = [] if prefill else ["delivery_matrix.ced_decode_boundary.ced_decode_guard"]
    return {"schema": "delivery_ced_service_plan_v1", "form": "a3-ced-pd", "role": args.role,
            "served_model": args.served_model, "source_root": str(args.source),
            "source_candidate_fingerprint": fingerprint(candidate),
            "ced_overlay_manifest_sha256": env["DELIVERY_CED_MANIFEST_SHA256"],
            "ced_compressor_manifest_sha256": env.get("DELIVERY_CED_COMPRESSOR_MANIFEST_SHA256"),
            "native_opp_receipt": getattr(args, "native_opp_receipt", None),
            "http_middleware": middleware,
            "model_kwargs": kw, "environment": env,
            "runtime_fingerprint": fingerprint({"model_kwargs": kw, "environment": env,
                                                "http_middleware": middleware}),
            "actual_quality_passed": False, "actual_performance_observed": False,
            "simulated_hardware": False}


async def serve_ced(args, candidate, plan):
    # The core and attention role overlays were installed before imports by
    # ced_bootstrap. Add the HTTP D boundary before accepting any generation.
    from vllm.entrypoints.openai import cli_args
    from vllm.config import KVTransferConfig
    original_validate = cli_args.validate_parsed_serve_args
    def validate(config):
        if isinstance(config.kv_transfer_config, dict):
            config.kv_transfer_config = KVTransferConfig(**config.kv_transfer_config)
        return original_validate(config)
    cli_args.validate_parsed_serve_args = validate
    try:
        await serve(args, candidate, plan)
    finally:
        cli_args.validate_parsed_serve_args = original_validate


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--role", choices=["prefill", "decode"], required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--kv-port", required=True, type=int)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--max-model-len", type=int, default=1048576)
    parser.add_argument("--kv-bytes", type=int, default=15728022528)
    parser.add_argument("--device-index", choices=[0, 1], type=int, default=0)
    parser.add_argument("--decode-multistream", choices=["best", "off"], default="best",
                        help="A correctness fallback changes the candidate fingerprint; no quality result is inherited")
    parser.add_argument("--compressor-overlay", type=Path,
                        help="D-only four-address residual candidate, selected before any SDK import")
    parser.add_argument("--shared-engram-overlay", type=Path,
                        help="Independent post-baseline candidate: P/D readonly MAP_SHARED INT8 tables")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.source, args.overlay = args.source.resolve(strict=True), args.overlay.resolve(strict=True)
    args.overlay_manifest_sha256 = hashlib.sha256((args.overlay / "manifest.json").read_bytes()).hexdigest()
    args.output = args.output.resolve()
    args.form = "a3-ced-pd"
    if os.environ.get("V41_CED_ROLE") != args.role:
        raise ValueError("CED role must be selected before any native model import")
    if args.compressor_overlay is not None:
        args.compressor_overlay = args.compressor_overlay.resolve(strict=True)
    selected_compressor = os.environ.get("DELIVERY_CED_COMPRESSOR")
    if (selected_compressor is not None) != (args.compressor_overlay is not None) or (
            selected_compressor is not None and Path(selected_compressor).resolve(strict=True) != args.compressor_overlay):
        raise ValueError("CLI compressor candidate differs from the pre-import environment selection")
    if args.shared_engram_overlay is not None:
        args.shared_engram_overlay = args.shared_engram_overlay.resolve(strict=True)
    selected_shared = os.environ.get("DELIVERY_SHARED_ENGRAM")
    if (selected_shared is not None) != (args.shared_engram_overlay is not None) or (
            selected_shared is not None and Path(selected_shared).resolve(strict=True) != args.shared_engram_overlay):
        raise ValueError("CLI shared Engram differs from the pre-import environment selection")
    candidate = json.loads((args.source / "candidate_contract.json").read_text())
    if args.execute:
        from .ced_native_contract import inspect_opp
        args.native_opp_receipt = inspect_opp(candidate["startup_env"]["ASCEND_CUSTOM_OPP_PATH"])
    plan = make_ced_plan(candidate, args)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "native").mkdir()
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.execute:
        asyncio.run(serve_ced(args, candidate, plan))
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
