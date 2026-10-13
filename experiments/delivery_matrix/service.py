"""Native OpenAI server using an immutable best-source snapshot.

The frozen candidate is verified, then explicit deployment overrides are
recorded as a *new* candidate. No old quality receipt is inherited.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def physical_chips(value):
    """One explicit TP8 subset of this 16-chip A3; never auto-pick devices."""
    if not isinstance(value, str):
        raise ValueError("An explicit comma-separated physical-chip list is required")
    parts = [part.strip() for part in value.split(",")]
    if (len(parts) != 8 or any(re.fullmatch(r"(?:[0-9]|1[0-5])", part) is None for part in parts)
            or len(set(parts)) != 8):
        raise ValueError("Exactly eight distinct physical chips in 0..15 are required")
    return ",".join(parts)


def make_plan(candidate, args):
    kw = copy.deepcopy(candidate["model_kwargs"])
    if kw.get("load_format") == "dummy" or kw.get("tensor_parallel_size") != 8:
        raise ValueError("A real full-model TP8 candidate is required")
    spec = kw.get("speculative_config", {})
    if spec.get("method") != "dspark" or spec.get("num_speculative_tokens") != 5:
        raise ValueError("The verified DSpark5 candidate must be retained")
    env = copy.deepcopy(candidate["startup_env"])
    # A generic full40 TP8 server is never a leftover P or D half-server.
    for key in ("V41_CED_ROLE", "DELIVERY_CED_OVERLAY", "DELIVERY_CED_COMPRESSOR",
                "DELIVERY_CED_COMPRESSOR_MANIFEST_SHA256", "DELIVERY_CED_MANIFEST_SHA256",
                "DELIVERY_CED_OPP_FINGERPRINT", "DELIVERY_SHARED_ENGRAM",
                "DELIVERY_SHARED_ENGRAM_MANIFEST_SHA256", "V41_ENGRAM_SHARED_HOST_MMAP"):
        env.pop(key, None)
    selected_chips = getattr(args, "physical_chips", None)
    if selected_chips is not None:
        selected_chips = physical_chips(selected_chips)
        env.update(STACK_PHYSICAL_CHIPS=selected_chips, STACK_EXPECTED_CHIPS=selected_chips,
                   ASCEND_RT_VISIBLE_DEVICES=selected_chips, ASCEND_VISIBLE_DEVICES=selected_chips)
    component = getattr(args, "physical_worker", None)
    for key in ("DELIVERY_PHYSICAL_WORKER", "DELIVERY_PHYSICAL_WORKER_MANIFEST_SHA256"):
        env.pop(key, None)
    base_chips = candidate["startup_env"].get("STACK_PHYSICAL_CHIPS")
    if selected_chips is not None and base_chips is not None and selected_chips != base_chips and component is None:
        raise ValueError("Changing physical chips requires a separately pinned worker guard derivation")
    if component is not None:
        from .build_physical_worker import BASE_CLASS, CHIPS, sha, verify
        verify(component)
        if kw["worker_cls"] != BASE_CLASS or env.get("STACK_PHYSICAL_CHIPS") != CHIPS:
            raise ValueError("Physical worker derivation does not match the original class/selected chips")
        env.update(DELIVERY_PHYSICAL_WORKER=str(component),
                   DELIVERY_PHYSICAL_WORKER_MANIFEST_SHA256=sha(Path(component) / "manifest.json"))
    if getattr(args, "dcp_det_reduce", None) is not None and args.form != "a3-tp8-dcp8":
        raise ValueError("DCP software reduction is a separate DCP-only candidate option")
    env.update(
        STACK_DSPARK_FORMAL_OUTPUT=str(args.output / "native"),
        VLLM_CACHE_ROOT=str(args.output / "cache"),
        DELIVERY_BASE_WORKER=kw["worker_cls"],
        DELIVERY_FORM=args.form,
        V41_ENGRAM_ROUTE_PROBE="",
        DSPARK_API_FROZEN_COUNTER=str(args.source / "stack/parallel_tiny/dspark_general_quality_launcher_cpu/assets/frozen_v4_counter.py"),
    )
    kw.update(worker_cls="delivery_matrix.worker.DeliveryWorker", max_model_len=args.max_model_len,
              max_num_batched_tokens=args.batched_tokens, max_num_seqs=1,
              enable_prefix_caching=False)
    if args.device_index is not None:
        env["V41_ENGRAM_DEVICE_INDEX"] = str(args.device_index)
    if args.form == "a3-tp8-dcp8":
        kw["decode_context_parallel_size"] = 8
        env["V41_DCP"] = "1"
        if not os.environ.get("DELIVERY_DCP_OVERLAY"):
            raise ValueError("DCP8 requires an explicit pinned overlay")
        env["DELIVERY_DCP_OVERLAY"] = os.environ["DELIVERY_DCP_OVERLAY"]
        env["DELIVERY_DCP_DRAFT"] = os.environ["DELIVERY_DCP_DRAFT"]
        env["V41_DCP_REPLICATE_INDEXER"] = os.environ.get("V41_DCP_REPLICATE_INDEXER", "1")
        if getattr(args, "dcp_det_reduce", None) is not None:
            env["V41_DCP_DET_REDUCE"] = str(args.dcp_det_reduce)
        else:
            env["V41_DCP_DET_REDUCE"] = os.environ.get("V41_DCP_DET_REDUCE", env.get("V41_DCP_DET_REDUCE", "0"))
        if env["V41_DCP_DET_REDUCE"] == "1":
            from .build_dcp_stability import verify_candidate
            verify_candidate(env["DELIVERY_DCP_OVERLAY"])
    else:
        kw["decode_context_parallel_size"] = 1
        for key in ("V41_DCP", "DELIVERY_DCP_OVERLAY", "DELIVERY_DCP_DRAFT",
                    "V41_DCP_DET_REDUCE", "V41_DCP_REPLICATE_INDEXER"):
            env.pop(key, None)
    if not kw.get("additional_config", {}).get("enable_engram"):
        raise ValueError("Engram must remain enabled")
    if env.get("DRAFT_GRAPH") != "1" or env.get("DSPARK_GRAPH_CAPTURE_METADATA") != "1":
        raise ValueError("Draft graph and its real attention metadata are required")
    if env.get("VLLM_BATCH_INVARIANT") != "1" or not env.get("ASCEND_CUSTOM_OPP_PATH"):
        raise ValueError("The quality-critical batch-invariant OPP must be selected")
    return {"schema": "delivery_service_plan_v1", "form": args.form,
            "served_model": args.served_model, "source_root": str(args.source),
            "source_candidate_fingerprint": fingerprint(candidate),
            "model_kwargs": kw, "environment": env,
            "runtime_fingerprint": fingerprint({"model_kwargs": kw, "environment": env}),
            "actual_quality_passed": False, "actual_performance_observed": False,
            "simulated_hardware": args.form == "a2-tp8-a3sim"}


async def serve(args, candidate, plan):
    # Import only after the snapshot's supported run_stack/sitecustomize chain.
    from parallel_tiny.dspark_general_quality_api_cpu.contract import verify_source_files
    from parallel_tiny.dspark_api_counter_bridge_cpu.runtime_guards import load_frozen_counter
    verify_source_files(candidate)
    os.environ.update(plan["environment"])
    load_frozen_counter()
    from vllm.entrypoints.openai import api_server as official
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    parser = make_arg_parser(FlexibleArgumentParser())
    config = parser.parse_args([
        "--model", plan["model_kwargs"]["model"], "--served-model-name", args.served_model,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--tokenizer-mode", "deepseek_v41", "--reasoning-parser", "deepseek_v41",
        "--tool-call-parser", "deepseek_v41", "--enable-auto-tool-choice",
        "--default-chat-template-kwargs", json.dumps({"enable_thinking": False}),
    ])
    for key, value in plan["model_kwargs"].items():
        setattr(config, key, value)
    config.enable_log_requests = False
    # Native build_app installs these before materializing middleware_stack.
    # Adding middleware to its returned app can already be too late.
    config.middleware = list(plan.get("http_middleware", []))
    config.api_server_count = 1
    validate_parsed_serve_args(config)
    selection = {**plan, "service_instance_id": uuid.uuid4().hex, "pid": os.getpid(),
                 "engine_initialized": False, "ASGI_startup_completed": False}
    def save():
        (args.output / "service_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    save()
    _, sock = official.setup_server(config, reuse_port=False)
    try:
        async with official.build_async_engine_client(config) as engine:
            selection["engine_initialized"] = True
            tasks = await engine.get_supported_tasks()
            app = official.build_app(config, tasks, engine.model_config)
            await official.init_app_state(engine, app.state, config, tasks)
            descriptions = await engine.collective_rpc("delivery_describe")
            if len(descriptions) != 8 or {r["rank"] for r in descriptions} != set(range(8)):
                raise ValueError("All eight actual workers must describe the selected form")
            selection["rank_descriptions"] = descriptions
            await engine.collective_rpc("delivery_begin")
            save()

            @app.get("/delivery/identity")
            async def identity():
                return selection

            @app.get("/delivery/metrics")
            async def metrics():
                ranks = await engine.collective_rpc("delivery_metrics")
                return {"schema": "delivery_runtime_metrics_v1", "trusted": True,
                        "counter_scope": "actual runner scheduled/request CPU-token bookkeeping",
                        "service_instance_id": selection["service_instance_id"],
                        "form": args.form, "served_model": args.served_model, "ranks": ranks}

            @app.post("/delivery/arm")
            async def arm(payload: dict):
                tag, budget = payload.get("tag"), payload.get("max_tokens")
                if not isinstance(tag, str) or not tag or len(tag) > 160 or type(budget) is not int or budget < 2:
                    from fastapi import HTTPException
                    raise HTTPException(status_code=400, detail="An explicit serial request tag and output budget >=2 are required")
                ranks = await engine.collective_rpc("delivery_arm", args=(tag, budget))
                return {"schema": "delivery_counter_arm_v1", "service_instance_id": selection["service_instance_id"],
                        "tag": tag, "max_tokens": budget, "ranks": ranks}

            original = app.router.lifespan_context
            from contextlib import asynccontextmanager
            @asynccontextmanager
            async def lifespan(actual_app):
                async with original(actual_app):
                    selection["ASGI_startup_completed"] = True
                    save()
                    yield
                    selection["ASGI_startup_completed"] = False
                    save()
            app.router.lifespan_context = lifespan
            shutdown = await official.serve_http(
                app, sock=sock, host=config.host, port=config.port,
                enable_ssl_refresh=config.enable_ssl_refresh, log_level=config.uvicorn_log_level,
                access_log=not config.disable_uvicorn_access_log,
                timeout_keep_alive=official.envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=config.ssl_keyfile, ssl_certfile=config.ssl_certfile,
                ssl_ca_certs=config.ssl_ca_certs, ssl_cert_reqs=config.ssl_cert_reqs,
                ssl_ciphers=config.ssl_ciphers,
                h11_max_incomplete_event_size=config.h11_max_incomplete_event_size,
                h11_max_header_count=config.h11_max_header_count)
        await shutdown
    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--form", choices=["a2-tp8-a3sim", "a3-tp8-dcp8"], required=True)
    parser.add_argument("--max-model-len", type=int, default=196608)
    parser.add_argument("--batched-tokens", type=int, default=8192)
    parser.add_argument("--device-index", type=int, choices=[0, 1])
    parser.add_argument("--physical-chips", type=physical_chips,
                        help="Explicit eight comma-separated A3 die IDs; overrides are fingerprinted")
    parser.add_argument("--physical-worker", type=Path,
                        help="Pinned geometry-only derivation required when moving the original worker")
    parser.add_argument("--dcp-det-reduce", type=int, choices=[0, 1],
                        help="Optional new DCP candidate; requires a bounded, all-branch reduction overlay")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.source = args.source.resolve(strict=True)
    if args.physical_worker is not None:
        args.physical_worker = args.physical_worker.resolve(strict=True)
    args.output = args.output.resolve()
    candidate = json.loads((args.source / "candidate_contract.json").read_text())
    plan = make_plan(candidate, args)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "native").mkdir()
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.execute:
        asyncio.run(serve(args, candidate, plan))
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
