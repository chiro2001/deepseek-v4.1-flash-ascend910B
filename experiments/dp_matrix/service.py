"""New native-DP OpenAI frontend; native model/SDK snapshots remain immutable."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import uuid

from .build_runtime import verify
from .launch_plan import make_plan
from .native_rpc import NativeDPRPC
from .metrics_contract import validate_metrics, validate_status
from .topology import Topology, integer


def verify_native_files(guards):
    required = {"core_client.py", "input_processor.py", "serving.py"}
    if isinstance(guards, dict) and guards.get("schema") == "native_source_guard_v1":
        sources = guards.get("sources", {})
        if not required.issubset(sources):
            raise ValueError("All three actual native DP/RPC/routing sources are required")
        guards = {record["path"]: record["sha256"] for record in sources.values()}
    if not isinstance(guards, dict) or not guards:
        raise ValueError("Explicit preflight native source guards are required")
    if not required.issubset({Path(path).name for path in guards if isinstance(path, str)}):
        raise ValueError("Native guards must cover core-client, input-processor and routing sources")
    for path, expected in guards.items():
        if (not isinstance(path, str) or not Path(path).is_absolute()
                or not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
            raise ValueError("Native guard must bind an absolute path to an exact SHA-256")
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Native source changed since preflight: " + path)
    return guards


async def serve(args, candidate, plan):
    from parallel_tiny.dspark_general_quality_api_cpu.contract import verify_source_files
    verify_source_files(candidate)
    verify(args.runtime)
    verify_native_files(plan["native_source_guards"])
    if args.layout == "dp4-tp2-dcp2":
        from .build_dcp2 import verify as verify_dcp2, verify_draft
        verify_dcp2(args.dcp2_overlay)
        verify_draft(args.dcp_draft)
    os.environ.update(plan["environment"])
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
    config.middleware = ["dp_matrix.http_routes.DPRouteMiddleware"]
    config.api_server_count = 1
    validate_parsed_serve_args(config)
    selection = {**plan, "service_instance_id": uuid.uuid4().hex, "pid": os.getpid(),
                 "engine_initialized": False, "ASGI_startup_completed": False}

    def save():
        (args.output / "service_selection.json").write_text(json.dumps(selection, indent=2) + "\n")

    save()
    _, sock = official.setup_server(config, reuse_port=False)
    topology = Topology.named(args.layout)
    try:
        async with official.build_async_engine_client(config) as engine:
            selection["engine_initialized"] = True
            rpc = NativeDPRPC(engine.engine_core, topology)
            actual_client_file = str(Path(inspect.getfile(type(engine.engine_core))).resolve())
            if actual_client_file not in plan["native_source_guards"]:
                raise ValueError("Actual native DP client module is not the preflight-guarded source")
            descriptions = [row for group in await rpc.all("dp_describe") for row in group]
            topology.validate_workers(descriptions)
            selection["rank_descriptions"] = descriptions
            selection["native_dp_client"] = {
                "class": type(engine.engine_core).__module__ + "." + type(engine.engine_core).__name__,
                "engine_ranks_managed": engine.engine_core.engine_ranks_managed,
                "core_engines_hex": [value.hex() for value in engine.engine_core.core_engines],
                "per_engine_utility_source_sha256": hashlib.sha256(
                    inspect.getsource(engine.engine_core._call_utility_async).encode()).hexdigest(),
            }
            await rpc.all("delivery_begin")
            selection["capacity_descriptions"] = [row for group in await rpc.all("dp_capacity") for row in group]
            save()
            tasks = await engine.get_supported_tasks()
            app = official.build_app(config, tasks, engine.model_config)
            await official.init_app_state(engine, app.state, config, tasks)
            from fastapi import HTTPException
            from .http_routes import WINDOWS
            locks = [asyncio.Lock() for _ in range(topology.dp)]
            used_tags = set()

            def rank(value):
                try:
                    integer(value, "dp_rank")
                    if value >= topology.dp:
                        raise ValueError("Foreign DP rank")
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                return value

            @app.get("/dp/identity")
            async def identity():
                return selection

            @app.get("/dp/status")
            async def status():
                return {"service_instance_id": selection["service_instance_id"],
                        "replicas": await rpc.all("dp_status")}

            @app.post("/dp/arm")
            async def arm(payload: dict):
                dp_rank = rank(payload.get("dp_rank"))
                tag, budget = payload.get("tag"), payload.get("max_tokens")
                if (not isinstance(tag, str) or not tag or len(tag) > 160
                        or type(budget) is not int or budget < 2):
                    raise HTTPException(status_code=400, detail="An explicit tag and budget >=2 are required")
                async with locks[dp_rank]:
                    if dp_rank in WINDOWS:
                        raise HTTPException(status_code=409, detail="Previous DP window remains active")
                    if tag in used_tags:
                        raise HTTPException(status_code=409, detail="Measurement tags cannot be reused in this service")
                    # Reserve before the awaited RPC, including failed attempts.
                    used_tags.add(tag)
                    replies = await rpc.call(dp_rank, "delivery_arm", args=(tag, budget))
                    WINDOWS[dp_rank] = {"tag": tag, "budget": budget, "generation_started": False}
                    return {"service_instance_id": selection["service_instance_id"],
                            "dp_rank": dp_rank, "tag": tag, "ranks": replies}

            @app.get("/dp/metrics")
            async def metrics(dp_rank: int, tag: str):
                dp_rank = rank(dp_rank)
                async with locks[dp_rank]:
                    window = WINDOWS.get(dp_rank)
                    if (window is None or window["tag"] != tag
                            or window.get("generation_complete") is not True
                            or window.get("generation_http_status") != 200):
                        raise HTTPException(status_code=409, detail="No matching actual DP generation window")
                    actual_id = validate_status(await rpc.call(dp_rank, "dp_status"), topology,
                                                dp_rank=dp_rank, tag=tag)
                    replies = await rpc.call(dp_rank, "delivery_metrics")
                    binding = validate_metrics(replies, topology, dp_rank=dp_rank,
                                               tag=tag, request_id=actual_id)
                    WINDOWS.pop(dp_rank)
                    return {"schema": "dp_request_metrics_v1", "trusted": True,
                            "service_instance_id": selection["service_instance_id"],
                            "runtime_fingerprint": selection["runtime_fingerprint"],
                            "dp_rank": dp_rank, "ranks": replies, "request_binding": binding,
                            "completion_scope": "active-DP TP completion; not global EP drain"}

            original = app.router.lifespan_context

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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--native-source-guards", type=Path, required=True)
    parser.add_argument("--layout", choices=["dp2-tp4", "dp4-tp2", "dp4-tp2-dcp2"], required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--loading-threads", type=int)
    parser.add_argument("--host-numa-nodes")
    parser.add_argument("--dcp2-overlay", type=Path)
    parser.add_argument("--dcp-draft", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    from .host_memory import ENV as NUMA_ENV, verify_interleave
    if os.environ.get(NUMA_ENV) != args.host_numa_nodes:
        raise ValueError("NUMA launch environment and service CLI differ")
    host_policy = verify_interleave(args.host_numa_nodes)
    candidate = json.loads((args.source / "candidate_contract.json").read_text())
    dcp_options = {}
    if args.layout == "dp4-tp2-dcp2":
        if args.dcp2_overlay is None or args.dcp_draft is None:
            parser.error("DCP2 requires both reviewed overlay and draft paths")
        from .build_dcp2 import verify as verify_dcp2, verify_draft
        verify_dcp2(args.dcp2_overlay)
        verify_draft(args.dcp_draft)
        dcp_options = {
            "dcp_overlay": str(args.dcp2_overlay.resolve(strict=True)),
            "dcp_draft": str(args.dcp_draft.resolve(strict=True)),
            "dcp2_overlay_manifest_sha256": hashlib.sha256((args.dcp2_overlay / "manifest.json").read_bytes()).hexdigest(),
            "dcp_draft_manifest_sha256": hashlib.sha256((args.dcp_draft / "manifest.json").read_bytes()).hexdigest(),
        }
    elif args.dcp2_overlay is not None or args.dcp_draft is not None:
        parser.error("DCP overlays are admitted only for the DCP2 layout")
    plan = make_plan(candidate, layout=args.layout, source=args.source, output=args.output,
                     served_model=args.served_model, max_model_len=args.max_model_len,
                     gpu_memory_utilization=args.gpu_memory_utilization,
                     loading_threads=args.loading_threads,
                     host_numa_nodes=args.host_numa_nodes, **dcp_options)
    plan["host_memory_policy_receipt"] = host_policy
    verify(args.runtime)
    guards = json.loads(args.native_source_guards.read_text())
    guards = verify_native_files(guards)
    plan["native_source_guards"] = guards
    plan["environment"]["DP_MATRIX_RUNTIME"] = str(args.runtime.resolve(strict=True))
    if args.layout == "dp4-tp2-dcp2":
        bootstrap = Path(os.environ["DP_MATRIX_DCP2_BOOTSTRAP"]).resolve(strict=True)
        expected = Path(__file__).resolve().parent / "dcp2_bootstrap"
        if bootstrap != expected or (bootstrap / "sitecustomize.py").is_symlink():
            raise ValueError("DCP2 bootstrap must be the reviewed file in this exact DP package")
        plan["environment"].update(
            DP_MATRIX_DCP2_BOOTSTRAP=str(bootstrap),
            DP_MATRIX_DCP2_BOOTSTRAP_SHA256=hashlib.sha256((bootstrap / "sitecustomize.py").read_bytes()).hexdigest())
    if __package__.startswith("experiments."):
        from experiments.delivery_matrix.service import fingerprint
    else:
        from delivery_matrix.service import fingerprint
    plan["runtime_fingerprint"] = fingerprint({
        "model_kwargs": plan["model_kwargs"], "environment": plan["environment"],
        "native_source_guards": guards,
        "runtime_derivation_manifest": hashlib.sha256((args.runtime / "manifest.json").read_bytes()).hexdigest()})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "native").mkdir()
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.execute:
        asyncio.run(serve(args, candidate, plan))
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
