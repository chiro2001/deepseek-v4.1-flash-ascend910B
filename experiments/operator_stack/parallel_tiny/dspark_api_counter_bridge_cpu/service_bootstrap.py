"""Deferred private formal DSpark service. Default CLI writes a CPU launch plan.

The --execute branch is for a later separately allocated TP8 window. This file
was prepared and verified without starting a service or calling a device.
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

from .async_adapter import make_async_llm_adapter
from .bridge_state import CounterBridge
from .http_isolation import PrivateOnlyASGI
from .runtime_guards import (COUNTER_SHA256, false_claims, load_frozen_counter,
    sha256_file, verify_installed_sources)

WORKER_CLASS = 'parallel_tiny.dspark_api_counter_bridge_cpu.worker_adapter.ApiCounterDSparkWorker'


def launch_plan(model, served_model, port, output, counter_file, draft_graph):
    from parallel_tiny.dspark_formal_prep.contract import model_kwargs, startup_env
    if not served_model.startswith('dsv41-dspark-private-') or type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError('An explicit private served model and unprivileged port are required')
    kwargs = model_kwargs(model, 5, audit=False, draft_graph=draft_graph)
    kwargs['worker_cls'] = WORKER_CLASS
    env = startup_env(Path(output) / 'native', audit=False, draft_graph=draft_graph)
    env['DSPARK_API_FROZEN_COUNTER'] = str(Path(counter_file).resolve())
    return {'cpu_preparation_only': True, 'service_started': False,
        'npu_calls': 0, 'actual_http_inference_requests': 0,
        'served_model': served_model, 'host': '127.0.0.1', 'port': port,
        'model_kwargs': kwargs, 'environment': env,
        'frozen_counter_sha256': COUNTER_SHA256,
        'http_protocol': ['GET /private/dspark/identity',
            'POST /private/dspark/completions', 'POST /private/dspark/finish'],
        'clock': 'same boot/time namespace CLOCK_MONOTONIC; end immediately after response.read()',
        'warmup_requests': 2, 'measured_requests_minimum': 8,
        'prompt_tokens': 2048, 'output_tokens': 256, **false_claims()}


def _guard_config(config):
    spec = config.speculative_config
    if (config.parallel_config.tensor_parallel_size != 8 or
            config.parallel_config.data_parallel_size != 1 or
            config.scheduler_config.async_scheduling or
            config.scheduler_config.max_num_seqs != 1 or
            config.cache_config.enable_prefix_caching or
            config.load_config.load_format == 'dummy' or
            spec is None or spec.method != 'dspark' or spec.num_speculative_tokens != 5):
        raise ValueError('Actual HTTP engine config is outside the formal synchronous DSpark TP8 contract')


@asynccontextmanager
async def build_private_engine(args, bridge):
    # Preserve the actual official factory arguments and lifecycle. Only the
    # class is private; no official module or global factory is overwritten.
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args._api_process_count = 1
    engine_args._api_process_rank = 0
    config = engine_args.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)
    _guard_config(config)
    engine = None
    try:
        cls = make_async_llm_adapter(AsyncLLM)
        engine = cls.from_vllm_config(vllm_config=config,
            usage_context=UsageContext.OPENAI_API_SERVER,
            enable_log_requests=engine_args.enable_log_requests,
            aggregate_engine_logging=engine_args.aggregate_engine_logging,
            disable_log_stats=engine_args.disable_log_stats,
            client_addresses={}, client_count=1, client_index=0)
        engine.dspark_api_bridge = bridge
        await engine.reset_mm_cache()
        await bridge.describe(engine)
        yield engine
    finally:
        if engine is not None:
            engine.shutdown(timeout=config.shutdown_timeout)


async def build_private_app(engine, args, bridge):
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from vllm.entrypoints.openai import api_server as official
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest, CompletionResponse
    from vllm.entrypoints.serve.utils.api_utils import with_cancellation
    supported_tasks = await engine.get_supported_tasks()
    if 'generate' not in supported_tasks:
        raise ValueError('The loaded native model does not support generation')
    app = official.build_app(args, supported_tasks, engine.model_config)
    await official.init_app_state(engine, app.state, args, supported_tasks)
    # Remove all official/public inference, administration and plugin routes.
    # The outer ASGI whitelist also denies any routes attached later.
    app.router.routes.clear()

    @app.get('/private/dspark/identity')
    async def identity():
        return bridge.identity()

    @app.post('/private/dspark/completions')
    @with_cancellation
    async def completion(envelope: dict, raw_request: Request):
        # FastAPI parses this body before with_cancellation starts its receive
        # listener, exactly as required by the actual decorator's contract.
        accepted = False
        try:
            request_dict = await bridge.begin(engine, envelope,
                raw_request.headers.get('X-Request-Id'))
            accepted = True
            request = CompletionRequest.model_validate(request_dict)
            response = await app.state.openai_serving_completion.create_completion(request, raw_request)
            if not isinstance(response, CompletionResponse):
                raise ValueError('Stock Completion did not return a full successful non-stream response')
            # Exclude optional None fields so usage is the exact observed triple.
            result = response.model_dump(exclude_none=True)
            metadata = raw_request.state.request_metadata
            binding = bridge.note_completion_response(result, metadata.request_id)
            raw_request.scope['dspark_api_completion_window'] = binding['window_id']
            return JSONResponse({'completion': result, 'bridge': binding})
        except BaseException as exc:
            # Rejected inputs without a window have no native state to poison.
            if accepted:
                bridge.fail('http_completion', exc)
            raise

    @app.post('/private/dspark/finish')
    async def finish(raw_request: Request):
        return JSONResponse(await bridge.finish(engine, await raw_request.json()))

    return PrivateOnlyASGI(app, bridge)


async def execute_service(args, plan):
    # This branch must not be invoked under a CPU-only allocation.
    from parallel_tiny.dspark_formal_prep.contract import inspect_dspark_checkpoint
    checkpoint = inspect_dspark_checkpoint(args.model, 5)
    source_guard = verify_installed_sources()
    load_frozen_counter(args.counter_file)
    native = Path(args.output) / 'native'
    native.mkdir(parents=True, exist_ok=True)
    os.environ.update(plan['environment'])
    from vllm.entrypoints.openai import api_server as official
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    parser = make_arg_parser(FlexibleArgumentParser())
    serving = parser.parse_args(['--model', args.model, '--served-model-name', args.served_model,
        '--host', '127.0.0.1', '--port', str(args.port),
        '--tokenizer-mode', 'deepseek_v41', '--disable-log-stats'])
    for key, value in plan['model_kwargs'].items():
        setattr(serving, key, value)
    serving.enable_log_requests = False
    serving.middleware = []
    serving.api_server_count = 1
    serving.root_path = ''
    serving.enable_request_id_headers = False
    validate_parsed_serve_args(serving)
    bridge = CounterBridge(served_model=args.served_model, draft_graph=args.draft_graph,
        output=Path(args.output) / 'bridge')
    selection = {**plan, 'cpu_preparation_only': False, 'service_started': False,
        'checkpoint': checkpoint, 'source_guard': source_guard,
        'service_id': bridge.service_id,
        'bootstrap_source_sha256': sha256_file(__file__), **false_claims()}
    # The CPU plan's zero-call statements are not runtime device counters.
    selection.pop('npu_calls')
    selection.pop('actual_http_inference_requests')
    selection['runtime_npu_calls_counted'] = False
    selection_path = Path(args.output) / 'service_selection.json'
    if selection_path.exists():
        raise FileExistsError('Do not overwrite an earlier private service selection')
    selection_path.write_text(json.dumps(selection, indent=2) + '\n')
    listen_address, sock = official.setup_server(serving, reuse_port=False)
    try:
        async with build_private_engine(serving, bridge) as engine:
            app = await build_private_app(engine, serving, bridge)
            selection['service_started'] = True
            selection['listen_address'] = listen_address
            selection['identity'] = bridge.identity()
            selection['worker_descriptions'] = bridge.worker_description
            selection_path.write_text(json.dumps(selection, indent=2) + '\n')
            shutdown_task = await official.serve_http(app, sock=sock,
                host=serving.host, port=serving.port, log_level=serving.uvicorn_log_level,
                access_log=not serving.disable_uvicorn_access_log,
                timeout_keep_alive=official.envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
                enable_ssl_refresh=serving.enable_ssl_refresh,
                ssl_keyfile=serving.ssl_keyfile, ssl_certfile=serving.ssl_certfile,
                ssl_ca_certs=serving.ssl_ca_certs, ssl_cert_reqs=serving.ssl_cert_reqs,
                ssl_ciphers=serving.ssl_ciphers,
                h11_max_incomplete_event_size=serving.h11_max_incomplete_event_size,
                h11_max_header_count=serving.h11_max_header_count)
        await shutdown_task
    finally:
        sock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--served-model', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--counter-file', required=True, type=Path)
    parser.add_argument('--draft-graph', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    plan = launch_plan(args.model, args.served_model, args.port, args.output,
        args.counter_file, args.draft_graph)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'cpu_launch_plan.json'
    if path.exists():
        raise FileExistsError('Do not overwrite an earlier CPU launch plan')
    path.write_text(json.dumps(plan, indent=2) + '\n')
    if args.execute:
        asyncio.run(execute_service(args, plan))
    else:
        print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
