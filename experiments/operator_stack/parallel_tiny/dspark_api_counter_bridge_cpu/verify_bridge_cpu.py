"""CPU-only contract checks. No HTTP server/socket, torch import, or NPU call.

The original installed request-ID, submit, cleanup, and collective-RPC functions
are AST-executed against explicit CPU fakes. The frozen v4 install_runner runs
unchanged against eight mocked runners and a threading.Barrier standing in for
the CPU Gloo group. These checks cannot grant actual performance/quality proof.
"""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import textwrap
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from .acceptance_client import aggregate, make_completion, read_http_response, validate_workload, verify_identity
from .async_adapter import make_async_llm_adapter
from .bridge_state import CounterBridge, validate_completion
from .http_isolation import PrivateOnlyASGI
from .quality_cpu import quality, repeat_loop
from .runtime_guards import COUNTER_SHA256, HERE, clock_scope, false_claims, load_frozen_counter, monotonic_ns, token_hash
from .service_bootstrap import launch_plan
from .worker_rpc import ApiCounterWorkerRpcMixin, RPC_BEGIN, RPC_BIND, RPC_FINISH

COUNTER_PATH = HERE.parent / 'dspark_formal_audit' / 'stock_compat_v4' / 'formal_dspark_counters.py'
os.environ['DSPARK_API_FROZEN_COUNTER'] = str(COUNTER_PATH)
FROZEN = load_frozen_counter(COUNTER_PATH)
GUARDS = json.loads((HERE / 'evidence' / 'installed_source_guards.json').read_text())
SOURCE_FUNCTIONS = []
NEGATIVE_CASES = []
CPU_WINDOWS = []
SPLIT_PREFILL_RESULTS = []
MODEL = 'dsv41-dspark-private-cpu-fixture'
PROMPT = list(range(2048))


def original(relative, qualified_name, namespace=None):
    file = next(item for item in GUARDS['files'] if item['relative'] == relative)
    fragment = file['fragments'][qualified_name]
    if fragment.get('missing'):
        raise ValueError('Cannot AST-execute an undiscovered actual function')
    source = textwrap.dedent(fragment['source'])
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
            node.decorator_list = []
            for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                arg.annotation = None
            if node.args.vararg:
                node.args.vararg.annotation = None
            if node.args.kwarg:
                node.args.kwarg.annotation = None
    ast.fix_missing_locations(tree)
    scope = dict(namespace or {})
    exec(compile(tree, file['path'], 'exec'), scope)
    receipt = {'relative': relative, 'function': qualified_name,
        'installed_file_sha256': file['sha256'], 'function_source_sha256': fragment['source_sha256']}
    if receipt not in SOURCE_FUNCTIONS:
        SOURCE_FUNCTIONS.append(receipt)
    return scope[tree.body[0].name]


ASSIGN = original('v1/engine/input_processor.py', 'InputProcessor.assign_request_id',
    {'envs': SimpleNamespace(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=False),
     'random_uuid': lambda: uuid.uuid4().hex})
BASE_ID = original('entrypoints/serve/engine/serving.py', 'BaseServing._base_request_id',
    {'random_uuid': lambda: uuid.uuid4().hex})
CORE_RPC = original('v1/engine/core.py', 'EngineCore.collective_rpc')
CORE_CLIENT_RPC = original('v1/engine/core_client.py', 'AsyncMPClient.collective_rpc_async')
LISTEN_DISCONNECT = original('entrypoints/serve/utils/api_utils.py', 'listen_for_disconnect')
WITH_CANCELLATION = original('entrypoints/serve/utils/api_utils.py', 'with_cancellation',
    {'asyncio': asyncio, 'functools': functools, 'listen_for_disconnect': LISTEN_DISCONNECT})


class CpuOutputProcessor:
    def __init__(self):
        self.request_states = {}
        self.external_req_ids = {}
        self.parent_requests = {}

    def add_request(self, request, prompt, parent_req, index, queue):
        state = SimpleNamespace(request_id=request.request_id,
            external_req_id=request.external_req_id, parent_req=parent_req)
        self.request_states[request.request_id] = state
        self.external_req_ids.setdefault(request.external_req_id, []).append(request.request_id)

    _finish_request = original('v1/engine/output_processor.py', 'OutputProcessor._finish_request')


class CpuGraphObserver:
    def begin_proposal(self):
        pass

    def cancel_proposal(self):
        pass

    def finish_proposal(self):
        return {'actual_graph_submission': False, 'cpu_mock': True}


class CpuDrafter:
    method = 'dspark'

    def __init__(self):
        self.calls = 0

    def _propose(self, num_speculative_tokens):
        self.calls += 1
        return SimpleNamespace(shape=(1, num_speculative_tokens))


class CpuRunner:
    use_async_scheduling = False

    def __init__(self):
        self.requests = {}
        self.execute_model_state = None
        self.drafter = CpuDrafter()
        self.vllm_config = SimpleNamespace(speculative_config=SimpleNamespace(method='dspark'))
        self.current = None
        self.next_tokens = None

    def execute_model(self, scheduler, *args, **kwargs):
        self.current = scheduler
        self.requests.update({r.req_id: r for r in scheduler.scheduled_new_reqs})
        self.execute_model_state = object()

    def _sample(self, logits, spec_decode_metadata):
        return None

    def _bookkeeping_sync(self):
        return (None, [self.next_tokens], None, list(self.current.num_scheduled_tokens), None, None)

    def sample_tokens(self):
        self._sample(None, object() if self.current.scheduled_spec_decode_tokens else None)
        self._bookkeeping_sync()
        self.drafter._propose(num_speculative_tokens=5)
        self.execute_model_state = None
        return SimpleNamespace(req_ids=list(self.current.num_scheduled_tokens),
            sampled_token_ids=[self.next_tokens])


class CpuWorker(ApiCounterWorkerRpcMixin):
    def __init__(self, rank):
        self.model_runner = CpuRunner()
        FROZEN.install_runner(self.model_runner, rank, audit=False)


class CpuExecutor:
    def __init__(self, workers, engine):
        self.workers = workers
        self.engine = engine

    def collective_rpc(self, method, timeout, args, kwargs):
        rows = [getattr(worker, method)(*args, **(kwargs or {})) for worker in self.workers]
        if self.engine.corrupt_rpc == method:
            rows[-1]['rank'] = 0
        return rows


class CpuCore:
    collective_rpc_async = CORE_CLIENT_RPC

    def __init__(self, engine):
        self.engine = engine
        self.actual_core = SimpleNamespace(model_executor=CpuExecutor(engine.workers, engine))

    async def call_utility_async(self, method, *args):
        if method != 'collective_rpc':
            raise ValueError('CPU fixture received an unexpected utility ABI')
        return CORE_RPC(self.actual_core, *args)

    async def add_request_async(self, request):
        self.engine.core_submits.append(request.request_id)
        self.engine.ledger.append({'event': 'actual_submit_ast_delegated_to_cpu_core',
            'ns': monotonic_ns(), 'core_request_id': request.request_id})
        await asyncio.to_thread(self.engine.simulate_native, request)


class CpuStockAsyncLLM:
    _add_request = original('v1/engine/async_llm.py', 'AsyncLLM._add_request')
    collective_rpc = original('v1/engine/async_llm.py', 'AsyncLLM.collective_rpc')
    log_requests = False

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        request = SimpleNamespace(request_id=request_id, external_req_id=None,
            prompt_token_ids=prompt, sampling_params=sampling_params)
        ASSIGN(request)
        await self._add_request(request, None, None, 0, None)
        # Execute the actual cleanup function before the final output is read.
        self.output_processor._finish_request(self.output_processor.request_states[request.request_id])
        self.ledger.append({'event': 'actual_output_mapping_deleted_before_final_output', 'ns': monotonic_ns()})
        yield SimpleNamespace(request_id=request.external_req_id, prompt_token_ids=prompt,
            finished=True, outputs=[SimpleNamespace(index=0, token_ids=list(self.final_token_ids),
                finish_reason='length', text='；'.join(f'第{i}项记录具有不同的上下文内容{i*i}' for i in range(32)))])

    async def abort(self, request_id, internal=False):
        self.native_aborts.append({'request_id': request_id, 'internal': internal})


CPU_ADAPTER = make_async_llm_adapter(CpuStockAsyncLLM)


def scheduler(core_id, done, *, new=False):
    state = SimpleNamespace(req_id=core_id, num_computed_tokens=done, prompt_token_ids=PROMPT)
    return SimpleNamespace(scheduled_new_reqs=[state] if new else [],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[] if new else [core_id],
            num_computed_tokens=[] if new else [done]),
        num_scheduled_tokens={core_id: 2048 if new else 6},
        scheduled_spec_decode_tokens={} if new else {core_id: [7, 8, 9, 10, 11]},
        num_invalid_spec_tokens={}, finished_req_ids=set(),
        total_num_scheduled_tokens=2048 if new else 6)


class CpuEngine(CPU_ADAPTER):
    def __init__(self, bridge):
        self.dspark_api_bridge = bridge
        self.core_submits = []
        self.native_aborts = []
        self.ledger = []
        self.corrupt_rpc = None
        self.workers = [CpuWorker(rank) for rank in range(8)]
        self.engine_core = CpuCore(self)
        self.output_processor = CpuOutputProcessor()
        self.rank_local = threading.local()
        self.group_barrier = None
        self.protocol_case = None
        self.final_token_ids = list(range(1000, 1256))
        self.prefill_cpu_snapshots = []

    def fence(self):
        rank = self.rank_local.rank
        if rank == 7:
            # A delayed last rank makes the publication ordering meaningful.
            time.sleep(0.002)
        self.ledger.append({'event': 'mock_terminal_device_fence', 'rank': rank,
            'window_id': self.dspark_api_bridge.active['window_id'], 'ns': monotonic_ns()})

    def tp_group(self):
        rank = self.rank_local.rank
        engine = self
        class CpuTPGroup:
            world_size = 8
            rank_in_group = rank
            cpu_group = 'CPU_ONLY_MOCK_GLOO_GROUP'

            def barrier(self):
                engine.ledger.append({'event': 'mock_cpu_gloo_entry', 'rank': rank, 'ns': monotonic_ns()})
                engine.group_barrier.wait(timeout=10)
                engine.ledger.append({'event': 'mock_cpu_gloo_return', 'rank': rank, 'ns': monotonic_ns()})
        return CpuTPGroup()

    def simulate_native(self, request):
        self.group_barrier = threading.Barrier(8)
        def run_rank(rank):
            self.rank_local.rank = rank
            runner = self.workers[rank].model_runner
            if self.protocol_case is not None:
                for step in self.protocol_case['rounds']:
                    sched = scheduler(request.request_id, step['computed_before'], new=step['computed_before'] == 0)
                    sched.num_scheduled_tokens = {request.request_id: step['scheduled_target_tokens']}
                    sched.total_num_scheduled_tokens = step['scheduled_target_tokens']
                    # This source recorded the actual count, not the draft IDs.
                    # These IDs are explicit placeholders in this CPU replay.
                    sched.scheduled_spec_decode_tokens = ({request.request_id: list(range(step['verified_draft_tokens']))}
                        if step['verified_draft_tokens'] else {})
                    runner.next_tokens = step['sampled_token_ids']
                    runner.execute_model(sched)
                    runner.sample_tokens()
                    if rank == 0 and not step['is_decode']:
                        state = runner._formal_dspark_counter.requests[request.request_id]
                        self.prefill_cpu_snapshots.append({'scheduled_tokens': step['scheduled_target_tokens'],
                            'sampled_token_ids': step['sampled_token_ids'], 'seed_token_ids': state['seed_token_ids'],
                            'decode_start_ns': state['decode_start_ns'], 'prefill_calls': state['prefill_calls']})
                runner.requests.clear()
                return
            runner.next_tokens = [1000]
            runner.execute_model(scheduler(request.request_id, 0, new=True))
            runner.sample_tokens()
            for i in range(85):
                runner.next_tokens = list(range(1001 + 3 * i, 1004 + 3 * i))
                runner.execute_model(scheduler(request.request_id, 2048 + 3 * i))
                runner.sample_tokens()
            runner.requests.clear()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(run_rank, rank) for rank in range(8)]
            for future in futures:
                future.result()


def mocked_runtime(engine=None):
    graph = ModuleType('parallel_tiny.dspark_formal_audit.stock_graph_observer')
    graph.install = lambda *args: CpuGraphObserver()
    torch = ModuleType('torch')
    torch.npu = SimpleNamespace(synchronize=(engine.fence if engine else lambda: None))
    distributed = ModuleType('torch.distributed')
    distributed.get_backend = lambda group: 'gloo'
    torch.distributed = distributed
    vllm = ModuleType('vllm')
    vllm.__path__ = []
    vdistributed = ModuleType('vllm.distributed')
    vdistributed.get_tp_group = engine.tp_group if engine else lambda: None
    return patch.dict(sys.modules, {'torch': torch, 'torch.distributed': distributed,
        'vllm': vllm, 'vllm.distributed': vdistributed,
        'parallel_tiny.dspark_formal_audit.stock_graph_observer': graph})


def envelope(phase, index, *, prompt=PROMPT):
    rid = 'http_' + uuid.uuid4().hex
    return {'window_id': 'window_' + uuid.uuid4().hex, 'phase': phase, 'index': index,
        'client_clock': clock_scope(), 'completion': make_completion(MODEL, prompt, rid, index)}, rid


async def roundtrip(bridge, engine, phase, index, *, before_finish=None):
    payload, rid = envelope(phase, index)
    await bridge.begin(engine, payload, rid)
    raw_request = SimpleNamespace(headers={'X-Request-Id': rid})
    completion_id = 'cmpl-' + BASE_ID(raw_request, 'different_body_default')
    params = SimpleNamespace(n=1, max_tokens=256, min_tokens=256, ignore_eos=True)
    with mocked_runtime(engine):
        native_outputs = [out async for out in engine.generate(PROMPT, params, completion_id + '-0')]
    if engine.output_processor.external_req_ids:
        raise AssertionError('The actual cleanup AST must already remove the mapping')
    native = native_outputs[-1]
    response = {'id': completion_id, 'model': MODEL,
        'choices': [{'index': 0, 'prompt_token_ids': PROMPT, 'token_ids': list(native.outputs[0].token_ids),
            'text': native.outputs[0].text, 'finish_reason': 'length'}],
        'usage': {'prompt_tokens': 2048, 'completion_tokens': 256, 'total_tokens': 2304}}
    binding = bridge.note_completion_response(response, completion_id)
    body = json.dumps({'completion': response, 'bridge': binding}, separators=(',', ':')).encode()
    bridge.note_http_send_begin(payload['window_id'], 200)
    bridge.note_http_body(payload['window_id'], body, True)
    read_order = []
    class CpuResponse:
        def read(self):
            read_order.append('read_completed')
            return body
    def end_clock():
        read_order.append('CLOCK_MONOTONIC_end')
        return monotonic_ns()
    raw, ended = read_http_response(CpuResponse(), now=end_clock)
    read_order.append('parse_response_after_end')
    parsed = json.loads(raw)
    finish_payload = {'window_id': payload['window_id'], 'service_id': bridge.service_id,
        'client_clock': clock_scope(), 'client_end_ns': ended, 'client_read_completed': True,
        'http_body_sha256': hashlib.sha256(raw).hexdigest(),
        'final_tokens_sha256': token_hash(parsed['completion']['choices'][0]['token_ids'])}
    if before_finish is not None:
        await before_finish(bridge, engine, finish_payload)
    read_order.append('finish_RPC_after_end')
    finished = await bridge.finish(engine, finish_payload)
    window = bridge.history[-1]
    event = {r['event']: r['observed_ns'] for r in window['events']}
    before_submit = ['begin_rpc_call', 'begin_all_tp8_acked', 'stock_generate_external_id_observed',
        'mapping_tap_before_core_submit', 'bind_all_tp8_acked', 'native_core_submit_call']
    if [event[k] for k in before_submit] != sorted(event[k] for k in before_submit):
        raise AssertionError('RPC begin/mapping/bind did not precede core submit')
    terminal = []
    for receipt in window['rank_receipts']:
        row = receipt['requests'][window['core_request_id']]['rounds'][-1]
        joined = row['terminal_tp_join']
        rank = receipt['rank']
        fences = [e for e in engine.ledger if e['event'] == 'mock_terminal_device_fence' and
            e['rank'] == rank and e['window_id'] == window['window_id']]
        if len(fences) != 1 or not fences[0]['ns'] <= row['sample_end_ns'] <= joined['entered_ns']:
            raise AssertionError('Frozen terminal hook did not fence/timestamp before CPU join')
        if not joined['entered_ns'] <= event['finished_native_request_output_observed'] <= event['http_output_send_begin'] <= ended:
            raise AssertionError('A delayed rank terminal entry occurred after output publication')
        if not ended <= event['finish_rpc_call_after_client_end'] <= receipt['api_finish_rpc_ns']:
            raise AssertionError('Retrieval RPC was inserted before the client read/end clock')
        terminal.append({'rank': rank, 'mock_fence_ns': fences[0]['ns'],
            'native_v4_sample_end_ns': row['sample_end_ns'], 'mock_gloo_enter_ns': joined['entered_ns'],
            'mock_gloo_return_ns': joined['returned_ns'], 'finish_rpc_ns': receipt['api_finish_rpc_ns']})
    evidence = {'phase': phase, 'index': index, 'cpu_mock': True,
        'mapping': finished['mapping'], 'exact_begin_rank_count': len(window['begin_acks']),
        'exact_bind_rank_count': len(window['bind_acks']), 'events': window['events'],
        'terminal_per_rank': terminal, 'client_read_order': read_order,
        'client_end_ns': ended, 'actual_rounds_in_cpu_fixture': finished['metrics']['rounds'],
        'decoded_tokens_in_cpu_fixture': finished['metrics']['decoded_tokens'],
        'prefill_seed_tokens': finished['metrics']['prefill_seed_tokens'],
        'HTTP_emitted_tokens_in_cpu_fixture': len(native.outputs[0].token_ids),
        'stock_output_map_deleted_before_final_observer': not bool(engine.output_processor.external_req_ids),
        'all_actual_performance_quality_formal_flags_false': all(finished[k] is False for k in false_claims()),
        'rank_receipts_sha256': hashlib.sha256(json.dumps(window['rank_receipts'], sort_keys=True).encode()).hexdigest(),
        **false_claims()}
    CPU_WINDOWS.append(evidence)
    return {'phase': phase, 'index': index, 'finish': finished,
        'quality': quality(native.outputs[0].text, native.outputs[0].token_ids)}, finish_payload


class BridgeContract(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.unhandled_async_errors = []
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: self.unhandled_async_errors.append(context))
        self.bridge = CounterBridge(served_model=MODEL, draft_graph=False, cpu_mock=True, frozen=FROZEN)
        with mocked_runtime():
            self.engine = CpuEngine(self.bridge)

    async def asyncTearDown(self):
        await asyncio.sleep(0)
        self.assertFalse(self.unhandled_async_errors, self.unhandled_async_errors)

    def rejected(self, label, call, error=(ValueError, RuntimeError)):
        with self.assertRaises(error):
            call()
        NEGATIVE_CASES.append(label)

    async def rejected_async(self, label, call, error=(ValueError, RuntimeError)):
        with self.assertRaises(error):
            await call()
        NEGATIVE_CASES.append(label)

    async def test_two_warmup_eight_serial_actual_source_roundtrips(self):
        warmup = [await roundtrip(self.bridge, self.engine, 'warmup', i) for i in range(2)]
        measured = [await roundtrip(self.bridge, self.engine, 'measured', i) for i in range(8)]
        result = aggregate([row for row, _ in measured])
        self.assertEqual(result['samples'], 8)
        self.assertEqual(result['total_decoded_tokens'], 8 * 255)
        self.assertEqual(result['total_actual_rounds'], 8 * 85)
        self.assertEqual(self.bridge.completed, {'warmup': 2, 'measured': 8})
        self.assertEqual(len(set(self.engine.core_submits)), 10)
        self.assertEqual(len(warmup), 2)
        self.assertTrue(all(not row['finish'][key] for row, _ in measured for key in false_claims()))

    async def test_input_rejections_happen_before_counter_begin(self):
        payload, rid = envelope('warmup', 0)
        base = payload['completion']
        changes = {'multiple_prompts': {'prompt': [PROMPT, PROMPT]}, 'string_prompt': {'prompt': 'hello'},
            'n_gt_one': {'n': 2}, 'n_bool': {'n': True}, 'stream': {'stream': True},
            'beam': {'use_beam_search': True}, 'echo': {'echo': True}, 'output_255': {'max_tokens': 255},
            'min_short': {'min_tokens': 0}, 'eos_stop': {'ignore_eos': False},
            'special_tokens': {'add_special_tokens': True}, 'no_ids': {'return_token_ids': False},
            'random_sampling': {'temperature': 1.0}, 'negative_token': {'prompt': [-1] * 2048},
            'bool_token': {'prompt': [True] * 2048}, 'short_prompt': {'prompt': [1] * 2047},
            'long_prompt': {'prompt': [1] * 2049}, 'another_model': {'model': 'other'},
            'another_body_id': {'request_id': 'other'}, 'extra_stop': {'stop': ['stop']}}
        for label, change in changes.items():
            request = {**base, **change}
            self.rejected(label, lambda r=request: validate_completion(r, rid, MODEL))
        self.rejected('no_header_id', lambda: validate_completion(base, None, MODEL))
        self.rejected('unsafe_header_id', lambda: validate_completion(base, 'bad\nID', MODEL))
        self.assertIsNone(self.bridge.active)
        self.assertFalse(self.engine.core_submits)

    async def test_clock_mismatch_and_measurement_before_warmup(self):
        payload, rid = envelope('warmup', 0)
        payload['client_clock'] = {**clock_scope(), 'boot_id': 'another_host'}
        await self.rejected_async('remote_host_clock', lambda: self.bridge.begin(self.engine, payload, rid))
        payload['client_clock'] = {**clock_scope(), 'time_namespace': 'time:[other]'}
        await self.rejected_async('other_time_namespace', lambda: self.bridge.begin(self.engine, payload, rid))
        payload, rid = envelope('measured', 0)
        await self.rejected_async('measured_before_two_warmup', lambda: self.bridge.begin(self.engine, payload, rid))
        self.assertFalse(self.engine.core_submits)

    async def test_mapping_must_ack_all_ranks_before_actual_submit(self):
        self.engine.corrupt_rpc = RPC_BIND
        payload, rid = envelope('warmup', 0)
        await self.bridge.begin(self.engine, payload, rid)
        params = SimpleNamespace(n=1, max_tokens=256, min_tokens=256, ignore_eos=True)
        async def generate():
            with mocked_runtime(self.engine):
                return [out async for out in self.engine.generate(PROMPT, params, 'observed-external-id')]
        await self.rejected_async('duplicate_rank_bind_ack', generate)
        self.assertFalse(self.engine.core_submits)
        self.assertTrue(self.bridge.failed)

    async def test_bad_begin_ack_poisoned_without_submit(self):
        self.engine.corrupt_rpc = RPC_BEGIN
        payload, rid = envelope('warmup', 0)
        await self.rejected_async('duplicate_rank_begin_ack', lambda: self.bridge.begin(self.engine, payload, rid))
        self.assertTrue(self.bridge.failed)
        self.assertFalse(self.engine.core_submits)

    async def test_active_window_parallel_request_is_rejected(self):
        payload, rid = envelope('warmup', 0)
        await self.bridge.begin(self.engine, payload, rid)
        payload2, rid2 = envelope('warmup', 0)
        await self.rejected_async('parallel_request_before_finish', lambda: self.bridge.begin(self.engine, payload2, rid2))

    async def test_abort_is_native_and_never_receipt(self):
        payload, rid = envelope('warmup', 0)
        await self.bridge.begin(self.engine, payload, rid)
        await self.engine.abort('actual-native-id', internal=True)
        self.assertTrue(self.bridge.failed)
        self.assertEqual(self.engine.native_aborts, [{'request_id': 'actual-native-id', 'internal': True}])
        await self.rejected_async('abort_no_finish_receipt', lambda: self.bridge.finish(self.engine, {}))
        payload2, rid2 = envelope('warmup', 0)
        await self.rejected_async('abort_no_automatic_retry', lambda: self.bridge.begin(self.engine, payload2, rid2))

    async def test_partial_output_generator_cannot_publish_receipt(self):
        payload, rid = envelope('warmup', 0)
        await self.bridge.begin(self.engine, payload, rid)
        params = SimpleNamespace(n=1, max_tokens=256, min_tokens=256, ignore_eos=True)
        native = CpuStockAsyncLLM.generate
        async def partial(instance, *args, **kwargs):
            async for out in native(instance, *args, **kwargs):
                out.finished = False
                yield out
        async def generate():
            with mocked_runtime(self.engine), patch.object(CpuStockAsyncLLM, 'generate', partial):
                return [out async for out in self.engine.generate(PROMPT, params, 'actual-partial-external')]
        await self.rejected_async('native_EOF_without_finished_output', generate)
        self.assertTrue(self.bridge.failed)
        self.assertNotIn('http_output_send_begin', [e['event'] for e in self.bridge.active['events']])
        await self.rejected_async('partial_output_no_finish_receipt', lambda: self.bridge.finish(self.engine, {}))

    async def test_actual_disconnect_decorator_reads_only_after_body_parsed(self):
        started = asyncio.Event()
        disconnected = asyncio.Event()
        cancelled = asyncio.Event()
        payload, rid = envelope('warmup', 0)
        class BodyParsedCpuRequest:
            app = SimpleNamespace(state=SimpleNamespace(enable_server_load_tracking=False))
            async def receive(self):
                await disconnected.wait()
                return {'type': 'http.disconnect'}
        @WITH_CANCELLATION
        async def handler(envelope, raw_request):
            try:
                await self.bridge.begin(self.engine, envelope, rid)
                started.set()
                await asyncio.Event().wait()
            except BaseException as exc:
                self.bridge.fail('CPU_disconnect', exc)
                cancelled.set()
                raise
        task = asyncio.create_task(handler(envelope=payload, raw_request=BodyParsedCpuRequest()))
        await asyncio.wait_for(started.wait(), 2)
        disconnected.set()
        self.assertIsNone(await asyncio.wait_for(task, 2))
        await asyncio.wait_for(cancelled.wait(), 2)
        self.assertTrue(self.bridge.failed)
        self.assertFalse(self.engine.core_submits)
        NEGATIVE_CASES.append('actual_disconnect_decorator_cancels_and_poisoned_window')

    async def test_actual_core_binding_rejects_changed_prompt_or_id(self):
        for label in ('rendered_prompt_changed', 'core_external_id_changed', 'randomization_disabled', 'parent_fanout'):
            bridge = CounterBridge(served_model=MODEL, draft_graph=False, cpu_mock=True, frozen=FROZEN)
            with mocked_runtime():
                engine = CpuEngine(bridge)
            payload, rid = envelope('warmup', 0)
            await bridge.begin(engine, payload, rid)
            params = SimpleNamespace(n=1, max_tokens=256, min_tokens=256, ignore_eos=True)
            bridge.note_generate('actual-external', params)
            request = SimpleNamespace(request_id='actually-randomized', external_req_id='actual-external',
                prompt_token_ids=PROMPT, sampling_params=params)
            parent = None
            if label == 'rendered_prompt_changed':
                request.prompt_token_ids = [99] * 2048
            elif label == 'core_external_id_changed':
                request.external_req_id = 'other'
            elif label == 'randomization_disabled':
                request.request_id = request.external_req_id
            else:
                parent = object()
            await self.rejected_async(label, lambda: bridge.bind_core(engine, request, parent, 0))
            self.assertFalse(engine.core_submits)

    async def test_worker_mapping_rejects_duplicate_or_after_execution(self):
        payload, rid = envelope('warmup', 0)
        await self.bridge.begin(self.engine, payload, rid)
        worker = self.engine.workers[0]
        binding = {'window_id': payload['window_id'], 'http_request_id': rid,
            'engine_external_request_id': 'actual-external', 'core_request_id': 'actual-core',
            'prompt_sha256': token_hash(PROMPT)}
        worker.dspark_api_v1_bind(binding)
        self.rejected('duplicate_worker_core_binding', lambda: worker.dspark_api_v1_bind(binding))
        self.engine.workers[1].model_runner._formal_dspark_counter.requests['foreign'] = {}
        self.rejected('mapping_after_native_execute', lambda: self.engine.workers[1].dspark_api_v1_bind(binding))

    async def test_finish_must_follow_http_read_and_exact_window(self):
        payload, rid = envelope('warmup', 0)
        await self.bridge.begin(self.engine, payload, rid)
        await self.rejected_async('finish_before_http_output', lambda: self.bridge.finish(self.engine, {}))
        worker = self.engine.workers[0]
        self.rejected('worker_finish_before_actual_core_binding', lambda: worker.dspark_api_v1_finish({
            'window_id': payload['window_id'], 'core_request_id': 'guessed'}))

    async def test_finish_rejects_clock_read_body_identity_and_future_stamp(self):
        async def controls(bridge, engine, valid):
            changes = {'finish_wrong_window': {'window_id': 'other'},
                'finish_wrong_service': {'service_id': 'other'},
                'finish_other_clock': {'client_clock': {**clock_scope(), 'boot_id': 'other'}},
                'finish_before_read': {'client_read_completed': False},
                'finish_other_body': {'http_body_sha256': '0' * 64},
                'finish_other_tokens': {'final_tokens_sha256': '0' * 64},
                'finish_future_clock': {'client_end_ns': monotonic_ns() + 10**12},
                'finish_early_clock': {'client_end_ns': 1}}
            for label, change in changes.items():
                await self.rejected_async(label, lambda p={**valid, **change}: bridge.finish(engine, p))
            core = bridge.active['core_request_id']
            rpc = {'window_id': valid['window_id'], 'core_request_id': core,
                'client_clock': valid['client_clock'], 'client_end_ns': valid['client_end_ns'],
                'client_read_completed': True}
            worker_changes = {'worker_finish_wrong_core': {'core_request_id': 'other'},
                'worker_finish_other_clock': {'client_clock': {**clock_scope(), 'time_namespace': 'other'}},
                'worker_finish_not_read': {'client_read_completed': False},
                'worker_finish_future_stamp': {'client_end_ns': monotonic_ns() + 10**12}}
            for label, change in worker_changes.items():
                self.rejected(label, lambda p={**rpc, **change}: engine.workers[0].dspark_api_v1_finish(p))
        await roundtrip(self.bridge, self.engine, 'warmup', 0, before_finish=controls)

    async def test_real_tiny_two_prefill_geometry_replays_only_as_CPU_protocol(self):
        case = json.loads((HERE / 'evidence' / 'tiny_two_prefill_protocol_input.json').read_text())
        self.assertFalse(case['source_formal_admitted'])
        self.assertTrue(case['source_dummy_period2'])
        self.assertEqual(case['source_actual_prefill_chunks'], [2044, 4])
        self.assertEqual(case['source_actual_decode_rounds'], 128)
        self.assertEqual(case['source_actual_helper_calls'], 130)
        self.engine.protocol_case = case
        self.engine.final_token_ids = case['token_ids']
        row, _ = await roundtrip(self.bridge, self.engine, 'warmup', 0)
        receipt = row['finish']['rank_receipts'][0]
        state = receipt['requests'][row['finish']['mapping']['core_request_id']]
        self.assertEqual(state['prefill_calls'], 2)
        self.assertEqual(state['seed_token_ids'], [87930])
        self.assertEqual(len(state['rounds']), 128)
        self.assertEqual(row['finish']['metrics']['decoded_tokens'], 255)
        self.assertEqual(self.engine.workers[0].model_runner.drafter.calls, 130)
        self.assertEqual(self.engine.prefill_cpu_snapshots[0]['seed_token_ids'], [])
        self.assertIsNone(self.engine.prefill_cpu_snapshots[0]['decode_start_ns'])
        self.assertEqual(self.engine.prefill_cpu_snapshots[1]['seed_token_ids'], [87930])
        self.assertEqual(self.engine.prefill_cpu_snapshots[1]['decode_start_ns'], state['decode_start_ns'])
        self.assertFalse(row['quality']['clean'])
        self.assertTrue(row['quality']['repeated_token_window'])
        self.assertTrue(all(row['finish'][key] is False for key in false_claims()))
        SPLIT_PREFILL_RESULTS.append({'cpu_mock': True, 'source_sha256': case['source_sha256'],
            'source_was_failed_dummy_tiny': True, 'not_formal_TP8_geometry_proof': True,
            'first_two_CPU_prefill_snapshots': self.engine.prefill_cpu_snapshots,
            'counted_decode_rounds': len(state['rounds']), 'CPU_mock_propose_calls': 130,
            'HTTP_tokens_in_CPU_replay': 256, 'decoded_tokens_excluding_seed': 255,
            'clean': False, 'repeated_token_window': True, **false_claims()})

    async def test_empty_prefill_is_not_seed_and_prefill_seed_must_be_unique(self):
        counter = FROZEN.RoundCounters(0)
        counter.begin('CPU-empty-prefill', 256)
        first = scheduler('CPU-core', 0, new=True)
        first.num_scheduled_tokens = {'CPU-core': 2044}
        first.total_num_scheduled_tokens = 2044
        counter.execute_begin(first, {})
        counter.sampler_finished(False)
        counter.bookkeeping_finished((None, [[]], None, ['CPU-core'], None, None))
        counter.draft_finished(k=5, output_shape=[1, 5], runtime_graph=False, method='dspark')
        counter.sample_finished()
        self.assertIsNone(counter.requests['CPU-core']['decode_start_ns'])
        self.assertEqual(counter.requests['CPU-core']['seed_token_ids'], [])
        missing_seed = scheduler('CPU-core', 2048)
        counter.execute_begin(missing_seed, {'CPU-core': SimpleNamespace(prompt_token_ids=PROMPT)})
        counter.sampler_finished(True)
        counter.bookkeeping_finished((None, [[1, 2]], None, ['CPU-core'], None, None))
        counter.draft_finished(k=5, output_shape=[1, 5], runtime_graph=False, method='dspark')
        self.rejected('empty_prefill_cannot_seed_a_decode_round', counter.sample_finished)
        counter = FROZEN.RoundCounters(0)
        counter.begin('CPU-multiple-seed', 256)
        counter.execute_begin(first, {})
        counter.sampler_finished(False)
        self.rejected('multiple_prefill_seed_tokens_rejected', lambda: counter.bookkeeping_finished(
            (None, [[87930, 28496]], None, ['CPU-core'], None, None)))

    async def test_summary_negative_receipts_and_seed_definition(self):
        row, _ = await roundtrip(self.bridge, self.engine, 'warmup', 0)
        original_rows = row['finish']['rank_receipts']
        tokens = list(range(1000, 1256))
        core_id = row['finish']['mapping']['core_request_id']
        ended = row['finish']['metrics']['client_end_ns']
        def summarize(rows, emitted=tokens, end=ended, expected=core_id, graph=False):
            return FROZEN.summarize(rows, emitted, end, clock_scope(), draft_graph=graph,
                expected_core_request_id=expected)
        variants = ['missing_rank', 'worker_clock', 'other_core_id', 'missing_fence', 'missing_join',
            'no_verifier', 'no_draft', 'divergent_rank_tokens', 'wrong_output_prefix']
        for label in variants:
            receipts = copy.deepcopy(original_rows)
            if label == 'missing_rank':
                receipts.pop()
            elif label == 'worker_clock':
                receipts[0]['clock_scope']['boot_id'] = 'other'
            else:
                rounds = receipts[0]['requests'][core_id]['rounds']
                if label == 'other_core_id':
                    receipts[0]['requests']['wrong-core'] = receipts[0]['requests'].pop(core_id)
                elif label == 'missing_fence':
                    rounds[-1]['terminal_device_synced'] = False
                elif label == 'missing_join':
                    rounds[-1].pop('terminal_tp_join')
                elif label == 'no_verifier':
                    rounds[0]['rejection_sampler_called'] = False
                elif label == 'no_draft':
                    rounds[0]['draft_calls'] = []
                elif label == 'divergent_rank_tokens':
                    rounds[0]['token_ids'][0] = 9999
                elif label == 'wrong_output_prefix':
                    receipts[0]['requests'][core_id]['seed_token_ids'] = [999]
            self.rejected(label, lambda r=receipts: summarize(r))
        self.rejected('client_end_before_final_sample', lambda: summarize(original_rows, end=1))
        self.rejected('eager_cannot_claim_draft_graph', lambda: summarize(original_rows, graph=True))
        self.rejected('wrong_expected_core_id', lambda: summarize(original_rows, expected='guess'))
        fake_rows = [copy.deepcopy(row) for _ in range(8)]
        for i, sample in enumerate(fake_rows):
            sample['phase'] = 'measured'
            sample['index'] = i
        self.assertEqual(aggregate(fake_rows)['total_decoded_tokens'], 8 * 255)
        fake_rows[0]['finish']['metrics']['decoded_tokens'] = 256
        self.rejected('completion_tokens_cannot_replace_decoded_tokens', lambda: aggregate(fake_rows))

    async def test_warmup_and_incomplete_measurement_not_aggregated(self):
        self.rejected('seven_measured_samples', lambda: aggregate([{}] * 7))
        self.rejected('warmup_cannot_enter_aggregate', lambda: aggregate([{'phase': 'warmup'}] * 8))

    async def test_all_public_inference_and_admin_routes_are_blocked(self):
        delegated = []
        async def app(scope, receive, send):
            delegated.append(scope)
        wrapper = PrivateOnlyASGI(app, self.bridge)
        forbidden = ['/v1/completions', '/v1/chat/completions', '/v1/responses', '/generate',
            '/pooling', '/tokenize', '/reset_prefix_cache', '/scale_elastic_ep', '/invocations',
            '/collective_rpc', '/start_profile', '/private/dspark/completions/extra']
        for path in forbidden:
            sent = []
            async def send(message):
                sent.append(message)
            await wrapper({'type': 'http', 'method': 'POST', 'path': path}, None, send)
            self.assertEqual(sent[0]['status'], 403)
            NEGATIVE_CASES.append('isolated_route_' + path)
        self.assertFalse(delegated)
        sent = []
        async def send(message):
            sent.append(message)
        await wrapper({'type': 'websocket', 'path': '/v1/realtime'}, None, send)
        self.assertEqual(sent, [{'type': 'websocket.close', 'code': 1008}])
        NEGATIVE_CASES.append('isolated_websocket')

    async def test_quality_detects_clean_and_repeated_or_corrupt_output(self):
        self.assertTrue(quality('正常回答，包含独立的内容。', list(range(256)))['clean'])
        self.assertTrue(repeat_loop('abcdefghijklmnopqrst0123456789ABCDEFGHIJ' * 6))
        self.assertFalse(repeat_loop('正常句子' * 6))
        for label, text, tokens in [('empty', '', list(range(256))),
                ('replacement', '错误\ufffd', list(range(256))), ('nul', '异常\x00', list(range(256))),
                ('char_repeat', 'abcdefghijklmnopqrst0123456789ABCDEFGHIJ' * 6, list(range(256))),
                ('token_repeat', '文字表面正常', [10] * 256)]:
            self.assertFalse(quality(text, tokens)['clean'])
            NEGATIVE_CASES.append('quality_' + label)

    async def test_source_guard_fragments_and_dry_plan(self):
        self.assertEqual(len(GUARDS['files']), 16)
        for item in GUARDS['files']:
            for name, fragment in item['fragments'].items():
                self.assertFalse(fragment.get('missing'), (item['relative'], name))
                self.assertEqual(hashlib.sha256(fragment['source'].encode()).hexdigest(), fragment['source_sha256'])
                ast.parse(textwrap.dedent(fragment['source']))
        plan = launch_plan('/deferred/real/model', 'dsv41-dspark-private-deferred', 19111,
            '/deferred/new/output', COUNTER_PATH, False)
        self.assertEqual(plan['model_kwargs']['speculative_config'],
            {'method': 'dspark', 'num_speculative_tokens': 5, 'enforce_eager': True})
        self.assertFalse(plan['service_started'])
        self.assertTrue(all(plan[key] is False for key in false_claims()))
        self.assertTrue(plan['model_kwargs']['worker_cls'].endswith('ApiCounterDSparkWorker'))

    async def test_workload_requires_real_provenance_and_eight_independent_prompts(self):
        fake = {'cpu_mock': True, 'prompt_kind': 'real_text_tasks', 'warmup': [], 'measured': []}
        self.rejected('cpu_mock_workload_rejected_for_actual_client', lambda: validate_workload(fake))
        workload = {'cpu_mock': False, 'prompt_kind': 'real_text_tasks',
            'tokenizer_provenance': {'fixture': True}, 'corpus_provenance': {'fixture': True},
            'warmup': [], 'measured': []}
        for phase, count in [('warmup', 2), ('measured', 8)]:
            for i in range(count):
                prompt = list(range(i * 2048, (i + 1) * 2048))
                workload[phase].append({'name': phase + str(i), 'task': 'fixture',
                    'token_ids': prompt, 'prompt_sha256': token_hash(prompt)})
        validate_workload(workload)
        for row in workload['measured']:
            row['token_ids'] = PROMPT
            row['prompt_sha256'] = token_hash(PROMPT)
        self.rejected('eight_duplicate_prompts_are_not_independent', lambda: validate_workload(workload))


def main():
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BridgeContract)
    result = runner.run(suite)
    report = {'cpu_preparation_only': True, 'cpu_mock': True,
        'actual_npu_calls': 0, 'actual_http_inference_requests': 0, 'services_started': 0,
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'unhandled_async_errors': 0 if result.wasSuccessful() else None,
        'passed': result.wasSuccessful(), 'negative_cases': NEGATIVE_CASES,
        'negative_case_count': len(NEGATIVE_CASES), 'ast_executed_actual_functions': SOURCE_FUNCTIONS,
        'unchanged_frozen_counter_sha256': hashlib.sha256(COUNTER_PATH.read_bytes()).hexdigest(),
        'guarded_actual_source_files': len(GUARDS['files']),
        'CPU_happens_before_windows': CPU_WINDOWS,
        'two_prefill_CPU_protocol_results': SPLIT_PREFILL_RESULTS,
        'claims': false_claims(),
        'limitations': ['All eight runners/device fences/Gloo group/HTTP bytes are explicit CPU fakes.',
            'Actual API performance, actual client quality, actual service launch and formal receipt remain unperformed.',
            'No device-consumer proof is granted by this bridge.']}
    (HERE / 'evidence' / 'cpu_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == '__main__':
    main()
