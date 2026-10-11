"""Single active HTTP counter window with explicit observed request identities."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path
import re
import uuid

from .runtime_guards import (COUNTER_SHA256, clock_scope, false_claims,
    load_frozen_counter, monotonic_ns, token_hash, token_list)
from .worker_rpc import RPC_BEGIN, RPC_BIND, RPC_DESCRIBE, RPC_FINISH

ID = re.compile(r'^[A-Za-z0-9_-]{1,96}$')
COMPLETION_KEYS = {'model', 'prompt', 'request_id', 'n', 'stream', 'echo',
    'max_tokens', 'min_tokens', 'ignore_eos', 'add_special_tokens',
    'return_token_ids', 'temperature', 'seed', 'use_beam_search'}


def validate_completion(payload, header_id, served_model):
    if type(payload) is not dict or set(payload) != COMPLETION_KEYS:
        raise ValueError('Private acceptance requires the exact fixed Completion fields')
    if type(header_id) is not str or ID.fullmatch(header_id) is None:
        raise ValueError('An explicit safe X-Request-Id is required')
    if payload['request_id'] != header_id or payload['model'] != served_model:
        raise ValueError('HTTP header/body identity or served model mismatch')
    token_list(payload['prompt'], size=2048)
    expected = {'n': 1, 'stream': False, 'echo': False, 'max_tokens': 256,
        'min_tokens': 256, 'ignore_eos': True, 'add_special_tokens': False,
        'return_token_ids': True, 'temperature': 0.0, 'use_beam_search': False}
    for key, value in expected.items():
        if payload[key] != value or (type(value) is bool and type(payload[key]) is not bool):
            raise ValueError(f'Unsupported Completion option: {key}')
    for key in ('n', 'max_tokens', 'min_tokens', 'seed'):
        if type(payload[key]) is not int:
            raise ValueError(f'Expected integer Completion field: {key}')
    return copy.deepcopy(payload)


class CounterBridge:
    def __init__(self, *, served_model, draft_graph, output=None, cpu_mock=False, frozen=None):
        self.served_model = served_model
        self.draft_graph = bool(draft_graph)
        self.cpu_mock = bool(cpu_mock)
        self.service_id = uuid.uuid4().hex
        self.clock = clock_scope()
        self.output = Path(output) if output is not None else None
        self.frozen = frozen
        self.active = None
        self.failed = False
        self.completed = {'warmup': 0, 'measured': 0}
        self.history = []
        self.used_window_ids = set()
        self.used_http_ids = set()
        self.lock = asyncio.Lock()
        self.worker_description = None

    def event(self, name, **fields):
        row = {'event': name, 'observed_ns': monotonic_ns(), **fields}
        if self.active is not None:
            self.active['events'].append(row)
        return row

    def _require(self, *stages):
        if self.failed or self.active is None or self.active['stage'] not in stages:
            raise RuntimeError('API window is absent, failed, or in another stage')
        return self.active

    def _write(self):
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
            evidence = {'service_id': self.service_id, 'cpu_mock': self.cpu_mock,
                'clock_scope': self.clock, 'failed': self.failed,
                'completed': self.completed, 'windows': self.history,
                'active': self.active, **false_claims()}
            (self.output / 'bridge_windows.json').write_text(json.dumps(evidence, indent=2) + '\n')

    def fail(self, stage, exc):
        self.failed = True
        if self.active is not None:
            self.event('failure', stage=stage, error_type=type(exc).__name__, error=str(exc))
            self.active['stage'] = 'failed'
            self.active['claims'] = false_claims()
        self._write()

    def _acks(self, rows, stage, *, window=None):
        if (type(rows) is not list or len(rows) != 8 or
                any(type(r.get('rank')) is not int for r in rows) or
                {r['rank'] for r in rows} != set(range(8))):
            raise ValueError('RPC requires one acknowledgement from every TP8 rank')
        for row in rows:
            if row.get('counter_sha256') != COUNTER_SHA256 or row.get('clock_scope') != self.clock:
                raise ValueError('Worker counter source or monotonic namespace mismatch')
            if stage != 'describe' and row.get('stage') != stage:
                raise ValueError('Worker acknowledgement has another RPC stage')
            if window is not None:
                key = 'tag' if stage == 'begin' else 'window_id'
                if row.get(key) != window['window_id']:
                    raise ValueError('Worker acknowledgement has another active window')
                if stage == 'bind':
                    for key in ('http_request_id', 'engine_external_request_id', 'core_request_id'):
                        if row.get(key) != window[key]:
                            raise ValueError('TP8 ranks disagree on the exact request mapping')
                    if row.get('before_any_native_execution') is not True:
                        raise ValueError('Mapping is not acknowledged before native execution')
                elif (row.get('phase') != window['phase'] or row.get('index') != window['index'] or
                        row.get('audit') is not False):
                    raise ValueError('Begin acknowledgement changed the timing window')
        return copy.deepcopy(rows)

    async def describe(self, engine):
        rows = await engine.collective_rpc(RPC_DESCRIBE)
        self._acks(rows, 'describe')
        if any(row.get('audit') is not False or row.get('counter_active') is not False for row in rows):
            raise ValueError('Private performance service needs idle audit-OFF workers')
        self.worker_description = copy.deepcopy(rows)
        return rows

    async def begin(self, engine, envelope, header_id):
        async with self.lock:
            if self.failed or self.active is not None:
                raise RuntimeError('Finish the previous window before submitting another request')
            if type(envelope) is not dict or set(envelope) != {'window_id', 'phase', 'index', 'client_clock', 'completion'}:
                raise ValueError('Invalid private Completion envelope')
            if envelope['client_clock'] != self.clock:
                raise ValueError('HTTP client and service monotonic clock namespaces differ')
            phase, index = envelope['phase'], envelope['index']
            if phase not in self.completed or type(index) is not int or index != self.completed[phase]:
                raise ValueError('Warmup/measured windows must each be serial and contiguous')
            if phase == 'measured' and self.completed['warmup'] < 2:
                raise ValueError('At least two separate warmup windows are required')
            if phase == 'warmup' and self.completed['measured']:
                raise ValueError('Warmup cannot enter the measurement sequence')
            window_id = envelope['window_id']
            if type(window_id) is not str or ID.fullmatch(window_id) is None:
                raise ValueError('Invalid window ID')
            if window_id in self.used_window_ids or header_id in self.used_http_ids:
                raise ValueError('Window and HTTP IDs must never be reused')
            request = validate_completion(envelope['completion'], header_id, self.served_model)
            window = {'window_id': window_id, 'phase': phase, 'index': index,
                'http_request_id': header_id, 'prompt_sha256': token_hash(request['prompt']),
                'stage': 'begin_pending', 'cpu_mock': self.cpu_mock,
                'events': [], 'claims': false_claims()}
            self.active = window
            self.used_window_ids.add(window_id)
            self.used_http_ids.add(header_id)
            payload = {k: window[k] for k in ('window_id', 'phase', 'index', 'http_request_id', 'prompt_sha256')}
            payload['client_clock'] = self.clock
            try:
                self.event('begin_rpc_call')
                rows = await engine.collective_rpc(RPC_BEGIN, kwargs={'payload': payload})
                window['begin_acks'] = self._acks(rows, 'begin', window=window)
                self.event('begin_all_tp8_acked')
                window['stage'] = 'armed'
                self._write()
                return request
            except BaseException as exc:
                self.fail('begin_rpc', exc)
                raise

    def note_generate(self, request_id, sampling_params):
        window = self._require('armed')
        if (type(request_id) is not str or not request_id or
                sampling_params.n != 1 or sampling_params.max_tokens != 256 or
                sampling_params.min_tokens != 256 or sampling_params.ignore_eos is not True):
            raise ValueError('Stock renderer changed the fixed single-request sampling contract')
        window['engine_external_request_id'] = request_id
        window['stage'] = 'generate_observed'
        self.event('stock_generate_external_id_observed', engine_external_request_id=request_id)

    async def bind_core(self, engine, request, parent_req, index):
        window = self._require('generate_observed')
        if parent_req is not None or index != 0:
            raise ValueError('Parent requests and n>1 fanout are unsupported')
        prompt = token_list(request.prompt_token_ids, size=2048)
        if request.external_req_id != window['engine_external_request_id']:
            raise ValueError('Randomized core request has another external engine identity')
        if type(request.request_id) is not str or not request.request_id:
            raise ValueError('Actual randomized core request ID is unavailable')
        if request.request_id == request.external_req_id:
            raise ValueError('Native request randomization must remain enabled')
        if token_hash(prompt) != window['prompt_sha256']:
            raise ValueError('Actual rendered prompt IDs differ from HTTP prompt IDs')
        params = request.sampling_params
        if (params.n != 1 or params.max_tokens != 256 or params.min_tokens != 256 or params.ignore_eos is not True):
            raise ValueError('Actual core sampling parameters changed')
        window['core_request_id'] = request.request_id
        payload = {k: window[k] for k in ('window_id', 'http_request_id',
            'engine_external_request_id', 'core_request_id', 'prompt_sha256')}
        self.event('mapping_tap_before_core_submit', **payload)
        rows = await engine.collective_rpc(RPC_BIND, kwargs={'payload': payload})
        window['bind_acks'] = self._acks(rows, 'bind', window=window)
        self.event('bind_all_tp8_acked')
        window['stage'] = 'bound'

    def note_core_submit(self, core_id):
        window = self._require('bound')
        if core_id != window['core_request_id']:
            raise ValueError('Core submit ID differs from the actual mapping tap')
        self.event('native_core_submit_call', core_request_id=core_id)
        window['stage'] = 'submitted'

    def note_native_output(self, output):
        window = self._require('submitted')
        if output.request_id != window['engine_external_request_id']:
            raise ValueError('RequestOutput has another external request identity')
        if not output.finished:
            return
        prompt = token_list(output.prompt_token_ids, size=2048)
        if token_hash(prompt) != window['prompt_sha256'] or len(output.outputs) != 1:
            raise ValueError('Final native prompt or output sequence diverged')
        choice = output.outputs[0]
        tokens = token_list(choice.token_ids, size=256)
        if choice.index != 0 or choice.finish_reason != 'length':
            raise ValueError('Only a full length-limited native output is accepted')
        window['final_tokens'] = tokens
        window['final_tokens_sha256'] = token_hash(tokens)
        self.event('finished_native_request_output_observed')
        window['stage'] = 'native_finished'

    def note_completion_response(self, response, metadata_id):
        window = self._require('native_finished')
        if response['id'] != metadata_id or window['engine_external_request_id'] != metadata_id + '-0':
            raise ValueError('Observed HTTP Completion metadata does not match stock generate ID')
        if len(response['choices']) != 1 or response['model'] != self.served_model:
            raise ValueError('HTTP response changed the actual native single output')
        choice = response['choices'][0]
        if (token_hash(token_list(choice['prompt_token_ids'], size=2048)) != window['prompt_sha256'] or
                token_list(choice['token_ids'], size=256) != window['final_tokens'] or
                choice['index'] != 0 or choice['finish_reason'] != 'length'):
            raise ValueError('HTTP output token IDs differ from the actual RequestOutput')
        if response['usage'] != {'prompt_tokens': 2048, 'completion_tokens': 256, 'total_tokens': 2304}:
            raise ValueError('HTTP usage does not report exact 2K/256 geometry')
        window['completion_response_id'] = response['id']
        window['stage'] = 'response_ready'
        self.event('stock_completion_response_ready', completion_response_id=response['id'])
        self._write()
        return {'window_id': window['window_id'], 'service_id': self.service_id,
            'phase': window['phase'], 'index': window['index'],
            'engine_external_request_id': window['engine_external_request_id'],
            'core_request_id': window['core_request_id'], 'clock_scope': self.clock,
            'cpu_mock': self.cpu_mock, **false_claims()}

    def note_http_send_begin(self, window_id, status):
        window = self._require('response_ready')
        if window['window_id'] != window_id or status != 200:
            raise ValueError('HTTP output publication is not the prepared successful response')
        self.event('http_output_send_begin')
        window['stage'] = 'http_sending'
        window['body_chunks'] = []

    def note_http_body(self, window_id, body, final):
        window = self._require('http_sending')
        if window['window_id'] != window_id or type(body) is not bytes:
            raise ValueError('HTTP body publication has another active window')
        window['body_chunks'].append(body.hex())
        if final:
            raw = b''.join(bytes.fromhex(chunk) for chunk in window.pop('body_chunks'))
            window['http_body_sha256'] = hashlib.sha256(raw).hexdigest()
            self.event('http_body_sent')
            window['stage'] = 'awaiting_client_end'
            self._write()

    async def finish(self, engine, payload):
        async with self.lock:
            window = self._require('awaiting_client_end')
            required = {'window_id', 'service_id', 'client_clock', 'client_end_ns',
                'client_read_completed', 'http_body_sha256', 'final_tokens_sha256'}
            if type(payload) is not dict or set(payload) != required:
                raise ValueError('Invalid finish envelope')
            if (payload['window_id'] != window['window_id'] or payload['service_id'] != self.service_id or
                    payload['client_clock'] != self.clock or payload['client_read_completed'] is not True or
                    payload['http_body_sha256'] != window['http_body_sha256'] or
                    payload['final_tokens_sha256'] != window['final_tokens_sha256']):
                raise ValueError('Finish is not bound to the exact client-read HTTP response')
            end = payload['client_end_ns']
            publication = next(e['observed_ns'] for e in window['events'] if e['event'] == 'http_output_send_begin')
            if type(end) is not int or end < publication or end > monotonic_ns():
                raise ValueError('Client read/end must follow HTTP publication in the same real clock')
            rpc_payload = {'window_id': window['window_id'],
                'core_request_id': window['core_request_id'], 'client_clock': self.clock,
                'client_end_ns': end, 'client_read_completed': True}
            try:
                window['client_end_ns'] = end
                self.event('finish_rpc_call_after_client_end', client_end_ns=end)
                window['stage'] = 'finish_pending'
                receipts = await engine.collective_rpc(RPC_FINISH, kwargs={'payload': rpc_payload})
                window['rank_receipts'] = copy.deepcopy(receipts)
                if type(receipts) is not list or len(receipts) != 8:
                    raise ValueError('Missing terminal TP8 counter receipts')
                for row in receipts:
                    binding = row.get('api_bridge_binding', {})
                    if (row.get('tag') != window['window_id'] or row.get('counter_sha256') != COUNTER_SHA256 or
                            row.get('audit') is not False or row.get('errors') or
                            row.get('api_client_end_ns') != end or row.get('api_finish_rpc_ns', 0) < end):
                        raise ValueError('Counter receipt changed the real API window')
                    for key in ('window_id', 'http_request_id', 'engine_external_request_id', 'core_request_id', 'prompt_sha256'):
                        if binding.get(key) != window[key]:
                            raise ValueError('Actual rank receipt does not bind the observed API mapping')
                frozen = self.frozen or load_frozen_counter()
                metrics = frozen.summarize(receipts, window['final_tokens'], end, self.clock,
                    draft_graph=self.draft_graph, expected_core_request_id=window['core_request_id'])
                # This bridge is not the source of actual device-consumer or
                # reviewed formal-performance permission, including in real runs.
                metrics.update(false_claims(), cpu_mock=self.cpu_mock,
                    timing_scope='terminal-prefill bookkeeping -> completed private HTTP response read; finish RPC excluded')
                if self.cpu_mock:
                    metrics['target_met'] = False
                window['metrics'] = metrics
                window['stage'] = 'finished'
                self.event('finish_all_tp8_retrieved')
                result = {'window_id': window['window_id'], 'service_id': self.service_id,
                    'phase': window['phase'], 'index': window['index'], 'metrics': metrics,
                    'mapping': {k: window[k] for k in ('http_request_id', 'completion_response_id',
                        'engine_external_request_id', 'core_request_id')},
                    'begin_acks': window['begin_acks'], 'bind_acks': window['bind_acks'],
                    'rank_receipts': receipts, 'events': copy.deepcopy(window['events']),
                    'cpu_mock': self.cpu_mock, **false_claims()}
                self.history.append(copy.deepcopy(window))
                self.completed[window['phase']] += 1
                self.active = None
                self._write()
                return result
            except BaseException as exc:
                self.fail('finish_rpc', exc)
                raise

    def identity(self):
        return {'service_id': self.service_id, 'served_model': self.served_model,
            'clock_scope': self.clock, 'counter_sha256': COUNTER_SHA256,
            'tensor_parallel_size': 8, 'single_active_window': True,
            'warmup_minimum': 2, 'measured_minimum': 8, 'prompt_tokens': 2048,
            'output_tokens': 256, 'audit': False, 'profiler': 'OFF',
            'draft_graph_requested': self.draft_graph, 'cpu_mock': self.cpu_mock,
            'completed': copy.deepcopy(self.completed), 'failed': self.failed,
            **false_claims()}
