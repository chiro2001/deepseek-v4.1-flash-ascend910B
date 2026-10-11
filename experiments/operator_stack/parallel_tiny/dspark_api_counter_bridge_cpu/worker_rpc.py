"""New worker RPC methods around the unchanged frozen v4 counter algorithm.

This mixin has no torch/vLLM import and never synchronizes a device. Terminal
device synchronization and CPU Gloo TP8 joining remain in the original v4 hook.
"""
from __future__ import annotations

import copy

from .runtime_guards import COUNTER_SHA256, clock_scope, load_frozen_counter, monotonic_ns

RPC_BEGIN = 'dspark_api_v1_begin'
RPC_BIND = 'dspark_api_v1_bind'
RPC_FINISH = 'dspark_api_v1_finish'
RPC_DESCRIBE = 'dspark_api_v1_describe'


class ApiCounterWorkerRpcMixin:
    def _api_counter(self):
        frozen = load_frozen_counter()
        counter = self.model_runner._formal_dspark_counter
        if type(counter) is not frozen.RoundCounters or counter.audit:
            raise ValueError('Expected the original audit-OFF v4 RoundCounters instance')
        if not set(('execute_model', '_sample', '_bookkeeping_sync', 'sample_tokens',
                'drafter._propose')).issubset(counter.sources):
            raise ValueError('Actual native counter installation sources are missing')
        return frozen, counter

    def dspark_api_v1_describe(self):
        _, counter = self._api_counter()
        return {'rank': counter.rank, 'clock_scope': clock_scope(),
            'counter_sha256': COUNTER_SHA256, 'audit': False,
            'native_sources': copy.deepcopy(counter.sources),
            'counter_active': counter.tag is not None}

    def dspark_api_v1_begin(self, payload):
        frozen, counter = self._api_counter()
        if payload['client_clock'] != clock_scope():
            raise ValueError('HTTP client and worker monotonic clock namespaces differ')
        if (payload['phase'] not in ('warmup', 'measured') or
                type(payload['index']) is not int or payload['index'] < 0 or
                not isinstance(payload['window_id'], str) or not payload['window_id']):
            raise ValueError('Invalid serial counter window identity')
        if getattr(self, '_dspark_api_binding', None) is not None:
            raise RuntimeError('Previous API binding has not been finished')
        if self.model_runner.requests:
            raise RuntimeError('Cannot begin with a native request still present')
        ack = frozen.begin(self, payload['window_id'], 256)
        self._dspark_api_binding = copy.deepcopy(payload)
        return {**ack, 'clock_scope': clock_scope(), 'counter_sha256': COUNTER_SHA256,
            'phase': payload['phase'], 'index': payload['index'], 'stage': 'begin',
            'ack_ns': monotonic_ns(), 'audit': False}

    def dspark_api_v1_bind(self, payload):
        _, counter = self._api_counter()
        current = getattr(self, '_dspark_api_binding', None)
        if (current is None or current['window_id'] != payload['window_id'] or
                counter.tag != payload['window_id'] or counter.requests or
                counter.pending is not None or self.model_runner.execute_model_state is not None):
            raise RuntimeError('Mapping must bind before the native request executes')
        if 'core_request_id' in current:
            raise ValueError('Duplicate or multiple core request mapping')
        for key in ('core_request_id', 'engine_external_request_id'):
            if type(payload[key]) is not str or not payload[key]:
                raise ValueError('Actual request identity is unavailable')
        if payload['http_request_id'] != current['http_request_id']:
            raise ValueError('HTTP identity changed before core submission')
        if payload['prompt_sha256'] != current['prompt_sha256']:
            raise ValueError('Rendered prompt differs from the HTTP token IDs')
        current.update(copy.deepcopy(payload))
        return {'rank': counter.rank, 'window_id': counter.tag, 'stage': 'bind',
            'http_request_id': current['http_request_id'],
            'engine_external_request_id': current['engine_external_request_id'],
            'core_request_id': current['core_request_id'],
            'counter_sha256': COUNTER_SHA256, 'clock_scope': clock_scope(),
            'ack_ns': monotonic_ns(), 'before_any_native_execution': True}

    def dspark_api_v1_finish(self, payload):
        frozen, counter = self._api_counter()
        current = getattr(self, '_dspark_api_binding', None)
        if (current is None or current.get('core_request_id') is None or
                payload['window_id'] != current['window_id'] or
                payload['core_request_id'] != current['core_request_id']):
            raise ValueError('Finish does not identify the bound actual core request')
        if payload['client_clock'] != clock_scope():
            raise ValueError('Finish client and worker clock namespaces differ')
        if (type(payload['client_end_ns']) is not int or payload['client_end_ns'] <= 0 or
                payload['client_end_ns'] > monotonic_ns() or
                payload.get('client_read_completed') is not True):
            raise ValueError('Finish must follow the real client response read/end clock')
        if counter.tag != current['window_id'] or set(counter.requests) != {current['core_request_id']}:
            raise ValueError('The counter contains another request or no mapped request')
        receipt = frozen.finish(self)
        receipt['api_bridge_binding'] = copy.deepcopy(current)
        receipt['api_finish_rpc_ns'] = monotonic_ns()
        receipt['api_client_end_ns'] = payload['client_end_ns']
        receipt['counter_sha256'] = COUNTER_SHA256
        self._dspark_api_binding = None
        return receipt
