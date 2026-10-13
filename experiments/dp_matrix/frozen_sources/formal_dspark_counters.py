"""CPU bookkeeping for real DSpark target/verifier rounds; no tensor copies.

This module imports no torch/vLLM. Runtime adapters consume the CPU lists that
the stock synchronous runner already materializes. It grants no device audit.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import inspect
import math
import os
from pathlib import Path
import time


def clock_scope():
    return {'clock': 'CLOCK_MONOTONIC',
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'time_namespace': os.readlink('/proc/self/ns/time')}


def source_receipt(fn):
    fn = inspect.unwrap(fn)
    path = Path(inspect.getfile(fn))
    return {'path': str(path), 'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'function': fn.__qualname__,
            'function_sha256': hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()}


def tensor_metadata(value):
    """Host metadata only; deliberately never calls cpu/item/tolist/numpy."""
    if not all(hasattr(value, key) for key in ('shape', 'stride', 'data_ptr')):
        return None
    return {'shape': list(value.shape), 'stride': list(value.stride()),
            'ptr': int(value.data_ptr()), 'dtype': str(value.dtype), 'device': str(value.device)}


def _token_list(tokens):
    if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('Expected existing CPU list[int], never device output or placeholders')
    return list(tokens)


class RoundCounters:
    def __init__(self, rank, *, audit=False, now=time.monotonic_ns):
        self.rank = rank
        self.audit = audit
        self.now = now
        self.tag = None
        self.pending = None
        self.requests = {}
        self.sources = {}
        self.errors = []
        self.max_tokens = None
        self.zero_token_executions = []
        # [async arm] env-gated (DSPARK_ASYNC_COUNTER). The stock async runner
        # never materialises per-round CPU token IDs; this arm counts real round
        # boundaries and fences the terminal submitted work instead.
        self.async_mode = False
        self.terminal_fence = None

    def begin(self, tag, max_tokens=None):
        if self.pending is not None or self.tag is not None:
            raise RuntimeError('Previous counter window is still active')
        self.tag = str(tag)
        # Every counted request must observe its own terminal fence; a stale
        # fence from the previous request would mis-date the async window.
        self.terminal_fence = None
        host_spans = getattr(self, 'host_span_state', None)
        if host_spans is not None:
            import host_span_probe
            host_span_probe.reset(host_spans)
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 2):
            raise ValueError('Exact output budget must include tokens after the seed')
        self.max_tokens = max_tokens
        self.zero_token_executions = []
        self.requests = {}
        self.errors = []

    def execute_begin(self, scheduler, runner_requests):
        if self.tag is None:
            return
        if self.pending is not None:
            raise RuntimeError('execute_model without previous sample completion')
        new = {r.req_id: r for r in scheduler.scheduled_new_reqs}
        cached = scheduler.scheduled_cached_reqs
        if len(cached.req_ids) != len(cached.num_computed_tokens):
            raise ValueError('Unknown cached-request metadata layout')
        if not scheduler.num_scheduled_tokens:
            # The stock runner updates/frees request state and returns EMPTY
            # before any target, sampler or draft execution for this ABI.
            # This commonly occurs after the final request has completed.
            if (type(scheduler.total_num_scheduled_tokens) is not int or
                    scheduler.total_num_scheduled_tokens != 0 or new or cached.req_ids or
                    scheduler.scheduled_spec_decode_tokens or
                    (getattr(scheduler, 'num_invalid_spec_tokens', None) or {}) or
                    (getattr(scheduler, 'scheduled_encoder_inputs', None) or {})):
                raise ValueError('Inconsistent zero-token cleanup scheduler')
            finished = sorted(scheduler.finished_req_ids)
            if any(type(req_id) is not str for req_id in finished):
                raise ValueError('Unknown cleanup request ID ABI')
            self.zero_token_executions.append({'finished_req_ids': finished,
                'total_num_scheduled_tokens': 0, 'execute_start_ns': self.now(),
                'counted_as_decode_round': False})
            return False
        computed = dict(zip(cached.req_ids, cached.num_computed_tokens))
        pending = {}
        for req_id, scheduled in scheduler.num_scheduled_tokens.items():
            if type(scheduled) is not int or scheduled <= 0:
                raise ValueError('Round requires a positive actual scheduled token count')
            state = new.get(req_id) or runner_requests.get(req_id)
            if state is None or state.prompt_token_ids is None:
                raise ValueError('Missing CPU prompt metadata; embedded-only input is unsupported')
            prompt_len = len(state.prompt_token_ids)
            done = state.num_computed_tokens if req_id in new else computed.get(req_id)
            if type(done) is not int or done < 0 or prompt_len <= 0:
                raise ValueError('Missing exact scheduled num_computed_tokens')
            request = self.requests.setdefault(req_id, {
                'prompt_tokens': prompt_len, 'seed_token_ids': [], 'decode_start_ns': None,
                'prefill_calls': 0, 'rounds': [], 'initial_draft': [],
                'prefill_sample_end_ns': None})
            if request['prompt_tokens'] != prompt_len:
                raise ValueError('Prompt binding changed within counter window')
            phase = 'prefill' if done < prompt_len else 'decode'
            raw_drafted = list(scheduler.scheduled_spec_decode_tokens.get(req_id, []))
            if self.async_mode:
                # Async scheduling pads the spec-token stream with -1 placeholders
                # (the worker substitutes the real previous-step drafts). Keep the
                # raw stream for TP-identity checks and record the placeholder
                # count explicitly instead of rejecting the round.
                if any(type(token) is not int for token in raw_drafted):
                    raise ValueError('Async scheduled spec tokens must be existing CPU ints')
                placeholders = sum(1 for token in raw_drafted if token < 0)
                drafted = list(raw_drafted)
                scheduled_draft_ids = raw_drafted
            else:
                placeholders = 0
                drafted = _token_list(raw_drafted)
                scheduled_draft_ids = drafted
            invalid = int((getattr(scheduler, 'num_invalid_spec_tokens', None) or {}).get(req_id, 0))
            if invalid < 0 or invalid > len(drafted):
                raise ValueError('Invalid draft accounting')
            if phase == 'decode' and request['decode_start_ns'] is None:
                boundary = request.get('last_bookkeeping_end_ns')
                if boundary is None:
                    raise ValueError('Decode round lacks the terminal-prefill bookkeeping boundary')
                request['decode_start_ns'] = boundary
                request['decode_start_source'] = 'bookkeeping immediately before the first decode round'
            pending[req_id] = {'phase': phase, 'computed_before': done,
                'scheduled_tokens': scheduled, 'scheduled_draft_token_ids': scheduled_draft_ids,
                'placeholder_draft_tokens': placeholders,
                'verified_draft_tokens': len(drafted)-invalid, 'sampler_called': False,
                'rejection_sampler_called': False, 'token_ids': None, 'draft_calls': [],
                'execute_start_ns': self.now()}
        if len(pending) != 1:
            raise ValueError('Formal single-request counter requires exactly one active request')
        self.pending = pending
        return True

    def sampler_finished(self, has_spec_metadata):
        if self.tag is None:
            return
        if self.pending is None:
            raise RuntimeError('Sampler without a real execute_model round')
        for row in self.pending.values():
            if row['sampler_called']:
                raise RuntimeError('Actual round sampler was counted twice')
            row['sampler_called'] = True
            row['rejection_sampler_called'] = bool(has_spec_metadata)

    def bookkeeping_finished(self, output):
        if self.tag is None:
            return
        if self.pending is None or not isinstance(output, tuple) or len(output) != 6:
            raise ValueError('Unknown synchronous bookkeeping return ABI')
        tokens, req_ids = output[1], output[3]
        if not isinstance(tokens, list) or not isinstance(req_ids, list):
            raise ValueError('CPU valid_sampled_token_ids/req_ids ABI is unavailable')
        now = self.now()  # Stock prefill target/bookkeeping has completed; draft follows.
        if self.async_mode:
            # Async scheduling keeps sampled tokens on the device; the stock
            # bookkeeping call returns an EMPTY valid-sampled-token list. Record
            # the real round boundary only; per-round CPU token materialisation
            # stays a synchronous-arm property and is reported as such.
            if set(req_ids) != set(self.pending) or len(req_ids) != len(self.pending):
                raise ValueError('Async bookkeeping request set differs from the scheduled round')
            if any(tokens):
                raise ValueError('Async bookkeeping unexpectedly materialised CPU tokens')
            for req_id, row in self.pending.items():
                if row.get('bookkeeping_end_ns') is not None:
                    raise RuntimeError('Actual round bookkeeping was counted twice')
                row['bookkeeping_end_ns'] = now
                request = self.requests[req_id]
                request['last_bookkeeping_end_ns'] = now
                row['placeholder_draft_tokens'] = row.get('placeholder_draft_tokens', 0)
            return
        if len(tokens) != len(req_ids):
            raise ValueError('CPU valid_sampled_token_ids/req_ids ABI is unavailable')
        by_id = {req_id: _token_list(row) for req_id, row in zip(req_ids, tokens)}
        for req_id, row in self.pending.items():
            if req_id not in by_id:
                raise ValueError('Scheduled request missing from bookkeeping')
            if row['token_ids'] is not None:
                raise RuntimeError('Actual round bookkeeping was counted twice')
            row['token_ids'] = by_id[req_id]
            row['bookkeeping_end_ns'] = now
            if row['phase'] == 'prefill' and row['token_ids']:
                request = self.requests[req_id]
                if request['seed_token_ids'] or len(row['token_ids']) != 1:
                    raise ValueError('Expected one seed token from terminal prefill')
                request['seed_token_ids'] = row['token_ids']
                request['decode_start_ns'] = now

    def draft_finished(self, *, k, output_shape, runtime_graph, method):
        if self.tag is None:
            return
        if self.pending is None or method != 'dspark' or type(k) is not int or k <= 0:
            raise ValueError('Actual DSpark proposal ABI is unavailable')
        shape = list(output_shape)
        if shape != [1, k]:
            raise ValueError(f'Unexpected real DSpark proposal shape: {shape}, K={k}')
        for row in self.pending.values():
            row['draft_calls'].append({'k': k, 'output_shape': shape,
                                       'runtime_graph': bool(runtime_graph), 'method': method})

    def terminal_round_pending(self):
        if self.async_mode:
            # Async arm: per-round CPU tokens are not materialised, so the
            # synchronous token-budget prediction cannot run; the terminal
            # fence is taken on the stock cleanup execution instead.
            return False
        if self.tag is None or self.pending is None or self.max_tokens is None:
            return False
        for req_id,row in self.pending.items():
            request=self.requests[req_id]
            raw=len(request['seed_token_ids'])+sum(len(r['token_ids']) for r in request['rounds'])
            if row['phase']=='decode' and row['token_ids'] is not None:
                raw+=len(row['token_ids'])
            if raw>=self.max_tokens:return True
        return False

    def post_client_device_fence(self, phase='post_client', sync=None):
        """Async arm: fence every submitted round, including the terminal draft.

        Called from the stock cleanup execution when the engine reports our own
        finished request, or as a bounded fallback just before the counter
        receipt is read. Marks the last recorded round as device-fenced so the
        terminal Gloo join keeps its synchronous-arm semantics.
        """
        if sync is None:
            import torch
            sync = torch.npu.synchronize
        entered = self.now()
        sync()
        self.terminal_fence = {'phase': phase, 'entered_ns': entered,
                               'sync_returned_ns': self.now(), 'rank': self.rank}
        for request in self.requests.values():
            if request['rounds']:
                request['rounds'][-1]['terminal_device_synced'] = True
        return self.terminal_fence

    def zero_token_cleanup_fence(self, cleanup, sync=None):
        """Async arm: fence on the stock cleanup execution of our own request."""
        if not self.async_mode or self.terminal_fence is not None:
            return False
        if not self.requests or any(not request['rounds'] for request in self.requests.values()):
            return False
        finished = set(cleanup.get('finished_req_ids') or [])
        if finished != set(self.requests):
            return False
        self.post_client_device_fence('cleanup_execute', sync=sync)
        return True

    def ensure_terminal_join(self):
        if not self.async_mode:
            return None
        for request in self.requests.values():
            if not request['rounds']:
                continue
            row = request['rounds'][-1]
            if not row.get('terminal_device_synced') or 'terminal_tp_join' in row:
                continue
            from vllm.distributed import get_tp_group
            import torch.distributed as distributed
            terminal_tp_completion_join(self, get_tp_group(), distributed)
        return None

    def sample_finished(self, *, route_receipt=None, terminal_device_synced=False):
        if self.tag is None:
            return
        if self.pending is None:
            raise RuntimeError('sample_tokens without an execution')
        now = self.now()
        for req_id, row in self.pending.items():
            request = self.requests[req_id]
            if self.async_mode:
                if row.get('bookkeeping_end_ns') is None or not row['sampler_called']:
                    raise ValueError('Actual sampler/bookkeeping did not finish')
            elif row['token_ids'] is None or not row['sampler_called']:
                raise ValueError('Actual sampler/bookkeeping did not finish')
            row['sample_end_ns'] = now
            row['terminal_device_synced'] = bool(terminal_device_synced)
            if route_receipt is not None:
                row['main_route_receipt'] = route_receipt
            if row['phase'] == 'prefill':
                request['prefill_calls'] += 1
                if self.async_mode:
                    if row['draft_calls']:
                        request['initial_draft'] = row['draft_calls']
                        request['prefill_sample_end_ns'] = now
                elif row['token_ids']:
                    request['initial_draft'] = row['draft_calls']
                    request['prefill_sample_end_ns'] = now
            else:
                if self.async_mode:
                    # Per-round accepted counts are not materialised by the
                    # stock async bookkeeping ABI; keep them explicitly None.
                    row['accepted_draft_tokens_raw'] = None
                else:
                    if not request['seed_token_ids'] or not row['token_ids']:
                        raise ValueError('Decode must follow a real seed and emit CPU tokens')
                    if row['verified_draft_tokens'] and not row['rejection_sampler_called']:
                        raise ValueError('Scheduled drafts did not reach the rejection sampler')
                    accepted = max(len(row['token_ids'])-1, 0)
                    if accepted > row['verified_draft_tokens']:
                        raise ValueError('Accepted token count exceeds actual verified drafts')
                    row['accepted_draft_tokens_raw'] = accepted
                row['round'] = len(request['rounds'])+1
                request['rounds'].append(row)
        self.pending = None

    def finish(self):
        if self.tag is None or self.pending is not None:
            raise RuntimeError('Counter window inactive or a round remains incomplete')
        host_spans = getattr(self, 'host_span_state', None)
        host_receipt = None
        if host_spans is not None:
            import host_span_probe
            host_receipt = host_span_probe.snapshot(host_spans)
        result = {'rank': self.rank, 'tag': self.tag, 'clock_scope': clock_scope(),
                  'requests': copy.deepcopy(self.requests), 'sources': self.sources,
                  'errors': list(self.errors), 'audit': self.audit,
                  'extra_device_copies': 0, 'capture_profile_counted': False,
                  'host_spans': host_receipt,
                  'counter_scope': ('actual async-scheduled target/sampler/bookkeeping/draft round boundaries'
                                    if self.async_mode else
                                    'actual synchronous target/sampler/bookkeeping/draft calls'),
                  'async_mode': self.async_mode,
                  'terminal_fence': None if self.terminal_fence is None else dict(self.terminal_fence),
                  'actual_device_consumer_proof': False,
                  'zero_token_executions': copy.deepcopy(self.zero_token_executions)}
        self.tag = None
        return result


def summarize(receipts, final_tokens, client_end_ns, client_clock, *, draft_graph, expected_core_request_id=None):
    final_tokens = _token_list(final_tokens)
    if len(receipts) != 8 or {r['rank'] for r in receipts} != set(range(8)):
        raise ValueError('Expected all eight TP ranks')
    if len(final_tokens) < 2:
        raise ValueError('Decode window needs tokens after the prefill seed')
    rows = []
    common = None
    starts = []
    fence_ns = []
    fence_phases = []
    for receipt in receipts:
        if receipt['clock_scope'] != client_clock:
            raise ValueError('Worker/client monotonic clock namespaces differ')
        if receipt['errors'] or len(receipt['requests']) != 1:
            raise ValueError('Incomplete or ambiguous per-request counters')
        if expected_core_request_id is not None:
            if (type(expected_core_request_id) is not str or
                    set(receipt['requests']) != {expected_core_request_id}):
                raise ValueError('Worker counters do not match actual core request ID')
            for cleanup in receipt.get('zero_token_executions', []):
                if not set(cleanup['finished_req_ids']).issubset({expected_core_request_id}):
                    raise ValueError('Cleanup counters include a different core request ID')
                if cleanup.get('execute_end_ns') is None:
                    raise ValueError('Zero-token cleanup did not return stock empty output')
        request = next(iter(receipt['requests'].values()))
        rounds = request['rounds']
        if receipt.get('async_mode'):
            # Async arm: real round boundaries + terminal fence, no per-round
            # CPU tokens (the stock async bookkeeping ABI keeps them on device).
            if not rounds:
                raise ValueError('Async arm recorded no decode round')
            for row in rounds:
                if row.get('bookkeeping_end_ns') is None or not row.get('sampler_called'):
                    raise ValueError('Async decode round lacks sampler/bookkeeping boundary')
            identity = [(r['scheduled_tokens'], r['scheduled_draft_token_ids'],
                         [{k:c[k] for k in ('k','output_shape','runtime_graph','method')} for c in r['draft_calls']])
                        for r in rounds]
            if common is None:
                common = identity
            elif identity != common:
                raise ValueError('TP ranks disagree on actual round/proposal coverage')
            if request['decode_start_ns'] is None:
                raise ValueError('Missing terminal-prefill decode boundary')
            if client_end_ns <= request['decode_start_ns']:
                raise ValueError('Client completion precedes the decode window opening')
            fence = receipt.get('terminal_fence')
            if not fence or fence.get('sync_returned_ns') is None:
                raise ValueError('Async terminal device fence was not observed')
            if fence['sync_returned_ns'] < rounds[-1]['sample_end_ns']:
                raise ValueError('Async terminal fence precedes the final sample boundary')
            join = rounds[-1].get('terminal_tp_join')
            if (not join or join.get('backend') != 'gloo' or join.get('world_size') != 8 or
                    join.get('rank') != receipt['rank'] or
                    not join.get('device_fence_preceded_entry') or
                    not join.get('sample_timestamp_preceded_entry') or
                    not rounds[-1]['sample_end_ns'] <= join['entered_ns'] <= join['returned_ns']):
                raise ValueError('Missing actual all-TP terminal completion join')
            proposals = request['initial_draft'] + [c for r in rounds for c in r['draft_calls']]
            if not proposals or any(not r['verified_draft_tokens'] for r in rounds):
                raise ValueError('DSpark was not actually proposed and verified in every decode round')
            steady = [c for row in rounds for c in row['draft_calls']]
            if any(not row['draft_calls'] for row in rounds):
                raise ValueError('A real decode round did not run its DSpark proposal')
            if draft_graph and any(not p['runtime_graph'] for p in steady):
                raise ValueError('Draft graph requested but actual steady DSpark proposal used eager')
            starts.append(request['decode_start_ns'])
            fence_ns.append(fence['sync_returned_ns'])
            fence_phases.append(fence.get('phase'))
            post_client = [r for r in rounds if r['sample_end_ns'] > client_end_ns]
            rows.append({'rank': receipt['rank'], 'rounds': len(rounds),
                         'draft_calls': len(proposals), 'proposed_tokens': sum(p['k'] for p in proposals),
                         'verified_draft_tokens': sum(r['verified_draft_tokens'] for r in rounds),
                         'accepted_draft_tokens_raw': None, 'accepted_draft_tokens_used': None,
                         'raw_decode_tokens': None, 'terminal_tokens_trimmed': None,
                         'async_per_round_tokens_materialized': False,
                         'post_client_rounds': len(post_client)})
            continue
        raw = request['seed_token_ids'] + [t for row in rounds for t in row['token_ids']]
        if not rounds or raw[:len(final_tokens)] != final_tokens or len(raw) < len(final_tokens):
            raise ValueError('Real round tokens do not match the final request output')
        # Only terminal scheduler trimming is allowed; intermediate rounds must
        # be represented completely in the accepted output sequence.
        if len(raw)-len(final_tokens) >= len(rounds[-1]['token_ids']):
            raise ValueError('Output truncation spans more than the final round')
        identity = [(r['scheduled_tokens'], r['scheduled_draft_token_ids'], r['token_ids'],
                     r['rejection_sampler_called'], [{k:c[k] for k in ('k','output_shape','runtime_graph','method')} for c in r['draft_calls']]) for r in rounds]
        if common is None:
            common = identity
        elif identity != common:
            raise ValueError('TP ranks disagree on actual round/proposal/acceptance coverage')
        if request['seed_token_ids'] != final_tokens[:1] or request['decode_start_ns'] is None:
            raise ValueError('Missing terminal-prefill seed boundary')
        if client_end_ns < rounds[-1]['sample_end_ns']:
            raise ValueError('Client completion precedes final worker sample completion')
        if not rounds[-1].get('terminal_device_synced'):
            raise ValueError('Final native draft/target work has no completion fence')
        join = rounds[-1].get('terminal_tp_join')
        if (not join or join.get('backend') != 'gloo' or join.get('world_size') != 8 or
                join.get('rank') != receipt['rank'] or
                not join.get('device_fence_preceded_entry') or
                not join.get('sample_timestamp_preceded_entry') or
                not rounds[-1]['sample_end_ns'] <= join['entered_ns'] <= join['returned_ns']):
            raise ValueError('Missing actual all-TP terminal completion join')
        proposals = request['initial_draft'] + [c for r in rounds for c in r['draft_calls']]
        if not proposals or any(not r['verified_draft_tokens'] for r in rounds):
            raise ValueError('DSpark was not actually proposed and verified in every decode round')
        steady=[c for row in rounds for c in row['draft_calls']]
        if any(not row['draft_calls'] for row in rounds):
            raise ValueError('A real decode round did not run its DSpark proposal')
        if draft_graph and any(not p['runtime_graph'] for p in steady):
            raise ValueError('Draft graph requested but actual steady DSpark proposal used eager')
        starts.append(request['decode_start_ns'])
        remaining = len(final_tokens)-1
        accepted_used = 0
        for row in rounds:
            used = min(remaining, len(row['token_ids']))
            accepted_used += min(used, row['accepted_draft_tokens_raw'])
            remaining -= used
        rows.append({'rank': receipt['rank'], 'rounds': len(rounds),
                     'draft_calls': len(proposals), 'proposed_tokens': sum(p['k'] for p in proposals),
                     'verified_draft_tokens': sum(r['verified_draft_tokens'] for r in rounds),
                     'accepted_draft_tokens_raw': sum(r['accepted_draft_tokens_raw'] for r in rounds),
                     'accepted_draft_tokens_used': accepted_used,
                     'raw_decode_tokens': len(raw)-1,
                     'terminal_tokens_trimmed': len(raw)-len(final_tokens)})
    async_arm = bool(receipts[0].get('async_mode'))
    if async_arm and not fence_ns:
        raise ValueError('Async arm has no terminal fence across ranks')
    window_end = max([client_end_ns] + fence_ns) if async_arm else client_end_ns
    if async_arm and window_end - client_end_ns > 50_000_000:
        raise ValueError('Async terminal fence lands more than 50 ms after the client clock')
    elapsed = (window_end-min(starts))/1e9
    count = rows[0]['rounds']
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('Invalid decode wall clock')
    decoded = len(final_tokens)-1
    metrics = {'decoded_tokens': decoded, 'rounds': count, 'decode_elapsed_s': elapsed,
               'A': decoded/count, 'ms_per_step': 1000*elapsed/count,
               'tokens_per_second': decoded/elapsed}
    if not math.isclose(metrics['ms_per_step'], 1000*metrics['A']/metrics['tokens_per_second'], rel_tol=1e-12):
        raise ValueError('A/ms/tok-s triple is inconsistent')
    metrics.update(emitted_tokens_including_prefill_seed=len(final_tokens), prefill_seed_tokens=1,
                   clock_scope=client_clock, decode_start_ns=min(starts), client_end_ns=client_end_ns,
                   rank_counters=rows, actual_device_consumer_proof=False,
                   core_request_id=expected_core_request_id,
                   exact_core_request_id_bound=expected_core_request_id is not None,
                   acceptance_zero_diagnostic_required=(False if async_arm else rows[0]['accepted_draft_tokens_used']==0),
                   timing_scope=('terminal-prefill decode boundary -> max(client RequestOutput clock, terminal device fence); includes draft/target/verifier/cache/scheduling and the terminal submitted draft completion'
                                 if async_arm else
                                 'terminal-prefill CPU bookkeeping before initial draft -> final client RequestOutput; includes draft/target/verifier/cache/scheduling'),
                   target_met=metrics['ms_per_step']<=20,
                   performance_receipt_granted=False)
    if async_arm:
        metrics.update(async_scheduling=True, async_per_round_tokens_available=False,
                       decode_window_end_ns=window_end,
                       decode_window_end_scope='max(client RequestOutput clock, terminal device fence)',
                       terminal_fence_delta_ms=(window_end-client_end_ns)/1e6,
                       terminal_fence_phases=fence_phases,
                       post_client_rounds=rows[0]['post_client_rounds'],
                       rounds_scope='all actually executed target/verifier decode rounds, including any round that finished after the client clock')
    metrics['terminal_device_completion_observed']=True
    metrics['terminal_tp_completion_join_observed']=True
    metrics['terminal_completion_scope']='each rank device fence + sample timestamp -> CPU Gloo TP8 join -> client output; included in wall clock'
    return metrics


def _cpu_route_receipt(output, expected_tokens):
    exported = getattr(output, 'routed_experts', None)
    if exported is None:
        raise ValueError('Actual target route export is missing')
    routes = exported.routing_data  # Stock runner already produced a numpy view.
    import numpy as np
    if not isinstance(routes, np.ndarray) or routes.shape != (expected_tokens, 40, 6):
        raise ValueError(f'Unexpected full40 target route layout: {getattr(routes, "shape", None)}')
    valid = (np.issubdtype(routes.dtype, np.integer) and
             bool(np.all((routes >= 0) & (routes < 384))) and
             bool(np.all(np.diff(np.sort(routes, axis=-1), axis=-1) > 0)))
    if not valid:
        raise ValueError('Actual route export violates full40/384/top6')
    return {'shape': list(routes.shape), 'dtype': str(routes.dtype), 'all_layers_present': True,
            'sha256': hashlib.sha256(routes.tobytes()).hexdigest(),
            'scope': 'stock target/verifier scheduled-token CPU route export; no draft route proof'}


def terminal_tp_completion_join(counter, tp_group, distributed):
    """Join already completed device work before publishing the terminal output.

    The sample timestamp precedes entry on every rank. A CPU barrier therefore
    guarantees that the output rank cannot publish before any rank's fence and
    timestamp. Per-rank barrier returns need not occur simultaneously.
    """
    if (tp_group.world_size != 8 or tp_group.rank_in_group != counter.rank or
            str(distributed.get_backend(tp_group.cpu_group)) != 'gloo'):
        raise ValueError('Terminal completion requires the actual eight-rank CPU Gloo TP group')
    if counter.pending is not None or len(counter.requests) != 1:
        raise ValueError('Terminal join requires one completed real request round')
    request = next(iter(counter.requests.values()))
    row = request['rounds'][-1]
    if not row.get('terminal_device_synced') or 'terminal_tp_join' in row:
        raise ValueError('Terminal join missing a device fence or counted twice')
    entered = counter.now()
    tp_group.barrier()  # Stock GroupCoordinator.barrier uses only cpu_group.
    row['terminal_tp_join'] = {'backend': 'gloo', 'world_size': 8,
        'rank': counter.rank, 'entered_ns': entered, 'returned_ns': counter.now(),
        'device_fence_preceded_entry': True, 'sample_timestamp_preceded_entry': True}


def install_runner(runner, rank, *, audit):
    """Patch only the new job's live instance; never edit existing SDK files."""
    async_arm = bool(getattr(runner, 'use_async_scheduling', True))
    if async_arm and os.environ.get('DSPARK_ASYNC_COUNTER', '0') != '1':
        raise ValueError('Async output placeholder ABI is outside this counter receipt')
    spec = runner.vllm_config.speculative_config
    if spec is None or spec.method != 'dspark' or runner.drafter is None:
        raise ValueError('DSpark must be configured and loaded')
    counter = RoundCounters(rank, audit=audit)
    if async_arm:
        counter.async_mode = True
    methods = ['execute_model', '_sample', '_bookkeeping_sync', 'sample_tokens']
    originals = {name: getattr(runner, name) for name in methods}
    counter.sources = {name: source_receipt(fn) for name, fn in originals.items()}
    drafter = runner.drafter
    draft_original = drafter._propose
    counter.sources['drafter._propose'] = source_receipt(draft_original)
    from parallel_tiny.dspark_formal_audit.stock_graph_observer import install as install_graph_observer
    graph_observer=install_graph_observer(counter,runner,draft_original)
    counter.actual_draft_graph_observer=graph_observer

    @functools.wraps(originals['execute_model'])
    def execute(scheduler_output, *args, **kwargs):
        active = counter.tag is not None
        counted = counter.execute_begin(scheduler_output, runner.requests)
        result = originals['execute_model'](scheduler_output, *args, **kwargs)
        if active and not counted:
            # A zero-token return must not have a pending sample or output
            # tokens. Check only the stock CPU ABI, never a device tensor.
            if counter.async_mode:
                counter.zero_token_executions[-1]['async_empty_output_check_skipped'] = True
                if counter.zero_token_cleanup_fence(counter.zero_token_executions[-1]):
                    counter.ensure_terminal_join()
            elif (result is None or runner.execute_model_state is not None or
                    getattr(result, 'req_ids', None) != [] or
                    getattr(result, 'sampled_token_ids', None) != []):
                raise ValueError('Zero-token cleanup did not return stock empty output')
            counter.zero_token_executions[-1]['execute_end_ns'] = counter.now()
        return result

    @functools.wraps(originals['_sample'])
    def sample(logits, spec_decode_metadata):
        result = originals['_sample'](logits, spec_decode_metadata)
        counter.sampler_finished(spec_decode_metadata is not None)
        return result

    @functools.wraps(originals['_bookkeeping_sync'])
    def book(*args, **kwargs):
        result = originals['_bookkeeping_sync'](*args, **kwargs)
        counter.bookkeeping_finished(result)
        return result

    @functools.wraps(draft_original)
    def draft(*args, **kwargs):
        active=counter.tag is not None
        if active:graph_observer.begin_proposal()
        try:
            result = draft_original(*args, **kwargs)
        except BaseException:
            if active:graph_observer.cancel_proposal()
            raise
        if active:
            evidence=graph_observer.finish_proposal()
            k = kwargs.get('num_speculative_tokens', args[0] if args else None)
            counter.draft_finished(k=k, output_shape=result.shape,
                runtime_graph=evidence['actual_graph_submission'],method=drafter.method)
            for row in counter.pending.values():
                row['draft_calls'][-1]['actual_execution_evidence']=evidence
        return result

    @functools.wraps(originals['sample_tokens'])
    def finish_sample(*args, **kwargs):
        result = originals['sample_tokens'](*args, **kwargs)
        terminal=counter.terminal_round_pending()
        if terminal:
            # Once per request, strictly outside capture. The native terminal
            # draft may still be asynchronous after sample_tokens returns.
            # Complete it before the client timestamp, not in the later RPC.
            import torch
            torch.npu.synchronize()
        route = None
        if counter.tag is not None and audit:
            if counter.pending is None:
                raise RuntimeError('Route receipt without pending real execution')
            route = _cpu_route_receipt(result, sum(r['scheduled_tokens'] for r in counter.pending.values()))
        counter.sample_finished(route_receipt=route,terminal_device_synced=terminal)
        if terminal:
            from vllm.distributed import get_tp_group
            import torch.distributed as distributed
            terminal_tp_completion_join(counter,get_tp_group(),distributed)
        return result

    runner.execute_model = execute
    runner._sample = sample
    runner._bookkeeping_sync = book
    runner.sample_tokens = finish_sample
    drafter._propose = draft
    runner._formal_dspark_counter = counter
    # Additive host-span probe (env-gated). Installed after the counter wrappers
    # so its spans enclose exactly what the counters observe; it performs no
    # tensor reads and no device synchronisation.
    import os as _os
    if _os.environ.get('DSPARK_HOST_SPANS', '0') == '1':
        import host_span_probe as _host_span_probe
        state = _host_span_probe.install(runner, rank, counter)
        counter.host_span_state = state
        _host_span_probe.reset(state)
    return {'installed': True, 'rank': rank, 'sources': counter.sources,
            'extra_device_copies': 0, 'actual_device_consumer_proof': False}


def begin(worker, tag, max_tokens):
    runner = worker.model_runner
    if runner.execute_model_state is not None:
        raise RuntimeError('Cannot begin counters while the runner has a pending sample')
    runner._formal_dspark_counter.begin(tag,max_tokens)
    return {'rank': runner._formal_dspark_counter.rank, 'tag': tag}


def finish(worker):
    counter = worker.model_runner._formal_dspark_counter
    if counter.async_mode:
        if counter.pending is not None:
            raise RuntimeError('Counter window inactive or a round remains incomplete')
        if counter.terminal_fence is None:
            if not counter.requests or any(not r['rounds'] for r in counter.requests.values()):
                raise RuntimeError('Async counter has no terminal round to fence')
            counter.post_client_device_fence('post_client')
        counter.ensure_terminal_join()
    return counter.finish()


def bank(worker):
    runner = worker.model_runner
    counter = runner._formal_dspark_counter
    drafter = runner.drafter
    target=runner._formal_dspark_native_target
    main=runner._formal_dspark_native_backbone
    model=runner._formal_dspark_native_draft_model
    draft=runner._formal_dspark_native_draft_backbone
    config = runner.vllm_config.model_config.hf_text_config
    draft_layers=draft.layers
    caches={}
    # Read the current post-initialization tensors from the registered native
    # cache objects. Metadata only; no replay/capture callback or device copy.
    for name,layer in runner.compilation_config.static_forward_context.items():
        cache=getattr(layer,'kv_cache',None)
        values=cache if isinstance(cache,(list,tuple)) else [cache]
        metadata=[tensor_metadata(value) for value in values if value is not None]
        metadata=[value for value in metadata if value is not None]
        if metadata:caches[name]=metadata
    return {'rank': counter.rank, 'sources': counter.sources, 'clock_scope': clock_scope(),
            'main_model_class': type(target).__module__+'.'+type(target).__name__,
            'main_backbone_class':type(main).__module__+'.'+type(main).__name__,
            'current_main_wrapper_class':type(runner.model).__module__+'.'+type(runner.model).__name__,
            'drafter_class': type(drafter).__module__+'.'+type(drafter).__name__,
            'draft_model_class': type(model).__module__+'.'+type(model).__name__,
            'current_draft_wrapper_class':type(drafter.model).__module__+'.'+type(drafter.model).__name__,
            'method': runner.vllm_config.speculative_config.method,
            'num_speculative_tokens': runner.vllm_config.speculative_config.num_speculative_tokens,
            'main_layers':len(main.layers), 'routed_experts': int(config.n_routed_experts),
            'experts_per_token': int(config.num_experts_per_tok),
            'engram_layers': list(config.engram_layer_ids),
            'draft_layer_ids': sorted(int(k) for k in draft_layers),
            'actual_draft_target_layer_ids':list(draft.target_layer_ids),
            'native_references_retained_at':'post-load/pre-capture',
            'current_cache_tensor_metadata':caches,
            'cache_metadata_is_device_consumer_proof':False,
            'draft_attn_layer_names': list(drafter.attn_layer_names),
            'draft_use_cuda_graph': bool(drafter.use_cuda_graph),
            'draft_graph_configured':bool(drafter.use_cuda_graph),
            'actual_graph_submission_proof':None,
            'graph_observer_sources':counter.actual_draft_graph_observer.sources,
            'async_scheduling': bool(runner.use_async_scheduling),
            'load_format': runner.vllm_config.load_config.load_format,
            'hccl_deterministic': os.getenv('HCCL_DETERMINISTIC'),
            'audit_only_route_export': counter.audit,
            'actual_device_consumer_proof': False,
            'pending_consumer_proofs': ['draft per-layer device routes/consumer coverage',
                                        'main/draft cache write/read and graph binding',
                                        'complete actual consumer marker receipt']}
