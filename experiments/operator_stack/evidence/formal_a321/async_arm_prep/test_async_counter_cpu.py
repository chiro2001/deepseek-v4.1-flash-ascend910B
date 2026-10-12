"""CPU-only contracts for the env-gated async counter arm (DSPARK_ASYNC_COUNTER).

No torch/vLLM/NPU import: the async path is exercised through the same CPU
bookkeeping ABI the stock async runner materialises (empty token lists), plus a
stubbed device fence.
"""
import sys
from types import SimpleNamespace as NS
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'testenv'))
from formal_dspark_counters import RoundCounters, clock_scope, summarize  # noqa: E402


class Clock:
    def __init__(self):
        self.tick = 1_000_000

    def __call__(self):
        self.tick += 1000
        return self.tick


def scheduler(computed, drafts=(), *, new=False):
    state = NS(req_id='r', prompt_token_ids=[1, 2, 3, 4], num_computed_tokens=computed)
    return NS(scheduled_new_reqs=[state] if new else [],
              scheduled_cached_reqs=NS(req_ids=[] if new else ['r'],
                                       num_computed_tokens=[] if new else [computed]),
              num_scheduled_tokens={'r': 2 if computed < 4 else 1 + len(drafts)},
              scheduled_spec_decode_tokens={'r': list(drafts)},
              num_invalid_spec_tokens={}), {'r': state}


def async_step(c, *, drafts=(), new=False, computed=None, draft=True):
    so, states = scheduler(computed, drafts, new=new)
    assert c.execute_begin(so, states)
    c.sampler_finished(bool(drafts))
    # The stock async bookkeeping call returns an EMPTY valid-sampled-token list.
    c.bookkeeping_finished((None, [], {}, ['r'], {'r': 0}, []))
    if draft:
        c.draft_finished(k=5, output_shape=[1, 5], runtime_graph=True, method='dspark')
    c.sample_finished()


def async_receipt(rank, *, rounds=3, fence_phase='cleanup_execute', with_join=True):
    c = RoundCounters(rank, now=Clock())
    c.async_mode = True
    c.begin('pair', 7)
    # terminal prefill (seed) with the initial MTP draft
    so, states = scheduler(0, new=True)
    assert c.execute_begin(so, states)
    c.sampler_finished(False)
    c.bookkeeping_finished((None, [], {}, ['r'], {'r': 0}, []))
    c.draft_finished(k=5, output_shape=[1, 5], runtime_graph=True, method='dspark')
    c.sample_finished()
    for index in range(rounds):
        async_step(c, drafts=[20, 21, 22, 23, 24], computed=4 + 6 * index)
    cleanup = {'finished_req_ids': ['r'], 'total_num_scheduled_tokens': 0,
               'execute_start_ns': c.now(), 'counted_as_decode_round': False}
    c.zero_token_executions.append(cleanup)
    assert c.zero_token_cleanup_fence(cleanup, sync=lambda: None) if fence_phase == 'cleanup_execute' else True
    if fence_phase == 'post_client':
        c.post_client_device_fence('post_client', sync=lambda: None)
    if with_join:
        row = c.requests['r']['rounds'][-1]
        row['terminal_tp_join'] = {'backend': 'gloo', 'world_size': 8, 'rank': rank,
                                   'entered_ns': row['sample_end_ns'] + 1,
                                   'returned_ns': row['sample_end_ns'] + 2,
                                   'device_fence_preceded_entry': True,
                                   'sample_timestamp_preceded_entry': True}
    return c.finish()


def main():
    rows = [async_receipt(i) for i in range(8)]
    fence_ns = max(r['terminal_fence']['sync_returned_ns'] for r in rows)
    client_end = max(r['requests']['r']['rounds'][-1]['sample_end_ns'] for r in rows) + 500
    client_end = max(client_end, fence_ns) + 500
    metrics = summarize(rows, [10, 20, 21, 30], client_end, clock_scope(), draft_graph=True)
    assert metrics['rounds'] == 3, metrics
    assert metrics['decoded_tokens'] == 3, metrics
    assert abs(metrics['A'] - 1.0) < 1e-12, metrics
    assert metrics['async_scheduling'] is True
    assert metrics['async_per_round_tokens_available'] is False
    assert metrics['terminal_fence_phases'][0] == 'cleanup_execute'
    assert metrics['rank_counters'][0]['accepted_draft_tokens_raw'] is None
    assert metrics['rank_counters'][0]['raw_decode_tokens'] is None
    assert metrics['terminal_device_completion_observed'] is True
    assert metrics['performance_receipt_granted'] is False

    # Fence must extend the window only when it lands after the client clock.
    rows2 = [async_receipt(i, fence_phase='post_client') for i in range(8)]
    late_client = min(r['requests']['r']['rounds'][-1]['sample_end_ns'] for r in rows2) + 100
    metrics2 = summarize(rows2, [10, 20, 21, 30], late_client, clock_scope(), draft_graph=True)
    fence2 = max(r['terminal_fence']['sync_returned_ns'] for r in rows2)
    assert metrics2['decode_window_end_ns'] == fence2, (metrics2['decode_window_end_ns'], fence2)
    assert metrics2['terminal_fence_delta_ms'] > 0

    # Rank divergence on the round stream must be rejected.
    rows3 = [async_receipt(i) for i in range(8)]
    rows3[7]['requests']['r']['rounds'][1]['scheduled_draft_token_ids'][0] = 999
    client_end3 = max(r['terminal_fence']['sync_returned_ns'] for r in rows3) + 200
    try:
        summarize(rows3, [10, 20, 21, 30], client_end3, clock_scope(), draft_graph=True)
        raise AssertionError('rank divergence accepted')
    except ValueError as exc:
        assert 'TP ranks disagree' in str(exc), exc

    # A missing terminal fence must fail closed.
    rows4 = [async_receipt(i) for i in range(8)]
    rows4[2]['terminal_fence'] = None
    client_end4 = max(r['requests']['r']['rounds'][-1]['sample_end_ns'] for r in rows4) + 400
    try:
        summarize(rows4, [10, 20, 21, 30], client_end4, clock_scope(), draft_graph=True)
        raise AssertionError('missing fence accepted')
    except ValueError as exc:
        assert 'fence was not observed' in str(exc), exc

    # An eager steady draft must still fail the graph-on gate.
    rows5 = [async_receipt(i) for i in range(8)]
    for r in rows5:
        for row in r['requests']['r']['rounds']:
            for call in row['draft_calls']:
                call['runtime_graph'] = False
    client_end5 = max(r['terminal_fence']['sync_returned_ns'] for r in rows5) + 200
    try:
        summarize(rows5, [10, 20, 21, 30], client_end5, clock_scope(), draft_graph=True)
        raise AssertionError('eager draft accepted')
    except ValueError as exc:
        assert 'used eager' in str(exc), exc

    # Sync-mode receipts keep the synchronous path: empty tokens still rejected.
    c = RoundCounters(0, now=Clock())
    c.begin('x', 7)
    so, states = scheduler(0, new=True)
    assert c.execute_begin(so, states)
    try:
        c.bookkeeping_finished((None, [], {}, ['r'], {'r': 0}, []))
        raise AssertionError('sync empty tokens accepted')
    except ValueError as exc:
        assert 'ABI is unavailable' in str(exc), exc

    print('ASYNC_COUNTER_CPU_CONTRACTS_PASSED 6/6')


if __name__ == '__main__':
    main()
