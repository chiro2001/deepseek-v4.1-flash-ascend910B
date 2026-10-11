"""CPU regressions for terminal TP8 joining; no device receipt granted."""
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from parallel_tiny.dspark_formal_audit.stock_compat_v3 import test_stock_v3 as prior
from parallel_tiny.dspark_formal_audit.stock_compat_v4 import formal_dspark_counters as counters

prior.counters = counters
prior.base.RoundCounters = counters.RoundCounters
prior.base.summarize = counters.summarize
prior.base.install_runner = counters.install_runner


class MockTP:
    world_size = 8
    cpu_group = object()
    def __init__(self, rank, barrier=None):
        self.rank_in_group = rank
        self._barrier = barrier
        self.joined = False
    def barrier(self):
        if self._barrier is not None:
            self._barrier.wait(timeout=2)
        self.joined = True


class MockDistributed:
    @staticmethod
    def get_backend(group):
        assert group is MockTP.cpu_group
        return 'gloo'


original_sampled = prior.base.sampled
def sampled(counter, tokens, **kwargs):
    original_sampled(counter, tokens, **kwargs)
    if kwargs.get('terminal'):
        counters.terminal_tp_completion_join(counter, MockTP(counter.rank), MockDistributed)
prior.base.sampled = sampled

ExistingCounterContract = prior.ExistingCounterContract
StockCleanupContract = prior.StockCleanupContract


def terminal_counter(rank=0, *, fenced=True):
    counter = counters.RoundCounters(rank, now=prior.base.Clock())
    counter.begin('x')
    so, states = prior.base.scheduler(0, new=True)
    counter.execute_begin(so, states)
    original_sampled(counter, [10])
    so, states = prior.base.scheduler(4, [20,21,22,23,24])
    counter.execute_begin(so, states)
    original_sampled(counter, [20,30], spec=True, terminal=fenced)
    return counter


class TerminalJoinContract(unittest.TestCase):
    def test_device_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, 'CPU Gloo'):
            counters.terminal_tp_completion_join(terminal_counter(), MockTP(0),
                                                NS(get_backend=lambda group: 'hccl'))

    def test_incomplete_tp_group_rejected(self):
        group = MockTP(0)
        group.world_size = 4
        with self.assertRaisesRegex(ValueError, 'eight-rank'):
            counters.terminal_tp_completion_join(terminal_counter(), group, MockDistributed)

    def test_wrong_rank_group_rejected(self):
        with self.assertRaisesRegex(ValueError, 'eight-rank'):
            counters.terminal_tp_completion_join(terminal_counter(), MockTP(7), MockDistributed)

    def test_join_cannot_replace_device_fence(self):
        with self.assertRaisesRegex(ValueError, 'missing a device fence'):
            counters.terminal_tp_completion_join(terminal_counter(fenced=False), MockTP(0), MockDistributed)

    def test_duplicate_terminal_join_rejected(self):
        counter = terminal_counter()
        counters.terminal_tp_completion_join(counter, MockTP(0), MockDistributed)
        with self.assertRaisesRegex(ValueError, 'counted twice'):
            counters.terminal_tp_completion_join(counter, MockTP(0), MockDistributed)

    def test_output_rank_waits_for_delayed_rank_timestamp(self):
        barrier = threading.Barrier(8)
        rows, errors, published = {}, [], []
        def worker(rank):
            try:
                counter = terminal_counter(rank)
                if rank == 7:
                    time.sleep(.02)
                row = counter.requests['r']['rounds'][-1]
                row['sample_end_ns'] = time.monotonic_ns()
                counter.now = time.monotonic_ns
                counters.terminal_tp_completion_join(counter, MockTP(rank, barrier), MockDistributed)
                rows[rank] = row
                if rank == 0:
                    published.append(time.monotonic_ns())
            except BaseException as exc:
                errors.append(exc)
        threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
        self.assertEqual(errors, [])
        self.assertEqual(set(rows), set(range(8)))
        self.assertEqual(len(published), 1)
        self.assertGreaterEqual(published[0], max(row['sample_end_ns'] for row in rows.values()))
        self.assertTrue(all(row['terminal_tp_join']['entered_ns'] >= row['sample_end_ns'] for row in rows.values()))


if __name__ == '__main__':
    unittest.main()
