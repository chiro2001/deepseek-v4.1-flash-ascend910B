"""Regressions for the observed stock cleanup ABI and real core-ID binding."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

STACK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STACK))
from parallel_tiny.dspark_formal_prep import test_counters as base
from parallel_tiny.dspark_formal_audit.stock_compat_v3 import formal_dspark_counters as counters
from parallel_tiny.dspark_formal_audit.stock_compat_v3.bench_formal_dspark import mapped_generation

base.RoundCounters = counters.RoundCounters
base.summarize = counters.summarize
base.install_runner = counters.install_runner


class ExistingCounterContract(base.CounterContract):
    def test_bank_uses_retained_native_modules_after_wrapping(self):
        original = base.bank
        def adapted_bank(worker):
            worker.model_runner._formal_dspark_counter.actual_draft_graph_observer = NS(sources={})
            return counters.bank(worker)
        base.bank = adapted_bank
        try:
            super().test_bank_uses_retained_native_modules_after_wrapping()
        finally:
            base.bank = original


def empty_scheduler(**changes):
    fields = dict(scheduled_new_reqs=[],
                  scheduled_cached_reqs=NS(req_ids=[], num_computed_tokens=[]),
                  num_scheduled_tokens={}, total_num_scheduled_tokens=0,
                  scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
                  num_invalid_spec_tokens=None, finished_req_ids=['r'])
    fields.update(changes)
    return NS(**fields)


class StockCleanupContract(unittest.TestCase):
    def test_zero_token_finished_cleanup_is_not_a_round(self):
        counter = counters.RoundCounters(0, now=base.Clock())
        counter.begin('actual-request')
        self.assertFalse(counter.execute_begin(empty_scheduler(), {}))
        self.assertIsNone(counter.pending)
        self.assertEqual(counter.requests, {})
        receipt = counter.finish()
        self.assertEqual(receipt['zero_token_executions'][0]['finished_req_ids'], ['r'])
        self.assertFalse(receipt['zero_token_executions'][0]['counted_as_decode_round'])

    def test_empty_map_with_nonzero_total_rejected(self):
        counter = counters.RoundCounters(0)
        counter.begin('x')
        with self.assertRaisesRegex(ValueError, 'Inconsistent zero-token'):
            counter.execute_begin(empty_scheduler(total_num_scheduled_tokens=6), {})

    def test_empty_map_with_proposals_rejected(self):
        counter = counters.RoundCounters(0)
        counter.begin('x')
        with self.assertRaisesRegex(ValueError, 'Inconsistent zero-token'):
            counter.execute_begin(empty_scheduler(scheduled_spec_decode_tokens={'r': [20]}), {})

    def test_empty_cleanup_cannot_claim_sampler(self):
        counter = counters.RoundCounters(0)
        counter.begin('x')
        counter.execute_begin(empty_scheduler(), {})
        with self.assertRaisesRegex(RuntimeError, 'without a real execute_model'):
            counter.sampler_finished(False)

    def test_exact_core_id_is_bound_on_all_eight_ranks(self):
        receipts = [base.receipt(i) for i in range(8)]
        metrics = counters.summarize(receipts, [10,20,21,30,22,23], 2_000_000_000,
                                    counters.clock_scope(), draft_graph=True,
                                    expected_core_request_id='r')
        self.assertTrue(metrics['exact_core_request_id_bound'])
        self.assertEqual(metrics['core_request_id'], 'r')

    def test_same_tokens_from_foreign_core_id_rejected(self):
        receipts = [base.receipt(i) for i in range(8)]
        receipts[7]['requests']['foreign-core'] = receipts[7]['requests'].pop('r')
        with self.assertRaisesRegex(ValueError, 'actual core request ID'):
            counters.summarize(receipts, [10,20,21,30,22,23], 2_000_000_000,
                               counters.clock_scope(), draft_graph=True, expected_core_request_id='r')

    def test_foreign_finished_cleanup_rejected(self):
        receipts = [base.receipt(i) for i in range(8)]
        receipts[0]['zero_token_executions'] = [{'finished_req_ids': ['foreign-core'], 'execute_end_ns': 1}]
        with self.assertRaisesRegex(ValueError, 'different core request ID'):
            counters.summarize(receipts, [10,20,21,30,22,23], 2_000_000_000,
                               counters.clock_scope(), draft_graph=True, expected_core_request_id='r')

    def test_uncompleted_cleanup_rejected(self):
        receipts = [base.receipt(i) for i in range(8)]
        receipts[0]['zero_token_executions'] = [{'finished_req_ids': ['r']}]
        with self.assertRaisesRegex(ValueError, 'did not return stock empty'):
            counters.summarize(receipts, [10,20,21,30,22,23], 2_000_000_000,
                               counters.clock_scope(), draft_graph=True, expected_core_request_id='r')

    def test_generate_uses_actual_add_request_return(self):
        engine = NS(output_processor=NS(external_req_ids={}))
        def add(request_id, *args, **kwargs):
            engine.output_processor.external_req_ids[request_id] = ['real-core-suffix']
            return 'real-core-suffix'
        engine.add_request = add
        def generate(prompts, sampling, use_tqdm):
            self.assertEqual(engine.add_request('external-id', prompts[0], sampling), 'real-core-suffix')
            return [NS(request_id='external-id')]
        output, core_id = mapped_generation(NS(llm_engine=engine, generate=generate), [1,2], object())
        self.assertEqual((output.request_id, core_id), ('external-id', 'real-core-suffix'))
        self.assertIs(engine.add_request, add)

    def test_generate_false_mapping_is_rejected_and_restored(self):
        engine = NS(output_processor=NS(external_req_ids={'external-id': ['wrong']}))
        def add(*args, **kwargs):
            return 'real-core'
        engine.add_request = add
        def generate(*args, **kwargs):
            engine.add_request('external-id')
        with self.assertRaisesRegex(ValueError, 'external/core request mapping'):
            mapped_generation(NS(llm_engine=engine, generate=generate), [1], object())
        self.assertIs(engine.add_request, add)


if __name__ == '__main__':
    unittest.main()
