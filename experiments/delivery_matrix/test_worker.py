"""CPU negative controls for asynchronous deployment measurements."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

stub = ModuleType("delivery_test_base")
stub.Worker = type("Worker", (), {})
sys.modules[stub.__name__] = stub
with patch.dict(os.environ, DELIVERY_BASE_WORKER="delivery_test_base.Worker"):
    spec = importlib.util.spec_from_file_location("delivery_test_worker", Path(__file__).with_name("worker.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


class Counter:
    def __init__(self):
        self.rank = 0
        self.async_mode = True
        self.tag = None
        self.pending = None
        self.terminal_fence = {"old": True}
        self.requests = {}
        self.errors = []

    def execute_begin(self, *_):
        return True

    def bookkeeping_finished(self, *_):
        return None

    def begin(self, tag, budget):
        self.tag = tag
        self.requests = {}


def make_worker():
    worker = module.DeliveryWorker()
    worker.model_runner = NS(_formal_dspark_counter=Counter(), execute_model_state=None)
    worker.delivery_begin()
    return worker


def receipt(tag="task-0", req_id="req-0"):
    return {"rank": 0, "tag": tag, "errors": [], "async_mode": True,
            "capture_profile_counted": False,
            "terminal_fence": {"sync_returned_ns": 1100000000},
            "requests": {req_id: {"prompt_tokens": 2048, "prefill_calls": 1,
                                  "decode_start_ns": 1000000000, "seed_token_ids": [],
                                  "initial_draft": [{"runtime_graph": True}],
                                  "rounds": [{"token_ids": None, "scheduled_tokens": 6,
                                              "scheduled_draft_token_ids": [1, 2, 3, 4, 5],
                                              "sample_end_ns": 1090000000,
                                              "terminal_device_synced": True,
                                              "terminal_tp_join": {"backend": "gloo"},
                                              "draft_calls": [{"runtime_graph": True}]}]}}}


class MeasurementTests(unittest.TestCase):
    def finish(self, worker, data):
        counter = worker.model_runner._formal_dspark_counter
        counter.requests = data["requests"]
        fake = ModuleType("formal_dspark_counters")
        def finish(_):
            counter.tag = None
            return data
        fake.finish = finish
        with patch.dict(sys.modules, formal_dspark_counters=fake):
            return worker.delivery_metrics()

    def test_warmup_is_not_armed_or_counted(self):
        worker = make_worker()
        self.assertIsNone(worker.model_runner._formal_dspark_counter.tag)
        self.assertEqual(worker.delivery_metrics()["cumulative"]["decode_rounds"], 0)

    def test_async_counts_actual_rounds_without_invented_cpu_tokens(self):
        worker = make_worker()
        worker.delivery_arm("task-0", 256)
        self.assertIsNone(worker.model_runner._formal_dspark_counter.terminal_fence)
        value = self.finish(worker, receipt())
        self.assertEqual(value["cumulative"]["decode_rounds"], 1)
        self.assertIsNone(value["cumulative"]["decode_tokens"])
        self.assertIsNone(value["requests"][0]["raw_decode_tokens"])
        self.assertTrue(value["requests"][0]["terminal_fence_observed"])
        self.assertAlmostEqual(value["cumulative"]["decode_seconds"], 0.1)
        self.assertEqual(worker.delivery_metrics()["cumulative"]["decode_rounds"], 1)

    def test_two_serial_requests_use_independent_windows(self):
        worker = make_worker()
        worker.delivery_arm("task-0", 256)
        self.finish(worker, receipt())
        worker.delivery_arm("task-1", 1024)
        value = self.finish(worker, receipt("task-1", "req-1"))
        self.assertEqual(value["cumulative"]["decode_rounds"], 2)
        self.assertEqual([r["tag"] for r in value["requests"]], ["task-0", "task-1"])

    def test_chunked_prefill_uses_terminal_chunk_boundary(self):
        worker = make_worker()
        worker.delivery_arm("task-0", 256)
        counter = worker.model_runner._formal_dspark_counter
        data = receipt()
        counter.requests = data["requests"]
        counter.pending = {"req-0": {"phase": "prefill", "computed_before": 0,
                                    "scheduled_tokens": 1024, "bookkeeping_end_ns": 1000000000}}
        counter.bookkeeping_finished(None)
        self.assertNotIn("req-0", worker._delivery_terminal_prefill)
        counter.pending = {"req-0": {"phase": "prefill", "computed_before": 1024,
                                    "scheduled_tokens": 1024, "bookkeeping_end_ns": 1050000000}}
        counter.bookkeeping_finished(None)
        counter.pending = None
        value = self.finish(worker, data)
        self.assertAlmostEqual(value["cumulative"]["decode_seconds"], 0.05)

    def test_double_arm_is_rejected(self):
        worker = make_worker()
        worker.delivery_arm("task-0", 256)
        with self.assertRaises(RuntimeError):
            worker.delivery_arm("task-1", 256)

    def test_missing_async_completion_fence_is_rejected(self):
        worker = make_worker()
        worker.delivery_arm("task-0", 256)
        data = receipt()
        data["terminal_fence"] = None
        with self.assertRaises(ValueError):
            self.finish(worker, data)

    def test_extra_request_contamination_is_rejected(self):
        worker = make_worker()
        worker.delivery_arm("task-0", 256)
        data = receipt()
        data["requests"].update(receipt("other", "other")["requests"])
        with self.assertRaises(ValueError):
            self.finish(worker, data)


if __name__ == "__main__":
    unittest.main()
