"""CPU controls for P warmup exclusion and terminal chunk timing."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

torch = ModuleType("torch")
torch.no_grad = lambda: (lambda function: function)
torch.npu = NS(synchronize=lambda: None)
native = ModuleType("vllm_ascend.worker.worker")
native.NPUWorker = type("NPUWorker", (), {})
distributed = ModuleType("vllm.distributed")
distributed.get_tensor_model_parallel_rank = lambda: 0
spec = importlib.util.spec_from_file_location("ced_prefill_counter_test", Path(__file__).with_name("ced_worker.py"))
module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"torch": torch, native.__name__: native}):
    spec.loader.exec_module(module)


def scheduler(done, size, request_id="request", new=False):
    state = NS(req_id=request_id, prompt_token_ids=list(range(2048)), num_computed_tokens=done)
    return NS(scheduled_new_reqs=[state] if new else [],
              scheduled_cached_reqs=NS(req_ids=[] if new else [request_id],
                                       num_computed_tokens=[] if new else [done]),
              num_scheduled_tokens={request_id: size}), state


class PTimingTests(unittest.TestCase):
    def worker(self):
        worker = module.PrefillDeliveryWorker()
        worker.vllm_config = NS(scheduler_config=NS(async_scheduling=True))
        worker.model_runner = NS(execute_model=lambda *_: None, _bookkeeping_sync=lambda: (None,) * 6, requests={})
        worker.delivery_describe = lambda: {"rank": 0}
        worker.delivery_begin()
        return worker

    def test_unarmed_warmup_is_not_counted(self):
        worker = self.worker()
        data, _ = scheduler(0, 2048, new=True)
        worker.model_runner.execute_model(data)
        worker.model_runner._bookkeeping_sync()
        self.assertEqual(worker.delivery_metrics()["cumulative"]["prefill_calls"], 0)

    def test_multi_chunk_waits_for_the_terminal_prompt_boundary(self):
        worker = self.worker()
        worker.delivery_arm("task", 256)
        first, state = scheduler(0, 1024, new=True)
        worker.model_runner.execute_model(first)
        worker.model_runner._bookkeeping_sync()
        self.assertIsNone(worker._p_requests["request"]["end_ns"])
        with self.assertRaises(ValueError):
            worker.delivery_metrics()
        worker.model_runner.requests["request"] = state
        second, _ = scheduler(1024, 1024)
        worker.model_runner.execute_model(second)
        worker.model_runner._bookkeeping_sync()
        result = worker.delivery_metrics()
        self.assertEqual(result["cumulative"]["prefill_calls"], 2)
        self.assertTrue(result["requests"][0]["profile_excluded"])
        self.assertIsNone(worker._p_tag)

    def test_two_serial_windows_accumulate_independently(self):
        worker = self.worker()
        for index in range(2):
            worker.delivery_arm("task-" + str(index), 256)
            data, _ = scheduler(0, 2048, "request-" + str(index), new=True)
            worker.model_runner.execute_model(data)
            worker.model_runner._bookkeeping_sync()
            worker.delivery_metrics()
        self.assertEqual(worker.delivery_metrics()["cumulative"]["prefill_calls"], 2)
        self.assertEqual([r["tag"] for r in worker._p_completed], ["task-0", "task-1"])

    def test_double_arm_is_rejected(self):
        worker = self.worker()
        worker.delivery_arm("first", 256)
        with self.assertRaises(RuntimeError):
            worker.delivery_arm("second", 256)


if __name__ == "__main__":
    unittest.main()
