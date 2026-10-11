"""Counter boundary tests with mocks, without importing torch or initializing NPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS


def main():
    syncs = []
    fake_torch = types.ModuleType("torch")
    fake_torch.npu = NS(synchronize=lambda: syncs.append("terminal"))
    sys.modules["torch"] = fake_torch
    weight = types.ModuleType("vllm.model_executor.model_loader.weight_utils")
    weight.initialize_single_dummy_weight = lambda *a, **k: None
    loader = types.ModuleType("vllm.model_executor.model_loader")
    loader.weight_utils = weight
    sys.modules["vllm.model_executor.model_loader"] = loader

    class Backbone:
        def __init__(self):
            self.layers = list(range(40))
            self.config = NS(hidden_size=5120)

    model_module = types.ModuleType("vllm_ascend.models.deepseek_v41.model")
    model_module.DeepseekV41Model = Backbone
    sys.modules[model_module.__name__] = model_module

    class Runner:
        def __init__(self):
            backbone = Backbone()
            self.model = NS(modules=lambda: [backbone])
            self.drafter = NS(_propose=lambda: [], model=NS(model=NS(
                layers={"40": object(), "41": object(), "42": object()},
                target_layer_ids=[37, 38, 39])), num_speculative_tokens=5, use_cuda_graph=False)
            self.compilation_config = NS(static_forward_context={})
            self.next_tokens = [7]

        def execute_model(self, scheduler_output):
            return None

        def _bookkeeping_sync(self, *args):
            return (None, [self.next_tokens], {}, ["r"], {"r": 0}, [])

        def sample_tokens(self):
            result = self._bookkeeping_sync()
            self.drafter._propose()
            return result

    class NPUWorker:
        def load_model(self, *args, **kwargs):
            return None

    module = types.ModuleType("vllm_ascend.worker.worker")
    module.NPUWorker = NPUWorker
    sys.modules[module.__name__] = module
    base = Path(__file__).resolve().parent
    path = base / "dspark_worker.py"
    spec = importlib.util.spec_from_file_location("dspark_worker", path)
    worker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker_module)
    worker = worker_module.DSparkTinyWorker()
    worker.vllm_config = NS(load_config=NS(load_format="real"))
    worker.model_runner = Runner()
    worker.load_model()
    runner = worker.model_runner
    new = NS(req_id="r", prompt_token_ids=[1, 2, 3, 4], num_computed_tokens=0)
    prefill = NS(scheduled_new_reqs=[new], scheduled_cached_reqs=NS(req_ids=[], num_computed_tokens=[]),
                 num_scheduled_tokens={"r": 4}, scheduled_spec_decode_tokens={})
    runner.execute_model(prefill)
    runner.sample_tokens()
    assert runner._tiny_dspark_receipts == []
    worker_module.reset_receipts(worker)
    runner.execute_model(prefill)
    runner.sample_tokens()
    seed = runner._tiny_dspark_receipts[-1]
    assert not seed["is_decode"] and seed["draft_calls"] == 1
    decode = NS(scheduled_new_reqs=[], scheduled_cached_reqs=NS(req_ids=["r"], num_computed_tokens=[4]),
                num_scheduled_tokens={"r": 6}, scheduled_spec_decode_tokens={"r": [8, 9, 10, 11, 12]})
    runner.next_tokens = [13, 14, 15]
    runner.execute_model(decode)
    runner.sample_tokens()
    row = runner._tiny_dspark_receipts[-1]
    assert row["is_decode"] and row["accepted_draft_tokens"] == 2 and row["verified_draft_tokens"] == 5
    assert row["draft_calls"] == 1 and syncs == []
    receipt = worker_module.read_receipts(worker)
    assert len(receipt["rounds"]) == 2 and syncs == ["terminal"]
    assert receipt["terminal_device_done_s"] >= row["sample_done_s"]
    assert seed["execute_begin_s"] <= seed["bookkeeping_done_s"] <= seed["sample_done_s"] <= row["execute_begin_s"]
    assert row["execute_begin_s"] <= row["bookkeeping_done_s"] <= row["sample_done_s"]
    # Replacing the target by a wrapper lacking .model/.modules reproduces the
    # real v4 topology failure; the saved native references must still work.
    runner.model = object()
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "4"
    os.environ["HCCL_DETERMINISTIC"] = "strict"
    topology = worker_module.topology(worker)
    assert topology["main_layers"] == 40 and topology["draft_layers"] == 3
    result = {"passed": True, "cpu_only_mock": True, "real_torch_imported": False,
              "startup_counts_excluded": True, "prefill_seed_excluded_from_decode_rounds": True,
              "actual_verified_proposal_count": 5, "native_cpu_raw_accept_count": 2,
              "steady_extra_syncs": 0, "terminal_extra_syncs": 1,
              "post_acl_wrapper_topology_passed": True,
              "worker_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (base / "evidence/cpu_counter_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
