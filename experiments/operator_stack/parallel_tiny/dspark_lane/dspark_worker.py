"""Private TP1 DSpark worker: native weights/ops with host-only round receipts."""
import os
import time
from types import MethodType

from vllm_ascend.worker.worker import NPUWorker


class DSparkTinyWorker(NPUWorker):
    def load_model(self, *args, **kwargs):
        # Previous tiny runs required bounded dummy-init submission. Keep this
        # startup-only guard; it does not alter weights or the decode schedule.
        import torch
        from vllm.model_executor.model_loader import weight_utils
        original = weight_utils.initialize_single_dummy_weight
        init_calls = 0

        def initialize(param, *a, **kw):
            nonlocal init_calls
            result = original(param, *a, **kw)
            if param.device.type == "npu":
                torch.npu.synchronize()
            init_calls += 1
            return result

        if self.vllm_config.load_config.load_format == "dummy":
            weight_utils.initialize_single_dummy_weight = initialize
        try:
            result = super().load_model(*args, **kwargs)
        finally:
            weight_utils.initialize_single_dummy_weight = original
        runner = self.model_runner
        from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model
        target = runner.model
        backbones = [m for m in target.modules() if isinstance(m, DeepseekV41Model)]
        assert len(backbones) == 1
        # ACLGraphWrapper replaces runner.model during warm-up and does not
        # expose the multimodal wrapper's model/modules attribute. Preserve the
        # actual native objects before wrapping; cache buffers are filled later.
        runner._tiny_dspark_native_target = target
        runner._tiny_dspark_native_backbone = backbones[0]
        runner._tiny_dspark_native_draft = runner.drafter.model.model
        runner._tiny_dspark_receipts = []
        runner._tiny_dspark_prompt_lens = {}
        runner._tiny_dspark_pending = None
        runner._tiny_dspark_draft_calls = 0
        runner._tiny_dspark_recording = False
        execute = runner.execute_model
        bookkeeping = runner._bookkeeping_sync
        sample = runner.sample_tokens
        # The DSpark branch bypasses .propose and calls ._propose directly
        # (NPUModelRunner.propose_draft_token_ids). This is outside capture.
        propose = runner.drafter._propose

        def execute_counted(this, scheduler_output, *a, **kw):
            if this._tiny_dspark_recording:
                assert this._tiny_dspark_pending is None
                for request in scheduler_output.scheduled_new_reqs:
                    assert request.prompt_token_ids is not None
                    this._tiny_dspark_prompt_lens[request.req_id] = len(request.prompt_token_ids)
                cached = scheduler_output.scheduled_cached_reqs
                computed = dict(zip(cached.req_ids, cached.num_computed_tokens))
                computed.update({r.req_id: r.num_computed_tokens for r in scheduler_output.scheduled_new_reqs})
                reqs = list(scheduler_output.num_scheduled_tokens)
                assert len(reqs) == 1, "This diagnostic lane requires serial requests"
                rid = reqs[0]
                this._tiny_dspark_pending = {
                    "req_id": rid, "execute_begin_s": time.perf_counter(),
                    "computed_before": computed[rid],
                    "prompt_tokens": this._tiny_dspark_prompt_lens[rid],
                    "is_decode": computed[rid] >= this._tiny_dspark_prompt_lens[rid],
                    "scheduled_target_tokens": scheduler_output.num_scheduled_tokens[rid],
                    "verified_draft_tokens": len(scheduler_output.scheduled_spec_decode_tokens.get(rid, [])),
                    "draft_calls_before": this._tiny_dspark_draft_calls,
                }
            return execute(scheduler_output, *a, **kw)

        def bookkeeping_counted(this, *a, **kw):
            result = bookkeeping(*a, **kw)
            receipt = this._tiny_dspark_pending
            if receipt is not None:
                # The original sync path has already produced CPU lists. There
                # is no additional device read/synchronize in these counters.
                valid, req_ids = result[1], result[3]
                index = req_ids.index(receipt["req_id"])
                receipt["bookkeeping_done_s"] = time.perf_counter()
                receipt["sampled_token_ids"] = list(valid[index])
                receipt["raw_generated_tokens"] = len(valid[index])
                receipt["accepted_draft_tokens"] = max(len(valid[index]) - 1, 0) if receipt["verified_draft_tokens"] else 0
            return result

        def propose_counted(*a, **kw):
            receipt = runner._tiny_dspark_pending
            if runner._tiny_dspark_recording:
                runner._tiny_dspark_draft_calls += 1
                if receipt is not None:
                    receipt["draft_propose_begin_s"] = time.perf_counter()
            result = propose(*a, **kw)
            if runner._tiny_dspark_recording and receipt is not None:
                receipt["draft_propose_done_s"] = time.perf_counter()
                receipt["draft_propose_entrypoint"] = "AscendDSparkProposer._propose"
            return result

        def sample_counted(this, *a, **kw):
            result = sample(*a, **kw)
            receipt = this._tiny_dspark_pending
            if receipt is not None:
                receipt["sample_done_s"] = time.perf_counter()
                receipt["draft_calls"] = this._tiny_dspark_draft_calls - receipt.pop("draft_calls_before")
                this._tiny_dspark_receipts.append(receipt)
                this._tiny_dspark_pending = None
            return result

        runner.execute_model = MethodType(execute_counted, runner)
        runner._bookkeeping_sync = MethodType(bookkeeping_counted, runner)
        runner.sample_tokens = MethodType(sample_counted, runner)
        runner.drafter._propose = propose_counted
        print("DSPARK_TINY_LOADED", {"dummy_init_calls": init_calls,
                                      "runner": type(runner).__name__,
                                      "proposer": type(runner.drafter).__name__,
                                      "draft_model": type(runner.drafter.model).__name__,
                                      "pid": os.getpid()}, flush=True)
        return result


def reset_receipts(worker):
    runner = worker.model_runner
    assert runner._tiny_dspark_pending is None
    runner._tiny_dspark_receipts.clear()
    runner._tiny_dspark_draft_calls = 0
    runner._tiny_dspark_recording = True
    return {"pid": os.getpid(), "clock": time.get_clock_info("perf_counter").implementation}


def read_receipts(worker):
    runner = worker.model_runner
    runner._tiny_dspark_recording = False
    assert runner._tiny_dspark_pending is None
    import torch
    torch.npu.synchronize()
    terminal_device_done_s = time.perf_counter()
    return {"pid": os.getpid(), "rounds": list(runner._tiny_dspark_receipts),
            "terminal_device_done_s": terminal_device_done_s,
            "terminal_sync_count": 1, "steady_extra_sync_count": 0,
            "proposer": type(runner.drafter).__name__,
            "draft_model": type(runner.drafter.model).__name__}


def topology(worker):
    runner = worker.model_runner
    model, drafter = runner._tiny_dspark_native_target, runner.drafter
    main = runner._tiny_dspark_native_backbone
    draft = runner._tiny_dspark_native_draft
    caches = {}
    for name, layer in runner.compilation_config.static_forward_context.items():
        cache = getattr(layer, "kv_cache", None)
        if cache is None:
            continue
        tensors = cache if isinstance(cache, (tuple, list)) else [cache]
        caches[name] = [{"shape": list(t.shape), "ptr": t.data_ptr()}
                        for t in tensors if hasattr(t, "data_ptr")]
    return {"main_model": type(model).__name__, "main_backbone": type(main).__name__,
            "main_layers": len(main.layers),
            "draft_model": type(drafter.model).__name__, "draft_layers": len(draft.layers),
            "target_layer_ids": list(draft.target_layer_ids),
            "main_hidden_size": main.config.hidden_size,
            "num_speculative_tokens": drafter.num_speculative_tokens,
            "draft_use_cuda_graph": drafter.use_cuda_graph,
            "proposer": type(drafter).__name__, "cache_consumers": caches,
            "physical_device": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
            "hccl_deterministic": os.environ["HCCL_DETERMINISTIC"]}
