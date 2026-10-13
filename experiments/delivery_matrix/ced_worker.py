"""Native CED P role: real weights/Engram, no architecturally impossible draft."""
import inspect
import os
from pathlib import Path
import json
import time

import torch
from vllm_ascend.worker.worker import NPUWorker


class PrefillDeliveryWorker(NPUWorker):
    @torch.no_grad()
    def load_model(self, *args, **kwargs):
        text = self.vllm_config.model_config.hf_text_config
        parallel = self.vllm_config.parallel_config
        if (os.environ.get("V41_CED_ROLE") != "prefill" or self.vllm_config.speculative_config is not None
                or self.vllm_config.load_config.load_format == "dummy"
                or text.num_hidden_layers != 40 or text.hidden_size != 5120
                or text.n_routed_experts != 384 or text.num_experts_per_tok != 6
                or parallel.tensor_parallel_size != 8 or parallel.decode_context_parallel_size != 1
                or os.environ.get("HCCL_DETERMINISTIC") != "strict"):
            raise ValueError("Real full-weight CED P, TP8/zero-DCP and strict transport are required")
        from parallel_tiny.dspark_formal_prep.sdk_gate import inspect_native_factory
        from vllm.distributed import get_tensor_model_parallel_rank
        sdk = inspect_native_factory()
        if not sdk["compatible"]:
            raise ValueError("Actual native full-model factory must preserve vision bias_vl support")
        sdk["rank"] = get_tensor_model_parallel_rank()
        (Path(os.environ["STACK_DSPARK_FORMAL_OUTPUT"]) / f"sdk_rank{sdk['rank']}.json").write_text(json.dumps(sdk) + "\n")
        return super().load_model(*args, **kwargs)

    def delivery_describe(self):
        from vllm.distributed import get_tensor_model_parallel_rank
        result = {"rank": get_tensor_model_parallel_rank(), "role": "prefill", "form": "a3-ced-pd",
                "tp": 8, "dcp": 1, "executed_encoder_layers": 20, "layer20_global_source": True,
                "loaded_model_layers": 40, "draft_graph": False, "speculative_tokens": 0,
                "async_scheduling": self.vllm_config.scheduler_config.async_scheduling,
                "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT"),
                "ced_overlay_manifest_sha256": os.environ.get("DELIVERY_CED_MANIFEST_SHA256"),
                "native_opp_fingerprint": os.environ.get("DELIVERY_CED_OPP_FINGERPRINT"),
                "engram_device_index": os.environ.get("V41_ENGRAM_DEVICE_INDEX"),
                "base_worker_source": inspect.getfile(NPUWorker)}
        if os.environ.get("DELIVERY_SHARED_ENGRAM"):
            from .shared_engram_contract import describe_selected
            result["shared_engram"] = describe_selected(self.model_runner.model)
        return result

    def delivery_begin(self):
        self._p_tag = None
        self._p_pending = None
        self._p_requests = {}
        self._p_totals = {"prefill_seconds": 0.0, "prefill_calls": 0,
                          "decode_rounds": 0, "decode_tokens": 0, "decode_seconds": 0.0}
        self._p_completed = []
        runner = self.model_runner
        execute, bookkeeping = runner.execute_model, runner._bookkeeping_sync

        def observed_execute(scheduler, *args, **kwargs):
            if self._p_tag is not None and scheduler.num_scheduled_tokens:
                now = time.monotonic_ns()
                new = {r.req_id: r for r in scheduler.scheduled_new_reqs}
                cached = scheduler.scheduled_cached_reqs
                computed = dict(zip(cached.req_ids, cached.num_computed_tokens))
                pending = {}
                for req_id, scheduled in scheduler.num_scheduled_tokens.items():
                    state = new.get(req_id) or runner.requests.get(req_id)
                    if state is None or state.prompt_token_ids is None:
                        raise ValueError("CED P measurement requires actual CPU prompt metadata")
                    done = state.num_computed_tokens if req_id in new else computed.get(req_id)
                    length = len(state.prompt_token_ids)
                    if type(done) is not int:
                        raise ValueError("Missing actual P computed-token boundary")
                    if done < length:
                        entry = self._p_requests.setdefault(req_id, {"start_ns": now, "end_ns": None,
                                                                    "prompt_tokens": length, "calls": 0})
                        entry["calls"] += 1
                        pending[req_id] = done + scheduled >= length
                if len(self._p_requests) > 1:
                    raise ValueError("CED P measurement must stay single-request")
                self._p_pending = pending
            return execute(scheduler, *args, **kwargs)

        def observed_bookkeeping(*args, **kwargs):
            result = bookkeeping(*args, **kwargs)
            now = time.monotonic_ns()
            for req_id, terminal in (self._p_pending or {}).items():
                if terminal:
                    self._p_requests[req_id]["end_ns"] = now
            self._p_pending = None
            return result

        runner.execute_model, runner._bookkeeping_sync = observed_execute, observed_bookkeeping
        return {"prefill_measurement_ready_after_warmup": True}

    def delivery_arm(self, tag, max_tokens):
        if self._p_tag is not None or self._p_pending is not None:
            raise RuntimeError("A previous P measurement remains active")
        self._p_tag, self._p_requests = tag, {}
        return {"rank": self.delivery_describe()["rank"], "tag": tag, "armed": True,
                "async_mode": self.vllm_config.scheduler_config.async_scheduling, "max_tokens": max_tokens}

    def delivery_metrics(self):
        if self._p_tag is not None:
            if self._p_pending is not None or len(self._p_requests) != 1:
                raise RuntimeError("Finish P measurements only after the producer HTTP response")
            req_id, request = next(iter(self._p_requests.items()))
            if request["end_ns"] is None:
                raise ValueError("P terminal prompt chunk was not observed")
            torch.npu.synchronize()  # Outside capture, once at measurement completion.
            self._p_totals["prefill_seconds"] += (request["end_ns"] - request["start_ns"]) / 1e9
            self._p_totals["prefill_calls"] += request["calls"]
            self._p_completed.append({"tag": self._p_tag, "request_id": req_id, **request,
                                      "profile_excluded": True, "terminal_fence_observed": True,
                                      "prefill_wall_scope": "P first scheduled prompt -> terminal P CPU bookkeeping"})
            self._p_tag = None
        return {"rank": self.delivery_describe()["rank"], "role": "prefill",
                "cumulative": dict(self._p_totals), "requests": list(self._p_completed),
                "pending": self._p_pending is not None, "errors": []}
