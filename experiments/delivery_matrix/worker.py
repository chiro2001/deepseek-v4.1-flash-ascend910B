"""Actual native worker bookkeeping, outside graph capture and warmup."""
from __future__ import annotations

import functools
import importlib
import inspect
import os
import time
import hashlib
import json

_physical_derivation = None

def base_worker():
    global _physical_derivation
    name = os.environ["DELIVERY_BASE_WORKER"]
    module, cls = name.rsplit(".", 1)
    if module.startswith("delivery_matrix"):
        raise ValueError("Recursive delivery worker")
    if os.environ.get("DELIVERY_PHYSICAL_WORKER"):
        from .physical_worker import load_base
        derived, _physical_derivation = load_base(os.environ["DELIVERY_PHYSICAL_WORKER"],
                                                 os.environ["STACK_SRC_ROOT"], name)
        return derived
    return getattr(importlib.import_module(module), cls)


Base = base_worker()


class DeliveryWorker(Base):
    def load_model(self, *args, **kwargs):
        self._delivery_dcp_derivation = None
        if os.environ.get("DELIVERY_DCP_DRAFT"):
            from .dcp_observer_contract import install
            from parallel_tiny.dspark_formal_audit import stock_graph_observer
            self._delivery_dcp_derivation = install(os.environ["DELIVERY_DCP_DRAFT"], stock_graph_observer)
        return super().load_model(*args, **kwargs)

    def delivery_describe(self):
        from vllm.distributed import get_tensor_model_parallel_rank
        config = self.vllm_config
        text = config.model_config.hf_text_config
        if config.load_config.load_format == "dummy" or text.num_hidden_layers != 40:
            raise ValueError("Actual full40 weights are required")
        if _physical_derivation is not None:
            from .build_physical_worker import sha as geometry_sha, verify as verify_geometry
            root = os.environ["DELIVERY_PHYSICAL_WORKER"]
            manifest = verify_geometry(root)
            if (geometry_sha(inspect.getfile(Base)) != manifest["files"]["worker.py"]["sha256"]
                    or geometry_sha(os.path.join(root, "manifest.json"))
                    != os.environ["DELIVERY_PHYSICAL_WORKER_MANIFEST_SHA256"]):
                raise ValueError("Actually loaded physical worker differs from the selected manifest")
        compressor_source = None
        if os.environ.get("DELIVERY_CED_COMPRESSOR"):
            import sys
            from .build_ced_compressor import MODULE, RELATIVE, sha, verify
            manifest = verify(os.environ["DELIVERY_CED_COMPRESSOR"])
            actual = sys.modules.get(MODULE)
            if actual is None or not getattr(actual, "__file__", None):
                raise ValueError("Selected compressor source was not actually loaded")
            compressor_source = sha(actual.__file__)
            if compressor_source != manifest["files"][RELATIVE]["sha256"]:
                raise ValueError("Actual compressor source is not the selected candidate")
        result = {"rank": get_tensor_model_parallel_rank(), "base_worker_source": inspect.getfile(Base),
                "form": os.environ["DELIVERY_FORM"], "tp": config.parallel_config.tensor_parallel_size,
                "physical_chips": os.environ.get("STACK_PHYSICAL_CHIPS"),
                "physical_worker_derivation": _physical_derivation,
                "dcp": config.parallel_config.decode_context_parallel_size,
                "dcp_software_ordered_reduction": os.environ.get("V41_DCP_DET_REDUCE", "0") == "1",
                "async_scheduling": config.scheduler_config.async_scheduling,
                "draft_graph": bool(self.model_runner.drafter.use_cuda_graph),
                "speculative_tokens": config.speculative_config.num_speculative_tokens,
                "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT"),
                "engram_device_index": os.environ.get("V41_ENGRAM_DEVICE_INDEX"),
                "ced_overlay_manifest_sha256": os.environ.get("DELIVERY_CED_MANIFEST_SHA256"),
                "ced_compressor_manifest_sha256": os.environ.get("DELIVERY_CED_COMPRESSOR_MANIFEST_SHA256"),
                "ced_compressor_source_sha256": compressor_source,
                "native_opp_fingerprint": os.environ.get("DELIVERY_CED_OPP_FINGERPRINT"),
                "dcp_proposer_derivation": getattr(self, "_delivery_dcp_derivation", None)}
        if os.environ.get("DELIVERY_SHARED_ENGRAM"):
            from .shared_engram_contract import describe_selected
            result["shared_engram"] = describe_selected(self.model_runner.model)
        return result

    def delivery_begin(self):
        counter = self.model_runner._formal_dspark_counter
        self._delivery_prefill_start = {}
        self._delivery_terminal_prefill = {}
        self._delivery_totals = {"decode_rounds": 0, "decode_tokens": None if getattr(counter, "async_mode", False) else 0,
                                 "prefill_calls": 0, "prefill_seconds": 0.0, "decode_seconds": 0.0,
                                 "draft_graph_calls": 0, "draft_eager_calls": 0}
        self._delivery_completed = []
        self._delivery_last_window = None
        original = counter.execute_begin

        @functools.wraps(original)
        def begin(scheduler, requests):
            now = time.monotonic_ns()
            new = {r.req_id: r for r in scheduler.scheduled_new_reqs}
            for req_id in scheduler.num_scheduled_tokens:
                state = new.get(req_id) or requests.get(req_id)
                if state is not None and req_id not in counter.requests:
                    self._delivery_prefill_start[req_id] = now
            return original(scheduler, requests)
        counter.execute_begin = begin
        original_bookkeeping = counter.bookkeeping_finished

        @functools.wraps(original_bookkeeping)
        def bookkeeping(output):
            result = original_bookkeeping(output)
            # Chunked prefill may sample/discard intermediate chunks. The
            # decode boundary is the LAST prompt chunk, never the first one.
            for req_id, row in (counter.pending or {}).items():
                request = counter.requests[req_id]
                if (row["phase"] == "prefill"
                        and row["computed_before"] + row["scheduled_tokens"] >= request["prompt_tokens"]):
                    self._delivery_terminal_prefill[req_id] = row["bookkeeping_end_ns"]
            return result
        counter.bookkeeping_finished = bookkeeping
        return {"rank": counter.rank, "armed_after_warmup": False, "measurement_ready_after_warmup": True}

    def delivery_arm(self, tag, max_tokens):
        counter = self.model_runner._formal_dspark_counter
        if counter.tag is not None or counter.pending is not None or self.model_runner.execute_model_state is not None:
            raise RuntimeError("A previous measurement or scheduled sample is still active")
        self._delivery_prefill_start = {}
        self._delivery_terminal_prefill = {}
        # Fences are request-window state, never reusable completion evidence.
        counter.terminal_fence = None
        counter.begin(tag, None if getattr(counter, "async_mode", False) else max_tokens)
        self._delivery_budget = max_tokens
        return {"rank": counter.rank, "tag": tag, "max_tokens": max_tokens,
                "async_mode": bool(getattr(counter, "async_mode", False)), "armed": True}

    def delivery_metrics(self):
        counter = self.model_runner._formal_dspark_counter
        if counter.tag is not None:
            if counter.pending is not None or not counter.requests:
                raise RuntimeError("Do not finish a measurement until its HTTP response has completed")
            import formal_dspark_counters
            receipt = formal_dspark_counters.finish(self)
            if len(receipt["requests"]) != 1 or receipt["errors"]:
                raise ValueError("A serial request window must bind one actual request and no errors")
            req_id, request = next(iter(receipt["requests"].items()))
            rounds = request["rounds"]
            async_mode = bool(receipt.get("async_mode"))
            raw_tokens = None if async_mode else sum(len(r["token_ids"]) for r in rounds)
            if not rounds or request["decode_start_ns"] is None:
                raise ValueError("No actual decode rounds or terminal-prefill boundary")
            fenced = bool(receipt.get("terminal_fence")) if async_mode else bool(rounds[-1].get("terminal_device_synced"))
            joined = rounds[-1].get("terminal_tp_join", {})
            if async_mode and (not fenced or not joined or not rounds[-1].get("terminal_device_synced")):
                raise ValueError("Async measurement requires actual per-rank device fence and terminal TP join")
            if not fenced:
                # Early EOS on the synchronous arm may precede the token budget.
                # Fence here and label its timing; never claim a pre-response fence.
                import torch
                torch.npu.synchronize()
            fenced_end = (receipt.get("terminal_fence") or {}).get("sync_returned_ns", time.monotonic_ns())
            seed = self._delivery_terminal_prefill.get(req_id, request["decode_start_ns"])
            start = self._delivery_prefill_start.get(req_id)
            counts = self._delivery_totals
            counts["decode_rounds"] += len(rounds)
            if raw_tokens is not None:
                counts["decode_tokens"] += raw_tokens
            counts["prefill_calls"] += request["prefill_calls"]
            if start is not None:
                counts["prefill_seconds"] += (seed - start) / 1e9
            counts["decode_seconds"] += (fenced_end - seed) / 1e9
            proposals = request["initial_draft"] + [p for r in rounds for p in r["draft_calls"]]
            counts["draft_graph_calls"] += sum(bool(p["runtime_graph"]) for p in proposals)
            counts["draft_eager_calls"] += sum(not p["runtime_graph"] for p in proposals)
            coverage = [{k: r[k] for k in ("scheduled_tokens", "scheduled_draft_token_ids")} for r in rounds]
            coverage_sha = hashlib.sha256(json.dumps(coverage, sort_keys=True).encode()).hexdigest()
            detail = {"request_id": req_id, "tag": receipt["tag"], "window_fingerprint": coverage_sha,
                      "async_mode": async_mode, "prompt_tokens": request["prompt_tokens"],
                      "profile_excluded": receipt.get("capture_profile_counted") is False,
                      "seed_tokens": 1 if async_mode else len(request["seed_token_ids"]), "decode_rounds": len(rounds),
                      "raw_decode_tokens": raw_tokens,
                      "last_round_tokens": None if async_mode else len(rounds[-1]["token_ids"]),
                      "terminal_fence_observed": True, "terminal_tp_join_observed": bool(joined),
                      "decode_start_ns": seed, "terminal_sample_end_ns": rounds[-1]["sample_end_ns"],
                      "fenced_end_ns": fenced_end,
                      "prefill_wall_scope": "first scheduled prefill -> terminal prefill CPU bookkeeping",
                      "decode_wall_scope": "terminal prefill CPU bookkeeping -> worker completion fence; may include metrics RPC delay"}
            self._delivery_completed.append(detail)
            self._delivery_last_window = {"tag": receipt["tag"], "window_fingerprint": coverage_sha}
        return {"rank": counter.rank, "cumulative": dict(self._delivery_totals), "requests": self._delivery_completed,
                "last_window": self._delivery_last_window,
                "pending": counter.pending is not None, "errors": list(counter.errors),
                "counter_scope": "actual runner scheduled/request CPU-token bookkeeping",
                "decode_tokens_scope": "HTTP final usage required for async; raw worker tokens before terminal trimming for sync"}
