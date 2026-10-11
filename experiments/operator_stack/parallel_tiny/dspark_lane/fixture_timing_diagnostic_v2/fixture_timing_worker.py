"""Frozen native arithmetic, fixed cache owner, and host-only timing counters."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
from types import MethodType

from dspark_worker import DSparkTinyWorker
from timing_controls import WindowGuard


def actual_source_guard():
    directory = Path(__file__).resolve().parent
    source = json.loads(os.environ["DSPARK_TIMING_EXPECTED_SOURCES_JSON"])
    native = json.loads(os.environ["DSPARK_TIMING_EXPECTED_NATIVE_JSON"])
    seen = {}
    for name, expected in source.items():
        path = directory / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, (name, actual, expected)
        seen[name] = actual
    for relative, expected in native.items():
        module = importlib.import_module(relative[:-3].replace("/", "."))
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        assert actual == expected, (relative, actual, expected)
        seen[relative] = actual
    return {"files": seen, "all_actual_source_sha256_match": True, "outside_timing_window": True}


class FixtureTimingWorker(DSparkTinyWorker):
    def load_model(self, *args, **kwargs):
        required = {"DRAFT_GRAPH": "1", "DSPARK_GRAPH_CAPTURE_METADATA": "1",
                    "DSPARK_CAPTURE_VALUE_FIX": "1", "DSPARK_CAPTURE_NCTX_FIX": "1",
                    "DSPARK_CAPTURE_DISPATCH": "1", "DSPARK_DRAFT_METADATA_MODE": "sync",
                    "DSPARK_DRAFT_USE_CUDAGRAPH": "1", "DSPARK_CAPTURE_PAD_SLOTS": "0",
                    "DSPARK_HOIST_CONTEXT_KV": "0", "DSPARK_DRAFT_SYNC_BEFORE": "0",
                    "DSPARK_DRAFT_SYNC_AFTER": "0", "DSPARK_FIXTURE_VALIDATE_CONSUMERS": "0",
                    "DSPARK_ROUTE_IMMEDIATE_DIAGNOSTIC": "0", "DSPARK_FIRST_VALUE_TRACE": "0"}
        assert all(os.environ.get(key) == value for key, value in required.items()), required
        assert not self.vllm_config.speculative_config.enforce_eager
        self._fixture_timing_sources = actual_source_guard()
        result = super().load_model(*args, **kwargs)
        import torch
        import graph_precision_audit as audit
        import o_proj_cache_fixture as fixture
        import draft_graph_observer as observer
        from vllm_ascend.spec_decode import llm_base_proposer
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        assert not audit.STATE["router_roles"] and not audit.STATE["enabled"]
        self._fixture_timing_guard = guard = WindowGuard()
        for name in ("_diagnostic_inspect", "_cpu", "_parameter_metadata", "_diagnostic_write"):
            setattr(audit, name, guard.protect("audit." + name, getattr(audit, name)))
        for name in ("check_entry", "validate_all"):
            setattr(fixture, name, guard.protect("fixture." + name, getattr(fixture, name)))
        torch.npu.synchronize = guard.observe_synchronize(torch.npu.synchronize)
        print("TIMING_ACTUAL_SOURCE_GUARD", self._fixture_timing_sources, flush=True)
        print("TIMING_FIXED_CACHE_BEFORE_COMPILE", fixture.initialize_before_compile(self), flush=True)
        print("TIMING_GRAPH_OBSERVER", observer.install(self, os.environ["DSPARK_EXPECTED_ACLGRAPH_SHA256"]), flush=True)
        self._fixture_timing_state = state = {"arm": "on", "helper_history": [], "helper_marker": 0}
        original_flag = llm_base_proposer._dspark_rt_flag

        def runtime_flag(name, default="0"):
            if name == "DRAFT_FORCE_EAGER":
                return "1" if state["arm"] == "off" else "0"
            return original_flag(name, default)

        llm_base_proposer._dspark_rt_flag = runtime_flag
        runner, drafter = self.model_runner, self.model_runner.drafter
        assert drafter.use_cuda_graph and isinstance(drafter._runnable, ACLGraphWrapper)
        original_helper = drafter._run_merged_draft
        assert drafter._runnable.runnable == original_helper

        def helper(this, *args, **kwargs):
            from vllm.forward_context import get_forward_context
            context = get_forward_context()
            pending = runner._tiny_dspark_pending
            receipt = {"capturing": bool(getattr(context, "capturing", False)),
                       "runtime_mode": str(context.cudagraph_runtime_mode),
                       "descriptor": repr(context.batch_descriptor),
                       "actual_round_at_entry": None if pending is None else dict(pending),
                       "begin_s": time.perf_counter(), "returned": False}
            try:
                result = original_helper(*args, **kwargs)
                receipt["returned"] = True
                return result
            finally:
                receipt["end_s"] = time.perf_counter()
                state["helper_history"].append(receipt)

        drafter._run_merged_draft = MethodType(helper, drafter)
        drafter._runnable.runnable = drafter._run_merged_draft
        return result

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = super().compile_or_warm_up_model(*args, **kwargs)
        import graph_precision_audit as audit
        import o_proj_cache_fixture as fixture
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        runnable = self.model_runner.drafter._runnable
        assert isinstance(runnable, ACLGraphWrapper)
        entries = [entry for entry in runnable.concrete_aclgraph_entries.values() if entry.aclgraph is not None]
        assert len(entries) == 1 and entries[0].batch_descriptor.num_tokens == 6
        self._fixture_timing_capture = {"actual_entries": len(entries),
                                        "descriptors": [repr(entry.batch_descriptor) for entry in entries],
                                        "constructor_use_cuda_graph": self.model_runner.drafter.use_cuda_graph,
                                        "bool_flip": False}
        self._fixture_timing_parameter_seal = audit._parameter_metadata(self)
        print("TIMING_NATIVE_DRAFT_CAPTURE", self._fixture_timing_capture, flush=True)
        print("TIMING_FIXED_CACHE_AFTER_CAPTURE", fixture.validate_all(self, "timing.after_capture"), flush=True)
        return result


def metadata(worker):
    assert not worker._fixture_timing_guard.active
    history = worker._fixture_timing_state["helper_history"]
    return {"source_guard": worker._fixture_timing_sources, "capture": worker._fixture_timing_capture,
            "parameter_metadata_views": len(worker._fixture_timing_parameter_seal),
            "cold_helper_history": [row for row in history if row["capturing"]],
            "fixture_consumer_d2h_checks_enabled": False, "route_logprob_audit": False,
            "first_value_trace": False, "profiler": False, "admitted": False, "formal_tp8_claim": False}


def begin_request(worker, arm, tag):
    import torch
    import graph_precision_audit as audit
    import o_proj_cache_fixture as fixture
    import dspark_worker
    import draft_graph_observer as observer
    guard, state = worker._fixture_timing_guard, worker._fixture_timing_state
    guard.require_outside("begin_request")
    assert arm in ("off", "on") and not worker.model_runner._tiny_dspark_recording
    assert audit._parameter_metadata(worker) == worker._fixture_timing_parameter_seal
    assert not audit.STATE["router_roles"] and not audit.STATE["enabled"]
    before = fixture.validate_all(worker, tag + ".before")
    torch.npu.synchronize()
    state["arm"] = arm
    state["helper_marker"] = len(state["helper_history"])
    state["tag"] = tag
    dspark_worker.reset_receipts(worker)
    observer.begin(worker)
    guard.begin()
    return before


def finish_request(worker):
    import graph_precision_audit as audit
    import o_proj_cache_fixture as fixture
    import dspark_worker
    import draft_graph_observer as observer
    guard, state = worker._fixture_timing_guard, worker._fixture_timing_state
    assert guard.active
    guard.terminal_fence_expected = True
    try:
        counter = dspark_worker.read_receipts(worker)
    finally:
        guard.terminal_fence_expected = False
        guard.end()
    graph = observer.read(worker)
    history = state["helper_history"][state["helper_marker"]:]
    helper = {"native_helper_calls_this_request": len(history),
              "native_helper_successful_returns_this_request": sum(row["returned"] for row in history),
              "cold_capture_helper_calls_global": sum(row["capturing"] for row in state["helper_history"]),
              "cold_capture_helper_successful_returns_global": sum(row["capturing"] and row["returned"] for row in state["helper_history"]),
              "history": history, "host_only_counter": True, "graph_replay_executes_no_python_helper": True}
    after = fixture.validate_all(worker, state["tag"] + ".after")
    assert audit._parameter_metadata(worker) == worker._fixture_timing_parameter_seal
    return {"counter": counter, "graph": graph, "helper": helper,
            "fixture_after": after, "window_guard": guard.receipt()}
