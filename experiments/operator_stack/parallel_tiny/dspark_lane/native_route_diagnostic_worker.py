"""Native no-draft-capture worker; immediate values only outside capture."""
import os
import hashlib
from pathlib import Path
from types import MethodType
from dspark_worker import DSparkTinyWorker


class NativeRouteDiagnosticWorker(DSparkTinyWorker):
    def load_model(self,*args,**kwargs):
        assert self.vllm_config.speculative_config.enforce_eager
        assert os.environ["DRAFT_GRAPH"] == "0"
        result = super().load_model(*args,**kwargs)
        drafter = self.model_runner.drafter
        assert not drafter.use_cuda_graph
        from vllm_ascend.spec_decode import dspark_proposer, llm_base_proposer
        for module, key in ((dspark_proposer, "DSPARK_EXPECTED_PROPOSER_SHA256"),
                            (llm_base_proposer, "DSPARK_EXPECTED_BASE_SHA256")):
            assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == os.environ[key]
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context
        self._native_helper_calls = 0
        original_helper = drafter._run_merged_draft
        assert drafter._runnable == original_helper

        def draft_context_guard(this, *a, **kw):
            context = get_forward_context()
            assert not getattr(context, "capturing", False), "Draft entered capture in eager-only process"
            assert context.cudagraph_runtime_mode == CUDAGraphMode.NONE, "Native draft inherited graph runtime"
            self._native_helper_calls += 1
            return original_helper(*a, **kw)

        # A module forward hook only sees the first compiler trace. The plain
        # native helper is outside that compiled region and runs on every call.
        drafter._run_merged_draft = MethodType(draft_context_guard, drafter)
        drafter._runnable = drafter._run_merged_draft
        import graph_precision_audit
        print("NATIVE_ROUTE_DIAGNOSTIC",graph_precision_audit.install(self),flush=True)
        if os.environ.get("DSPARK_FIRST_VALUE_TRACE") == "1":
            from native_first_value_trace import FirstValueTrace
            self._tiny_first_value_trace = FirstValueTrace(self)
            print("NATIVE_FIRST_VALUE_TRACE", self._tiny_first_value_trace.install(), flush=True)
        return result

    def compile_or_warm_up_model(self,*args,**kwargs):
        result=super().compile_or_warm_up_model(*args,**kwargs)
        drafter = self.model_runner.drafter
        assert not drafter.use_cuda_graph
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        assert not isinstance(drafter._runnable, ACLGraphWrapper)
        assert drafter._runnable == drafter._run_merged_draft
        print("NATIVE_DRAFT_CAPTURE_ABSENT", {"native_helper_calls": self._native_helper_calls,
              "graph_wrapper": False, "graph_entries": 0, "all_observed_draft_contexts_none": True,
              "constructor_use_cuda_graph": False}, flush=True)
        import graph_precision_audit
        print("NATIVE_ROUTE_PARAMETER_SEAL",graph_precision_audit.freeze_parameters(self),flush=True)
        return result


def coverage(worker):
    return {"native_draft_helper_calls": worker._native_helper_calls,
            "draft_graph_wrapper": False, "all_observed_draft_contexts_none": True,
            "guard_at": "actual native _run_merged_draft entry",
            "draft_capture": False, "constructor_use_cuda_graph": worker.model_runner.drafter.use_cuda_graph}
