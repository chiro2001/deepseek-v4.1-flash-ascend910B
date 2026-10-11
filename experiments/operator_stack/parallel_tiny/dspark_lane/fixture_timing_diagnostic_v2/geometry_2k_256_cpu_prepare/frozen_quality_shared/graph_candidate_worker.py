"""Construct the private complete draft graph path; never flip use_cuda_graph."""
import os
from pathlib import Path
import hashlib

from dspark_worker import DSparkTinyWorker


class GraphCandidateWorker(DSparkTinyWorker):
    def load_model(self, *args, **kwargs):
        assert not self.vllm_config.speculative_config.enforce_eager
        required = {"DRAFT_GRAPH": "1", "DSPARK_GRAPH_CAPTURE_METADATA": "1",
                    "DSPARK_CAPTURE_VALUE_FIX": "1", "DSPARK_CAPTURE_NCTX_FIX": "1",
                    "DSPARK_CAPTURE_DISPATCH": "1", "DSPARK_DRAFT_METADATA_MODE": "sync",
                    "DSPARK_DRAFT_USE_CUDAGRAPH": "1", "DSPARK_CAPTURE_PAD_SLOTS": "0",
                    "DSPARK_HOIST_CONTEXT_KV": "0", "DSPARK_DRAFT_SYNC_BEFORE": "0"}
        assert all(os.environ.get(k) == v for k, v in required.items()), required
        result = super().load_model(*args, **kwargs)
        from vllm_ascend.spec_decode import dspark_proposer, llm_base_proposer
        for module, key in ((dspark_proposer, "DSPARK_EXPECTED_PROPOSER_SHA256"),
                            (llm_base_proposer, "DSPARK_EXPECTED_BASE_SHA256")):
            assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == os.environ[key]
        assert self.model_runner.drafter.use_cuda_graph, "Real constructor gate rejected draft graph"
        import draft_graph_observer
        print("DSPARK_REAL_GRAPH_OBSERVER", draft_graph_observer.install(
            self, os.environ["DSPARK_EXPECTED_ACLGRAPH_SHA256"]), flush=True)
        import graph_precision_audit
        print("DSPARK_GRAPH_AUDIT_INSTALL", graph_precision_audit.install(self), flush=True)
        if os.environ.get("DSPARK_FIRST_VALUE_TRACE") == "1":
            from native_first_value_trace import FirstValueTrace
            self._tiny_first_value_trace = FirstValueTrace(self)
            print("DSPARK_FIRST_VALUE_TRACE", self._tiny_first_value_trace.install(), flush=True)
        return result

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = super().compile_or_warm_up_model(*args, **kwargs)
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        draft = self.model_runner.drafter._runnable
        assert isinstance(draft, ACLGraphWrapper)
        entries = [e for e in draft.concrete_aclgraph_entries.values() if e.aclgraph is not None]
        assert entries, "No real draft graph was captured"
        print("DSPARK_DRAFT_CAPTURED", {"entries": len(entries),
              "descriptors": [repr(e.batch_descriptor) for e in entries],
              "constructor_use_cuda_graph": self.model_runner.drafter.use_cuda_graph,
              "no_bool_flip": True}, flush=True)
        import graph_precision_audit
        print("DSPARK_GRAPH_PARAMETER_SEAL", graph_precision_audit.freeze_parameters(self), flush=True)
        return result
