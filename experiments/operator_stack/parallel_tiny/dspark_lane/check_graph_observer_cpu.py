"""Negative controls: fake entries/booleans never count as successful replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS


def main():
    context = NS(batch_descriptor="6x1", cudagraph_runtime_mode="FULL", capturing=False)

    class Tensor:
        def __init__(self, ptr):
            self.ptr = ptr

        def data_ptr(self):
            return self.ptr

    class Graph:
        def __init__(self):
            self.throws = False

        def replay(self):
            if self.throws:
                raise RuntimeError("native replay failed")
            return None

    class Wrapper:
        def __init__(self):
            self.runtime_mode = "FULL"
            self.concrete_aclgraph_entries = {}
            self.other_graph = None

        def __call__(self, *args, **kwargs):
            if context.cudagraph_runtime_mode != self.runtime_mode:
                return "eager"
            if context.batch_descriptor not in self.concrete_aclgraph_entries:
                self.concrete_aclgraph_entries[context.batch_descriptor] = NS(
                    aclgraph=Graph(), input_addresses=[x.data_ptr() for x in args])
            entry = self.concrete_aclgraph_entries[context.batch_descriptor]
            (self.other_graph or entry.aclgraph).replay()
            return "graph-output"

    torch = types.ModuleType("torch")
    torch.Tensor = Tensor
    torch.npu = NS(NPUGraph=Graph)
    sys.modules["torch"] = torch
    config = types.ModuleType("vllm.config")
    config.CUDAGraphMode = NS(NONE="NONE")
    sys.modules[config.__name__] = config
    forward = types.ModuleType("vllm.forward_context")
    forward.get_forward_context = lambda: context
    sys.modules[forward.__name__] = forward
    acl = types.ModuleType("vllm_ascend.compilation.acl_graph")
    acl.ACLGraphWrapper = Wrapper
    acl.__file__ = __file__
    compilation = types.ModuleType("vllm_ascend.compilation")
    compilation.acl_graph = acl
    sys.modules[compilation.__name__] = compilation
    base = Path(__file__).resolve().parent
    path = base / "draft_graph_observer.py"
    spec = importlib.util.spec_from_file_location("observer", path)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    draft, target = Wrapper(), Wrapper()
    worker = NS(model_runner=NS(model=target, drafter=NS(_runnable=draft, use_cuda_graph=True)))
    expected = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        observer.install(worker, "0" * 64)
        raise AssertionError("Source mismatch incorrectly accepted")
    except AssertionError:
        assert not observer.STATE["installed"]
    observer.install(worker, expected)
    ptr = Tensor(11)
    checks = []

    def setup():
        observer.begin(worker)
        context.cudagraph_runtime_mode = "FULL"
        draft.other_graph = None
        draft.concrete_aclgraph_entries = {"6x1": NS(aclgraph=Graph(), input_addresses=[11])}

    setup()
    draft(ptr)
    assert observer.read(worker)["events"][-1]["native_replay_returned"]
    checks.append("existing matching entry plus native replay return counts")
    for mode in ("NONE", "PARTIAL"):
        setup()
        context.cudagraph_runtime_mode = mode
        draft(ptr)
        assert not observer.read(worker)["events"][-1]["native_replay_returned"]
        checks.append(f"runtime {mode} does not count")
    setup()
    draft.concrete_aclgraph_entries.clear()
    draft(ptr)
    assert not observer.read(worker)["events"][-1]["native_replay_returned"]
    checks.append("new entry/capture does not count as existing replay")
    setup()
    draft(Tensor(12))
    assert not observer.read(worker)["events"][-1]["native_replay_returned"]
    checks.append("address mismatch does not count")
    setup()
    draft.other_graph = Graph()
    draft(ptr)
    assert not observer.read(worker)["events"][-1]["native_replay_returned"]
    checks.append("unrelated graph object does not count")
    setup()
    draft.concrete_aclgraph_entries["6x1"].aclgraph.throws = True
    try:
        draft(ptr)
        raise AssertionError("Expected native replay error")
    except RuntimeError:
        assert not observer.read(worker)["events"][-1]["native_replay_returned"]
    checks.append("native replay exception does not count")
    result = {"passed": True, "cpu_mock_only_no_npu": True, "checks": checks,
              "source_mismatch_rejected": True, "device_consumer_or_cache_proof": False,
              "kwargs_address_guard_is_native": False,
              "observer_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (base / "evidence/graph_observer_cpu_gate.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
