"""Observe existing private ACL entries and successful native replay submissions."""
import hashlib
from pathlib import Path
import time

STATE = {"installed": False, "enabled": True, "roles": {}, "events": [], "active": None}


def install(worker, expected_sha256):
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm_ascend.compilation import acl_graph
    path = Path(acl_graph.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
    assert not STATE["installed"]
    original_call = acl_graph.ACLGraphWrapper.__call__
    original_replay = torch.npu.NPUGraph.replay
    runner = worker.model_runner
    draft = runner.drafter._runnable
    assert runner.drafter.use_cuda_graph, "Constructor has not configured the real draft graph path"
    assert isinstance(draft, acl_graph.ACLGraphWrapper), "Missing actual draft ACLGraphWrapper"
    STATE["roles"][id(draft)] = "draft"
    if isinstance(runner.model, acl_graph.ACLGraphWrapper):
        STATE["roles"][id(runner.model)] = "target"

    def call(this, *args, **kwargs):
        role = STATE["roles"].get(id(this))
        if role is None or not STATE["enabled"]:
            return original_call(this, *args, **kwargs)
        context = get_forward_context()
        descriptor = context.batch_descriptor
        mode = context.cudagraph_runtime_mode
        entry = this.concrete_aclgraph_entries.get(descriptor)
        graph = None if entry is None else entry.aclgraph
        addresses = [arg.data_ptr() for arg in args if isinstance(arg, torch.Tensor)]
        matching = bool(entry is not None and graph is not None and addresses == entry.input_addresses)
        event = {"role": role, "wrapper_id": id(this), "mode": str(mode),
                 "wrapper_mode": str(this.runtime_mode), "descriptor": repr(descriptor),
                 "capturing": bool(getattr(context, "capturing", False)),
                 "existing_entry": graph is not None, "positional_address_match": matching,
                 "positional_tensor_arguments": len(addresses),
                 "kwargs_native_address_guard": False,
                 "native_replay_returned": False, "begin_s": time.perf_counter()}
        active = {"event": event, "graph": graph, "eligible": bool(
            mode != CUDAGraphMode.NONE and mode == this.runtime_mode and matching)}
        previous = STATE["active"]
        STATE["active"] = active
        try:
            result = original_call(this, *args, **kwargs)
            event["call_returned"] = True
        finally:
            event["end_s"] = time.perf_counter()
            STATE["active"] = previous
            STATE["events"].append(event)
        return result

    def replay(this, *args, **kwargs):
        result = original_replay(this, *args, **kwargs)
        active = STATE["active"]
        if (STATE["enabled"] and active is not None and active["eligible"]
                and this is active["graph"]):
            active["event"]["native_replay_returned"] = True
        return result

    acl_graph.ACLGraphWrapper.__call__ = call
    torch.npu.NPUGraph.replay = replay
    STATE["installed"] = True
    STATE["source_sha256"] = expected_sha256
    return {"installed": True, "acl_graph_source_sha256": expected_sha256,
            "submission_only_not_device_or_precision_proof": True,
            "private_legacy_helper_scope": True, "real_existing_entries_only": True,
            "kwargs_guard_requires_separate_consumer_seals": True}


def begin(worker):
    STATE["events"].clear()
    return {"enabled": STATE["enabled"]}


def read(worker):
    assert STATE["active"] is None
    return {"events": list(STATE["events"]), "source_sha256": STATE["source_sha256"],
            "submission_only_not_device_or_precision_proof": True}
