"""Exercise exact native RPC method ASTs with CPU-only worker payloads."""
import ast
from contextlib import contextmanager
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace

BASE = Path(__file__).resolve().parent
NATIVE = BASE / "source_origin_readonly/native_source_read"
EXPECTED = {
    "vllm/v1/serial_utils.py": "d75c17a098231b2398cbdb7da6f9983e5b9229c0ba138b2874934d45fafd1f28",
    "vllm/v1/worker/worker_base.py": "7da44338c2645ebf03d23394e452b31a8e3da1011fd1b42fcfcccfe99551b3fe",
    "vllm/v1/executor/uniproc_executor.py": "4625c8e59aef1482600b02204701d348d7f716427581d3cf23b06bff9b056abe",
    "vllm/forward_context.py": "985adacc9e7ed2a7695819ac2e2ad9cb264a7d886806d3091ad75bfda7392242",
}


def execute(nodes, namespace, filename):
    # Method bodies are unchanged. Deferred annotations avoid SDK/type imports.
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(filename), "exec"), namespace)


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def methods(tree, class_name, names):
    source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    # CPU class shells omit unrelated base classes; exact native methods execute.
    return ast.ClassDef(name=class_name, bases=[], keywords=[], body=selected, decorator_list=[])


@contextmanager
def temporary_modules(modules):
    before = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, old in before.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def expect_rejection(call, message):
    try:
        call()
    except AssertionError as error:
        assert str(error) == message, (str(error), message)
        return
    raise AssertionError("Required entry rejection did not occur: " + message)


def run_rpc_guard_controls():
    controls = []
    sources = {}
    trees = {}
    for name, expected in EXPECTED.items():
        path = NATIVE / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, name
        sources[name] = actual
        trees[name] = ast.parse(path.read_text())
    namespace = {"partial": partial, "AsyncModelRunnerOutput": type("CPUAsyncOutput", (), {})}
    execute([function(trees["vllm/v1/serial_utils.py"], "run_method"),
             methods(trees["vllm/v1/worker/worker_base.py"], "WorkerWrapperBase", ("__init__", "__getattr__")),
             methods(trees["vllm/v1/executor/uniproc_executor.py"], "UniProcExecutor", ("collective_rpc",))],
            namespace, "exact_native_RPC_method_AST")
    wrapper_class, executor_class = namespace["WorkerWrapperBase"], namespace["UniProcExecutor"]
    execute([methods(ast.parse((BASE / "geometry_quality_worker.py").read_text()),
                     "GeometryQualityWorker", ("snapshot_geometry_step",))],
            namespace, BASE / "geometry_quality_worker.py")
    installed, wrong = namespace["GeometryQualityWorker"](), namespace["GeometryQualityWorker"]()
    wrapper, executor = wrapper_class(rpc_rank=0), executor_class()
    wrapper.worker = installed
    executor.driver_worker = wrapper
    assert wrapper is not installed
    assert executor.collective_rpc(lambda received: received)[0] is wrapper
    assert wrapper.snapshot_geometry_step.__self__ is installed
    controls.append("actual_UniProcExecutor_run_method_callable_receives_WorkerWrapper_not_inner_worker")

    forward = ModuleType("vllm.forward_context")
    execute([function(trees["vllm/forward_context.py"], "get_forward_context")],
            forward.__dict__, NATIVE / "vllm/forward_context.py")
    forward._forward_context = SimpleNamespace(capturing=False)
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    vllm.forward_context = forward
    audit = ModuleType("graph_precision_audit")
    audit.STATE = {"enabled": True, "current": {"tag": "CPU-entry-only"}}
    shared = BASE / "frozen_quality_shared/graph_precision_audit.py"
    execute([function(ast.parse(shared.read_text()), "_capturing"),
             function(ast.parse(shared.read_text()), "_diagnostic_write")],
            audit.__dict__, shared)
    audit.Path, audit.os = Path, os
    actual_write, writes = audit._diagnostic_write, []

    def counted_write(label, payload):
        assert not audit._capturing(), "CPU detected diagnostic I/O during capture"
        assert all(value is None or isinstance(value, (str, bool, int)) for value in payload.values())
        writes.append((label, dict(payload)))
        return actual_write(label, payload)

    audit._diagnostic_write = counted_write
    probe = ModuleType("geometry_cache_probe")
    probe.STATE = {"worker": installed, "snapshot_entry_host_guard_written": False}
    probe_tree = ast.parse((BASE / "geometry_cache_probe.py").read_text())
    execute([function(probe_tree, "validate_snapshot_entry")], probe.__dict__, BASE / "geometry_cache_probe.py")
    payloads = []

    def CPU_entry_payload(worker, is_decode, draft_replayed):
        # Only the device payload is substituted. RPC methods, worker adapter,
        # strict three-gate function and capture-state readers are actual ASTs.
        probe.validate_snapshot_entry(worker)
        payloads.append(worker)
        return worker

    probe.snapshot_step = CPU_entry_payload
    old_tree = ast.parse((BASE / "source_origin_readonly/frozen_v2_entry/geometry_cache_probe.py").read_text())
    old_snapshot = function(old_tree, "snapshot_step")
    old_guard = next(node for node in old_snapshot.body if isinstance(node, ast.Assert))
    old_namespace = {"STATE": probe.STATE, "audit": audit}
    old_entry = ast.FunctionDef(name="old_entry", args=old_snapshot.args,
                               body=[old_guard], decorator_list=[])
    execute([old_entry], old_namespace, "unaltered_v2_three_conjunct_assertion_AST")

    with TemporaryDirectory(prefix="geometry3_entry_cpu_") as temp:
        previous_output = os.environ.get("DSPARK_AUDIT_OUTPUT_PATH")
        os.environ["DSPARK_AUDIT_OUTPUT_PATH"] = temp
        try:
            with temporary_modules({"vllm": vllm, "vllm.forward_context": forward,
                                    "graph_precision_audit": audit, "geometry_cache_probe": probe}):
                expect_rejection(lambda: executor.collective_rpc(old_namespace["old_entry"], args=(True, False)), "")
                controls.append("old_callable_wrapper_rejected_by_exact_v2_identity_assertion")
                assert executor.collective_rpc("snapshot_geometry_step", args=(True, False))[0] is installed
                assert payloads[-1] is installed and writes[-1][1]["worker_identity_equal"]
                assert writes[-1][1]["worker_id"] == writes[-1][1]["installed_worker_id"] == id(installed)
                controls.append("new_string_RPC_exact_bound_worker_adapter_passes_strict_identity_audit_capture_gates")
                assert executor.collective_rpc("snapshot_geometry_step", args=(True, True))[0] is installed
                assert len(writes) == 1 and len(list(Path(temp).glob("*.json"))) == 1
                record = json.loads(next(Path(temp).glob("*.json")).read_text())
                assert record["host_scalars_only"] and not record["device_reads"]
                controls.append("noncapture_entry_host_scalar_identity_receipt_written_once")
                wrapper.worker = wrong
                probe.STATE["snapshot_entry_host_guard_written"] = False
                expect_rejection(lambda: executor.collective_rpc("snapshot_geometry_step", args=(True, False)),
                                 "geometry_snapshot.worker_identity_mismatch")
                controls.append("string_RPC_different_real_worker_owner_rejected_without_identity_normalization")
                wrapper.worker = installed
                probe.STATE["snapshot_entry_host_guard_written"] = False
                audit.STATE["enabled"] = False
                expect_rejection(lambda: executor.collective_rpc("snapshot_geometry_step", args=(True, False)),
                                 "geometry_snapshot.audit_disabled")
                controls.append("string_RPC_disabled_audit_rejected_with_distinct_reason")
                # Simultaneous wrong owner + disabled audit must still reject
                # capture first, before any host evidence write or device body.
                wrapper.worker = wrong
                probe.STATE["snapshot_entry_host_guard_written"] = False
                forward._forward_context = SimpleNamespace(capturing=True)
                writes_before, payloads_before = len(writes), len(payloads)
                expect_rejection(lambda: executor.collective_rpc("snapshot_geometry_step", args=(True, False)),
                                 "geometry_snapshot.capture_forbidden")
                assert len(writes) == writes_before and len(payloads) == payloads_before
                assert not probe.STATE["snapshot_entry_host_guard_written"]
                controls.append("capture_rejected_before_identity_audit_diagnostic_IO_or_device_payload")
                wrapper.worker, audit.STATE["enabled"], forward._forward_context = installed, True, None
                assert executor.collective_rpc("snapshot_geometry_step", args=(True, False))[0] is installed
                controls.append("actual_get_forward_context_missing_context_AssertionError_keeps_existing_false_fallback")
        finally:
            if previous_output is None:
                os.environ.pop("DSPARK_AUDIT_OUTPUT_PATH", None)
            else:
                os.environ["DSPARK_AUDIT_OUTPUT_PATH"] = previous_output

    new_snapshot = function(probe_tree, "snapshot_step")
    assert ast.unparse(new_snapshot.body[0]) == "validate_snapshot_entry(worker)"
    assert ast.dump(ast.Module(body=old_snapshot.body[4:], type_ignores=[]), include_attributes=False) == \
        ast.dump(ast.Module(body=new_snapshot.body[4:], type_ignores=[]), include_attributes=False)
    controls.append("complete_geometry_sync_reader665_cache_finite_receipt_body_AST_identical_after_entry_guard")
    assert "torch_npu" not in sys.modules and "acl" not in sys.modules
    return controls, sources


if __name__ == "__main__":
    controls, sources = run_rpc_guard_controls()
    print(json.dumps({"passed": True, "controls_passed": len(controls), "controls": controls,
                      "actual_RPC_source_sha256": sources,
                      "CPU_worker_payload_only_not_device_geometry_proof": True}, indent=2))
