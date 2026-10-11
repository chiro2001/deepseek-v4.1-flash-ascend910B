"""Use the source-guarded native runner capture context; no SDK import or repair."""
import ast
import hashlib
import inspect
from pathlib import Path
import sys
from types import ModuleType

if __package__:
    from .contracts import digest,assert_borrowed_scope_stable
else:
    from contracts import digest,assert_borrowed_scope_stable


RUNNER_MODULE='vllm_ascend.worker.model_runner_v1'
STABLE_DEVICE_FIELDS=('current_initialized','context_ptr','runtime_logical_device','torch_current_device',
                      'worker_device_index','physical_device','tp_rank','tp_world_size','worker_rank','global_rank')


def native_factory_source(worker):
    module=sys.modules.get(RUNNER_MODULE)
    if not isinstance(module,ModuleType) or not isinstance(worker.model_runner,module.NPUModelRunner):
        raise ValueError('Need this actual normal native NPUModelRunner owner')
    factory=module.graph_capture;function=inspect.unwrap(factory)
    if function.__module__!=RUNNER_MODULE or function.__qualname__!='graph_capture':
        raise ValueError('Native capture factory identity differs from the normal runner')
    receipt=digest(inspect.getfile(function));source=Path(receipt['path']).read_text()
    nodes=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='graph_capture']
    if len(nodes)!=1:raise ValueError('Native graph_capture source is missing or ambiguous')
    receipt['function_sha256']=hashlib.sha256(ast.get_source_segment(source,nodes[0]).encode()).hexdigest()
    receipt['qualified_function']=RUNNER_MODULE+'.graph_capture'
    return receipt,factory,module.GraphCaptureContext


def validate_native_entry(baseline, actual, context_stream_ptr, default_stream_ptr):
    if any(actual.get(key)!=baseline.get(key) for key in STABLE_DEVICE_FIELDS):
        raise ValueError('Native capture context changed initialized device/context/rank scope')
    if (type(context_stream_ptr) is not int or context_stream_ptr<=0
            or actual.get('stream_ptr')!=context_stream_ptr or context_stream_ptr==default_stream_ptr
            or actual.get('capturing') is not False or actual.get('runner_idle') is not True):
        raise ValueError('Native context must select its returned non-default stream outside capture')


class NativeCaptureScope:
    def __init__(self, worker, runtime, torch, npu, distributed, baseline, frozen_source, receipt, resources):
        source,factory,self.context_type=native_factory_source(worker)
        if source!=frozen_source:raise ValueError('Native capture factory source differs from controller frozen source')
        self.worker=worker;self.runtime=runtime;self.torch=torch;self.npu=npu;self.distributed=distributed
        self.baseline=baseline;self.receipt=receipt;self.resources=resources
        self.manager=factory(worker.device)
        receipt.update(owner=RUNNER_MODULE+'.graph_capture',source=source,
                       native_context_creates_stream=True,init_return_stream_borrowed_unchanged=False,
                       default_stream_ptr=int(torch.npu.default_stream(worker.device).npu_stream),
                       init_return_baseline=dict(baseline),normal_exit_restoration_verified=False,
                       gate_stream_constructor_called=False,gate_stream_patch_applied=False)

    def __enter__(self):
        self.context=self.manager.__enter__()
        try:
            if type(self.context) is not self.context_type:
                raise ValueError('Native graph_capture returned a different SDK context owner type')
            self.stream=self.context.stream
            self.scope,_=self.runtime.snapshot(self.worker,self.torch,self.npu,self.distributed)
            validate_native_entry(self.baseline,self.scope,int(self.stream.npu_stream),self.receipt['default_stream_ptr'])
            self.resources['native_capture_context']=self.context
            self.resources['native_capture_context_manager']=self.manager
            self.resources['native_capture_stream']=self.stream
            self.receipt.update(context_object_id=id(self.context),stream_object_id=id(self.stream),
                                native_scope=dict(self.scope),native_stream_ptr=int(self.stream.npu_stream))
            return self
        except BaseException:
            self.manager.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, exc, traceback):
        # SDK lifecycle restores its previous stream. No setter/repair/retry.
        result=self.manager.__exit__(exc_type,exc,traceback)
        self.receipt['native_context_exit_completed']=True
        if exc_type is None:
            after,_=self.runtime.snapshot(self.worker,self.torch,self.npu,self.distributed)
            assert_borrowed_scope_stable(self.baseline,after)
            if after['capturing'] or not after['runner_idle']:
                raise ValueError('Normal native context exit did not restore idle noncapturing baseline')
            self.receipt.update(normal_exit_restoration_verified=True,restored_baseline=after)
        else:
            self.receipt['failure_exit_uses_native_lifecycle_only']=True
        return result
