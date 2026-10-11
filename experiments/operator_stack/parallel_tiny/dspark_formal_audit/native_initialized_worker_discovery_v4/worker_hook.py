"""Minimal private new-snapshot hook; never patch an existing worker class."""
import os
import sys

if __package__:
    from .contracts import WorkerGateLifecycle
    from .initialized_worker_gate import run_in_initialized_worker
    from .runtime_scope import already_imported_worker_modules,describe_source_metadata
else:
    from contracts import WorkerGateLifecycle
    from initialized_worker_gate import run_in_initialized_worker
    from runtime_scope import already_imported_worker_modules,describe_source_metadata


def make_worker_class(base_worker_class):
    """The controller's new snapshot imports this factory for its worker-cls."""
    class NativeInitializedCopyGateWorker(base_worker_class):
        def init_device(self, *args, **kwargs):
            if hasattr(self,'_native_initialized_gate_lifecycle'):
                raise ValueError('Normal init-return lifecycle was already installed')
            result=super().init_device(*args,**kwargs)
            self._native_initialized_gate_lifecycle=WorkerGateLifecycle(id(self))
            return result

        def describe_native_initialized_copy_gate(self, *, library_paths=None):
            lifecycle=self._native_initialized_gate_lifecycle
            npu=sys.modules.get('torch_npu')
            if (lifecycle.worker_object_id!=id(self) or lifecycle.pid!=os.getpid()
                    or npu is None or not npu.npu.is_initialized()):
                raise ValueError('Describe only the actual normally initialized new worker')
            torch,npu,distributed=already_imported_worker_modules()
            result={**lifecycle.identity(),'tp_rank':distributed.get_tensor_model_parallel_rank(),
                    'worker_rank':self.rank,'worker_device':str(self.device),
                    'scope':'read-only worker/source/mapped ABI identity after normal init return, no device receipt',
                    'device_audit_passed':False,'safe_formal_receipt':False}
            if library_paths is not None:
                manifest,runtime=describe_source_metadata(self,torch,npu,distributed,library_paths)
                result['source_manifest']=manifest
                result['loaded_abi_proofs']={name:receipt['loaded_abi_proof']
                                             for name,receipt in runtime.receipts.items()}
                lifecycle.resources={'discovery_runtime_handles':runtime}
            return result

        def run_native_initialized_copy_gate(self, **kwargs):
            return run_in_initialized_worker(self,**kwargs)

    return NativeInitializedCopyGateWorker
