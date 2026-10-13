"""Original MP executor with immediate owned birth capture and verified cleanup."""
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import types

from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from parallel_tiny.dspark_formal_audit.native_initialized_worker_discovery_v8.owned_worker_cleanup import (
    owner_identity,capture_owned_worker,finish_owned_workers)
from .executor_source import derive_executor_init


class Full43AuditExecutor(MultiprocExecutor):
    def _init_executor(self):
        self._v10_owner = owner_identity();self._v10_owned = [];self._v10_failed = False
        original = MultiprocExecutor._init_executor
        path = Path(inspect.getfile(original));data = path.read_bytes()
        node,self._v10_bootstrap_source = derive_executor_init(data.decode())
        node.decorator_list = [];tree = ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
        namespace = dict(original.__globals__);exec(compile(tree,'/private-v10/MP.init.py','exec'),namespace)
        function = namespace[node.name]
        try:return types.MethodType(function,self)()
        except BaseException:
            self._v10_failed = True
            raise

    def _v10_capture(self,handle):
        self._v10_owned.append(capture_owned_worker(handle.proc,handle.rank,self._v10_owner))

    def execute_model(self,*args,**kwargs):
        try:return super().execute_model(*args,**kwargs)
        except BaseException:self._v10_failed = True;raise

    def sample_tokens(self,*args,**kwargs):
        try:return super().sample_tokens(*args,**kwargs)
        except BaseException:self._v10_failed = True;raise

    def _ensure_worker_termination(self,worker_procs):
        # The ORIGINAL shutdown closes each original death_writer first. This
        # exact parent owns the captured Process handles, so actual reaped codes
        # and protected pidfd signals are available without descendant scans.
        root = Path(os.environ['V10_AUDIT_OUTPUT']).parent
        def write_error(exc):
            try:
                (root/'owned_worker_cleanup_error.json').write_text(json.dumps(
                    {'error_type':type(exc).__name__,'error':str(exc),
                     'shutdown_pids':[p.pid for p in worker_procs],
                     'captured_pids':[w.process.pid for w in self._v10_owned],
                     'device_audit_passed':False},indent=2)+'\n')
            except Exception:pass
        if {p.pid for p in worker_procs} != {w.process.pid for w in self._v10_owned}:
            exc = ValueError('Standard shutdown process set differs from captured owned workers')
            write_error(exc);raise exc
        if getattr(self,'_v10_cleanup',None) is not None:
            if self._v10_cleanup['all_terminal']:return
            exc = RuntimeError('Previously owned worker cleanup is still incomplete')
            write_error(exc);raise exc
        try:
            report = finish_owned_workers(self._v10_owned,self._v10_owner,
                success=not self._v10_failed and not getattr(self,'is_failed',False),success_grace_seconds=45)
        except BaseException as exc:
            write_error(exc);raise
        (root/'owned_worker_cleanup.json').write_text(json.dumps(report,indent=2)+'\n')
        self._v10_cleanup = report
        if not report['all_terminal']:raise RuntimeError('Actual owned worker cleanup is incomplete')
