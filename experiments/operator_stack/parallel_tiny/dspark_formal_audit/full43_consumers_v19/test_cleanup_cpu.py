"""Run the actual v10 cleanup methods on this harness's ordinary Linux children.

The SDK module is parsed, never imported. Its executor base is a CPU placeholder;
the v10 capture and cleanup method bodies and frozen v8 pidfd helper are actual.
"""
import __future__
import ast
import copy
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent


def ordinary_child(seconds):time.sleep(seconds)


def executor_class():
    path=HERE.parent/'native_initialized_worker_discovery_v8/owned_worker_cleanup.py'
    spec=importlib.util.spec_from_file_location('_actual_v8_owned_cleanup_for_v10_CPU',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    tree=ast.parse((HERE/'sdk_executor.py').read_text())
    node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Full43AuditExecutor'))
    scope={'MultiprocExecutor':object,'owner_identity':module.owner_identity,
        'capture_owned_worker':module.capture_owned_worker,'finish_owned_workers':module.finish_owned_workers,
        'json':json,'Path':Path,'os':os}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(HERE/'sdk_executor.py'),
        'exec',flags=__future__.annotations.compiler_flag,dont_inherit=True),scope)
    return scope[node.name],module


def run_case(cls,helper,output,*,success):
    executor=cls();executor._v10_owner=helper.owner_identity();executor._v10_owned=[]
    executor._v10_failed=not success;executor.is_failed=False
    context=multiprocessing.get_context('spawn');children=[]
    os.environ['V10_AUDIT_OUTPUT']=str(output/'rank_exports');output.mkdir()
    for rank,seconds in enumerate([.2,.35] if success else [60,60]):
        process=context.Process(target=ordinary_child,args=(seconds,));process.start();children.append(process)
        executor._v10_capture(SimpleNamespace(proc=process,rank=rank))
    start=time.monotonic()
    executor._ensure_worker_termination(children)
    elapsed=time.monotonic()-start
    report=json.loads((output/'owned_worker_cleanup.json').read_text())
    expected=0 if success else -signal.SIGKILL
    assert report['all_terminal'] and report['worker_exit_codes']=={'0':expected,'1':expected}
    assert all(p.exitcode==expected and not p.is_alive() for p in children)
    assert report['normal_success_grace_seconds']==(45 if success else 0)
    assert all(row['hard_kill_sent'] is (not success) for row in report['workers'])
    assert all(worker.pidfd is None for worker in executor._v10_owned)
    executor._ensure_worker_termination(children)  # Idempotent: no second signal/reap.
    assert success or elapsed<10
    wrong=SimpleNamespace(pid=99999999)
    try:executor._ensure_worker_termination([wrong])
    except ValueError:pass
    else:raise AssertionError('Uncaptured process set was accepted')
    return {'success':success,'elapsed_seconds':elapsed,'report':report}


def main():
    assert hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'A real Linux pidfd run is required; no skip counts as pass'
    cls,helper=executor_class();helper.require_pidfd_support()
    with tempfile.TemporaryDirectory(prefix='v10_owned_CPU_') as tmp:
        results=[run_case(cls,helper,Path(tmp)/name,success=success) for name,success in [('normal',True),('failure',False)]]
    forbidden=[name for name in sys.modules if any(name==p or name.startswith(p+'.') for p in ('torch','torch_npu','vllm','vllm_ascend','acl'))]
    assert not forbidden
    print(json.dumps({'scope':'v10 actual cleanup method / harness-owned ordinary Linux children only',
        'python':sys.version,'passed':True,'skipped':0,'SDK_executor_constructed':False,
        'NPU_calls':0,'SDK_imports':forbidden,'cases':results,'sdk_executor_source_sha256':hashlib.sha256((HERE/'sdk_executor.py').read_bytes()).hexdigest(),
        'device_audit_passed':False,'full_consumer_audit_passed':False,'safe_formal_receipt':False},indent=2))

if __name__=='__main__':main()
