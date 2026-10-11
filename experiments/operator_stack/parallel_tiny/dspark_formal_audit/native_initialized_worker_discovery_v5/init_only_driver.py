"""Fresh TP8 normal WorkerWrapper init -> SDK RPC -> exit; no model loading.

The root host launcher owns fresh device/health authorization. This driver
never creates a container, chooses replacement chips, or repairs SDK state.
"""
import argparse
import hashlib
import json
import multiprocessing
from multiprocessing.connection import wait
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

# run_stack.sh executes a file, while children use spawn. Both routes load one
# package namespace; never import flat contracts/worker_hook duplicates.
if not __package__:
    stack_root=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(stack_root))
    __package__='parallel_tiny.dspark_formal_audit.native_initialized_worker_discovery_v5'

from .contracts import digest
from .controller_cpu import write_controller_bundle


STANDARD_SOURCES = {
    '/vllm-workspace/vllm/vllm/v1/worker/worker_base.py':
        '7da44338c2645ebf03d23394e452b31a8e3da1011fd1b42fcfcccfe99551b3fe',
    '/vllm-workspace/vllm/vllm/v1/executor/multiproc_executor.py':
        'ceb3477473bdb1de36e01687e0b2083ad239d42b1ddc2ec8cec16ba2c477efa6',
    '/vllm-workspace/vllm/vllm/engine/arg_utils.py':
        '165f9ec6e3b067bb7e6c72d68adedde7fa4d5ed7932fa7568064b9d9086e6830',
    '/vllm-workspace/vllm/vllm/config/speculative.py':
        '83cbb26932b9ec0633753b58e8fe666f55c779580b32fba750cf569cde67c40c',
    '/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker.py':
        '2eb424498fd1a908dece44a56451a2e42bb4697614327aaa91aaa171e3f0f127',
}
WORKER_CLS = 'parallel_tiny.dspark_formal_audit.native_initialized_worker_discovery_v5.sdk_worker.InitializedSdkGateWorker'


def guard_standard_sources():
    receipts={}
    for path,sha in STANDARD_SOURCES.items():
        record=digest(path)
        if record['sha256']!=sha:raise ValueError('Standard normal-init bootstrap source differs: '+path)
        receipts[path]=record
    return receipts


def init_standard_wrapper(wrapper_class, config, rank, distributed_init_method, shared_worker_lock):
    """Use the standard public split init APIs, as in WorkerProc lines619–640."""
    wrapper=wrapper_class(rpc_rank=rank,global_rank=rank)
    kwargs=[{} for _ in range(8)]
    kwargs[rank]={'vllm_config':config,'local_rank':rank,'rank':rank,
                  'distributed_init_method':distributed_init_method,
                  'is_driver_worker':rank==0,'shared_worker_lock':shared_worker_lock}
    wrapper.init_worker(kwargs)
    wrapper.init_device()
    return wrapper


def _rank_main(rank, config, distributed_init_method, shared_worker_lock, connection, library_paths):
    wrapper=None;quarantined=False
    try:
        guard_standard_sources()
        from vllm.v1.worker.worker_base import WorkerWrapperBase
        wrapper=init_standard_wrapper(WorkerWrapperBase,config,rank,distributed_init_method,shared_worker_lock)
        # This is an actual absence check after the standard init method, not
        # a metadata bool used to bypass a model-load requirement.
        if getattr(wrapper.worker.model_runner,'model',None) is not None:
            raise ValueError('Standard init unexpectedly loaded a model; init-only scope is invalid')
        descriptor=wrapper.describe_native_initialized_copy_gate(library_paths=library_paths)
        descriptor['standard_wrapper_source']=digest(inspect_wrapper_file(WorkerWrapperBase))
        descriptor['model_load_called']=False
        connection.send({'kind':'initialized','rank':rank,'descriptor':descriptor})
        message=connection.recv()
        if message['operation']!='run_one_variant':raise ValueError('Unexpected controller operation')
        result=wrapper.run_native_initialized_copy_gate(**message['kwargs'])
        result['bootstrap_scope']='normal WorkerWrapper init_device only; load_model never invoked'
        result['model_load_called']=False
        connection.send({'kind':'result','rank':rank,'result':result})
        quarantined=not result['passed']
        if not quarantined:
            stop=connection.recv()
            if stop['operation']!='stop':raise ValueError('One case only; no additional SDK command')
    except BaseException as exc:
        quarantined=True
        try:
            failure={'kind':'failure','rank':rank,'exception':str(exc),'traceback':traceback.format_exc()}
            if getattr(exc,'loaded_abi_proof',None) is not None:
                failure['rejected_loaded_abi_proof']=exc.loaded_abi_proof
            connection.send(failure)
        except (BrokenPipeError,EOFError,OSError):pass
    finally:
        # No new device operations after a runtime failure. Probe tensors stay
        # retained by lifecycle until this fresh process exits. Normal success
        # uses the SDK owner's standard shutdown; no explicit device reset.
        if wrapper is not None and not quarantined:
            try:wrapper.shutdown()
            except BaseException:quarantined=True
        connection.close()
    if quarantined:raise SystemExit(1)


def inspect_wrapper_file(wrapper_class):
    import inspect
    return inspect.getfile(wrapper_class)


def _collect(connections, processes, kind, timeout, *, raw_root):
    pending=set(connections);messages=[];deadline=time.monotonic()+timeout
    raw_root=Path(raw_root);raw_root.mkdir(parents=True,exist_ok=False)
    while pending:
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('New SDK workers timed out during '+kind)
        ready=wait(list(pending),timeout=min(remaining,1))
        for connection in ready:
            message=connection.recv();pending.remove(connection);messages.append(message)
            # Persist before validating/throwing: a failed rank must not erase
            # already-received successful or failed actual RPC messages.
            rank=connections.index(connection)
            (raw_root/f'rank{rank}.json').write_text(json.dumps(message,indent=2,default=str)+'\n')
            if message['kind']!=kind:raise RuntimeError(json.dumps(message))
            if kind=='result' and not message['result']['passed']:
                raise RuntimeError('Actual SDK gate failed rank'+str(message['rank'])+': '+json.dumps(message['result']))
        if not ready:
            for process in processes:
                if process.exitcode not in (None,0):raise RuntimeError('An owned SDK worker exited before '+kind)
    return sorted(messages,key=lambda message:message['rank'])


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model',required=True)
    parser.add_argument('--physical-chips',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=('static_default','nonstatic_default','static_keep','nonstatic_keep'),required=True)
    parser.add_argument('--generations',type=int,default=32)
    parser.add_argument('--acl-library',default='/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so')
    parser.add_argument('--runtime-library',default='/usr/local/Ascend/ascend-toolkit/latest/lib64/libruntime.so')
    parser.add_argument('--timeout-seconds',type=int,default=600)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    args.output=args.output.resolve()
    audit_job_id=args.output.parent.name
    if not audit_job_id:raise ValueError('Output needs an explicit new audit job parent directory')
    audit_job_root=str(args.output.parent)
    controller_nonce=uuid.uuid4().hex
    controller_path=str(args.output/'metadata/controller.json')
    if args.physical_chips!='8,9,10,11,12,13,14,15' or args.generations<32:
        raise ValueError('Only the root-authorized formal TP8 assignment and at least32 generations')
    config_path=Path(args.model)/'config.json'
    if digest(config_path)['sha256']!='40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424':
        raise ValueError('Actual unchanged formal checkpoint config SHA mismatch')
    if args.dry_run:
        print(json.dumps({'scope':'CPU command/config preparation only','worker_cls':WORKER_CLS,
                         'audit_job_id':audit_job_id,'audit_job_root':audit_job_root,
                         'driver_output_root':str(args.output),'controller_path':controller_path,
                         'variant':args.variant,'normal_init_api':'WorkerWrapperBase.init_worker -> init_device',
                         'load_model_called':False,'npu_jobs':0,'device_audit_passed':False},indent=2))
        return 0
    # The separate root host launcher supplies and guards this allocation.
    if any(os.environ.get(key)!=args.physical_chips for key in ('ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES')):
        raise ValueError('Process visibility differs from the explicit main-thread8–15 assignment')
    sources=guard_standard_sources()
    args.output.mkdir(parents=True,exist_ok=False)
    from parallel_tiny.dspark_formal_prep.contract import model_kwargs,startup_env
    env=startup_env(args.output,audit=False,draft_graph=False)
    for key,value in env.items():os.environ[key]=value
    for key,relative in (('ASCEND_CACHE_PATH','cache/ascend'),('TRITON_CACHE_DIR','cache/triton'),('TMPDIR','tmp')):
        path=args.output/relative;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path.resolve())
    # Standard EngineArgs creates the complete formal config; no LLM/Executor
    # is constructed, since the stock WorkerProc would force weight loading.
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.executor.multiproc_executor import set_multiprocessing_worker_envs
    from vllm.utils.network_utils import get_distributed_init_method,get_loopback_ip,get_open_port
    import torch_npu
    if torch_npu.npu.is_initialized():raise ValueError('Controller process must not own an initialized NPU context')
    kwargs=model_kwargs(args.model,5,audit=False,draft_graph=False);kwargs['worker_cls']=WORKER_CLS
    config=EngineArgs(**kwargs).create_engine_config()
    if torch_npu.npu.is_initialized():raise ValueError('Config construction unexpectedly initialized the controller NPU')
    set_multiprocessing_worker_envs()
    init_method=get_distributed_init_method(get_loopback_ip(),get_open_port())
    context=multiprocessing.get_context('spawn');shared_lock=context.Lock()
    processes=[];connections=[];rank_results=[]
    result={'scope':'normal-init-only TP8 SDK gate; no model weights loaded or formal performance claim',
            'audit_job_id':audit_job_id,'audit_job_root':audit_job_root,
            'driver_output_root':str(args.output),'controller_path':controller_path,
            'controller_nonce':controller_nonce,
            'variant':args.variant,'checkpoint_config':digest(config_path),'bootstrap_sources':sources,
            'model_kwargs':kwargs,'device_audit_passed':False,'full_consumer_audit_passed':False,
            'safe_formal_receipt':False,'formal_model_validated':False,
            'actual_prepack_consumer_order_validated':False,'passed':False,'model_load_called':False,
            'performance_claim':None,'rank_results':rank_results}
    try:
        for rank in range(8):
            parent,child=context.Pipe()
            process=context.Process(target=_rank_main,args=(rank,config,init_method,shared_lock,child,
                {'acl':args.acl_library,'runtime':args.runtime_library}),name=f'NativeSDKRank{rank}')
            process.start();child.close();processes.append(process);connections.append(parent)
        result['owned_worker_pids']=[process.pid for process in processes]
        (args.output/'launch.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        ready=_collect(connections,processes,'initialized',args.timeout_seconds,raw_root=args.output/'raw_initialized')
        descriptions=[message['descriptor'] for message in ready]
        (args.output/'initialized_workers.json').write_text(json.dumps(descriptions,indent=2)+'\n')
        controller=write_controller_bundle(descriptions,metadata_root=args.output/'metadata',
            output_root=args.output/'sdk_results',checkpoint_config_path=config_path,
            audit_job_id=audit_job_id,nonce=controller_nonce,variant=args.variant,
            physical_chips=list(range(8,16)),generations=args.generations)
        result['controller']=controller
        for connection in connections:connection.send({'operation':'run_one_variant','kwargs':{
            'controller_path':controller['controller_path'],'controller_sha256':controller['controller_sha256']}})
        received=_collect(connections,processes,'result',args.timeout_seconds,raw_root=args.output/'raw_results')
        rank_results.extend(message['result'] for message in received)
        result['passed']=len(rank_results)==8 and all(receipt['passed'] for receipt in rank_results)
        for connection in connections:connection.send({'operation':'stop'})
    except BaseException as exc:
        result.update(passed=False,exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc())
    finally:
        # Only processes created by this driver are stopped; no tenant scan,
        # kill by name, chip migration, retry, pagecache clear or device reset.
        for connection in connections:connection.close()
        for process in processes:process.join(timeout=5 if result['passed'] else 0)
        for process in processes:
            if process.is_alive():process.terminate()
        for process in processes:process.join(timeout=10)
        result['worker_exit_codes']={str(rank):process.exitcode for rank,process in enumerate(processes)}
        if any(process.exitcode!=0 for process in processes):result['passed']=False
        (args.output/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps({'passed':result['passed'],'variant':args.variant,'model_load_called':False,
                      'device_audit_passed':False,'full_consumer_audit_passed':False,'output':str(args.output)},indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
