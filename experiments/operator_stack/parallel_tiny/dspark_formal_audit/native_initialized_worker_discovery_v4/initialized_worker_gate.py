"""One real SDK case inside a newly initialized formal audit worker.

No device selection/init/repair, no new context or stream, no retry. Import is
stdlib-only. Actual execution requires a controller-frozen new audit snapshot.
"""
import importlib
import inspect
import json
from pathlib import Path
import time
import traceback

if __package__:
    from .contracts import (VARIANTS,WorkerGateLifecycle,read_frozen_json,validate_controller,
                            validate_initialized_facts,assert_borrowed_scope_stable,
                            validate_export,validate_copy_aliases,digest)
    from .runtime_scope import (already_imported_worker_modules,guard_worker_sources,
                                guard_formal_worker_config,AlreadyLoadedRuntime)
    from .graphex_source_identity import guard_graphex_imports
    from .tensor_tuple_transport import (FIELDS,SnapshotSpec,allocate_buffers,
                                         snapshot_tuple,reset_after_prior_fence)
else:
    from contracts import (VARIANTS,WorkerGateLifecycle,read_frozen_json,validate_controller,
                           validate_initialized_facts,assert_borrowed_scope_stable,
                           validate_export,validate_copy_aliases,digest)
    from runtime_scope import (already_imported_worker_modules,guard_worker_sources,
                               guard_formal_worker_config,AlreadyLoadedRuntime)
    from graphex_source_identity import guard_graphex_imports
    from tensor_tuple_transport import (FIELDS,SnapshotSpec,allocate_buffers,
                                        snapshot_tuple,reset_after_prior_fence)


def run_in_initialized_worker(worker, *, controller_path, controller_sha256, output=None):
    """RPC entry after the ordinary NPUWorker.init_device returned in a NEW job."""
    controller = read_frozen_json(controller_path,controller_sha256)
    validate_controller(controller)
    if str(Path(controller_path).resolve())!=controller['controller_path']:
        raise ValueError('Actual controller file differs from the full new-job path binding')
    lifecycle = getattr(worker,'_native_initialized_gate_lifecycle',None)
    if not isinstance(lifecycle,WorkerGateLifecycle):
        raise ValueError('Install the private init-return lifecycle hook in the new audit snapshot')
    lifecycle.begin(worker,controller['nonce'])
    static,keep = VARIANTS[controller['variant']]
    r = {'scope':'small initialized-worker SDK copy/epoch/alias gate only; no full43/model/performance receipt',
         'variant':controller['variant'],'static_kernel':static,
         'stage':'graphex',
         'outer_keep_inference_input_mutations':keep,'audit_job_id':controller['audit_job_id'],
         'audit_job_root':controller['audit_job_root'],'driver_output_root':controller['driver_output_root'],
         'controller_output_root':controller['output_root'],
         'controller_nonce':controller['nonce'],'controller':digest(controller_path),
         'process_identity':lifecycle.identity(),'phase':'initialized-scope-guard',
         'passed':False,'stage_semantics_passed':False,'actual_copy_alias_gate_passed':False,
         'device_audit_passed':False,'safe_formal_receipt':False,'formal_model_validated':False,
         'full_consumer_audit_passed':False,'actual_prepack_consumer_order_validated':False,
         'performance_claim':None,'generations':[],'copy_calls':[],
         'own_context_created':False,'own_stream_created':False,'device_selected_by_gate':False,
         'probe_allocation_reached':False,'source_gate':digest(__file__)}
    resources = dict(lifecycle.resources or {})
    output = Path(output) if output is not None else None
    compiler_original = compiler_owner = None
    prior_functionalization = None
    torch = None
    try:
        torch,npu,distributed = already_imported_worker_modules()
        guard_formal_worker_config(worker)
        manifest,sources = guard_worker_sources(worker,controller,torch,npu,distributed)
        r['sources'] = sources
        runtime = AlreadyLoadedRuntime(manifest['libraries'])
        r['runtime_library_sources'] = runtime.receipts
        before,stream = runtime.snapshot(worker,torch,npu,distributed)
        validate_initialized_facts(controller,before,lifecycle.identity())
        rank = before['tp_rank']
        process=next(item for item in controller['worker_processes'] if item['tp_rank']==rank)
        prior_proofs=read_frozen_json(process['loaded_abi_proofs_path'],process['loaded_abi_proofs_sha256'])
        fields=('pid','resolved_path','segments','map_files_link','mapped_file_stat','current_path_stat',
                'mapped_file_sha256','mapped_file_bytes','overlay_mount','executable_memory','symbol_addresses')
        if set(prior_proofs)!=set(runtime.receipts) or any(
                any(prior_proofs[name].get(key)!=receipt['loaded_abi_proof'].get(key) for key in fields)
                for name,receipt in runtime.receipts.items()):
            raise ValueError('Actual mapped ABI identity differs from this rank controller-bound discovery proof')
        r['controller_bound_loaded_abi_proofs']={
            'path':process['loaded_abi_proofs_path'],'sha256':process['loaded_abi_proofs_sha256']}
        expected_output = Path(controller['output_root'])/controller['variant']/f'rank{rank}'
        if output is None:output=expected_output
        if output.resolve() != expected_output.resolve():
            raise ValueError('Case output differs from the controller-bound rank/variant directory')
        output.mkdir(parents=True,exist_ok=False)
        r['initialized_worker_scope'] = before
        r['torch'] = torch.__version__;r['torch_npu'] = npu.__version__
        resources['runtime_handles'] = runtime
        resources['borrowed_stream'] = stream

        def assert_current_scope():
            actual,_ = runtime.snapshot(worker,torch,npu,distributed)
            assert_borrowed_scope_stable(before,actual)
            if actual['capturing'] or not actual['runner_idle']:
                raise ValueError('Model/capture activity overlaps the private SDK case')
            return actual

        from torch.utils._python_dispatch import TorchDispatchMode
        from torch._subclasses.fake_tensor import FakeTensor
        def stamp(tensor):
            if not isinstance(tensor,torch.Tensor) or isinstance(tensor,FakeTensor) or tensor.device.type != 'npu':
                raise ValueError('Actual probe pointer/layout must come from a real NPU Tensor')
            return {'ptr':tensor.data_ptr(),'shape':list(tensor.shape),'stride':list(tensor.stride()),
                    'dtype':str(tensor.dtype),'device':str(tensor.device),
                    'format':int(npu.get_npu_format(tensor))}
        class CopyMode(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                result = func(*args,**(kwargs or {}))
                if str(func) in ('aten.copy_.default','aten.copy.default') and len(args) >= 2:
                    if all(isinstance(t,torch.Tensor) and not isinstance(t,FakeTensor) and t.device.type=='npu' for t in args[:2]):
                        destination = args[0] if str(func)=='aten.copy_.default' else result
                        r['copy_calls'].append({'phase':r['phase'],'op':str(func),
                            'destination':stamp(destination),'logical_destination':stamp(args[0]),
                            'source':stamp(args[1]),'stream':int(torch.npu.current_stream().npu_stream)})
                return result

        assert_current_scope()
        r['phase']='probe-allocation';r['probe_allocation_reached']=True
        spec = SnapshotSpec(6,64)
        with torch.no_grad():
            x=torch.zeros((6,64),device=worker.device,dtype=torch.bfloat16)
            weight=torch.eye(64,dtype=torch.bfloat16,device='cpu').to(worker.device)
            positions=torch.arange(6,device=worker.device,dtype=torch.int64).reshape(6,1)
            valid=torch.ones((6,),device=worker.device,dtype=torch.bool)
            epoch=torch.zeros((1,),device=worker.device,dtype=torch.int64)
            indices=torch.arange(6,device=worker.device,dtype=torch.int64)
            cache=torch.zeros((16,64),device=worker.device,dtype=torch.bfloat16)
            producer=allocate_buffers(torch,spec,device=worker.device,dtype=torch.bfloat16)
            reader=allocate_buffers(torch,spec,device=worker.device,dtype=torch.bfloat16)
        resources.update(inputs=(x,weight,positions,valid,epoch,indices,cache),producer=producer,reader=reader)
        torch.npu.synchronize()
        assert_current_scope()
        buffers={f'{owner}.{field}':stamp(tensor) for owner,tensors in (('producer',producer),('reader',reader))
                 for field,tensor in zip(FIELDS,tensors)}
        r['buffers_before']=buffers
        marker_sources={buffers[f'{owner}.{field}']['ptr']:stamp(source)
                        for owner in ('producer','reader') for field,source in
                        (('positions',positions),('valid',valid),('epoch',epoch))}

        def program(*actual):
            x,weight,positions,valid,epoch,indices,cache=actual[:7]
            producer_inputs=actual[7:13];reader_inputs=actual[13:19]
            value=torch.nn.functional.linear(x,weight)
            producer_outputs=snapshot_tuple(torch,value,value,positions,valid,epoch,producer_inputs)
            value=producer_outputs[0]
            cache.index_copy_(0,indices,value)
            consumed=cache.index_select(0,indices)
            reader_outputs=snapshot_tuple(torch,value,consumed,positions,valid,epoch,reader_inputs)
            return (reader_outputs[0]*2,cache,*producer_outputs[1:],*reader_outputs[1:])

        # Compiler flags are scoped to this case in this private process.
        # A failed case still stops the entire audit job; restoration is CPU-only.
        prior_functionalization=torch._inductor.config.enable_auto_functionalized_v2
        torch._inductor.config.enable_auto_functionalized_v2=False
        nge=importlib.import_module('npugraph_ex')
        nfx=importlib.import_module('npugraph_ex.npu_fx_compiler')
        r['graphex_source_identity']=guard_graphex_imports(nge,nfx,sources)
        from npugraph_ex.configs.compiler_config import _process_kwargs_options
        from torch._dynamo.backends.common import aot_autograd
        config=nge.CompilerConfig()
        options={'force_eager':True,'inplace_pass':False,'clone_input':False,'clone_output':False,
                 'static_kernel_compile':static}
        if static:options['_vllm_aclnn_static_kernel_sym_range']=[1]
        _process_kwargs_options(config,{'options':options})
        r['compiler_options']=config.as_dict()
        raw=nge.get_npu_backend(compiler_config=config)
        generated=[]
        def fw(gm,args):
            (output/'aot_graph.py').write_text(gm.code)
            nonlocal compiler_original,compiler_owner
            compiler_owner=nfx._NpuFxCompiler
            compiler_original=compiler_owner._get_compiled_gm
            def save(compiler,graph,inputs):
                result=compiler_original(compiler,graph,inputs)
                code=result.get_code()
                if isinstance(code,str):
                    path=output/f'generated_kernel_{len(generated)}.py'
                    path.write_text(code);generated.append(digest(path))
                return result
            compiler_owner._get_compiled_gm=save
            try:return raw(gm,args)
            finally:
                compiler_owner._get_compiled_gm=compiler_original
                compiler_original=None
        outer=aot_autograd(fw_compiler=fw,keep_inference_input_mutations=keep)
        def backend(gm,args):
            (output/'dynamo_graph.py').write_text(gm.code)
            inner=outer(gm,args)
            def invoke(*actual):
                # This mode sees real runtime ATen calls after Dynamo/AOT trace,
                # and records source pointers to detect marker copybacks too.
                with CopyMode():return inner(*actual)
            return invoke
        compiled=torch.compile(program,backend=backend,fullgraph=True,dynamic=False)
        resources['compiled']=compiled
        r['generated_kernels']=generated
        actual_inputs=(*resources['inputs'],*producer,*reader)
        r['phase']='real-warmup'
        assert_current_scope()
        with torch.no_grad():returned=compiled(*actual_inputs)
        torch.npu.synchronize();assert_current_scope()
        graph=torch.npu.NPUGraph();resources['graph']=graph
        r['phase']='cold-capture'
        with torch.no_grad(),torch.npu.graph(graph,stream=stream):returned=compiled(*actual_inputs)
        torch.npu.synchronize();assert_current_scope()
        resources['returned']=returned
        r['capture_calls']=[call for call in r['copy_calls'] if call['phase']=='cold-capture']
        returned_stamps={f'{owner}.{field}':stamp(tensor) for owner,tensors in
                         (('producer',returned[2:8]),('reader',returned[8:14]))
                         for field,tensor in zip(FIELDS,tensors)}
        r['returned_buffers_after_capture']=returned_stamps
        alias_error=None
        try:validate_copy_aliases(buffers,r['capture_calls'],returned_stamps,before['stream_ptr'],marker_sources=marker_sources)
        except ValueError as exc:alias_error=str(exc)
        r['alias_error']=alias_error
        for generation in range(1,controller['generations']+1):
            r['phase']=f'epoch-{generation}-prepare'
            torch.npu.synchronize()  # Prior device fence BEFORE reset/reuse.
            assert_current_scope()
            expected=torch.arange(384,dtype=torch.bfloat16).reshape(6,64)/512+generation/8+rank/32
            expected_positions=torch.arange(6,dtype=torch.int64).reshape(6,1)+generation*8
            expected_indices=torch.arange(6,dtype=torch.int64)+(generation%2)*6
            with torch.no_grad():
                reset_after_prior_fence(producer);reset_after_prior_fence(reader)
                x.copy_(expected);positions.copy_(expected_positions);epoch.fill_(generation);indices.copy_(expected_indices)
                graph.replay()
            r['phase']=f'epoch-{generation}-fence'
            torch.npu.synchronize()
            fence_ns=time.monotonic_ns()
            assert_current_scope()
            exports=[]
            for owner,tensors in (('producer',producer),('reader',reader)):
                values,pos,mask,mark,count,rows=tensors
                payload={'owner':owner,'values':values.cpu().tolist(),'positions':pos.cpu().tolist(),
                         'valid':mask.cpu().tolist(),'epoch':int(mark.item()),
                         'copy_count':int(count.item()),'rows':int(rows.item())}
                validate_export(payload,generation=generation,expected_values=expected.tolist(),
                                expected_positions=expected_positions.tolist())
                exports.append(payload)
            if not torch.equal(returned[0].cpu(),expected*2):raise ValueError('Actual returned native compute values differ')
            if not torch.equal(cache.cpu()[expected_indices],expected):raise ValueError('Actual native cache write/read differs')
            current={f'{owner}.{field}':stamp(tensor) for owner,tensors in (('producer',producer),('reader',reader))
                     for field,tensor in zip(FIELDS,tensors)}
            if current!=buffers:raise ValueError('Original owned buffer pointers/layout changed')
            r['generations'].append({'epoch':generation,'device_fence_monotonic_ns':fence_ns,
                'payload_equal':True,'positions_equal':True,'device_write_count_one_each':True,
                'native_cache_write_read_equal':True,'returned_value_equal':True,
                'original_buffers_stable':True,'borrowed_scope_stable':True,'exports':exports})
            (output/'progress.json').write_text(json.dumps(r,indent=2)+'\n')
        r.update(stage_semantics_passed=True,actual_copy_alias_gate_passed=alias_error is None,
                 passed=alias_error is None,phase='terminal-complete',
                 action='Terminate this new audit worker normally; a different variant needs another fresh initialized job.')
    except BaseException as exc:
        r.update(passed=False,phase_at_exception=r['phase'],phase='terminal-failed',
                 exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc(),
                 action='Terminate the new audit job. Do not repair context/stream, retry, or continue model execution in this worker.')
        if getattr(exc,'loaded_abi_proof',None) is not None:
            r['rejected_loaded_abi_proof']=exc.loaded_abi_proof
    finally:
        if compiler_original is not None:compiler_owner._get_compiled_gm=compiler_original
        if torch is not None and prior_functionalization is not None:
            torch._inductor.config.enable_auto_functionalized_v2=prior_functionalization
        lifecycle.finish(r['passed'],resources)
        r['worker_lifecycle']=lifecycle.state
        # Only a controller-bound directory created after the current-scope
        # guard is written. Preallocation failures still return raw RPC receipt.
        if 'initialized_worker_scope' in r:
            (output/'result.json').write_text(json.dumps(r,indent=2)+'\n')
    return r
