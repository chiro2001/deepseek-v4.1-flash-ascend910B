"""Single-chip real SDK gate, run only by the authorized main thread.

32 changing epochs, real MatMul, bounded native copies and cache write/read.
Formal-aligned AOT/GraphEx static/nonstatic are separate fresh processes.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

from exporter import PointSpec,make_exporter_class,assert_export


def digest(path):
    path=Path(path);return {'path':str(path),'bytes':path.stat().st_size,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--stage',choices=('eager','npugraph','graphex'),required=True)
    p.add_argument('--physical-chip',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--static-kernel',action='store_true')
    p.add_argument('--outer-keep-mutations',action='store_true')
    p.add_argument('--generations',type=int,default=32)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    r={'scope':'small real SDK native bounded copy/epoch only; not full43/model/formal performance',
       'stage':a.stage,'physical_chip':a.physical_chip,'static_kernel':a.static_kernel,
       'outer_keep_inference_input_mutations':a.outer_keep_mutations,'passed':False,
       'actual_copy_alias_gate_passed':False,
       'device_audit_passed':False,'safe_formal_receipt':False,'formal_model_validated':False,
       'phase':'pre-init','generations':[],'copy_calls':[]}
    try:
        if a.generations<32:raise ValueError('Need at least32 real epochs')
        visible=str(a.physical_chip)
        assert os.environ['ASCEND_RT_VISIBLE_DEVICES']==os.environ['ASCEND_VISIBLE_DEVICES']==visible
        assert os.environ['SDK_NATIVE_COPY_PHYSICAL_CHIP']==visible
        assert os.environ['HCCL_DETERMINISTIC']=='strict' and os.environ['LOCAL_WORLD_SIZE']=='1'
        import torch,torch_npu
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch._subclasses.fake_tensor import FakeTensor
        r['environment']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,
            'initialized_before':torch_npu.npu.is_initialized(),'device_count_before':torch.npu.device_count(),
            'process_identity':{'container_pid':os.getpid(),
                'host_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'gate_start_monotonic_ns':time.monotonic_ns()},
            'source_gate':digest(__file__),'source_exporter':digest(Path(__file__).with_name('exporter.py'))}
        assert not r['environment']['initialized_before'] and r['environment']['device_count_before']==1
        torch._inductor.config.enable_auto_functionalized_v2=False
        r['phase']='device-initialization'
        torch.npu.set_device(0)
        stream=torch.npu.Stream()
        def stamp(t):
            return {'ptr':t.data_ptr(),'shape':list(t.shape),'stride':list(t.stride()),
                    'dtype':str(t.dtype),'device':str(t.device),'format':int(torch_npu.get_npu_format(t))}
        class CopyMode(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                result=func(*args,**(kwargs or {}))
                if str(func) in ('aten.copy_.default','aten.copy.default') and len(args)>=2:
                    if all(isinstance(t,torch.Tensor) and not isinstance(t,FakeTensor) and t.device.type=='npu' for t in args[:2]):
                        destination=args[0] if str(func)=='aten.copy_.default' else result
                        r['copy_calls'].append({'phase':r['phase'],'op':str(func),'destination':stamp(destination),
                            'logical_destination':stamp(args[0]),'source':stamp(args[1]),
                            'stream':int(torch.npu.current_stream().npu_stream)})
                return result
        Exporter=make_exporter_class(torch)
        with torch.npu.stream(stream):
            x=torch.zeros((6,64),device='npu',dtype=torch.bfloat16)
            weight=torch.eye(64,dtype=torch.bfloat16,device='cpu').npu()
            positions=torch.arange(6,device='npu',dtype=torch.int64).reshape(6,1)
            valid=torch.ones((6,),device='npu',dtype=torch.bool)
            epoch=torch.zeros((1,),device='npu',dtype=torch.int64)
            indices=torch.arange(6,device='npu',dtype=torch.int64)
            cache=torch.zeros((16,64),device='npu',dtype=torch.bfloat16)
            producer=Exporter(PointSpec('matmul','producer',0,6,64),device='npu',dtype=torch.bfloat16)
            reader=Exporter(PointSpec('cache-read','cache-read',0,6,64),device='npu',dtype=torch.bfloat16)
        torch.npu.synchronize()
        buffers={f'{name}.{field}':stamp(tensor) for name,observer in [('producer',producer),('reader',reader)]
                 for field,tensor in observer.named_buffers()}
        r['buffers_before']=buffers;r['stream']=int(stream.npu_stream)
        def program(x,weight,positions,valid,epoch,indices,cache):
            value=torch.nn.functional.linear(x,weight)
            value=producer(value,value,positions,valid,epoch)
            cache.index_copy_(0,indices,value)
            consumed=cache.index_select(0,indices)
            value=reader(value,consumed,positions,valid,epoch)
            return (value*2,cache,*producer.tensor_tuple(),*reader.tensor_tuple())
        compiled=program
        if a.stage=='graphex':
            import npugraph_ex as nge
            import npugraph_ex.npu_fx_compiler as nfx
            from npugraph_ex.configs.compiler_config import _process_kwargs_options
            from torch._dynamo.backends.common import aot_autograd
            config=nge.CompilerConfig()
            options={'force_eager':True,'inplace_pass':False,'clone_input':False,'clone_output':False,
                     'static_kernel_compile':a.static_kernel}
            if a.static_kernel:options['_vllm_aclnn_static_kernel_sym_range']=[1]
            _process_kwargs_options(config,{'options':options})
            r['compiler_options']=config.as_dict()
            r['sdk_sources']={'npugraph_ex':digest(nge.__file__),'compiler':digest(inspect.getfile(nfx))}
            raw=nge.get_npu_backend(compiler_config=config)
            generated=[]
            def fw(gm,args):
                (a.output/'aot_graph.py').write_text(gm.code)
                original=nfx._NpuFxCompiler._get_compiled_gm
                def save(compiler,graph,inputs):
                    out=original(compiler,graph,inputs);code=out.get_code()
                    if isinstance(code,str):
                        file=a.output/f'generated_kernel_{len(generated)}.py';file.write_text(code);generated.append(digest(file))
                    return out
                nfx._NpuFxCompiler._get_compiled_gm=save
                try:return raw(gm,args)
                finally:nfx._NpuFxCompiler._get_compiled_gm=original
            outer=aot_autograd(fw_compiler=fw,keep_inference_input_mutations=a.outer_keep_mutations)
            def backend(gm,args):
                (a.output/'dynamo_graph.py').write_text(gm.code)
                inner=outer(gm,args)
                def runtime(*actual):
                    # After Dynamo/AOT tracing: observe actual ATen writes and
                    # copybacks, not FakeTensor calls or Python callback counts.
                    with CopyMode():return inner(*actual)
                return runtime
            compiled=torch.compile(program,backend=backend,fullgraph=True,dynamic=False)
            r['generated_kernels']=generated
        inputs=(x,weight,positions,valid,epoch,indices,cache)
        r['phase']='real-warmup'
        with torch.no_grad(),torch.npu.stream(stream):
            if a.stage=='graphex':output=compiled(*inputs)
            else:
                with CopyMode():output=compiled(*inputs)
        torch.npu.synchronize()
        if a.stage!='eager':
            r['phase']='cold-capture'
            graph=torch.npu.NPUGraph()
            with torch.no_grad(),torch.npu.graph(graph,stream=stream):
                if a.stage=='graphex':output=compiled(*inputs)
                else:
                    with CopyMode():output=compiled(*inputs)
            torch.npu.synchronize()
        capture_copies=[row for row in r['copy_calls'] if row['phase']=='cold-capture']
        r['actual_capture_copy_destinations_observed']=bool(capture_copies)
        owned={value['ptr'] for value in buffers.values()}
        observer_copies=[row for row in capture_copies if row['logical_destination']['ptr'] in owned]
        other_known={stamp(t)['ptr'] for t in inputs}
        r['capture_observer_copy_calls']=observer_copies
        r['capture_nonowned_copy_destinations']=[row for row in capture_copies
            if row['destination']['ptr'] not in owned|other_known]
        r['capture_all_observer_destinations_original']=(bool(observer_copies)
            and all(row['destination']['ptr'] in owned for row in observer_copies)
            and not r['capture_nonowned_copy_destinations'])
        for generation in range(1,a.generations+1):
            r['phase']=f'epoch-{generation}-prepare'
            torch.npu.synchronize()  # Prior generation fence before any reset/reuse.
            expected=torch.arange(384,dtype=torch.bfloat16).reshape(6,64)/512+generation/8
            expected_positions=torch.arange(6,dtype=torch.int64).reshape(6,1)+generation*8
            with torch.no_grad(),torch.npu.stream(stream):
                producer.reset_outside_capture();reader.reset_outside_capture()
                x.copy_(expected);positions.copy_(expected_positions);epoch.fill_(generation)
                if a.stage=='eager':
                    with CopyMode():output=compiled(*inputs)
                else:graph.replay()
            r['phase']=f'epoch-{generation}-fence'
            torch.npu.synchronize()
            exports=[]
            for name,observer in [('producer',producer),('reader',reader)]:
                export={'name':name,'epoch':int(observer.epoch.item()),'copy_count':int(observer.copy_count.item()),
                        'rows':int(observer.row_count.item()),'values':observer.values.cpu().tolist(),
                        'positions':observer.positions.cpu().tolist(),'valid':observer.valid.cpu().tolist()}
                assert_export(export,epoch=generation,expected_writes=1,values=expected.tolist(),
                              positions=expected_positions.tolist(),valid=[True]*6)
                exports.append(export)
            assert torch.equal(output[0].cpu(),expected*2)
            assert torch.equal(cache[:6].cpu(),expected)
            current={f'{name}.{field}':stamp(tensor) for name,observer in [('producer',producer),('reader',reader)]
                     for field,tensor in observer.named_buffers()}
            assert current==buffers,'Static observer pointer/layout changed'
            r['generations'].append({'epoch':generation,'payload_equal':True,'positions_equal':True,
                'device_write_count_one_each':True,'cache_write_read_equal':True,'returned_value_equal':True,
                'original_buffer_pointers_stable':True,'full_rank_device_fence_observed':True})
            (a.output/'progress.json').write_text(json.dumps(r,indent=2,default=str)+'\n')
            print('NATIVE_COPY_EPOCH_OK',generation,flush=True)
        r.update(passed=True,phase='complete',stage_semantics_passed=True,
            actual_copy_alias_gate_passed=r['capture_all_observer_destinations_original'] if a.stage!='eager' else None,
            meaning='32 hardware epochs validate owned values/epoch/count; actual copy alias evidence is a separate gate. No formal consumer receipt.')
    except BaseException as exc:
        r.update(passed=False,exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc(),
                 action='Terminal failed process. No tensor/event/graph repair or reuse.')
        print('NATIVE_COPY_TERMINAL_REJECTION',r['phase'],str(exc),flush=True)
    finally:
        (a.output/'result.json').write_text(json.dumps(r,indent=2,default=str)+'\n')
    return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
