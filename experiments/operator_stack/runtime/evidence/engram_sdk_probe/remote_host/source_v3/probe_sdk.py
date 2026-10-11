"""Small real SDK/AOT/GraphEx probe. Never a model or formal performance result."""
import argparse
import faulthandler
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import torch
import torch.nn.functional as F
import torch_npu

EVENTS = {}
PROTOTYPE_CONTEXT_SHA256 = '3f8176a05201d61fe5cb4b39c94c0715e95cb6f776ee827ab0a9e0110cd01ccc'
OP_CALLS = {'publish': [], 'wait_copy': [], 'mark': []}


@torch.library.custom_op('engram_sdk_probe::publish', mutates_args={'cache'})
def publish(value: torch.Tensor, cache: torch.Tensor, event_id: int) -> None:
    cache.copy_(value)
    EVENTS[event_id].record(torch.npu.current_stream())
    OP_CALLS['publish'].append(int(torch.npu.current_stream().npu_stream))


@publish.register_fake
def publish_fake(value, cache, event_id): return None


@torch.library.custom_op('engram_sdk_probe::wait_copy', mutates_args={'out'})
def wait_copy(cache: torch.Tensor, out: torch.Tensor, event_id: int) -> None:
    EVENTS[event_id].wait(torch.npu.current_stream())
    out.copy_(cache)
    OP_CALLS['wait_copy'].append(int(torch.npu.current_stream().npu_stream))


@wait_copy.register_fake
def wait_copy_fake(cache, out, event_id): return None


@torch.library.custom_op('engram_sdk_probe::mark', mutates_args={'marker', 'epoch'})
def mark(dependency: torch.Tensor, marker: torch.Tensor, epoch: torch.Tensor, event_id: int) -> None:
    marker.copy_(dependency.reshape(-1)[:1])
    epoch.add_(1)
    EVENTS[event_id].record(torch.npu.current_stream())
    OP_CALLS['mark'].append(int(torch.npu.current_stream().npu_stream))


@mark.register_fake
def mark_fake(dependency, marker, epoch, event_id): return None


def digest(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def stamp(tensor):
    return {'ptr': tensor.data_ptr(), 'shape': list(tensor.shape), 'stride': list(tensor.stride()),
            'dtype': str(tensor.dtype), 'device': str(tensor.device),
            'format': int(torch_npu.get_npu_format(tensor))}


def freeze_environment(out):
    import npugraph_ex as nge
    import npugraph_ex.npu_fx_compiler as nfx
    import npugraph_ex._acl_concrete_graph.acl_graph as acl
    import npugraph_ex._acl_concrete_graph.static_kernel as static
    import npugraph_ex.configs.compiler_config as config
    sources = {'torch': digest(inspect.getfile(torch)), 'torch_npu': digest(inspect.getfile(torch_npu)),
               'ExternalEvent': digest(inspect.getfile(torch_npu.npu.ExternalEvent)),
               'npugraph_ex': digest(nge.__file__), 'npu_fx_compiler': digest(inspect.getfile(nfx)),
               'acl_graph_enpu': digest(inspect.getfile(acl)), 'static_kernel': digest(inspect.getfile(static)),
               'compiler_config': digest(inspect.getfile(config)), 'probe': digest(__file__),
               'formal_compiler_interface': digest('/vllm-workspace/vllm-ascend/vllm_ascend/compilation/compiler_interface.py'),
               'formal_runner_enpu': digest('/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')}
    result = {'sources': sources, 'versions': {'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
                                             'npugraph_ex': str(getattr(nge, '__version__', 'unreported'))},
              'env': {k: os.getenv(k) for k in ('ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','HCCL_DETERMINISTIC','LOCAL_WORLD_SIZE')},
              'device_count_before_init': torch.npu.device_count(), 'npu_initialized_before': torch_npu.npu.is_initialized(),
              'container_pid': os.getpid(), 'process_status': Path('/proc/self/status').read_text().split('NSpid:')[-1].splitlines()[0]}
    (out/'environment.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'ExternalEvent_source.txt').write_text(inspect.getsource(torch_npu.npu.ExternalEvent))
    print('ENVIRONMENT_FROZEN', json.dumps(result), flush=True)
    return result


def make_graph_ex(fn, tag, out, static_kernel):
    import npugraph_ex as nge
    from npugraph_ex.configs.compiler_config import _process_kwargs_options
    from torch._dynamo.backends.common import aot_autograd
    config = nge.CompilerConfig()
    options = {'force_eager': True, 'inplace_pass': False, 'clone_input': False, 'clone_output': False,
               'static_kernel_compile': static_kernel}
    if static_kernel: options['_vllm_aclnn_static_kernel_sym_range'] = [1]
    _process_kwargs_options(config, {'options': options})
    (out/f'{tag}_compiler_options.json').write_text(json.dumps(config.as_dict(), indent=2, default=str)+'\n')
    raw = nge.get_npu_backend(compiler_config=config)
    evidence = {'dynamo': [], 'aot': [], 'returned_callable': []}

    def fw(gm, args):
        index = len(evidence['aot'])
        (out/f'{tag}_aot_{index}.py').write_text(gm.code)
        targets = [str(n.target) for n in gm.graph.nodes if n.op == 'call_function']
        evidence['aot'].append(targets)
        compiled = raw(gm, args)
        evidence['returned_callable'].append(type(compiled).__module__+'.'+type(compiled).__name__)
        return compiled

    aot = aot_autograd(fw_compiler=fw)

    def backend(gm, args):
        index = len(evidence['dynamo'])
        (out/f'{tag}_dynamo_{index}.py').write_text(gm.code)
        evidence['dynamo'].append([str(n.target) for n in gm.graph.nodes if n.op == 'call_function'])
        return aot(gm, args)

    return torch.compile(fn, backend=backend, fullgraph=True, dynamic=False), evidence


@torch.no_grad()
def run(args):
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    result={'scope':'Small BF16 SDK/graph semantics only; not model/full Engram/formal performance',
            'stage':args.stage, 'static_kernel':args.static_kernel, 'formal_model_validated':False,
            'prototype_context_sha256':PROTOTYPE_CONTEXT_SHA256,'prototype_executed':False,
            'enqueue_order':args.enqueue_order,'host_consumed_wait':args.host_consumed_wait,
            'safe_formal_receipt':False,'passed':False,'phase':'source-freeze','generations':[]}
    partial=False
    try:
        assert os.environ['SDK_PROBE_PHYSICAL_CHIP']=='4'
        assert os.environ['ASCEND_RT_VISIBLE_DEVICES']==os.environ['ASCEND_VISIBLE_DEVICES']=='4'
        assert os.environ['HCCL_DETERMINISTIC']=='strict' and args.generations>=32
        result['environment']=freeze_environment(out)
        assert result['environment']['device_count_before_init']==1
        result['phase']='set-device'
        torch.npu.set_device(0)
        main=torch.npu.current_stream(); aux=torch.npu.Stream()
        result['streams']={'main':int(main.npu_stream),'aux':int(aux.npu_stream),
                           'runtime_device':str(torch.npu.current_device()), 'device_name':torch.npu.get_device_name(0),
                           'soc_version':int(torch_npu.npu.get_soc_version())}
        # Exact representable data and identity weight isolate event/alias defects from numerical tiling.
        x=torch.zeros((1,64),dtype=torch.bfloat16,device='npu')
        weight=torch.eye(64,dtype=torch.bfloat16,device='cpu').npu()
        cache=torch.empty_like(x); output=torch.empty_like(x); observed=torch.empty_like(x)
        marker=torch.empty((1,),dtype=torch.bfloat16,device='npu')
        epoch=torch.zeros((1,),dtype=torch.int64,device='npu')
        result['tensors']={name:stamp(t) for name,t in [('x',x),('weight',weight),('cache',cache),('out',output),
                                                     ('observed',observed),('marker',marker),('epoch',epoch)]}
        assert len({t['ptr'] for t in result['tensors'].values()})==7
        ready=torch_npu.npu.ExternalEvent(); consumed=torch_npu.npu.ExternalEvent()
        EVENTS[id(ready)]=ready; EVENTS[id(consumed)]=consumed
        input_done=torch.npu.Event(); producer_done=torch.npu.Event(); consumer_done=torch.npu.Event(); reset_done=torch.npu.Event()
        result['events']={'ready_object_id':id(ready),'consumed_object_id':id(consumed),
                          'ExternalEvent_graph_external':True,'ordinary_Event_graph_external':False}
        result['phase']='pristine-wait-before-first-record'
        pristine=torch_npu.npu.ExternalEvent()
        pristine.wait(main)
        pristine.record(aux)
        main.synchronize(); aux.synchronize()
        result['pristine_wait_before_record_passed']=True

        def producer(a,w,c):
            publish(F.linear(a,w),c,id(ready))
            return (c,)

        def consumer(c,o,v,m,e):
            wait_copy(c,o,id(ready))
            v.copy_(o*2)
            mark(v,m,e,id(consumed))
            return (v,)

        result['phase']='small-matmul-warmup'
        x.fill_(0.25)
        with torch.npu.stream(aux): producer(x,weight,cache)
        aux.synchronize()
        consumer(cache,output,observed,marker,epoch); main.synchronize()
        assert torch.equal(cache.cpu(),torch.full((1,64),0.25,dtype=torch.bfloat16))
        assert torch.equal(observed.cpu(),torch.full((1,64),0.5,dtype=torch.bfloat16))
        result['small_matmul_passed']=True
        print('SMALL_MATMUL_OK',flush=True)

        if args.stage=='npugraph':
            result['phase']='npugraph-capture'
            pg=torch.npu.NPUGraph()
            with torch.npu.graph(pg,stream=aux): producer(x,weight,cache)
            aux.synchronize()
            cg=torch.npu.NPUGraph()
            with torch.npu.graph(cg,stream=main): consumer(cache,output,observed,marker,epoch)
            main.synchronize()
            def call_producer():
                with torch.npu.stream(aux): pg.replay()
            def call_consumer(): cg.replay()
            result['compiler_scope']='eager NPUGraph only; not a GraphEx receipt'
        elif args.stage=='graphex':
            result['phase']='aot-graphex-producer-compile'
            pf,pe=make_graph_ex(producer,'producer',out,args.static_kernel)
            with torch.npu.stream(aux): pf(x,weight,cache)
            aux.synchronize()
            result['phase']='aot-graphex-consumer-compile'
            cf,ce=make_graph_ex(consumer,'consumer',out,args.static_kernel)
            cf(cache,output,observed,marker,epoch); main.synchronize()
            result['compiler_graphs']={'producer':pe,'consumer':ce}
            result['phase']='outer-capture-actual-graphex-producer'
            pg=torch.npu.NPUGraph()
            with torch.npu.graph(pg,stream=aux): pf(x,weight,cache)
            aux.synchronize()
            result['phase']='outer-capture-actual-graphex-consumer'
            cg=torch.npu.NPUGraph()
            with torch.npu.graph(cg,stream=main): cf(cache,output,observed,marker,epoch)
            main.synchronize()
            result['compiler_scope']='torch.compile fullgraph -> aot_autograd -> actual npugraph_ex backend -> outer capture/replay, matching formal force_eager/static options'
            def call_producer():
                with torch.npu.stream(aux): pg.replay()
            def call_consumer(): cg.replay()
        else:
            def call_producer():
                with torch.npu.stream(aux): producer(x,weight,cache)
            def call_consumer(): consumer(cache,output,observed,marker,epoch)
            result['compiler_scope']='real SDK eager event/None-op semantics; not a graph receipt'

        result['phase']='cold-capture-fences'
        producer_done.record(aux); consumer_done.record(main)
        producer_done.synchronize(); consumer_done.synchronize()
        base_epoch=int(epoch.item())  # Outside all capture and after both fences.
        result['capture_op_streams']={name:list(streams) for name,streams in OP_CALLS.items()}
        result['direct_fence_stream_matches_last_impl']=(OP_CALLS['mark'][-1]==int(main.npu_stream)
                                                       and OP_CALLS['wait_copy'][-1]==int(main.npu_stream))
        stable=[t['ptr'] for t in result['tensors'].values()]
        result['initial_epoch']=base_epoch
        for generation in range(1,args.generations+1):
            result['phase']=f'generation-{generation}-reset'
            partial=True
            ready.reset(aux); consumed.reset(aux); reset_done.record(aux); reset_done.synchronize()
            assert reset_done.query() and producer_done.query() and consumer_done.query()
            expected=torch.arange(64,dtype=torch.bfloat16).reshape(1,64)/64 + generation/8
            x.copy_(expected); input_done.record(main)
            result['phase']=f'generation-{generation}-main-wait-before-aux-record'
            # Explicit host record-before-wait and graph-only wait-before-record are separate capabilities.
            if args.enqueue_order=='producer-first':
                input_done.wait(aux); call_producer(); producer_done.record(aux)
            call_consumer()
            if args.host_consumed_wait: consumed.wait(main)
            consumer_done.record(main)
            early_fence_complete=None
            if args.enqueue_order=='consumer-first':
                time.sleep(0.005)  # Producer deliberately withheld; no device synchronization/capture D2H.
                early_fence_complete=bool(consumer_done.query())
                assert not early_fence_complete,'ordinary main fence completed before current ready producer was submitted'
                input_done.wait(aux); call_producer(); producer_done.record(aux)
            result['phase']=f'generation-{generation}-joint-fence'
            producer_done.synchronize(); consumer_done.synchronize()
            assert producer_done.query() and consumer_done.query()
            # D2H is strictly after graph replay and both completion fences, never capture.
            a,b,v,m=cache.cpu(),output.cpu(),observed.cpu(),marker.cpu()
            current_epoch=int(epoch.item())
            assert torch.equal(a,expected) and torch.equal(b,expected) and torch.equal(v,expected*2)
            assert torch.equal(m,(expected*2).reshape(-1)[:1])
            assert current_epoch==base_epoch+generation
            current=[t.data_ptr() for t in (x,weight,cache,output,observed,marker,epoch)]
            assert current==stable
            result['generations'].append({'generation':generation,'cache_equal':True,'out_equal':True,
                                          'consumer_equal':True,'epoch':current_epoch,'stable_pointers':True,
                                          'joint_fences_observed':True,'wait_enqueued_before_record':True})
            result['generations'][-1]['consumer_fence_before_producer']=early_fence_complete
            partial=False
            (out/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
            print('GENERATION_OK',generation,current_epoch,flush=True)
        if args.stage=='graphex':
            assert result['compiler_graphs']['producer']['aot'] and result['compiler_graphs']['consumer']['aot']
            # Only initial Python implementation calls plus changing tensors/epoch prove actual replay; callback counts never label generations.
            result['aot_stage_reached']=True
        result.update(passed=True,phase='complete',op_implementation_calls=OP_CALLS,
                      meaning='This stage passed 32+ real generations. No production/ENPU/model safety receipt is granted.')
    except BaseException as exc:
        result.update(phase=result['phase'],passed=False,exception_type=type(exc).__name__,
                      exception=str(exc),traceback=traceback.format_exc(),poisoned_after_partial_enqueue=partial,
                      action='Terminal failure; no tensor/event/graph reuse, no reset/retry')
        print('TERMINAL_REJECTION',result['phase'],type(exc).__name__,str(exc),flush=True)
    finally:
        (out/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    if not result['passed'] and partial:
        # An external wait may already be queued without its producer. Do not
        # enqueue a repair/reset or let destructor synchronization deadlock.
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--stage',choices=('sdk','npugraph','graphex'),required=True)
    parser.add_argument('--static-kernel',action='store_true')
    parser.add_argument('--generations',type=int,default=32)
    parser.add_argument('--enqueue-order',choices=('producer-first','consumer-first'),default='consumer-first')
    parser.add_argument('--host-consumed-wait',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    faulthandler.enable(); faulthandler.register(signal.SIGUSR1,all_threads=True)
    raise SystemExit(run(parser.parse_args()))
