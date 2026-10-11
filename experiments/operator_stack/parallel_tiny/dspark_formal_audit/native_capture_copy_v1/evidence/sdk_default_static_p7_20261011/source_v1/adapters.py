"""New-job integration helpers. No torch/SDK import or patch on import.

The hardware gate is mandatory before installing these into a fresh audit job.
Source/bindings alone never prove43 device consumers or cache correctness.
"""
import hashlib
import inspect
from pathlib import Path

from exporter import PointSpec,make_exporter_class


def require_sdk_receipts(static,nonstatic,exporter_sha256):
    for receipt,is_static in ((static,True),(nonstatic,False)):
        if receipt.get('stage')!='graphex' or receipt.get('static_kernel') is not is_static:
            raise ValueError('Need separate actual static/nonstatic GraphEx receipts')
        if not receipt.get('passed') or receipt.get('actual_copy_alias_gate_passed') is not True:
            raise ValueError('Native copy alias/current-epoch SDK gate has not passed')
        if len(receipt.get('generations',[]))<32:
            raise ValueError('Need32 changing hardware epochs, not callback counts')
        if receipt.get('environment',{}).get('source_exporter',{}).get('sha256')!=exporter_sha256:
            raise ValueError('Exporter differs from hardware-validated source')
        if receipt.get('environment',{}).get('initialized_before') is not False:
            raise ValueError('Hardware receipts need fresh source-frozen processes')
    if static['outer_keep_inference_input_mutations']!=nonstatic['outer_keep_inference_input_mutations']:
        raise ValueError('Static/nonstatic used different mutation policies')
    return {'sdk_gate_checked':True,'requires_outer_keep_inference_input_mutations':static['outer_keep_inference_input_mutations'],
            'full43_device_audit_passed':False,'safe_formal_receipt':False}


def bounded_cache_rows(cache,coordinates):
    # Fixed-row gather, never Nonzero/data-dependent allocation. Invalid rows
    # are gathered at0 solely for bounded transport and masked by the exporter.
    blocks=coordinates[:,0].long().clamp_min(0)
    offsets=coordinates[:,1].long().clamp_min(0)
    valid=(coordinates[:,0]>=0)&(coordinates[:,1]>=0)
    values=cache[blocks,offsets].reshape(coordinates.shape[0],-1)
    return values,valid


def observe_cache_reader(observer,q,cache,coordinates,epoch):
    values,valid=bounded_cache_rows(cache,coordinates)
    return observer(q,values,coordinates.to(dtype=observer.positions.dtype),valid,epoch)


def observe_cache_write_input(observer,updates,coordinates,epoch):
    valid=(coordinates[:,0]>=0)&(coordinates[:,1]>=0)
    # The live updates carrier is returned unchanged and then consumed by the
    # original native scatter. No new None-return opaque mutation is created.
    return observer(updates,updates.reshape(coordinates.shape[0],-1),
                    coordinates.to(dtype=observer.positions.dtype),valid,epoch)


def accepted_input_mask(torch,rows,accepted_draft_count,*,device):
    # Actual committed target cache rows: prior actual token + accepted drafts.
    # The bonus token is a future input. This is device-native, never item().
    return torch.arange(rows,device=device)<(accepted_draft_count+1)


def install_layer_outputs(torch,modules,epoch,*,source_sha256,static_receipt,nonstatic_receipt,max_rows=6):
    exporter_sha=hashlib.sha256(Path(__file__).with_name('exporter.py').read_bytes()).hexdigest()
    gate=require_sdk_receipts(static_receipt,nonstatic_receipt,exporter_sha)
    expected={('main',i) for i in range(40)}|{('draft',i) for i in (40,41,42)}
    if set(modules)!=expected or len({id(m) for m in modules.values()})!=43:
        raise ValueError('Need43 distinct actual native loaded modules, not wrapper/config counts')
    classes={type(module) for module in modules.values()}
    originals={cls:cls.forward for cls in classes}
    for function in originals.values():
        path=Path(inspect.getfile(inspect.unwrap(function)))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source_sha256:
            raise ValueError('Actual native decoder source differs from guarded source')
    Native=make_exporter_class(torch)
    observers={}
    for key,module in modules.items():
        role,layer=key
        # All buffers are created before compile/capture. Their actual layout
        # is sealed separately after allocation; no parameter math is changed.
        parameter=next(module.parameters())
        observers[id(module)]=Native(PointSpec(f'{role}.{layer}.output',role,layer,max_rows,20480),
                                     device=parameter.device,dtype=torch.bfloat16)
    for cls,original in originals.items():
        def create(original):
            def forward(module,*args,**kwargs):
                result=original(module,*args,**kwargs)
                observer=observers.get(id(module))
                if observer is None:return result
                positions=kwargs.get('positions',args[0] if args else None)
                if not isinstance(result,tuple) or len(result)!=2 or positions is None:
                    raise ValueError('Actual decoder return/positions ABI changed')
                hidden,mix=result;rows=hidden.shape[0]
                if rows>max_rows:return result  # Prefill/profile are outside bounded decode scope.
                values=hidden.reshape(rows,-1)
                valid=torch.ones((rows,),device=hidden.device,dtype=torch.bool)
                returned=observer(hidden,values,positions[:rows].reshape(rows,1).to(torch.int64),valid,epoch)
                return returned,mix
            return forward
        cls.forward=create(original)
    return {'observers':observers,'original_forwards':originals,'sdk_gate':gate,
            'registered_native_modules':43,'registration_is_device_coverage':False,
            'device_audit_passed':False,'safe_formal_receipt':False,
            'scope':'new private audit process only; real hardware epoch/payload exports still required'}
