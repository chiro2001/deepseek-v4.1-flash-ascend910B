"""New-job integration helpers. No torch/SDK import or patch on import.

The hardware gate is mandatory before installing these into a fresh audit job.
Source/bindings alone never prove43 device consumers or cache correctness.
"""
import hashlib
import inspect
from pathlib import Path

from exporter import PointSpec,make_exporter_class


def _require_actual_copy_addresses(receipt):
    buffers=receipt.get('buffers_before',{})
    fields=('values','positions','valid','epoch')
    required={f'{owner}.{field}' for owner in ('producer','reader') for field in fields}
    if not required.issubset(buffers):
        raise ValueError('Original preallocated device buffer address receipts are missing')
    originals={}
    for name in required:
        stamp=buffers[name]
        if (type(stamp.get('ptr')) is not int or stamp['ptr']<=0
                or not stamp.get('device','').startswith('npu:')
                or not stamp.get('shape') or len(stamp['shape'])!=len(stamp.get('stride',[]))):
            raise ValueError('Original buffer address/layout is not a device allocation')
        originals[stamp['ptr']]=stamp
    if len(originals)!=len(required):
        raise ValueError('Distinct observer buffers unexpectedly share original addresses')
    observed=set()
    for call in receipt['capture_observer_copy_calls']:
        destination=call.get('destination',{})
        logical=call.get('logical_destination',{})
        source=call.get('source',{})
        pointer=destination.get('ptr')
        if (call.get('phase')!='cold-capture' or call.get('op') not in ('aten.copy_.default','aten.copy.default')
                or pointer not in originals or destination!=originals[pointer] or logical!=destination
                or type(source.get('ptr')) is not int or source['ptr']<=0
                or source.get('device')!=destination['device']
                or type(call.get('stream')) is not int or call['stream']<=0):
            raise ValueError('Actual native copy source/destination/layout/stream evidence is incomplete')
        observed.add(pointer)
    if observed!=set(originals):
        raise ValueError('Actual payload/positions/valid/epoch copy destinations are missing')


def require_sdk_receipts(static,nonstatic,exporter_sha256):
    processes=[]
    gate_sha=hashlib.sha256(Path(__file__).with_name('gate_sdk.py').read_bytes()).hexdigest()
    for receipt,is_static in ((static,True),(nonstatic,False)):
        if receipt.get('stage')!='graphex' or receipt.get('static_kernel') is not is_static:
            raise ValueError('Need separate actual static/nonstatic GraphEx receipts')
        if not receipt.get('passed') or receipt.get('actual_copy_alias_gate_passed') is not True:
            raise ValueError('Native copy alias/current-epoch SDK gate has not passed')
        generations=receipt.get('generations',[])
        if len(generations)<32:
            raise ValueError('Need32 changing hardware epochs, not callback counts')
        required=('payload_equal','positions_equal','device_write_count_one_each',
                  'cache_write_read_equal','returned_value_equal',
                  'original_buffer_pointers_stable','full_rank_device_fence_observed')
        if ([row.get('epoch') for row in generations]!=list(range(1,len(generations)+1))
                or any(row.get(field) is not True for row in generations for field in required)):
            raise ValueError('Missing/stale/duplicate device epoch or consumer-value/fence proof')
        if (receipt.get('actual_capture_copy_destinations_observed') is not True
                or receipt.get('capture_all_observer_destinations_original') is not True
                or receipt.get('capture_nonowned_copy_destinations')!=[]
                or not receipt.get('capture_observer_copy_calls')):
            raise ValueError('Actual capture destination/alias evidence is incomplete')
        _require_actual_copy_addresses(receipt)
        if receipt.get('environment',{}).get('source_exporter',{}).get('sha256')!=exporter_sha256:
            raise ValueError('Exporter differs from hardware-validated source')
        if receipt.get('environment',{}).get('source_gate',{}).get('sha256')!=gate_sha:
            raise ValueError('SDK gate differs from the current frozen gate source')
        if receipt.get('environment',{}).get('initialized_before') is not False:
            raise ValueError('Hardware receipts need fresh source-frozen processes')
        if receipt['environment'].get('device_count_before')!=1:
            raise ValueError('SDK receipts must use the single-visible-device gate')
        process=receipt['environment'].get('process_identity',{})
        if (type(process.get('container_pid')) is not int or process['container_pid']<=0
                or not process.get('host_boot_id') or type(process.get('gate_start_monotonic_ns')) is not int
                or process['gate_start_monotonic_ns']<=0):
            raise ValueError('Fresh process identity is missing from the actual SDK gate')
        processes.append((process['host_boot_id'],process['container_pid'],process['gate_start_monotonic_ns']))
    if processes[0][:2]==processes[1][:2] or processes[0]==processes[1]:
        raise ValueError('Static/nonstatic need distinct fresh processes')
    if static.get('physical_chip')!=nonstatic.get('physical_chip'):
        raise ValueError('Static/nonstatic receipts refer to different physical chips')
    if set(static.get('sdk_sources',{}))!={'npugraph_ex','compiler'} or static['sdk_sources']!=nonstatic.get('sdk_sources'):
        raise ValueError('Static/nonstatic actual SDK source receipts are absent or different')
    for version in ('torch','torch_npu'):
        if not static['environment'].get(version) or static['environment'][version]!=nonstatic['environment'].get(version):
            raise ValueError('Static/nonstatic actual SDK versions differ or are missing')
    if type(static.get('outer_keep_inference_input_mutations')) is not bool or type(nonstatic.get('outer_keep_inference_input_mutations')) is not bool:
        raise ValueError('Actual AOT mutation policy is missing')
    if static['outer_keep_inference_input_mutations']!=nonstatic['outer_keep_inference_input_mutations']:
        raise ValueError('Static/nonstatic used different mutation policies')
    return {'sdk_gate_checked':True,'requires_outer_keep_inference_input_mutations':static['outer_keep_inference_input_mutations'],
            'full43_device_audit_passed':False,'safe_formal_receipt':False}


def bounded_cache_rows(cache,coordinates):
    # Fixed-row gather, never Nonzero/data-dependent allocation. Invalid rows
    # are gathered at0 solely for bounded transport and masked by the exporter.
    if coordinates.ndim!=2 or coordinates.shape[1]!=2 or cache.ndim<3:
        raise ValueError('Need native formatted block+offset coordinates and PA cache')
    if cache.shape[0]<=0 or cache.shape[1]<=0 or coordinates.device!=cache.device:
        raise ValueError('Bounded cache transport needs a nonempty same-device cache')
    valid=((coordinates[:,0]>=0)&(coordinates[:,1]>=0)
           &(coordinates[:,0]<cache.shape[0])&(coordinates[:,1]<cache.shape[1]))
    blocks=coordinates[:,0].long().clamp(0,cache.shape[0]-1)*valid.long()
    offsets=coordinates[:,1].long().clamp(0,cache.shape[1]-1)*valid.long()
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


def observe_route_pair(id_observer,weight_observer,result,positions,valid,epoch):
    # The guarded _compute_routing ABI is (weights, ids). Keep native dtypes:
    # BF16 would lose some expert IDs in the formal range0..383.
    if not isinstance(result,tuple) or len(result)!=2:
        raise ValueError('Actual native route return ABI changed')
    weights,ids=result
    weights=weight_observer(weights,weights,positions,valid,epoch)
    ids=id_observer(ids,ids,positions,valid,epoch)
    return weights,ids


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


def install_from_loaded_refs(torch,native_main,native_draft,epoch,*,source_plan,static_receipt,nonstatic_receipt,format_getter=None):
    # Reject the current initialization-failure receipt before binding,
    # allocation, or class mutation. This is a separate fresh-job entry only.
    exporter_sha=hashlib.sha256(Path(__file__).with_name('exporter.py').read_bytes()).hexdigest()
    require_sdk_receipts(static_receipt,nonstatic_receipt,exporter_sha)
    from consumer_bindings import bind_native_decoders,seal_native_parameters
    bindings=bind_native_decoders(native_main,native_draft,source_plan)
    if format_getter is None:
        raise ValueError('Actual SDK format getter is required for loaded weight seals')
    weight_seals=seal_native_parameters(bindings,format_getter)
    decoder_entry=next(entry for entry in source_plan['hooks'] if entry['qualified_function']=='DeepseekV41DecoderLayer.forward')
    installed=install_layer_outputs(torch,bindings.modules,epoch,source_sha256=decoder_entry['file_sha256'],
                                    static_receipt=static_receipt,nonstatic_receipt=nonstatic_receipt)
    installed['native_bindings']=bindings
    installed['native_binding_manifest']=bindings.manifest
    installed['native_weight_seals']=weight_seals
    return installed
