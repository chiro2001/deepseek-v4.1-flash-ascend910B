"""Role/owner/layout and source correspondence; no torch or SDK imports."""
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = json.loads((HERE/'evidence/installed_sources.json').read_text())['sources']
GEOMETRY = {'main': (384,6,48,6,36), 'draft': (128,3,16,5,15)}


def no_actual_claims():
    return {'actual_prepack_consumer_order_validated':False,
        'actual_device_consumer_proof':False,'device_audit_passed':False,
        'full_consumer_audit_passed':False,'safe_formal_receipt':False,
        'formal_model_validated':False,'performance_receipt_granted':False,
        'performance_benefit_observed':False}


def metadata(tensor, format_getter):
    # Metadata only. A version is recorded, never an activation content seal.
    try: version=int(tensor._version)
    except RuntimeError: version=None
    return {'object_id':id(tensor),'ptr':int(tensor.data_ptr()),
        'storage_ptr':int(tensor.untyped_storage().data_ptr()),
        'storage_bytes':int(tensor.untyped_storage().nbytes()),
        'storage_offset':int(tensor.storage_offset()),'shape':tuple(tensor.shape),
        'stride':tuple(tensor.stride()),'dtype':str(tensor.dtype),
        'device':str(tensor.device),'format':int(format_getter(tensor)),'version':version}


def guard_function(function, qualified_name, label='w4a8'):
    fn=inspect.unwrap(getattr(function,'__func__',function))
    entry=SOURCES[label]['functions'][qualified_name]
    path=Path(inspect.getfile(fn))
    if fn.__qualname__!=qualified_name or hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCES[label]['sha256']:
        raise ValueError('The actual loaded W4A8 function/source differs from the frozen source')
    # The full file SHA binds the exact original function; the function SHA is
    # retained separately for reviewer/source-plan correspondence.
    return {'qualified_function':qualified_name,'path':str(path),
        'file_sha256':SOURCES[label]['sha256'],'function_sha256':entry['sha256']}


@dataclass(frozen=True)
class OwnerBinding:
    role:str
    global_layer:int
    local_layer:int
    moe:object
    layer:object
    router:object
    scheme:object
    ep_rank:int
    hidden:int
    ffn:int
    capacity:int
    parameter_objects:dict
    parameter_seals:dict
    cpu_mock:bool


def bind_owner(*, role, global_layer, retained_decoder, quant_layer, router, scheme,
        format_getter, cpu_mock=False):
    if role not in GEOMETRY:
        raise ValueError('Explicit main/draft role is required')
    local=global_layer if role=='main' else global_layer-40
    if (type(global_layer) is not int or (role=='main' and not 0<=global_layer<40) or
            (role=='draft' and not 40<=global_layer<=42)):
        raise ValueError('Actual main/global or draft/global/local index changed')
    moe=retained_decoder.mlp
    if retained_decoder.layer_idx!=local or moe.layer_idx!=local:
        raise ValueError('The supplied owner is not the retained role/layer decoder')
    descendants=list(moe.experts.modules())
    if not any(module is quant_layer for module in descendants):
        raise ValueError('The actual mlp_compute_input.layer is outside the retained expert owner')
    adapter=quant_layer.quant_method
    scheme_matches=adapter is scheme or getattr(adapter,'quant_method',None) is scheme
    if not scheme_matches or quant_layer.router is not router:
        raise ValueError('Actual quant scheme or selected router differs from the loaded owner')
    experts,topk,local_experts,max_tokens,capacity=GEOMETRY[role]
    observed=(moe.n_routed_experts,router.top_k,moe.n_local_physical_experts)
    if (observed!=(experts,topk,local_experts) or moe.ep_size!=8 or
            moe.n_physical_experts!=experts or not 0<=moe.ep_rank<8 or
            tuple(moe.gate.weight.shape)!=(experts,5120)):
        raise ValueError('Loaded MLP/router/EP geometry differs from its explicit role')
    if scheme.use_expert_weight_list:
        raise ValueError('This component binds the actual single E48/E16 weight bank only')
    parameters={name:getattr(quant_layer,name) for name in
        ('w13_weight','w2_weight','w13_weight_scale','w2_weight_scale','w13_scale_bias','w2_scale_bias')}
    seals={name:metadata(t,format_getter) for name,t in parameters.items()}
    w13,w2=seals['w13_weight'],seals['w2_weight']
    if len(w13['shape'])!=3 or len(w2['shape'])!=3:
        raise ValueError('Actual packed expert weight rank changed')
    hidden=w13['shape'][1]
    # The GMM2 K comes from the actual loaded role-specific W2 dimension, and
    # must agree with its loaded FFN partition. 2304 is NOT a generic constant.
    ffn=w2['shape'][1]
    if (hidden!=5120 or ffn<=0 or ffn%4 or quant_layer.intermediate_size_per_partition!=ffn or
            w13['shape']!=(local_experts,hidden,ffn//4) or
            w2['shape']!=(local_experts,ffn,hidden//8) or
            any(m['dtype']!='torch.int32' or m['format']!=29 for m in (w13,w2))):
        raise ValueError('Loaded FFN/output/INT32 NZ weight dimensions disagree')
    expected_shapes={'w13_weight_scale':(local_experts,2*ffn),
        'w2_weight_scale':(local_experts,1,hidden),
        'w13_scale_bias':(local_experts,2*ffn),'w2_scale_bias':(local_experts,hidden)}
    for name,shape in expected_shapes.items():
        expected_dtype='torch.int64' if 'weight_scale' in name else 'torch.float32'
        if seals[name]['shape']!=shape or seals[name]['dtype']!=expected_dtype:
            raise ValueError('Loaded role-specific scale/assist shape or dtype changed: '+name)
    if not cpu_mock and any(not m['device'].startswith('npu:') for m in seals.values()):
        raise ValueError('Production binding must use actual initialized NPU tensors')
    if not cpu_mock:
        for method in ('apply_gmm1_act_quant','apply_gmm2','_get_mlp_weights','_get_mlp_scale_bias','_maybe_convert_group_list'):
            guard_function(getattr(scheme,method),'AscendW4A8DynamicFusedMoEMethod.'+method)
        quant_name=next(iter(SOURCES['quant_base']['functions']))
        guard_function(scheme._quant_hidden_states,quant_name,'quant_base')
    return OwnerBinding(role,global_layer,local,moe,quant_layer,router,scheme,moe.ep_rank,
        hidden,ffn,capacity,parameters,seals,bool(cpu_mock))


def check_parameters(binding, format_getter):
    adapter=binding.layer.quant_method
    if (not (adapter is binding.scheme or getattr(adapter,'quant_method',None) is binding.scheme) or
            binding.layer.router is not binding.router):
        raise ValueError('The original selected quant scheme/router owner was replaced')
    for name,original in binding.parameter_objects.items():
        actual=getattr(binding.layer,name)
        if actual is not original or metadata(actual,format_getter)!=binding.parameter_seals[name]:
            raise ValueError('An original immutable parameter owner/GM/layout/version changed')


def check_call(binding, stage, mlp_input, kwargs, format_getter):
    if stage not in ('gmm1','gmm2') or mlp_input.layer is not binding.layer:
        raise ValueError('Call stage or actual native quant-layer owner changed')
    if getattr(mlp_input,'dynamic_eplb',False) or getattr(mlp_input,'lora_context',None) is not None:
        raise ValueError('EPLB/LoRA is outside the guarded native prepack role')
    check_parameters(binding,format_getter)
    x=kwargs['x'] if stage=='gmm1' else kwargs['x'][0]
    width=binding.hidden if stage=='gmm1' else binding.ffn
    x_meta=metadata(x,format_getter)
    if (x_meta['shape']!=(binding.capacity,width) or x_meta['dtype']!='torch.int8' or
            x_meta['format']!=2 or x_meta['stride']!=(width,1) or x_meta['storage_offset']!=0):
        raise ValueError('Actual raw ND INT8 carrier extent/format/stride differs from its role')
    group=kwargs['group_list']
    if tuple(group.shape)!=(GEOMETRY[binding.role][2],) or str(group.dtype)!='torch.int64' or kwargs['group_list_type']!=1:
        raise ValueError('Actual native dense counts ABI changed')
    scale=kwargs['x_scale'] if stage=='gmm1' else kwargs['per_token_scale'][0]
    if tuple(scale.shape)!=(binding.capacity,) or str(scale.dtype)!='torch.float32':
        raise ValueError('Actual per-token scale extent/dtype differs from the raw carrier')
    names=('w13_weight','w13_weight_scale','w13_scale_bias') if stage=='gmm1' else ('w2_weight','w2_weight_scale','w2_scale_bias')
    fields=('weight','weight_scale','weight_assist_matrix') if stage=='gmm1' else ('weight','scale','bias')
    for field,name in zip(fields,names):
        values=kwargs[field]
        if type(values) is not list or len(values)!=1 or values[0] is not binding.parameter_objects[name]:
            raise ValueError('Native weight/scale/assist argument changed owner: '+field)
    if stage=='gmm1' and kwargs['dequant_mode']!=0:
        raise ValueError('Native fused dequant mode changed')
    if stage=='gmm2' and (kwargs['split_item']!=2 or kwargs['group_type']!=0):
        raise ValueError('Native GMM2 split/group semantics changed')
    return x,x_meta


def pack_target_source(stage):
    labels=('gmm1_pre','gmm1_pipeline','gmm1_torch','native_stream_and_GM') if stage=='gmm1' else ('gmm2_pre','gmm2_kernel_selector')
    return {'kind':'workspace_doublebank' if stage=='gmm1' else 'ND_original_x_GM_alias',
        'sources':{label:{'path':SOURCES[label]['path'],'sha256':SOURCES[label]['sha256']} for label in labels},
        'actual_workspace_GM_observed':False,'actual_selected_kernel_or_tiling_observed':False,
        'actual_pack_target_binding_requires_device_job':True,**no_actual_claims()}
