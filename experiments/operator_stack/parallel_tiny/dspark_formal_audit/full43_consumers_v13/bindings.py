"""Actual native object/role/source bindings; no SDK import or live install."""
import ast
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path


@dataclass(frozen=True)
class RoleGeometry:
    role: str
    global_experts: int
    topk: int
    local_experts: int
    max_decode_tokens: int

    @property
    def max_expanded_rows(self):
        return self.topk*self.max_decode_tokens


GEOMETRY = {'main':RoleGeometry('main',384,6,48,6),
            'draft':RoleGeometry('draft',128,3,16,5)}


def guard_native_function(function, entry):
    function=inspect.unwrap(function)
    function=getattr(function,'__func__',function)
    if function.__qualname__!=entry['qualified_function']:
        raise ValueError('Actual native function identity differs from frozen hook')
    path=Path(inspect.getfile(function));data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=entry['file_sha256']:
        raise ValueError('Actual native file/source SHA mismatch')
    source=data.decode();cls_name,fn_name=function.__qualname__.split('.')
    nodes=[fn for cls in ast.parse(source).body if isinstance(cls,ast.ClassDef) and cls.name==cls_name
           for fn in cls.body if isinstance(fn,ast.FunctionDef) and fn.name==fn_name]
    if len(nodes)!=1 or hashlib.sha256(ast.get_source_segment(source,nodes[0]).encode()).hexdigest()!=entry['function_sha256']:
        raise ValueError('Actual native function/source SHA mismatch')
    return {'path':str(path),'qualified_function':function.__qualname__,
            'file_sha256':entry['file_sha256'],'function_sha256':entry['function_sha256']}


def bind_actual_moe(role, mlp, router):
    """Use loaded MLP/router values, not HF defaults or a class-name count."""
    geometry=GEOMETRY[role]
    observed={'global_experts':int(mlp.n_routed_experts),'topk':int(router.top_k),
              'ep_size':int(mlp.ep_size),'local_experts':int(mlp.n_local_physical_experts),
              'physical_experts':int(mlp.n_physical_experts)}
    if (observed['global_experts']!=geometry.global_experts or observed['topk']!=geometry.topk
            or observed['local_experts']!=geometry.local_experts or observed['ep_size']!=8
            or observed['physical_experts']!=geometry.global_experts):
        raise ValueError('Actual native MLP/router role geometry differs; main and draft cannot share guards')
    if tuple(mlp.gate.weight.shape)!=(geometry.global_experts,5120):
        raise ValueError('Actual loaded route gate weight shape differs from its role')
    return {'role':role,'mlp_object_id':id(mlp),'router_object_id':id(router),
            **observed,'max_decode_tokens':geometry.max_decode_tokens,
            'max_expanded_rows':geometry.max_expanded_rows,
            'loaded_geometry_binding_only':True,'device_audit_passed':False,'full_consumer_audit_passed':False}


def validate_actual_gmm_extent(binding, *, token_rows, x_shape, counts_shape):
    if token_rows<=0 or token_rows>binding['max_decode_tokens']:
        raise ValueError('Actual GMM decode token extent is outside its role bound')
    expected_rows=token_rows*binding['topk']
    if tuple(x_shape)!=(expected_rows,5120) or tuple(counts_shape)!=(binding['local_experts'],):
        raise ValueError('Actual expanded GMM/count extent differs from the bound module role')
    return {'actual_gmm_extent_structure_passed':True,'expanded_rows':expected_rows,
            'device_audit_passed':False,'full_consumer_audit_passed':False}


def bind_native_layers(native_main, native_draft, routers, source_plan):
    entries={entry['qualified_function']:entry for entry in source_plan['hooks']}
    if len(entries)!=17:raise ValueError('Need exact frozen17 native hook source plan')
    modules={('main',i):module for i,module in enumerate(native_main.layers)}
    modules.update({('draft',int(key)):module for key,module in native_draft.layers.items()})
    expected={('main',i) for i in range(40)}|{('draft',i) for i in (40,41,42)}
    if set(modules)!=expected or set(routers)!=expected or len({id(module) for module in modules.values()})!=43:
        raise ValueError('Need43 distinct actual native modules and their selected routers')
    if set(native_main.aux_hidden_state_layers)!={38,39,40} or list(native_draft.target_layer_ids)!=[37,38,39]:
        raise ValueError('Actual aux37-39/one-based38-40 mapping differs')
    bindings={};producers=[]
    for key,module in modules.items():
        role,global_layer=key;local_layer=global_layer if role=='main' else global_layer-40
        if module.layer_idx!=local_layer or module.mlp.layer_idx!=local_layer or module.self_attn.layer_idx!=local_layer:
            raise ValueError('Actual role/global/MTP prefix-local indices differ')
        source=guard_native_function(module.forward,entries['DeepseekV41DecoderLayer.forward'])
        geometry=bind_actual_moe(role,module.mlp,routers[key])
        if role=='draft':producers.append((module.self_attn.wkv,module.self_attn.kv_norm,module.self_attn.rotary_emb))
        bindings[key]={'module':module,'router':routers[key],'global_layer':global_layer,
                       'checkpoint_mtp_index':local_layer if role=='draft' else None,
                       'source':source,'geometry':geometry}
    if any(len({id(producer[i]) for producer in producers})!=3 for i in range(3)):
        raise ValueError('Three draft WKV/norm/RoPE producer objects must be distinct')
    return {'bindings':bindings,'registered_actual_native_objects':43,
            'registration_is_device_coverage':False,'device_audit_passed':False,
            'full_consumer_audit_passed':False,'safe_formal_receipt':False,'formal_model_validated':False}


def bind_gmm_operator(operator, *, expected_schema, actual_moe_binding, consumer_source_receipt):
    # Actual runtime selects this precise overload/packet. No matching by an
    # ancestor profiler label, and no Tensor._version activation proof.
    schema=getattr(operator,'_schema',None)
    if schema is None or str(schema)!=expected_schema or not consumer_source_receipt.get('file_sha256'):
        raise ValueError('Need actual selected GMM overload schema and source-guarded consumer')
    return {'operator':operator,'operator_object_id':id(operator),'schema':str(schema),
            'moe_role':actual_moe_binding['role'],'consumer_source':consumer_source_receipt,
            'prepack_tap_requires_actual_device_order_proof':True,
            'device_audit_passed':False,'full_consumer_audit_passed':False}
