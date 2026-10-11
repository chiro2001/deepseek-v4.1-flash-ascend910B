"""Bind retained native objects in a fresh audit job; no SDK import or patch.

Source and object bindings are CPU facts. They never grant device coverage.
Keep the real native backbones before graph wrappers replace runner.model.
"""
from dataclasses import dataclass
import ast
import hashlib
import inspect
from pathlib import Path


def guard_function(function, entry):
    function = inspect.unwrap(function)
    function = getattr(function, '__func__', function)
    if function.__qualname__ != entry['qualified_function']:
        raise ValueError('Actual native function identity differs from source plan')
    path = Path(inspect.getfile(function))
    source = path.read_text()
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['file_sha256']:
        raise ValueError('Actual native source file differs from frozen plan')
    cls_name, fn_name = entry['qualified_function'].split('.')
    nodes = [fn for cls in ast.parse(source).body
             if isinstance(cls, ast.ClassDef) and cls.name == cls_name
             for fn in cls.body if isinstance(fn, ast.FunctionDef) and fn.name == fn_name]
    if len(nodes) != 1:
        raise ValueError('Actual native function is absent or ambiguous')
    segment = ast.get_source_segment(source, nodes[0])
    if hashlib.sha256(segment.encode()).hexdigest() != entry['function_sha256']:
        raise ValueError('Actual native function differs from frozen plan')
    return {'path': str(path), 'file_sha256': entry['file_sha256'],
            'qualified_function': function.__qualname__,
            'function_sha256': entry['function_sha256']}


def _entries(plan):
    entries = {entry['qualified_function']: entry for entry in plan['hooks']}
    if len(entries) != 17 or len(plan['hooks']) != 17:
        raise ValueError('Need the exact frozen 17 consumer source hooks')
    return entries


@dataclass
class NativeDecoderBindings:
    modules: dict
    source_guards: dict
    manifest: dict
    native_main: object
    native_draft: object

    def assert_same_loaded_objects(self):
        actual = {('main', index): module for index, module in enumerate(self.native_main.layers)}
        actual.update({('draft', int(key)): module for key, module in self.native_draft.layers.items()})
        if set(actual) != set(self.modules) or any(actual[key] is not module for key, module in self.modules.items()):
            raise ValueError('Retained native decoder objects were replaced after binding')
        return {'loaded_object_bindings_stable': True, 'device_audit_passed': False}


def bind_native_decoders(native_main, native_draft, plan):
    """Receive the load-time native references explicitly, without wrapper guessing."""
    entries = _entries(plan)
    guards = {}
    for name, function in (
        ('DeepseekV41Model.forward', native_main.forward),
        ('DeepseekV41DSparkModel.forward', native_draft.forward),
        ('DeepseekV4DSparkModel.combine_hidden_states', native_draft.combine_hidden_states),
        ('DeepseekV4DSparkModel._project_shared_kv', native_draft._project_shared_kv),
        ('DeepseekV4DSparkModel._store_standard_swa_kv', native_draft._store_standard_swa_kv),
    ):
        guards[name] = guard_function(function, entries[name])
    if set(native_main.aux_hidden_state_layers) != {38, 39, 40}:
        raise ValueError('Actual one-based target aux entries differ from layers37-39')
    if (list(native_draft.target_layer_ids) != [37, 38, 39]
            or native_draft.mtp_start_layer_idx != 40 or native_draft.num_dspark_layers != 3
            or native_draft.block_size != 5 or native_draft.hidden_size != 5120
            or native_draft.hc_mult != 4):
        raise ValueError('Actual loaded draft differs from formal full40/K5/hidden5120')
    modules = {('main', index): module for index, module in enumerate(native_main.layers)}
    modules.update({('draft', int(key)): module for key, module in native_draft.layers.items()})
    expected = {('main', index) for index in range(40)} | {('draft', index) for index in (40, 41, 42)}
    if set(modules) != expected or len({id(module) for module in modules.values()}) != 43:
        raise ValueError('Need43 distinct actual native main/draft objects')
    rows = []
    producer_objects = {'wkv': [], 'kv_norm': [], 'rotary_emb': []}
    for (role, global_layer), module in modules.items():
        local_layer = global_layer if role == 'main' else global_layer - 40
        if module.self_attn.layer_idx != local_layer or module.mlp.layer_idx != local_layer:
            raise ValueError('Actual attention/MoE prefix index differs from loaded role/layer')
        forward_guard = guard_function(module.forward, entries['DeepseekV41DecoderLayer.forward'])
        if role == 'main':
            reader = module.self_attn.v41_impl
            reader_guard = guard_function(reader._attention, entries['DeepseekV41EagerAttentionImpl._attention'])
            guard_function(reader._write_compressed_source, entries['DeepseekV41EagerAttentionImpl._write_compressed_source'])
        else:
            reader = module.self_attn.dsa_attn.dsa_attn.impl
            reader_guard = guard_function(reader._forward_attention, entries['AscendDSAImpl._forward_attention'])
            for field in producer_objects:
                producer_objects[field].append(id(getattr(module.self_attn, field)))
        rows.append({'role': role, 'global_layer': global_layer,
                     'checkpoint_mtp_index': local_layer if role == 'draft' else None,
                     'native_attention_layer_idx': module.self_attn.layer_idx,
                     'native_moe_layer_idx': module.mlp.layer_idx,
                     'module_object_id': id(module), 'attention_object_id': id(module.self_attn),
                     'reader_object_id': id(reader), 'decoder_forward': forward_guard,
                     'actual_reader': reader_guard})
    if any(len(set(objects)) != 3 for objects in producer_objects.values()):
        raise ValueError('The three native draft context producers must be distinct objects')
    manifest = {'scope': 'load-time actual native objects/source only; no device exports',
                'module_bindings': rows, 'distinct_draft_producer_objects': producer_objects,
                'registered_native_modules': 43, 'registration_is_device_coverage': False,
                'device_audit_passed': False, 'safe_formal_receipt': False,
                'formal_model_validated': False, 'performance_claim': None}
    return NativeDecoderBindings(modules, guards, manifest, native_main, native_draft)


def bind_called_dispatch(router, dispatcher, *, owner_key, native_moe_owner, bindings, plan):
    """Called outside capture with the objects selected by the real MoE call.

    The caller must supply the actual selected dispatcher, not a config-derived
    class. This binding remains preparation; device route payloads are required.
    """
    module = bindings.modules.get(owner_key)
    if module is None or native_moe_owner is not module.mlp:
        raise ValueError('Actual MoE caller is not the retained role/layer owner')
    entries = _entries(plan)
    router_guard = guard_function(router._compute_routing, entries['AscendFusedTopKRouter._compute_routing'])
    function = inspect.unwrap(dispatcher.token_dispatch)
    qualified = getattr(function, '__func__', function).__qualname__
    choices = ('TokenDispatcherWithAllGather.token_dispatch', 'TokenDispatcherWithMC2.token_dispatch')
    if qualified not in choices:
        raise ValueError('Actual selected dispatch path is outside the guarded source plan')
    dispatch_guard = guard_function(function, entries[qualified])
    return {'owner_key': list(owner_key), 'native_moe_owner_id': id(native_moe_owner),
            'actual_router_object_id': id(router), 'actual_dispatcher_object_id': id(dispatcher),
            'router_source': router_guard, 'dispatch_source': dispatch_guard,
            'selection_scope': 'actual objects supplied by the native MoE call outside capture',
            'device_audit_passed': False, 'safe_formal_receipt': False}


def bind_worker_consumers(runner, cache_plan, bindings, plan):
    """Bind the retained native worker, proposer, sampler and cache writer."""
    if (runner._formal_dspark_native_backbone is not bindings.native_main
            or runner._formal_dspark_native_draft_backbone is not bindings.native_draft):
        raise ValueError('Worker native load-time references differ from decoder bindings')
    spec = runner.vllm_config.speculative_config
    if spec.method != 'dspark' or spec.num_speculative_tokens != 5:
        raise ValueError('Actual worker is outside the formal DSpark/K5 configuration')
    entries = _entries(plan)
    functions = (
        ('AscendDSparkProposer.set_inputs_first_pass', runner.drafter.set_inputs_first_pass),
        ('NPUModelRunner._sample', runner._sample),
        ('NPUModelRunner._bookkeeping_sync', runner._bookkeeping_sync),
        ('DsaAttnKvPlan.dsa_kv_compress_scatter', cache_plan.dsa_kv_compress_scatter),
    )
    sources = {name: guard_function(function, entries[name]) for name, function in functions}
    return {'runner_object_id': id(runner), 'proposer_object_id': id(runner.drafter),
            'cache_plan_object_id': id(cache_plan), 'sources': sources,
            'scope': 'actual worker source/object binding; no cache/rejection device export',
            'bookkeeping_is_device_coverage': False,
            'device_audit_passed': False, 'safe_formal_receipt': False}


def bind_scheduler_consumer(scheduler, plan):
    # The scheduler belongs to the frontend process. Do not infer it from a
    # worker callback or pretend it shares the worker's device fence.
    entry = _entries(plan)['Scheduler.update_from_output']
    source = guard_function(scheduler.update_from_output, entry)
    return {'scheduler_object_id': id(scheduler), 'source': source,
            'scope': 'frontend native scheduler source/object correspondence only',
            'scheduler_is_device_coverage': False,
            'device_audit_passed': False, 'safe_formal_receipt': False}


def tensor_layout_metadata(tensor, format_getter):
    # This reads metadata of an already-loaded tensor. The SDK format getter
    # is supplied by the initialized worker; this module never imports it.
    try:
        version = int(tensor._version)
    except RuntimeError:
        version = None
    return {'ptr': tensor.data_ptr(), 'storage_ptr': tensor.untyped_storage().data_ptr(),
            'storage_offset': tensor.storage_offset(), 'shape': list(tensor.shape),
            'stride': list(tensor.stride()), 'dtype': str(tensor.dtype),
            'device': str(tensor.device), 'format': int(format_getter(tensor)), 'version': version}


def seal_native_parameters(bindings, format_getter):
    bindings.assert_same_loaded_objects()
    retained = {}
    seals = {}
    for key, module in bindings.modules.items():
        parameters = dict(module.named_parameters())
        if not parameters:
            raise ValueError('Actual loaded decoder has no native weight parameters')
        retained[key] = parameters
        seals[key] = {}
        for name, tensor in parameters.items():
            metadata = tensor_layout_metadata(tensor, format_getter)
            if (metadata['ptr'] <= 0 or metadata['storage_ptr'] <= 0
                    or not metadata['device'].startswith('npu:') or metadata['version'] is None):
                raise ValueError('Actual native weight lacks device pointer or version seal')
            seals[key][name] = metadata
    return {'retained_parameters': retained, 'seals': seals,
            'device_audit_passed': False, 'safe_formal_receipt': False}


def assert_native_parameter_seals(bindings, sealed, format_getter):
    bindings.assert_same_loaded_objects()
    for key, module in bindings.modules.items():
        current = dict(module.named_parameters())
        retained = sealed['retained_parameters'][key]
        if set(current) != set(retained) or any(current[name] is not tensor for name, tensor in retained.items()):
            raise ValueError('Actual native weight objects changed after load')
        for name, tensor in current.items():
            if tensor_layout_metadata(tensor, format_getter) != sealed['seals'][key][name]:
                raise ValueError('Actual native weight pointer/version/storage/layout changed')
    return {'native_weight_metadata_stable': True,
            'device_audit_passed': False, 'safe_formal_receipt': False}


def _cache_tensors(layer):
    cache = layer.kv_cache
    while isinstance(cache, (list, tuple)) and len(cache) == 1:
        cache = cache[0]
    return list(cache) if isinstance(cache, (list, tuple)) else [cache]


def bind_cache_tensor(layer, tensor, metadata, *, owner_key, logical_group, registered_prefix):
    """Seal one actual cache tensor with logical ownership even if GM is shared."""
    if owner_key[0] not in ('main', 'draft') or type(logical_group) is not int or logical_group < 0:
        raise ValueError('Need actual role/layer and logical cache group')
    if getattr(layer, 'prefix', None) != registered_prefix:
        raise ValueError('Actual registered cache prefix differs from logical owner')
    tensors = _cache_tensors(layer)
    indices = [index for index, value in enumerate(tensors) if value is tensor]
    if len(indices) != 1:
        raise ValueError('Cache tensor is absent or ambiguous in the actual cache layer')
    required = {'ptr', 'storage_ptr', 'storage_offset', 'shape', 'stride', 'dtype', 'device', 'format', 'version'}
    if set(metadata) != required or metadata['ptr'] <= 0 or metadata['storage_ptr'] <= 0:
        raise ValueError('Need actual cache pointer/storage/offset/stride/format metadata')
    if metadata['storage_offset'] < 0 or len(metadata['shape']) != len(metadata['stride']):
        raise ValueError('Actual cache storage layout is invalid')
    if not metadata['device'].startswith('npu:'):
        raise ValueError('Actual device cache binding must be on an NPU')
    return {'owner_key': list(owner_key), 'logical_group': logical_group,
            'registered_prefix': registered_prefix, 'cache_layer_object_id': id(layer),
            'cache_tensor_object_id': id(tensor), 'cache_tensor_index': indices[0],
            'cache_block_size': layer.block_size, 'tensor': dict(metadata),
            'shared_pointer_allowed': True,
            'coordinate_scope': 'actual positions and formatted block+offset are separate device payloads',
            'cache_write_read_proven': False, 'device_audit_passed': False,
            'safe_formal_receipt': False}
