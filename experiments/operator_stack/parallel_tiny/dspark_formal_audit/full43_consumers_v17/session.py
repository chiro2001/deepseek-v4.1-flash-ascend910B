"""Private load/bind/install and post-fence diagnostic exports for real workers."""
import hashlib
import inspect
import json
from pathlib import Path

if __package__:
    from .native_refs import bind_native_decoders,seal_native_parameters,assert_native_parameter_seals,guard_function,retain_owned_backbones
    from .bindings import bind_actual_moe
    from .runtime_callbacks import ConsumerCallbacks
    from .source_factory import install_private_method
    from .lifecycle import extend_native_root_forward,extend_compiled_dispatch,NativeResultAdapter,make_result_adapter
    from .prepack_integration import PrepackIntegration
else:
    from native_refs import bind_native_decoders,seal_native_parameters,assert_native_parameter_seals,guard_function,retain_owned_backbones
    from bindings import bind_actual_moe
    from runtime_callbacks import ConsumerCallbacks
    from source_factory import install_private_method
    from lifecycle import extend_native_root_forward,extend_compiled_dispatch,NativeResultAdapter,make_result_adapter
    from prepack_integration import PrepackIntegration


def owned_objects(root,max_depth=6):
    """Walk actual loaded ownership attributes, not profiler/config labels."""
    seen = set();pending = [(root,'',0)]
    while pending:
        obj,path,depth = pending.pop()
        if id(obj) in seen or obj is None:continue
        seen.add(id(obj))
        if isinstance(obj,(str,int,float,bool,bytes,type)) or inspect.ismodule(obj) or inspect.isroutine(obj):continue
        if hasattr(obj,'data_ptr') and hasattr(obj,'shape'):continue
        yield obj,path
        if depth >= max_depth:continue
        data = obj if isinstance(obj,dict) else vars(obj) if hasattr(obj,'__dict__') else {}
        for name,value in data.items():
            if any(label in str(name).lower() for label in ('config','process','executor','hook')):continue
            if isinstance(value,dict):
                pending.extend((child,path+'.'+str(name)+'.'+str(key),depth+1) for key,child in value.items())
            elif isinstance(value,(list,tuple)):
                pending.extend((child,path+'.'+str(name)+'.'+str(index),depth+1) for index,child in enumerate(value))
            else:pending.append((value,path+'.'+str(name),depth+1))


class AuditSession:
    def __init__(self,torch,npu,runner,output,plan):
        self.torch = torch;self.npu = npu;self.runner = runner;self.output = Path(output);self.plan = plan
        self.entries = {entry['qualified_function']:entry for entry in plan['hooks']}
        self.target_root = None;self.draft_root = None;self.native_bindings = None
        self.actual_spec_decode_round = False
        self.callbacks = None;self.weight_seals = None;self.generation = 0;self.returned_outputs = {}
        self.envelope_sources = {'main':[],'draft':[]}
        self.dispatch_counts = {'main':{},'draft':{}}
        self.prepack = PrepackIntegration(self);self.prepack_binding_failures = []
        self.installed = [];self.coverage = {name:False for name in self.entries}
        self.output.mkdir(parents=True,exist_ok=False)

    def record_envelope(self,role,source):
        self.envelope_sources.setdefault(role,[]).append(source)

    def note_dispatch(self,role,source):
        counts = self.dispatch_counts.setdefault(role,{})
        counts[source] = counts.get(source,0)+1

    def retain_target(self,runner,actual_model):
        if runner is not self.runner or self.target_root is not None:raise ValueError('Actual target load identity reused')
        self.target_root = actual_model

    def before_graph(self,runner):
        if runner is not self.runner or self.target_root is not runner.model:
            raise ValueError('Actual native target changed before graph wrapping')
        if runner.drafter is None:raise ValueError('Formal DSpark actual drafter is absent')
        self.draft_root = runner.drafter.model
        native_main,native_draft,root_ownership = retain_owned_backbones(self.target_root,self.draft_root,self.plan)
        (self.output/'retained_root_ownership.json').write_text(json.dumps(root_ownership,indent=2)+'\n')
        self.native_bindings = bind_native_decoders(native_main,native_draft,self.plan)
        self.native_bindings.manifest.update(root_ownership)
        runner._formal_dspark_native_backbone = native_main;runner._formal_dspark_native_draft_backbone = native_draft
        self.weight_seals = seal_native_parameters(self.native_bindings,self.npu.get_npu_format)
        (self.output/'weight_seals.json').write_text(json.dumps({
            'module_count':len(self.weight_seals['seals']),
            'parameter_count':sum(len(rows) for rows in self.weight_seals['seals'].values()),
            'host_resident_parameter_count':self.weight_seals['host_resident_parameter_count'],
            'host_resident_parameters':self.weight_seals['host_resident_parameters'],
            'seal_scope':self.weight_seals['seal_scope'],
            'host_resident_npu_format_queried':False,
            'device_audit_passed':False,'safe_formal_receipt':False},indent=2)+'\n')
        epoch = self.torch.zeros((1,),device=runner.device,dtype=self.torch.int64)
        self.callbacks = ConsumerCallbacks(self.torch,device=runner.device,epoch=epoch,native_bindings=self.native_bindings)
        self.callbacks.register_owner(native_draft,'draft',40)
        for label in ('combine.proj','combine.norm','combine.output'):
            self.callbacks.add_point('draft',40,label,5120,self.torch.bfloat16,max_rows=6)
        for (role,layer),module in self.native_bindings.modules.items():
            self.callbacks.register_owner(module,role,layer)
            self.callbacks.register_owner(module.self_attn,role,layer)
            self.callbacks.register_owner(module.mlp,role,layer)
            self.callbacks.add_point(role,layer,'decoder.output',20480,self.torch.bfloat16)
            readers = module.self_attn.v41_impl if role == 'main' else module.self_attn.dsa_attn.dsa_attn.impl
            self.callbacks.register_owner(readers,role,layer)
            if role == 'main':
                self.callbacks.add_point(role,layer,'cache.long.read',512,self.torch.bfloat16,position_width=2)
                self.callbacks.add_point(role,layer,'cache.write',512,self.torch.bfloat16,position_width=2)
            else:
                self.callbacks.add_point(role,layer,'cache.draft.read',512,self.torch.bfloat16,position_width=2)
                self.callbacks.add_point(role,layer,'shared.kv',512,self.torch.bfloat16)
                for label,width in [('shared.wkv',512),('shared.norm',512),('shared.rope',64)]:
                    self.callbacks.add_point(role,layer,label,width,self.torch.bfloat16)
            self.install(module,'forward','DeepseekV41DecoderLayer.forward')
            if role == 'main':
                self.install(readers,'_attention','DeepseekV41EagerAttentionImpl._attention')
                self.install(readers,'_write_compressed_source','DeepseekV41EagerAttentionImpl._write_compressed_source')
            else:self.install(readers,'_forward_attention','AscendDSAImpl._forward_attention')
            for obj,path in owned_objects(module.mlp):
                adapter = getattr(obj,'quant_method',None)
                scheme = getattr(adapter,'quant_method',adapter)
                if scheme is not None and hasattr(scheme,'apply_gmm1_act_quant') and hasattr(scheme,'apply_gmm2') and hasattr(obj,'router'):
                    try:self.prepack.bind_quant_layer(role,layer,module,obj,scheme)
                    except ValueError as exc:self.prepack_binding_failures.append({'role':role,'layer':layer,'owner_path':path,'reason':str(exc)})
                if hasattr(obj,'_compute_routing'):
                    function = inspect.unwrap(getattr(obj._compute_routing,'__func__',obj._compute_routing))
                    if function.__qualname__ == 'AscendFusedTopKRouter._compute_routing':
                        self.callbacks.register_owner(obj,role,layer);bind_actual_moe(role,module.mlp,obj)
                        self.callbacks.add_point(role,layer,'route.ids',6 if role == 'main' else 3,self.torch.int32)
                        self.callbacks.add_point(role,layer,'route.weights',6 if role == 'main' else 3,self.torch.float32)
                        self.install(obj,'_compute_routing',function.__qualname__)
                if hasattr(obj,'token_dispatch'):
                    function = inspect.unwrap(getattr(obj.token_dispatch,'__func__',obj.token_dispatch))
                    if function.__qualname__ in ('TokenDispatcherWithAllGather.token_dispatch','TokenDispatcherWithMC2.token_dispatch'):
                        self.callbacks.register_owner(obj,role,layer)
                        self.callbacks.add_point(role,layer,'dispatch.ids',6 if role == 'main' else 3,self.torch.int32)
                        self.callbacks.add_point(role,layer,'dispatch.weights',6 if role == 'main' else 3,self.torch.float32)
                        self.callbacks.add_point(role,layer,'dispatch.hidden',5120,self.torch.int8,max_rows=36 if role == 'main' else 15)
                        self.install(obj,'token_dispatch',function.__qualname__)
        for name in ('combine_hidden_states','_project_shared_kv','_store_standard_swa_kv'):
            qualified = 'DeepseekV4DSparkModel.'+name
            self.install(native_draft,name,qualified)
        self.install(native_draft,'forward','DeepseekV41DSparkModel.forward')
        self.install(native_main,'forward','DeepseekV41Model.forward')
        # Explicit buffers live on the native roots before compiler/capture.
        for role,root in [('main',self.target_root),('draft',self.draft_root)]:
            points = [point for point in self.callbacks.points.values() if point.spec.role == role]
            root.add_module('_v10_observer_buffers',self.torch.nn.ModuleList(points))
            self.installed.append(extend_native_root_forward(root,self,role))
            self.installed.append(extend_compiled_dispatch(root,self,role))
        (self.output/'loaded_bindings.json').write_text(json.dumps(self.native_bindings.manifest,indent=2)+'\n')
        (self.output/'installed_sources.json').write_text(json.dumps({'sources':[item['source'] for item in self.installed if 'source' in item],
            'registered_source_hooks':self.coverage,'source_registration_is_device_coverage':False},indent=2)+'\n')
        (self.output/'dynamo_alias_audit.json').write_text(json.dumps({
            'scope':'native-module attribute aliases that satisfy dynamo LOAD_GLOBAL guards for installed audit taps',
            'modules':self.callbacks.dynamo_alias_modules,
            'rebind_events':self.callbacks.dynamo_alias_rebinds,
            'rebind_count':len(self.callbacks.dynamo_alias_rebinds),
            'device_audit_passed':False,'safe_formal_receipt':False},indent=2)+'\n')

    def install(self,owner,method,qualified):
        if any(item.get('owner') is owner and item.get('method_name') == method for item in self.installed):return
        self.installed.append(install_private_method(owner,method,self.entries[qualified],self.callbacks))
        self.coverage[qualified] = True

    def adapter_mismatch(self,role,result):
        """Compact structure-only diagnostic for an unexpected outer return.

        Written before the adapter raises so the next analysis has the exact
        container types and tensor metadata instead of a bare message.
        """
        def summarize(value):
            if self.torch.is_tensor(value):
                return {'kind':'tensor','shape':list(value.shape),'dtype':str(value.dtype),
                        'device':str(value.device),'ptr':value.data_ptr()}
            if isinstance(value,(tuple,list)):
                return {'kind':type(value).__name__,'length':len(value),
                        'items':[summarize(item) for item in value[:8]]}
            return {'kind':type(value).__name__}
        try:
            (self.output/f'adapter_mismatch_{role}.json').write_text(json.dumps({
                'scope':'unexpected outer return structure; no payload/tensor values read',
                'role':role,'outer':summarize(result),
                'wrapped_id':id(self.wrapped),'wrapped_type':type(self.wrapped).__name__,
                'draft_root_id':id(self.draft_root) if self.draft_root is not None else None,
                'target_root_id':id(self.target_root) if self.target_root is not None else None,
                'wrapped_is_draft_root':self.wrapped is self.draft_root,
                'wrapped_is_target_root':self.wrapped is self.target_root,
                'wrapped_forward_tag':getattr(dict(vars(self.wrapped)).get('forward',None),'_v10_tag',None),
                'wrapped_has_forward_in_dict':'forward' in vars(self.wrapped) if hasattr(self.wrapped,'__dict__') else None,
                'wrapped_has_observer_buffers':'_v10_observer_buffers' in getattr(self.wrapped,'_modules',{}),
                'wrapped_has_unwrap':hasattr(self.wrapped,'unwrap'),
                'drafter_model_is_wrapped':getattr(getattr(self.session.runner,'drafter',None),'model',None) is self.wrapped,
                'device_audit_passed':False,'safe_formal_receipt':False},indent=2)+'\n')
        except Exception:pass

    def after_normal_load(self):
        if self.callbacks is None:raise ValueError('Original before-graph load boundary was not reached')
        draft_model_at_entry = self.runner.drafter.model
        def json_safe_receipt(receipt):
            return {key:(value if isinstance(value,(str,int,float,bool,type(None),dict,list)) else repr(value))
                    for key,value in receipt.items()}
        (self.output/'post_load_identity.json').write_text(json.dumps({
            'scope':'object identity/extension audit at the post-load boundary; no payload reads',
            'target_root_id':id(self.target_root),'draft_root_id':id(self.draft_root),
            'runner_model_id':id(self.runner.model),
            'runner_model_is_target_root':self.runner.model is self.target_root,
            'runner_model_type':type(self.runner.model).__name__,
            'runner_drafter_id':id(self.runner.drafter),
            'drafter_model_id':id(draft_model_at_entry),
            'drafter_model_is_draft_root':draft_model_at_entry is self.draft_root,
            'drafter_model_type':type(draft_model_at_entry).__name__,
            'drafter_model_repr':repr(draft_model_at_entry)[:200],
            'drafter_model_dict_keys':sorted(list(vars(draft_model_at_entry)))[:30]
                if hasattr(draft_model_at_entry,'__dict__') else None,
            'drafter_model_forward_tag':getattr(dict(vars(draft_model_at_entry)).get('forward',None),'_v10_tag',None)
                if hasattr(draft_model_at_entry,'__dict__') else None,
            'target_root_forward_tag':getattr(dict(vars(self.target_root)).get('forward',None),'_v10_tag',None),
            'draft_root_forward_tag':getattr(dict(vars(self.draft_root)).get('forward',None),'_v10_tag',None),
            'device_audit_passed':False,'safe_formal_receipt':False},indent=2)+'\n')
        # Re-assert the envelope if anything replaced an extended forward after
        # before_graph; idempotent when the tag is intact.
        reapply = {}
        for role,root in [('main',self.target_root),('draft',self.draft_root)]:
            reapply[role] = extend_native_root_forward(root,self,role)
            reapply[role+'_compiled_dispatch'] = extend_compiled_dispatch(root,self,role)
        (self.output/'extension_identity.json').write_text(json.dumps({
            'scope':'envelope extension re-application audit at after-normal-load',
            'target_root_id':id(self.target_root),'draft_root_id':id(self.draft_root),
            'receipts':{role:json_safe_receipt(receipt) for role,receipt in reapply.items()},
            'dispatch_counts':self.dispatch_counts,
            'device_audit_passed':False,'safe_formal_receipt':False},indent=2)+'\n')
        self.runner.model = make_result_adapter(self.runner.model,self,'main')
        self.runner.drafter.model = make_result_adapter(self.runner.drafter.model,self,'draft')
        original_get_model = self.runner.get_model
        def get_model(owner):
            value = original_get_model()
            return value.unwrap_native() if getattr(value,'_v10_adapter',False) else value
        import types
        self.runner.get_model = types.MethodType(get_model,self.runner)

    def role_tensor_outputs(self,role):
        outputs = tuple(tensor for point in self.callbacks.points.values() if point.spec.role == role for tensor in point.tensor_tuple())
        return (*outputs,*self.callbacks.role_metadata_outputs(role))

    def begin_execute(self,scheduler_output):
        # Called at the standard runner boundary, before a real model graph.
        self.torch.npu.synchronize()
        self.generation += 1
        self.actual_spec_decode_round = False
        self.returned_outputs.clear()
        rows = int(scheduler_output.total_num_scheduled_tokens)
        rows = min(rows,6)
        self.callbacks.begin_after_prior_fence(self.generation,actual_target_rows=rows)

    def end_sample(self):
        self.torch.npu.synchronize()
        assert_native_parameter_seals(self.native_bindings,self.weight_seals,self.npu.get_npu_format)
        record = self.callbacks.export_after_fence(self.generation,self.output/f'epoch_{self.generation:04d}_raw.pt')
        record['actual_graph_output_exports'] = self.validate_returned_outputs_after_fence()
        record['actual_spec_decode_round'] = self.actual_spec_decode_round
        record['registered_hook_functions'] = self.coverage
        record['uncovered_hook_functions'] = [name for name,installed in self.coverage.items() if not installed]
        record['prepack_actual'] = False
        record['prepack_binding_failures'] = self.prepack_binding_failures
        record['envelope_sources'] = {role:{'count':len(events),
            'last':events[-1] if events else None,'synthesised':sum(1 for e in events if e != 'wrapped_callable_envelope')}
            for role,events in self.envelope_sources.items()}
        record['strict_full43_admission'] = False
        (self.output/f'epoch_{self.generation:04d}.json').write_text(json.dumps(record,indent=2)+'\n')

    def validate_returned_outputs_after_fence(self):
        records = {}
        for role in ('main','draft'):
            expected = self.role_tensor_outputs(role)
            actual = self.returned_outputs.get(role,())
            rows = []
            for tensor,original in zip(actual,expected):
                same_storage = tensor.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
                same_layout = (tensor.data_ptr() == original.data_ptr() and tensor.storage_offset() == original.storage_offset()
                    and tensor.shape == original.shape and tensor.stride() == original.stride()
                    and tensor.dtype == original.dtype and tensor.device == original.device)
                same_values = same_storage and same_layout
                if not same_values:same_values = bool(self.torch.equal(tensor,original))
                rows.append({'ptr':tensor.data_ptr(),'owned_ptr':original.data_ptr(),
                    'shape':list(tensor.shape),'stride':list(tensor.stride()),'dtype':str(tensor.dtype),
                    'same_original_storage':same_storage,'same_original_layout':same_layout,
                    'same_original_values':same_values})
            records[role] = {'expected_tensor_outputs':len(expected),'actual_tensor_outputs':len(actual),
                'all_explicit_outputs_observed':len(actual) == len(expected),
                'all_values_equal_owned_after_fence':len(actual) == len(expected) and all(row['same_original_values'] for row in rows),
                'all_outputs_alias_original_storage':len(actual) == len(expected) and all(row['same_original_storage'] and row['same_original_layout'] for row in rows),
                'tensor_metadata_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
                'actual_output_ptrs':[row['ptr'] for row in rows],
                'owned_output_ptrs':[row['owned_ptr'] for row in rows],
                'mismatched_tensor_outputs':[dict(index=index,**row) for index,row in enumerate(rows)
                    if not row['same_original_storage'] or not row['same_original_layout'] or not row['same_original_values']]}
        return records
