"""Use the immutable IDS pure Tensor entries with explicit six-buffer outputs."""
import inspect
import types


class PrepackIntegration:
    def __init__(self,session):
        self.session = session;self.bindings = {};self.buffers = {};self.installed = []

    def bind_quant_layer(self,role,layer,decoder,quant_layer,scheme):
        from parallel_tiny.dspark_prepack_audit_cpu.contracts import bind_owner
        from parallel_tiny.dspark_prepack_audit_cpu.pure_graph_factory import make_pure_graph_entry
        binding = bind_owner(role=role,global_layer=layer,retained_decoder=decoder,
            quant_layer=quant_layer,router=quant_layer.router,scheme=scheme,
            format_getter=self.session.npu.get_npu_format,cpu_mock=False)
        self.bindings[id(quant_layer)] = binding
        for stage,width in [('gmm1',binding.hidden),('gmm2',binding.ffn)]:
            point = self.session.callbacks.add_point(role,layer,'prepack.'+stage,width,
                self.session.torch.int8,max_rows=binding.capacity)
            self.buffers[(id(quant_layer),stage)] = point.tensor_tuple()
        for stage,method in [('gmm1','apply_gmm1_act_quant'),('gmm2','apply_gmm2')]:
            if any(item['owner'] is scheme and item['method'] == method for item in self.installed):continue
            original = getattr(scheme,method);function = inspect.unwrap(getattr(original,'__func__',original))
            entry,text,receipt = make_pure_graph_entry(stage,function.__globals__)
            integration = self
            def make_call(original,entry,stage):
                def call(owner,mlp_compute_input,*actual,**kwargs):
                    bound = integration.bindings.get(id(mlp_compute_input.layer))
                    if bound is None:return original(mlp_compute_input,*actual,**kwargs)
                    source = mlp_compute_input.hidden_states if stage == 'gmm1' else actual[0]
                    expected_width = bound.hidden if stage == 'gmm1' else bound.ffn
                    # Geometry is selected from actual metadata outside data
                    # inspection. Unsupported dynamic/prefill/weight-list cases
                    # use the exact original native method and remain uncovered.
                    supported = (source.ndim == 2 and source.shape[0] == bound.capacity
                        and source.shape[1] == expected_width and not owner.use_expert_weight_list
                        and not getattr(mlp_compute_input,'dynamic_eplb',False)
                        and getattr(mlp_compute_input,'lora_context',None) is None)
                    activation = getattr(mlp_compute_input,'activation',None)
                    if stage == 'gmm1' and getattr(activation,'name',str(activation)).lower() != 'silu':supported = False
                    if not supported:return original(mlp_compute_input,*actual,**kwargs)
                    callbacks = integration.session.callbacks
                    positions = callbacks.target_positions if bound.role == 'main' else callbacks.draft_positions
                    valid = callbacks.target_valid if bound.role == 'main' else callbacks.draft_valid
                    topk = 6 if bound.role == 'main' else 3
                    positions = positions.repeat_interleave(topk,dim=0)
                    valid = valid.repeat_interleave(topk,dim=0)
                    result = entry(owner,mlp_compute_input,*actual,**kwargs,
                        audit_positions=positions,audit_valid=valid,audit_epoch=callbacks.epoch,
                        audit_buffers=integration.buffers[(id(mlp_compute_input.layer),stage)])
                    if stage == 'gmm1':
                        q,scale,*outputs = result
                        if len(outputs) != 6:raise ValueError('GMM1 actual pure entry dropped snapshot outputs')
                        # The original two native carriers are returned to the
                        # original g1->record_event->g2 caller. The same six live
                        # mutated tensors are returned by the owning graph root.
                        return q,scale
                    output,*outputs = result
                    if len(outputs) != 6:raise ValueError('GMM2 actual pure entry dropped snapshot outputs')
                    return output
                return call
            setattr(scheme,method,types.MethodType(make_call(original,entry,stage),scheme))
            self.installed.append({'owner':scheme,'method':method,'original':original,'source':receipt,
                                   'explicit_snapshot_outputs':6,'prepack_actual':False})
        return binding
