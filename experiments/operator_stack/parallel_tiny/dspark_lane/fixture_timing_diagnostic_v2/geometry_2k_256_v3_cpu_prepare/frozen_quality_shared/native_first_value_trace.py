"""Diagnostic-only native MTP producer/reader trace, with no capture device reads.

HcPre/HcPost observations sit at the actual Torch operator dispatcher used by
the compiled eager FX program. They preserve the native callable and compiler
partition, but diagnostic reads change timing and cannot prove graph consumers.
"""
import hashlib
from pathlib import Path
import time
from types import MethodType

import graph_precision_audit as audit


def bind_operator_arguments(schema, args, kwargs):
    """Bind the observed native ABI; the actual call receives the original arguments."""
    names = [argument.name for argument in schema.arguments]
    assert len(args) <= len(names)
    result = dict(zip(names, args))
    for name, value in kwargs.items():
        assert name in names and name not in result, (name, "Native argument binding mismatch")
        result[name] = value
    return result


def target_replay_matches(active, graph):
    return bool(active is not None and active["eligible"] and active["graph"] is graph
                and active["event"]["role"] == "target" and not active["event"]["capturing"])


def select_visible_cache(cache, indices, query_count):
    """Select exactly the slots named by active query metadata, including duplicates."""
    import torch
    assert not audit._capturing()
    assert cache.ndim == 4 and tuple(cache.shape[1:]) == (128, 1, 512)
    assert cache.dtype == torch.bfloat16 and indices.dtype == torch.int32
    assert indices.ndim == 3 and indices.shape[1] == 1
    assert indices.shape[0] >= query_count > 0
    active, info = audit._diagnostic_inspect(indices[:query_count])
    assert bool((active >= -1).all()), "Unknown sparse-attention pad sentinel"
    valid = active >= 0  # The native ABI's -1 mask, never a finite-value filter.
    selected = active[valid].to(torch.long)
    assert selected.numel() > 0
    capacity = int(cache.shape[0] * cache.shape[1])
    assert int(selected.max()) < capacity, "Active sparse-attention cache slot is out of range"
    values = cache.view(capacity, *cache.shape[2:]).index_select(0, selected.to(cache.device))
    return values, {"cache_ptr": cache.data_ptr(), "cache_shape": list(cache.shape),
                    "indices": info, "actual_query_rows": query_count,
                    "selected_slots_count_with_duplicates": selected.numel(),
                    "selected_slots_first64": selected[:64].tolist(),
                    "selected_slots_min": int(selected.min()), "selected_slots_max": int(selected.max()),
                    "pad_entries_excluded_by_native_minus_one_abi": int((~valid).sum()),
                    "finite_filter_used": False, "source_cache_unchanged": True}


class FirstValueTrace:
    def __init__(self, worker):
        self.worker = worker
        self.runner = worker.model_runner
        self.active_attention = None
        self.active_o_proj = None
        self.sequence = 0
        self.labels = {}
        self.helper_active = False
        self.hc_pre_index = 0
        self.hc_post_index = 0
        self.original_operator_calls = {}
        self.helper_history = []
        self.helper_marker = 0

    def cache_checkpoint(self, label, extra=None):
        """Observe a non-consuming boundary; actual O-projection gates remain strict."""
        if not self.enabled():
            return
        import torch
        observed = {}
        for index, layer in enumerate(self.runner._tiny_dspark_native_draft.layers.values()):
            impl = layer.self_attn.dsa_attn.dsa_attn.impl
            cached = getattr(impl, "_dummy_wo_a_3d", None)
            assert cached is not None, "Expected warmed native derived-weight cache"
            value, info = audit._diagnostic_inspect(cached)
            source, source_info = audit._diagnostic_inspect(impl.wo_a.weight)
            expected = source.view(impl.n_local_groups, impl.o_lora_rank, -1).transpose(2, 1).contiguous()
            info["native_source_weight"] = source_info
            info["bitwise_matches_native_loader_permutation"] = audit._bitwise(value, expected)
            observed[f"draft.{index}"] = info
            if not info["finite"]:
                path = audit._diagnostic_write("boundary." + label + f".draft.{index}.nonfinite", {
                    "boundary": label, "tensor": info, "non_consuming_boundary": True,
                    "source_values_unchanged": True, "actual_consumer_gate_not_relaxed": True,
                    "actual_scheduled_round": self.runner._tiny_dspark_pending, **(extra or {})})
                torch.save(value, path.with_suffix(".pt"))
        audit._diagnostic_write("boundary." + label, {"cached_weights": observed,
            "actual_scheduled_round": self.runner._tiny_dspark_pending,
            "non_consuming_boundary": True, "actual_consumer_gate_not_relaxed": True,
            "capture": False, "source_values_unchanged": True, **(extra or {})})

    def install_target_boundaries(self):
        import torch
        import draft_graph_observer as observer
        execute = self.runner.execute_model

        def target_execute(this, scheduler_output, *args, **kwargs):
            cached = scheduler_output.scheduled_cached_reqs
            computed = dict(zip(cached.req_ids, cached.num_computed_tokens))
            incoming = {"num_scheduled_tokens": dict(scheduler_output.num_scheduled_tokens),
                        "computed_before": computed}
            decode = bool(computed and all(value >= this._tiny_dspark_prompt_lens[rid]
                                          for rid, value in computed.items()))
            active = decode and self.enabled()
            if active:
                self.cache_checkpoint("target_execute.before", {"incoming_scheduler": incoming})
            result = execute(scheduler_output, *args, **kwargs)
            if active:
                torch.npu.synchronize()
                self.cache_checkpoint("target_execute.after", {"incoming_scheduler": incoming,
                                       "diagnostic_device_fence": True})
            return result

        self.runner.execute_model = MethodType(target_execute, self.runner)
        replay = torch.npu.NPUGraph.replay

        def target_replay(graph, *args, **kwargs):
            active = self.enabled() and target_replay_matches(observer.STATE["active"], graph)
            if active:
                self.cache_checkpoint("target_replay.before", {"native_target_graph_id": id(graph)})
            result = replay(graph, *args, **kwargs)
            if active:
                torch.npu.synchronize()
                self.cache_checkpoint("target_replay.after", {"native_target_graph_id": id(graph),
                                       "native_replay_returned": True, "diagnostic_device_fence": True})
            return result

        torch.npu.NPUGraph.replay = target_replay

    def enabled(self):
        return audit.STATE["enabled"] and not audit._capturing()

    def event(self, label, tensors, extra=None):
        if not self.enabled():
            return
        observed = {name: audit._diagnostic_inspect(tensor) for name, tensor in tensors.items() if tensor is not None}
        assert observed
        values = {name: value[0] for name, value in observed.items()}
        summaries = {name: value[1] for name, value in observed.items()}
        self.sequence += 1
        self.labels[label] = self.labels.get(label, 0) + 1
        context = {"capture": False, "arm": audit.STATE["arm"], "sequence": self.sequence,
                   "actual_scheduled_round": self.runner._tiny_dspark_pending,
                   "point": label, "native_arithmetic_unchanged": True,
                   "host_schedule_modified_by_diagnostic_reads": True,
                   "full_consumer_precision_proof": False, **(extra or {})}
        audit._diagnostic_write("firstvalue." + label, {"context": context, "tensors": summaries})
        audit._diagnostic_reject(values, summaries, "firstvalue." + label, context)
        return values

    def observe_operator(self, original, operator, args, kwargs, role):
        import torch
        schema = operator._schema if hasattr(operator, "_schema") else operator.default._schema
        bound = bind_operator_arguments(schema, args, kwargs)
        tensors = {name: value for name, value in bound.items() if isinstance(value, torch.Tensor)}
        scalars = {name: value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
                   for name, value in bound.items() if name not in tensors}
        if role == "hc_pre":
            index = self.hc_pre_index
            self.hc_pre_index += 1
            assert index < 6, "Unexpected extra MTP HcPre in one native helper call"
            label = f"draft.{index // 2}.hc_pre." + ("attn" if index % 2 == 0 else "ffn")
        elif role == "hc_post":
            index = self.hc_post_index
            self.hc_post_index += 1
            assert index < 6, "Unexpected extra MTP HcPost in one native helper call"
            label = f"draft.{index // 2}.hc_post." + ("attn" if index % 2 == 0 else "ffn")
        else:
            label = role
        context = {"native_operator_schema": str(schema), "native_scalar_arguments": scalars,
                   "actual_operator_dispatch_observed": True, "compiler_partition_changed": False}
        self.event(label + ".inputs", tensors, context)
        result = original(operator, *args, **kwargs)
        outputs = result if isinstance(result, (tuple, list)) else [result]
        self.event(label + ".outputs", {str(index): value for index, value in enumerate(outputs)
                                        if isinstance(value, torch.Tensor)}, context)
        return result

    def install_operator_observer(self):
        import torch
        roles = {"_C_ascend::npu_hc_pre_v2": "hc_pre", "_C_ascend::npu_hc_post": "hc_post",
                 "_C_ascend::npu_rms_norm_cast": "rms_norm_cast"}
        for cls in (torch._ops.OpOverload, torch._ops.OpOverloadPacket):
            original = cls.__call__
            self.original_operator_calls[cls] = original

            def make_observer(original):
                def observed(operator, *args, **kwargs):
                    if not self.helper_active or not self.enabled():
                        return original(operator, *args, **kwargs)
                    name = (operator._schema.name if hasattr(operator, "_schema")
                            else operator._qualified_op_name)
                    if name not in roles:
                        return original(operator, *args, **kwargs)
                    return self.observe_operator(original, operator, args, kwargs, roles[name])
                return observed

            cls.__call__ = make_observer(original)

    def install(self):
        import torch
        from vllm_ascend.models.deepseek_v41 import model as model_v41, dspark as dspark_v41
        from vllm_ascend.models.deepseek_v4 import dspark as dspark_v4
        from vllm_ascend.attention import dsa_v1, dsa_attn_kv_plan
        expected = {model_v41: "dfbc3c0d3b0f9ce3349509de5dfb3799a47f834eec26a951769bf4a54e6c022b",
                    dspark_v41: "3fdc5c8afd777b493761cd9e646561e7ad56157d369db6d5c02df9579d4074a0",
                    dspark_v4: "840cfde5eecc559e63283f8bc33666f8767569a0b7c4ad7e7719d662c1ea9946",
                    dsa_v1: "24cb4e1ad3b2fcc9f0524153f22c0c6818aba396d290c4b93cdf4f12ced217d5",
                    dsa_attn_kv_plan: "cbb50f37e92aec5a39628ce01af89aa77e5b4b9e9d5b141cd1a2ef74d7fbe795"}
        for module, sha in expected.items():
            assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == sha
        draft = self.runner._tiny_dspark_native_draft
        layers = list(draft.layers.values())
        assert len(layers) == 3
        attention_labels = {id(layer.self_attn): f"draft.{index}" for index, layer in enumerate(layers)}
        drafter = self.runner.drafter
        original_helper = drafter._run_merged_draft

        def helper(this, *args, **kwargs):
            assert not self.helper_active
            from vllm.forward_context import get_forward_context
            context = get_forward_context()
            pending = self.runner._tiny_dspark_pending
            history = {"capturing": bool(getattr(context, "capturing", False)),
                       "runtime_mode": str(context.cudagraph_runtime_mode), "arm": audit.STATE["arm"],
                       "actual_round_at_entry": None if pending is None else dict(pending),
                       "recording_real_request": self.runner._tiny_dspark_recording,
                       "begin_s": time.perf_counter(), "native_helper_returned": False}
            self.helper_active = True
            self.hc_pre_index = self.hc_post_index = 0
            try:
                result = original_helper(*args, **kwargs)
                history["native_helper_returned"] = True
                return result
            finally:
                self.helper_active = False
                history["end_s"] = time.perf_counter()
                self.helper_history.append(history)

        drafter._run_merged_draft = MethodType(helper, drafter)
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        if isinstance(drafter._runnable, ACLGraphWrapper):
            assert drafter._runnable.runnable == original_helper
            drafter._runnable.runnable = drafter._run_merged_draft
        else:
            assert drafter._runnable == original_helper
            drafter._runnable = drafter._run_merged_draft
        self.install_operator_observer()
        # Same-instance actual target entry comes from the existing guarded
        # observer; foreign, draft and capture replays never trigger snapshots.
        import sys
        observer = sys.modules.get("draft_graph_observer")
        if observer is not None and observer.STATE["installed"]:
            self.install_target_boundaries()
        original_combine, original_project = draft.combine_hidden_states, draft._project_shared_kv

        def combine(this, hidden):
            self.event("context.combine.inputs", {"hidden": hidden, "main_proj_weight": this.main_proj.weight,
                                                   "main_norm_weight": this.main_norm.weight})
            result = original_combine(hidden)
            self.event("context.combine.output", {"hidden": result})
            return result

        def project(this, hidden, positions, attn=None):
            label = attention_labels[id(attn)]
            self.event(label + ".context_kv.inputs", {"hidden": hidden, "positions": positions,
                         "wkv_weight": attn.wkv.weight, "kv_norm_weight": attn.kv_norm.weight})
            result = original_project(hidden, positions, attn)
            self.event(label + ".context_kv.output", {"shared_kv": result})
            return result

        draft.combine_hidden_states = MethodType(combine, draft)
        draft._project_shared_kv = MethodType(project, draft)
        original_attention = dsa_v1.AscendDSAImpl._forward_attention

        def attention(this, layer_name, hidden_states, kv_cache, layer_metadata, *args, **kwargs):
            active = str(layer_name).startswith("mtp.") and self.enabled()
            if not active:
                return original_attention(this, layer_name, hidden_states, kv_cache, layer_metadata, *args, **kwargs)
            assert self.active_attention is None
            self.active_attention = str(layer_name)
            try:
                self.event(str(layer_name) + ".attention.input", {"hidden": hidden_states})
                result = original_attention(this, layer_name, hidden_states, kv_cache, layer_metadata, *args, **kwargs)
                self.event(str(layer_name) + ".attention.output", {"output": result})
                return result
            finally:
                self.active_attention = None

        dsa_v1.AscendDSAImpl._forward_attention = attention
        original_o_proj = dsa_v1.AscendDSAImpl._forward_o_proj

        def o_proj(this, attn_output, output, *args, **kwargs):
            # The outer attention backend executes this before returning its
            # projected hidden buffer. Match its real per-layer sink binding.
            label = next((f"mtp.{index}" for index, layer in enumerate(layers)
                          if this.attn_sink.data_ptr() == layer.self_attn.attn_sink.data_ptr()), None)
            active = label is not None and self.helper_active and self.enabled()
            if not active:
                return original_o_proj(this, attn_output, output, *args, **kwargs)
            assert self.active_o_proj is None
            self.active_o_proj = label
            try:
                tensors = {"attention_output": attn_output, "wo_a_weight": this.wo_a.weight,
                           "wo_b_weight": this.wo_b.weight}
                cached = getattr(this, "_dummy_wo_a_3d", None)
                if cached is not None:
                    tensors["actual_dummy_wo_a_3d"] = cached
                values = self.event(label + ".o_proj.inputs", tensors,
                                    {"n_local_groups": this.n_local_groups, "o_lora_rank": this.o_lora_rank})
                if cached is not None:
                    expected = values["wo_a_weight"].view(this.n_local_groups, this.o_lora_rank, -1).transpose(2, 1).contiguous()
                    matches = audit._bitwise(values["actual_dummy_wo_a_3d"], expected)
                    audit._diagnostic_write("firstvalue." + label + ".cached_wo_a_match", {
                        "native_cached_weight_matches_original_loader_permutation": matches,
                        "actual_cache_ptr": cached.data_ptr(), "source_weight_ptr": this.wo_a.weight.data_ptr(),
                        "actual_scheduled_round": self.runner._tiny_dspark_pending,
                        "source_and_cache_values_unchanged": True})
                    assert matches, "Actual cached dummy wo_a does not match its native source parameter"
                result = original_o_proj(this, attn_output, output, *args, **kwargs)
                self.event(label + ".o_proj.output", {"projected_hidden": output})
                return result
            finally:
                self.active_o_proj = None

        dsa_v1.AscendDSAImpl._forward_o_proj = o_proj
        import torch_npu
        original_batchmatmul = torch_npu.npu_transpose_batchmatmul

        def batchmatmul(*args, **kwargs):
            active = self.active_o_proj is not None and self.enabled()
            if active:
                assert len(args) == 2
                tensors = {"x1": args[0], "x2_actual_cached_weight": args[1]}
                tensors.update({name: value for name, value in kwargs.items() if isinstance(value, torch.Tensor)})
                controls = {name: value if isinstance(value, (str, int, float, bool, tuple, list, type(None))) else str(value)
                            for name, value in kwargs.items() if name not in tensors}
                self.event(self.active_o_proj + ".wo_a_native_bmm.inputs", tensors,
                           {"native_function": "torch_npu.npu_transpose_batchmatmul", "kwargs": controls})
            result = original_batchmatmul(*args, **kwargs)
            if active:
                self.event(self.active_o_proj + ".wo_a_native_bmm.output", {"projected_low_rank": result})
            return result

        torch_npu.npu_transpose_batchmatmul = batchmatmul
        for index, layer in enumerate(layers):
            original_linear = layer.self_attn.wo_b.forward

            def make_linear(original, label):
                def linear(this, x, *args, **kwargs):
                    active = self.active_o_proj == label and self.enabled()
                    if active:
                        self.event(label + ".wo_b.inputs", {"hidden": x, "actual_weight": this.weight})
                    result = original(x, *args, **kwargs)
                    if active:
                        assert isinstance(result, torch.Tensor)
                        self.event(label + ".wo_b.output", {"hidden": result})
                    return result
                return linear

            layer.self_attn.wo_b.forward = MethodType(make_linear(original_linear, f"mtp.{index}"), layer.self_attn.wo_b)
        original_prolog = dsa_v1.AscendDSAImpl._mla_prolog_single_stream

        def prolog(this, hidden_states, cos, sin, swa_kv_cache, slot_mapping, *args, **kwargs):
            active = self.active_attention is not None and self.enabled()
            if active:
                self.event(self.active_attention + ".query_kv_prolog.inputs", {"hidden": hidden_states,
                    "cos": cos, "sin": sin, "slot_mapping": slot_mapping, "wq_a_weight": this.wq_a.weight,
                    "wq_b_weight": this.wq_b.weight, "q_norm_weight": this.q_norm.weight,
                    "wkv_weight": this.wkv.weight, "kv_norm_weight": this.kv_norm.weight})
            result = original_prolog(this, hidden_states, cos, sin, swa_kv_cache, slot_mapping, *args, **kwargs)
            if active:
                self.event(self.active_attention + ".query_kv_prolog.outputs",
                           {"q": result[0], "qr": result[1], "qr_pertoken_scale": result[2]})
            return result

        dsa_v1.AscendDSAImpl._mla_prolog_single_stream = prolog
        original_get = dsa_attn_kv_plan.DsaAttnKvPlan.get_dsa_sparse_attn_op

        def get_sparse(this):
            original = original_get(this)

            def sparse(q, **kwargs):
                active = self.active_attention is not None and self.enabled()
                if active:
                    assert kwargs["layout_q"] == "TND" and kwargs["layout_kv"] == "PA_ND"
                    assert kwargs["ori_mask_mode"] == 4 and "ori_sparse_indices" in kwargs
                    count = self.runner.drafter.num_query_per_req
                    assert q.shape[0] == count
                    selected, cache_info = select_visible_cache(kwargs["ori_kv"], kwargs["ori_sparse_indices"], count)
                    self.event(self.active_attention + ".sparse_reader.inputs", {
                        "q": q, "sinks": kwargs["sinks"], "selected_cache": selected,
                        "ori_sparse_indices": kwargs["ori_sparse_indices"][:count],
                        "ori_block_table": kwargs["ori_block_table"], "query_start_loc": kwargs["cu_seqlens_q"],
                        "seq_lens": kwargs["seqused_kv"]}, cache_info)
                result = original(q, **kwargs)
                if active:
                    self.event(self.active_attention + ".sparse_reader.output", {"output": result[0]})
                return result
            return sparse

        dsa_attn_kv_plan.DsaAttnKvPlan.get_dsa_sparse_attn_op = get_sparse
        return {"draft_hc_layers": 3, "draft_context_projection_layers": 3,
                "actual_reader_hook": "DsaAttnKvPlan native SparseAttnSharedkv invocation",
                "selected_cache_by_actual_sparse_indices": True, "capture_device_reads": False,
                "hc_method_compiler_inlining_disabled_for_observation": False,
                "hc_observation": "actual native Torch OpOverload/OpOverloadPacket dispatcher",
                "native_arithmetic_unchanged": True, "host_schedule_modified": True,
                "performance_or_full_consumer_precision_proof": False}


def begin_request(worker):
    trace = worker._tiny_first_value_trace
    trace.helper_marker = len(trace.helper_history)
    return {"previous_helper_calls": trace.helper_marker}


def read_helper_receipt(worker):
    trace = worker._tiny_first_value_trace
    history = trace.helper_history[trace.helper_marker:]
    return {"native_helper_calls_this_request": len(history),
            "native_helper_successful_returns_this_request": sum(row["native_helper_returned"] for row in history),
            "cold_capture_helper_calls_global": sum(row["capturing"] for row in trace.helper_history),
            "cold_capture_helper_successful_returns_global": sum(row["capturing"] and row["native_helper_returned"]
                                                                   for row in trace.helper_history),
            "helper_history_this_request": list(history), "submission_and_context_only": True,
            "graph_replay_helper_executes_no_python": True, "device_completion_requires_terminal_fence": True}
