"""Read actual native SparseAttnSharedkv inputs after real steps; never during capture."""
import hashlib
from pathlib import Path

from geometry_contract import K, MAX_BATCHED_TOKENS, MAX_MODEL_LEN, CAPTURE_SEQ_LEN, WINDOW
from geometry_contract import validate_configuration, validate_query_positions, validate_reader_indices

STATE = {"active_role": None, "capture": {}, "eager": {}, "worker": None,
         "snapshot_entry_host_guard_written": False}


def install(worker):
    import importlib
    import os
    import graph_precision_audit as audit
    from vllm_ascend.attention import dsa_v1, dsa_attn_kv_plan
    assert hashlib.sha256(Path(dsa_v1.__file__).read_bytes()).hexdigest() == "24cb4e1ad3b2fcc9f0524153f22c0c6818aba396d290c4b93cdf4f12ced217d5"
    extra_sources = {
        "vllm_ascend.core.deepseek_v41": "d0d89cb4a0ef22672e1de92a80a2846ac86f8fd4ec8e7d14ab050bf1539d7acf",
        "vllm_ascend.core.kv_cache_interface": "1283cb2f69ba2cc5a4c1a50d6dacfc5e35396203add971bbe998263939081dc1",
        "vllm_ascend.models.deepseek_v4.model": "9ef83c7c2879f7b6115f5ac64556703858afa3786bab04ee05f9a65a15353061",
        "vllm.v1.core.block_pool": "51cad2fd425128a0ff433ca4685acfc022e40659153ea5d7180fc586e1eebb6c",
    }
    for name, expected in extra_sources.items():
        module = importlib.import_module(name)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected
    validate_configuration(worker.vllm_config.model_config.max_model_len,
                           worker.vllm_config.scheduler_config.max_num_batched_tokens,
                           int(os.environ["DSPARK_CAPTURE_SEQ_LEN"]),
                           int(worker.model_runner._tiny_dspark_native_draft.config.sliding_window),
                           worker.model_runner.drafter.num_speculative_tokens)
    assert os.environ["DSPARK_HOIST_CONTEXT_KV"] == "0"
    STATE["worker"] = worker
    original_attention = dsa_v1.AscendDSAImpl._forward_attention

    def attention(this, layer_name, *args, **kwargs):
        role = next((f"draft.{index}" for index in range(3)
                     if str(layer_name).startswith(f"mtp.{index}.")), None)
        previous = STATE["active_role"]
        STATE["active_role"] = role
        try:
            return original_attention(this, layer_name, *args, **kwargs)
        finally:
            STATE["active_role"] = previous

    dsa_v1.AscendDSAImpl._forward_attention = attention
    original_get = dsa_attn_kv_plan.DsaAttnKvPlan.get_dsa_sparse_attn_op

    def get_sparse(this):
        native = original_get(this)

        def sparse(q, **kwargs):
            role = STATE["active_role"]
            if role is not None:
                assert kwargs["layout_q"] == "TND" and kwargs["layout_kv"] == "PA_ND"
                assert kwargs["ori_mask_mode"] == 4 and "ori_sparse_indices" in kwargs
                bank = STATE["capture"] if audit._capturing() else STATE["eager"]
                bank[role] = {"q": q, "cache": kwargs["ori_kv"],
                              "indices": kwargs["ori_sparse_indices"], "table": kwargs["ori_block_table"],
                              "seq_lens": kwargs["seqused_kv"], "query_start_loc": kwargs["cu_seqlens_q"],
                              "capture_reference": audit._capturing()}
            return native(q, **kwargs)
        return sparse

    dsa_attn_kv_plan.DsaAttnKvPlan.get_dsa_sparse_attn_op = get_sparse
    # The old finish() reads every prefix block, which is not valid after a
    # SlidingWindowManager evicts old pages. This diagnostic observes every
    # actual native visible row and compares it by logical position instead.
    audit.finish = finish
    original_compare = audit.compare

    def compare(this_worker, left_tag, right_tag):
        result = original_compare(this_worker, left_tag, right_tag)
        left, right = (audit.STATE["requests"][tag] for tag in (left_tag, right_tag))
        for a, b in zip(left["steps"], right["steps"]):
            assert a["is_decode"] == b["is_decode"]
            if not a["is_decode"]:
                assert "visible_cache" not in a and "visible_cache" not in b
                continue
            assert set(a["visible_cache"]) == set(b["visible_cache"]) == {f"draft.{i}" for i in range(3)}
            for role in a["visible_cache"]:
                x, y = a["visible_cache"][role], b["visible_cache"][role]
                assert x["logical_positions"] == y["logical_positions"]
                assert x["kernel_block_size"] == y["kernel_block_size"]
                assert audit._bitwise(x["values"], y["values"]), (role, "Actual visible cache differs")
        result.update({"all_step_actual_visible_cache_bitwise": True,
                       "unretained_prefix_not_read": True, "independent_2048_256_gate": True})
        return result

    audit.compare = compare
    return {"actual_native_reader_capture_reference_only": True, "capture_device_reads": False,
            "original_sparse_call_arguments_and_return_preserved": True,
            "SWA_visible_context": WINDOW, "K": K, "math_or_mtp_order_changed": False,
            "actual_extra_source_guards": extra_sources, "native_null_block_id": 0}


def validate_snapshot_entry(worker):
    """Keep three entry gates; capture rejects before any diagnostic I/O."""
    import graph_precision_audit as audit
    assert not audit._capturing(), "geometry_snapshot.capture_forbidden"
    identity_equal = worker is STATE["worker"]
    audit_enabled = bool(audit.STATE["enabled"])
    if not STATE["snapshot_entry_host_guard_written"]:
        current = audit.STATE["current"]
        audit._diagnostic_write("geometry2k.snapshot_entry_host_guard", {
            "worker_id": id(worker), "installed_worker_id": id(STATE["worker"]),
            "worker_type": type(worker).__name__,
            "installed_worker_type": type(STATE["worker"]).__name__,
            "worker_identity_equal": identity_equal, "audit_enabled": audit_enabled,
            "capture": False, "current_tag": current["tag"] if current is not None else None,
            "host_scalars_only": True, "device_reads": False,
            "once_per_process": True, "entry_state_is_not_geometry_or_precision_proof": True})
        STATE["snapshot_entry_host_guard_written"] = True
    assert identity_equal, "geometry_snapshot.worker_identity_mismatch"
    assert audit_enabled, "geometry_snapshot.audit_disabled"


def snapshot_step(worker, is_decode, draft_replayed):
    validate_snapshot_entry(worker)
    import torch
    import graph_precision_audit as audit
    import native_first_value_trace as trace
    torch.npu.synchronize()
    runner, drafter = worker.model_runner, worker.model_runner.drafter
    scheduled = runner._tiny_dspark_receipts[-1]
    assert scheduled["is_decode"] and bool(is_decode), "Geometry snapshots require a real decode step"
    positions_tensor = drafter._get_positions(drafter.num_query_per_req)
    assert positions_tensor.dtype == torch.int32
    positions = audit._cpu(positions_tensor, "geometry2k.query_positions").reshape(-1).tolist()
    bank = STATE["capture"] if draft_replayed else STATE["eager"]
    assert set(bank) == {f"draft.{index}" for index in range(3)}
    receipts, values_by_role = {}, {}
    for index, layer in enumerate(runner._tiny_dspark_native_draft.layers.values()):
        role, binding = f"draft.{index}", bank[f"draft.{index}"]
        gid = drafter._layer_group_idx[index]
        kernel_block_size = int(drafter._per_group_kernel_block_sizes[gid])
        cache = layer.self_attn.dsa_attn.swa_cache_layer.kv_cache
        while isinstance(cache, (tuple, list)) and len(cache) == 1:
            cache = cache[0]
        assert binding["cache"].data_ptr() == cache.data_ptr()
        assert tuple(cache.shape[1:]) == (kernel_block_size, 1, 512) and cache.dtype == torch.bfloat16
        assert kernel_block_size == 128 and binding["indices"].dtype == torch.int32
        native_table = audit._cpu(binding["table"], role + ".native_reader_block_table")
        live_table = audit._cpu(drafter._per_group_block_tables[gid], role + ".live_group_block_table")
        assert torch.equal(native_table[0], live_table[0]), "Captured reader block table is stale"
        seq_len = int(audit._cpu(binding["seq_lens"], role + ".native_seq_len").reshape(-1)[0])
        validate_query_positions(positions, scheduled, seq_len)
        indices = audit._cpu(binding["indices"][:K], role + ".actual_sparse_indices")
        assert indices.ndim == 3 and indices.shape[1] == 1
        receipt = validate_reader_indices(indices[:, 0].tolist(), native_table[0].tolist(), seq_len,
                                           kernel_block_size, int(cache.shape[0]), null_block_id=0)
        selected, selected_info = trace.select_visible_cache(binding["cache"], binding["indices"], K)
        values = audit._cpu(selected, role + ".all_actual_native_selected_cache_values")
        assert values.shape[0] == receipt["selected_slots_count_with_duplicates"]
        receipt.update({"role": role, "gid": gid, "cache_ptr": cache.data_ptr(),
                        "native_indices_ptr": binding["indices"].data_ptr(),
                        "native_table_ptr": binding["table"].data_ptr(),
                        "live_table_ptr": drafter._per_group_block_tables[gid].data_ptr(),
                        "positions": positions, "finite": True, "source_cache_unchanged": True,
                        "capture_reference": bool(draft_replayed), "actual_scheduled_round": scheduled,
                        "value_sha256": hashlib.sha256(values.view(torch.uint8).numpy().tobytes()).hexdigest()})
        receipts[role] = receipt
        values_by_role[role] = {"logical_positions": receipt["logical_positions"],
                               "kernel_block_size": kernel_block_size, "logical_rows": len(receipt["logical_positions"]),
                               "values": values, "receipt": receipt}
    audit.STATE["current"]["steps"][-1]["visible_cache"] = values_by_role
    audit._diagnostic_write("geometry2k." + audit.STATE["current"]["tag"] + ".step." + str(len(audit.STATE["current"]["steps"])), receipts)
    return {"per_group": receipts, "positions": positions, "independent_2048_256": True,
            "performance_evaluated": False, "capture_device_reads": False}


def finish(worker):
    import torch
    import graph_precision_audit as audit
    assert audit.STATE["enabled"] and audit.STATE["current"] is not None and not audit._capturing()
    torch.npu.synchronize()
    current = audit.STATE["current"]
    assert current["steps"] and "visible_cache" in current["steps"][-1]
    current["cache"] = current["steps"][-1]["visible_cache"]
    current["query_positions"] = next(iter(current["cache"].values()))["receipt"]["positions"]
    assert audit._parameter_metadata(worker) == audit.STATE["parameter_seal"]
    current["parameter_seal_passed"] = True
    audit.STATE["requests"][current["tag"]] = current
    audit.STATE["current"], audit.STATE["enabled"] = None, False
    return {"tag": current["tag"], "actual_route_layers": 43, "actual_draft_attention_layers": 3,
            "actual_step_snapshots": len(current["steps"]), "draft_cache_layers": 3,
            "query_positions": current["query_positions"], "parameter_seal_passed": True,
            "native_SWA_visible_cache_only": True, "unretained_prefix_not_read": True,
            "geometry_cache_gate_is_diagnostic_only": True}
