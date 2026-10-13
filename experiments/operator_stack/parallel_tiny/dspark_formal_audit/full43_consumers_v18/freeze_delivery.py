"""Freeze exactly this owned diagnostic text package; no SDK/device actions."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
UNCOVERED=[
 'all17_actual_device_consumer_coverage_not_granted_by_CPU_or_source_registration',
 'actual_target_aux_residual_mean_entering_layers37_38_39_not_yet_tapped',
 'actual_per_group_proposer_positions_block_slot_sequence_metadata_not_yet_exported',
 'actual_rejection_sampler_device_output_not_yet_exported',
 'accepted_prefix_bonus_and_bookkeeping_CPU_correspondence_not_yet_connected',
 'frontend_native_scheduler_rejected_rollback_stop_clip_not_yet_connected',
 'dispatch_expanded_row_to_actual_token_position_mapping_not_yet_bound',
 'shared_context_valid_rows_and_three_serial_delayed_mHC_consumer_correspondence_not_yet_complete',
 'actual_formatted_C1_C2_indexer_and_SWA_cache_writer_bindings_not_yet_complete',
 'main_SparseFlashMla_actual_long_source_prefix_logical_group_and_PA_reader_mapping_not_yet_bound',
 'draft_actual_unpack_SWA_cache_plane_and_reader_mapping_not_yet_bound',
 'actual_cache_GM_storage_offset_stride_and_NPU_format_not_yet_fully_exported',
 'actual_native_GMM_selected_overload_dense_counts_and_raw_ND_input_ABI_not_yet_verified',
 'actual_borrowed_stream_GMM1_workspace_doublebank_and_GMM2_original_x_GM_lifecycle_not_yet_verified',
 'actual_device_copy_before_opaque_native_pack_consumer_order_not_yet_verified',
 'dynamic_prefill_other_format_activation_weight_list_EPLB_LoRA_prepack_paths_native_fallback_uncovered',
 'actual_fresh_TP8_loaded_weights_graph_outputs_32_rounds_and_reaped0_pending_root_diagnostic',
 'native_AA_math_and_logprob_correspondence_not_yet_run',
 'independent_general_quality_not_yet_run',
 'whole_main_plus_DSpark_step_performance_target20ms_not_yet_achieved',
]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE/'freeze.json').exists(),'Do not rewrite an already frozen version'
    same=json.loads((HERE/'evidence/CPU_same_version.json').read_text())
    execution=json.loads((HERE/'evidence/CPU_same_version_execution.json').read_text())
    cleanup=json.loads((HERE/'evidence/CPU_owned_cleanup.json').read_text())
    assert same['passed'] and same['tests']==25 and same['skipped']==0 and same['torch']=='2.10.0+cpu'
    assert same['original_consumer_functions_AST_verified']==17 and not same['SDK_imports'] and not same['NPU_dynamic_libraries']
    assert execution['exitcode']==0 and cleanup['passed'] and cleanup['skipped']==0
    assert digest(HERE/'evidence/CPU_same_version.json')==execution['stdout_sha256']
    assert digest(HERE/'evidence/CPU_same_version.log')==execution['stderr_sha256']
    input_proof=json.loads((HERE/'evidence/CPU_same_version_input_manifest.json').read_text())
    stack=HERE.parents[2]
    for rel,expected in input_proof['files'].items():
        if rel.startswith('stack/parallel_tiny/dspark_formal_audit/full43_consumers_v18/'):
            assert digest(stack/rel.removeprefix('stack/'))==expected,'CPU-tested v10 source changed: '+rel
    dependencies={
        'formal_consumer_sdk_bridge_v9_cpu':'1bbdc613969add1f47d9e678ac9ade9d35ad9e187a9985e8c21b67ff881623e5',
        'native_initialized_worker_discovery_v8':'73f865b8ce39a631db991cac1d6f8b39b35e9b7ab1e5cfdc1d89e7cadae33432',
        'dspark_prepack_audit_cpu':'bf2becb2f9e9e87e1d72ff1bbb23c88a5414f1d914e5ce8dec7b035182a902bd'}
    for name,expected in dependencies.items():
        base=HERE.parent/name if name!='dspark_prepack_audit_cpu' else stack/'parallel_tiny'/name
        manifest=base/('frozen_files.json' if name=='dspark_prepack_audit_cpu' else 'freeze.json')
        assert digest(manifest)==expected
        proof=json.loads(manifest.read_text());files=proof['text_files' if name=='dspark_prepack_audit_cpu' else 'files']
        for rel,spec in files.items():assert digest(base/rel)==(spec['sha256'] if isinstance(spec,dict) else spec)
    contract=stack/'parallel_tiny/dspark_formal_prep/contract.py'
    spec=importlib.util.spec_from_file_location('_v10_stdlib_contract_for_freeze',contract)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    env=module.startup_env('/work/ROOT_SELECTED_NEW_JOB/gate',audit=True,draft_graph=False)
    for name in ('VLLM_CACHE_ROOT','STACK_DSPARK_FORMAL_OUTPUT'):env.pop(name)
    env.update(ASCEND_VISIBLE_DEVICES=module.PHYSICAL_CHIPS,ASCEND_RT_VISIBLE_DEVICES=module.PHYSICAL_CHIPS,V10_AUDIT_ENABLED='1')
    no_actual={'device_audit_passed':False,'full_consumer_audit_passed':False,'safe_formal_receipt':False,
        'formal_model_validated':False,'actual_prepack_consumer_order_validated':False,'performance_claim':None}
    readiness={'scope':'source-guarded multimodal root ownership plus host-resident Engram seal repair for private full43 diagnostic only','ready_for_root_diagnostic':True,
        'NPU_permissions':0,'NPU_calls':0,'CPU_contracts_same_version_passed':25,'CPU_contracts_local_passed':25,
        'source_preservation_functions':17,'same_version_torch':same['torch'],'real_ordinary_children_reaped':4,
        'normal_child_exitcodes':[0,0],'failure_child_exitcodes':[-9,-9],'skip_count':0,
        'strict_uncovered_points':UNCOVERED,'diagnostic_pass_is_formal_admission':False,**no_actual}
    (HERE/'readiness.json').write_text(json.dumps(readiness,indent=2)+'\n')
    files={str(path.relative_to(HERE)):digest(path) for path in sorted(HERE.rglob('*')) if path.is_file()
        and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.tgz','.gz','.pt')
        and path.name not in ('freeze.json','SHA256SUMS')}
    assert all((HERE/name).stat().st_size<1_000_000 for name in files)
    freeze={'scope':readiness['scope'],'ready':True,'previous_v17_freeze_sha256':'7ea553eb31db58e102a9b6a51ce9cf94bda04bc3ca6ec7ff08cacb3d870034d4',
        'previous_v16_freeze_sha256':'da6786203cb3522a04bcc380002baad198a858e662f5b8ec421948ef17f533a6',
        'previous_v15_freeze_sha256':'12d7e76ca82ec76d1874645839ff581cf9c8e7a3b1c55e0885e1e1a96f966834',
        'previous_v14_freeze_sha256':'0fb7094e5180f0c864c4a042dfd609563d1d92ee0bbcd1d7e53dd1e148caf654',
        'previous_v13_freeze_sha256':'6bc5229f4ddceeca3c1414d29ba1b2f89a7ad8b8836b7bde23651bf91d425a10',
        'previous_v12_freeze_sha256':'64e5fae9a2f2d4269515b642e13aaf5762a8289fbfdcc7212969ed038e44ce3d',
        'previous_v11_freeze_sha256':'8f582de3e5c46337c7e002d4b0a1c94a1338acef33980875024f65f1ef752e4d',
        'previous_v10_freeze_sha256':'d6ea1badd7e3ec1ef62601ae5b2ac8716d9874cb7074cb8fc309cd350c7fea40',
        'repair_scope':'keep the K5 draft on its native eager path for the audit window (unwrapping the draft _runnable ACLGraphWrapper) so the draft decoder/attention consumer taps execute every round, and give the SDK shutdown a bounded 150s teardown window so owned_worker_cleanup.json is written (v18); keep the v17 subclass adapter for native isinstance guards, the v16 compiled-dispatch envelope and JSON-safe receipts, the v15 post-load identity audit and recorded adapter synthesis, the v14 list-container tolerance, the v13 dynamo module-attribute aliases, the v12 host-resident Engram seal and the v11 guarded multimodal root ownership',
        'prior_v13_failure':'formal_dspark_full43_diag_v4 passed all eight ranks through before_graph (weight_seals.json + dynamo_alias_audit.json with rebind_count=0 on every rank) and passed the dynamo guard, then failed in DSparkProposer.dummy_run: the draft outer return crossed the compiled/ACL boundary and NativeResultAdapter rejected a non-tuple container.',
        'prior_v14_failure':'formal_dspark_full43_diag_v5 wrote adapter_mismatch_draft.json on all eight ranks: the wrapped draft callable returned a raw [5,5120] bf16 npu tensor, i.e. the observer-buffer envelope was not produced by the callable the adapter wrapped.',
        'prior_v15_failure':'formal_dspark_full43_diag_v6 confirmed via post_load_identity.json that drafter.model is the draft root with the envelope tag on its instance forward, then failed while writing extension_identity.json: a raw bound method reached json.dumps (TypeError). The identity audit also showed _compiled_callable/compiled/evaluate_guards keys, i.e. @support_torch_compile dispatch never touches the instance forward.',
        'prior_v16_failure':'formal_dspark_full43_diag_v7 passed profile_run and the draft compiled-callable envelope (no envelope/adapter errors), then failed in DSparkProposer._propose: assert isinstance(model, _HIDDEN_STATE_DRAFTER_TYPES) rejects the plain proxy adapter stored at drafter.model.',
        'prior_v17_result':'formal_dspark_full43_diag_v8 completed the actual 32-generation diagnostic request (38 epochs, 37 spec rounds per rank) with complete main 40/40 decoder outputs and complete graph exports, but every draft decoder.output/shared/cache.draft.read point stayed at epoch 0: the draft _runnable ACLGraphWrapper replays recorded ops without running Python taps. passed=false also because worker_exit_codes_complete=false (no owned_worker_cleanup.json: the default 5s ProcessManager grace force-killed EngineCore before the executor teardown).',
        'prior_v12_failure':'formal_dspark_full43_diag_v3 passed 8-rank recovery, all-rank weight_seals.json and root ownership, then died in profile_run when Dynamo evaluated a module-attribute guard for "_v10_enter" on vllm_ascend.models.deepseek_v41.model (AttributeError; the name only existed in the transformed function globals).',
        'prior_job_passed':False,'prior_v11_failure':'formal_dspark_full43_diag_v2 rank1 before_graph seal invoked npu::get_npu_format on the CPU host-resident Engram embed.weight (NotImplementedError from the NPU dispatcher); rank1 wrote retained_root_ownership.json before failing. Evidence: evidence/formal_a321/full43_diag_v2/.',
        'prior_cleanup_reaped_exitcodes_proven':False,'NPU_permissions':0,'NPU_calls':0,
        'entry':'parallel_tiny/dspark_formal_audit/full43_consumers_v18/formal_audit_driver.py',
        'mode':'diagnostic','actual_sdk_gate_passed':False,'strict_full43_admission':False,
        'CPU_contracts_passed':20,'same_version_torch':same['torch'],'source_AST_preserved_functions':17,
        'startup_env':env,'startup_env_dynamic_from_root_and_driver':[
            'STACK_SRC_ROOT','V10_AUDIT_OUTPUT','STACK_DSPARK_FORMAL_OUTPUT','VLLM_CACHE_ROOT','TMPDIR','ASCEND_CACHE_PATH','TRITON_CACHE_DIR'],
        'startup_env_basis':{'source_path':str(contract),'sha256':digest(contract),'audit':True,'draft_graph':False},
        'dependency_freezes':dependencies,'strict_uncovered_points':UNCOVERED,
        'implemented_points':['actual_native_load_retention_before_original_ACLGraph_wrapper',
            'original_multimodal_vl_model_source_class_and_registered_language_model_model_owner_guard',
            'source_guarded_actual_40_main_3_draft_decoder_binding_and_parameter_seals',
            'private_source_derived_decoder_route_dispatch_shared_projection_and_partial_cache_taps',
            'IDS_pure_gmm1_gmm2_six_buffer_role_root_output_integration_for_supported_geometry',
            'original_root_positions_and_all_observer_tensor_outputs',
            'post_real_fence_raw_payload_and_compact_SHA_exports',
            'actual_spec_decode_round_count_and_strict_43_decoder_output_diagnostic_gate',
            'original_MP_immediate_birth_capture_and_actual_parent_owned_pidfd_cleanup',
            'host_resident_cpu_engram_parameter_seal_without_npu_format_query',
            'dynamo_module_attribute_guard_aliases_for_installed_audit_taps',
            'compiled_list_outer_observer_buffer_envelope_and_structure_diagnostic',
            'post_load_identity_and_envelope_extension_audit_with_recorded_adapter_synthesis',
            'compiled_dispatch_point_envelope_and_json_safe_extension_receipts',
            'adapter_subclass_of_wrapped_class_for_native_isinstance_guards',
            'draft_forced_native_eager_with_runnable_unwrap_receipt',
            'bounded_150s_sdk_shutdown_window_for_owned_worker_cleanup_receipt'],
        'diagnostic_pass_is_formal_admission':False,'files':files,
        'file_count_excluding_freeze_and_checksum':len(files),'file_count_including_freeze_and_checksum':len(files)+2,
        'max_file_bytes':max((HERE/name).stat().st_size for name in files),**no_actual}
    (HERE/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    checksums=dict(files,**{'freeze.json':digest(HERE/'freeze.json')})
    (HERE/'SHA256SUMS').write_text(''.join(f'{checksums[name]}  {name}\n' for name in sorted(checksums)))
    assert all(digest(HERE/name)==expected for name,expected in files.items())
    print(json.dumps({'freeze_sha256':digest(HERE/'freeze.json'),'SHA256SUMS_sha256':digest(HERE/'SHA256SUMS'),
        'files':len(files)+2,'max_file_bytes':freeze['max_file_bytes'],'strict_uncovered_points':len(UNCOVERED),'NPU_calls':0},indent=2))

if __name__=='__main__':main()
