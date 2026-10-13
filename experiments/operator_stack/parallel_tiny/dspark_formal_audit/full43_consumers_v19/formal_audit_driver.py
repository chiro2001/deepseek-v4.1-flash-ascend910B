"""Root-controlled fresh real-weight diagnostic/audit entry; default CPU dry-run."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    __package__ = 'parallel_tiny.dspark_formal_audit.full43_consumers_v19'

from .provenance import candidate_config


def summarize_rank_exports(output,generations):
    result = {}
    expected_decoders = {f'main.{layer}.decoder.output' for layer in range(40)} | {f'draft.{layer}.decoder.output' for layer in (40,41,42)}
    for rank in range(8):
        directory = Path(output)/'rank_exports'/f'rank{rank}'
        paths = sorted(directory.glob('epoch_*.json'))
        records = [json.loads(path.read_text()) for path in paths]
        qualifying = [record for record in records if record.get('actual_spec_decode_round') is True]
        binding_path = directory/'loaded_bindings.json'
        loaded = json.loads(binding_path.read_text()) if binding_path.is_file() else {}
        module_rows = loaded.get('module_bindings',[])
        loaded43 = (loaded.get('registered_native_modules') == 43 and len(module_rows) == 43
            and len({row['module_object_id'] for row in module_rows}) == 43
            and {(row['role'],row['global_layer']) for row in module_rows}
                == {('main',layer) for layer in range(40)} | {('draft',layer) for layer in (40,41,42)})
        complete_epochs = []
        for record in qualifying:
            generation = record['generation'];points = record.get('records',{})
            decoder_complete = all(name in points and points[name]['epoch'] == generation
                and points[name]['copy_count'] == 1 and points[name]['rows'] > 0
                and any(points[name]['valid'])
                and all(pos[0] >= 0 for pos,valid in zip(points[name]['positions'],points[name]['valid']) if valid)
                for name in expected_decoders)
            graph = record.get('actual_graph_output_exports',{})
            graphs_complete = all(role in graph and graph[role]['all_explicit_outputs_observed']
                and graph[role]['all_values_equal_owned_after_fence'] and graph[role]['all_outputs_alias_original_storage']
                for role in ('main','draft'))
            if decoder_complete and graphs_complete:complete_epochs.append(generation)
        result[str(rank)] = {'all_export_files':len(records),'actual_spec_decode_rounds':len(qualifying),
            'actual_generation_ids':[record['generation'] for record in qualifying],
            'actual_loaded43_bindings':loaded43,'complete_43_decoder_and_graph_output_generations':complete_epochs,
            'at_least_requested_generations':len(qualifying) >= generations,
            'diagnostic_export_contract_complete':loaded43 and len(complete_epochs) >= generations}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model',required=True)
    parser.add_argument('--physical-chips',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--generations',type=int,default=32)
    parser.add_argument('--max-output-tokens',type=int,default=192)
    parser.add_argument('--mode',choices=['diagnostic','audit'],default='diagnostic')
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args(argv)
    if args.physical_chips != '8,9,10,11,12,13,14,15' or args.generations < 32:
        raise ValueError('Only fresh root-assigned formal TP8 8–15 / at least32 generations')
    config = candidate_config(args.model,args.output,audit_enabled=args.execute)
    worker_cls = 'parallel_tiny.dspark_formal_audit.full43_consumers_v19.sdk_worker.Full43AuditWorker'
    if not args.execute:
        print(json.dumps({'scope':'CPU preparation only; no worker/model/NPU construction',
                         'driver_package':__package__,'worker_cls':worker_cls,'config':config,'npu_jobs':0,
                         'strict_full43_admission':False},indent=2))
        return 0
    if any(os.environ.get(name) != args.physical_chips for name in ['ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES']):
        raise ValueError('Actual host visibility differs from root fresh8–15 assignment')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'candidate_contract.json').write_text(json.dumps(config,indent=2)+'\n')
    os.environ.update(config['startup_env'])
    os.environ['V10_AUDIT_OUTPUT'] = str(args.output/'rank_exports')
    os.environ['V10_AUDIT_ENABLED'] = '1'
    for name,relative in [('ASCEND_CACHE_PATH','cache/ascend'),('TRITON_CACHE_DIR','cache/triton'),('TMPDIR','tmp')]:
        path = args.output/relative;path.mkdir(parents=True,exist_ok=True);os.environ[name] = str(path)
    result = {'scope':'partial private real-weight binding/graph/export diagnostic; not formal admission',
              'mode':args.mode,'passed':False,'strict_full43_admission':False,'full_consumer_audit_passed':False,
              'device_audit_passed':False,'safe_formal_receipt':False,'formal_model_validated':False,
              'actual_prepack_consumer_order_validated':False,'performance_claim':None}
    llm = None
    try:
        from vllm import LLM, SamplingParams
        from .sdk_executor import Full43AuditExecutor
        kwargs = dict(config['model_kwargs']);kwargs['worker_cls'] = worker_cls
        kwargs['distributed_executor_backend'] = Full43AuditExecutor
        llm = LLM(**kwargs)
        sampling = SamplingParams(temperature=0,max_tokens=args.max_output_tokens,min_tokens=args.max_output_tokens,
                                  ignore_eos=True,logprobs=5)
        outputs = llm.generate(['Explain a queue and a stack, then give short examples.'],sampling,use_tqdm=False)
        result['request_outputs'] = [{'request_id':item.request_id,
            'outputs':[{'token_ids':list(output.token_ids),'text':output.text,'finish_reason':output.finish_reason}
                       for output in item.outputs]} for item in outputs]
        result['diagnostic_request_completed'] = True
        result['passed'] = False  # Set only after actual exports AND reaped codes.
        result['strict_uncovered'] = ['frontend_scheduler_rollback','actual_prepack_before_pack_order',
                                     'all17_consumer_device_coverage','native_AA_math_logprob','quality','performance']
        # Audit mode is deliberately closed until all exported 17-point and
        # frontend correspondence checks are completed, not merely registered.
        if args.mode == 'audit':result['failure'] = 'Full17 device/rollback/math admission not implemented in this first diagnostic window'
    except BaseException as exc:
        result.update(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc())
    finally:
        if llm is not None:
            try:
                # The v8 diagnostic died with worker_exit_codes_complete=false
                # because the default 5s ProcessManager grace force-killed
                # EngineCore before the executor teardown could finish writing
                # owned_worker_cleanup.json. Give the teardown a bounded window
                # (45s success grace inside finish_owned_workers + reaping).
                llm.llm_engine.engine_core.shutdown(timeout=150)
                result['normal_SDK_shutdown_called'] = True
                result['shutdown_timeout_seconds'] = 150
            except BaseException as exc:
                result['shutdown_error'] = type(exc).__name__+': '+str(exc)
        cleanup_path = args.output/'owned_worker_cleanup.json'
        if cleanup_path.is_file():
            cleanup = json.loads(cleanup_path.read_text());result['owned_worker_cleanup'] = cleanup
            result['worker_exit_codes'] = cleanup['worker_exit_codes']
            result['worker_exit_codes_complete'] = cleanup['all_terminal'] and len(cleanup['worker_exit_codes']) == 8
        else:result['worker_exit_codes_complete'] = False
        actual_exports = summarize_rank_exports(args.output,args.generations)
        result['actual_exports'] = actual_exports
        complete = all(record['diagnostic_export_contract_complete'] for record in actual_exports.values())
        terminal = result['worker_exit_codes_complete'] and all(code == 0 for code in result.get('worker_exit_codes',{}).values())
        result['passed'] = bool(args.mode == 'diagnostic' and result.get('diagnostic_request_completed')
                                and complete and terminal and not result.get('shutdown_error'))
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ['passed','mode','strict_full43_admission','safe_formal_receipt']},indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':raise SystemExit(main())
