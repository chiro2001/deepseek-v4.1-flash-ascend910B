"""Formal W4A8 native/core/stack banks in the same eight workers."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
os.environ['VLLM_DISABLE_COMPILE_CACHE'] = '1'
os.environ['STACK_REAL_WEIGHTS'] = '1'
os.environ['STACK_TP8_ARM'] = 'tp8base'


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--output', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--physical-chips', required=True)
    p.add_argument('--pairs', type=int, default=10)
    p.add_argument('--arms', default='tp8base,tp8core,tp8act,tp8stack')
    p.add_argument('--cpu-bind', action='store_true',
                   help='Enable NUMA binding only after validating shared-host memory migration')
    p.add_argument('--audit', action='store_true')
    p.add_argument('--fp32-decode-reduction', action='store_true')
    p.add_argument('--hccl-deterministic', choices=('false','true','strict'))
    p.add_argument('--reduction-evidence', type=Path)
    p.add_argument('--profile', action='store_true',
                   help='Collect bounded per-rank metrics after unprofiled paired timing')
    p.add_argument('--profile-metrics', default='PipeUtilization,ArithmeticUtilization,Memory,MemoryL0,MemoryUB,L2Cache,ResourceConflictRatio')
    args = p.parse_args()
    assert not (args.audit and args.profile), 'Route/clone audit and profiling run separately'
    assert (os.getenv('STACK_FP32_DECODE_REDUCTION') == '1') == args.fp32_decode_reduction
    assert bool(args.reduction_evidence) == args.fp32_decode_reduction
    if args.hccl_deterministic is not None:
        assert os.getenv('HCCL_DETERMINISTIC') == args.hccl_deterministic
    from formal_tp8_profile import METRICS
    profile_metrics = args.profile_metrics.split(',')
    assert set(profile_metrics) <= set(METRICS) and len(set(profile_metrics)) == len(profile_metrics)
    assert os.getenv('TINY_PERF_RANDOM_VALIDATION') != '1'
    os.environ['STACK_REAL_AUDIT'] = '1' if args.audit else '0'
    chips = [int(x) for x in args.physical_chips.split(',')]
    assert len(chips) == len(set(chips)) == 8 and min(chips) >= 0
    assert os.getenv('STACK_PHYSICAL_CHIPS', os.environ['ASCEND_RT_VISIBLE_DEVICES']) == args.physical_chips
    assert args.pairs > 0
    arms = args.arms.split(',')
    assert arms[0] == 'tp8base' and len(set(arms)) == len(arms)
    assert set(arms) <= {'tp8base','tp8core','tp8act','tp8stack'}
    os.environ['OPT_BLOCKMAP_VERIFY'] = '1' if args.audit else '0'
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    os.environ['VLLM_CACHE_ROOT'] = str(out / 'cache')
    from formal_model_contract import inspect_checkpoint
    contract = inspect_checkpoint(args.model)
    (out/'checkpoint.json').write_text(json.dumps(contract, indent=2)+'\n')
    if args.fp32_decode_reduction:
        from reduction_evidence import verify
        validated=verify(args.reduction_evidence,contract['config_sha256'],args.physical_chips)
        (out/'validated_reduction_evidence.json').write_text(json.dumps(validated,indent=2)+'\n')
    import tp8_patches as patches
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tokenizer=args.model, load_format='auto', dtype='bfloat16',
              safetensors_load_strategy='lazy',
              model_loader_extra_config={'enable_multithread_load':True,'num_threads':128},
              worker_cls='real_tp8_worker.RealTP8StackWorker', tensor_parallel_size=8,
              distributed_executor_backend='mp', enable_expert_parallel=True, seed=0,
              trust_remote_code=True, async_scheduling=False, limit_mm_per_prompt={'image': 4},
              max_model_len=8192, max_num_seqs=1, max_num_batched_tokens=2048,
              gpu_memory_utilization=.70, kv_cache_memory_bytes=4*1024**3,
              block_size=128, enable_prefix_caching=False, enable_return_routed_experts=args.audit,
              profiler_config={'profiler':'torch','torch_profiler_dir':str(out/'prof'),
                               'torch_profiler_with_stack':False} if args.profile else None,
              compilation_config={'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': [1]},
              additional_config={'enable_engram': True, 'engram_storage':'int8',
                  'enable_cpu_binding': args.cpu_bind,
                  'ascend_compilation_config': {'enable_npugraph_ex': True, 'enable_static_kernel': True},
                  'multistream_dsv4_dsa_overlap': False})
    banks = {}
    banks[arms[0]] = llm.collective_rpc(patches.save, args=(arms[0],))
    def record_banks():
        for rows in banks.values():
            assert len(rows) == 8 and {r['rank'] for r in rows} == set(range(8)), rows
        (out/'banks.json').write_text(json.dumps(banks, indent=2)+'\n')
    record_banks()
    engine = llm.llm_engine; records = []; counter = 0
    tokenizer = llm.get_tokenizer()
    topics = ['请分析延迟受限计算的优化原理，给出判断依据。',
              '请解释缓存命中、数据搬运和流水线重叠的关系。',
              '请比较矩阵向量运算与矩阵乘法的性能特点。']
    prompts = [tokenizer.encode((topic + '这是供分析的上下文，保持事实准确。')*512,
                               add_special_tokens=False)[:2048] for topic in topics]
    assert all(len(prompt) == 2048 for prompt in prompts)
    (out/'prompts.json').write_text(json.dumps(prompts)+'\n')

    def request(arm, tag, seed, metric=None):
        nonlocal counter
        llm.collective_rpc(patches.switch, args=(arm,)); counter += 1
        count = 47 + seed % 2 if args.audit else 48
        prompt = prompts[seed % len(prompts)]
        engine.add_request(f'stack8-{counter}', {'prompt_token_ids': prompt},
            SamplingParams(temperature=0, max_tokens=count, ignore_eos=True,
                           detokenize=False, logprobs=5 if args.audit else None))
        times = []; final = None
        profiling = False
        try:
            while engine.has_unfinished_requests():
                if metric and len(times) == 9:
                    from formal_tp8_profile import configure
                    receipt = llm.collective_rpc(configure, args=(metric, str(out/'prof'/arm/metric), 5, 10))
                    (out/f'profile_receipt_{arm}_{metric}.json').write_text(json.dumps(receipt,indent=2)+'\n')
                    llm.start_profile(); profiling = True
                start = time.perf_counter(); outputs = engine.step()
                times.append((time.perf_counter()-start)*1000)
                if profiling:
                    from formal_tp8_profile import advance
                    llm.collective_rpc(advance)
                    if len(times) == 24:
                        llm.stop_profile(); profiling = False
                for value in outputs:
                    if value.finished: final = value
        finally:
            if profiling: llm.stop_profile()
        assert final is not None and len(final.outputs[0].token_ids) == count
        output = final.outputs[0]
        record = {'arm': arm, 'tag': tag, 'seed': seed, 'step_ms': times,
                  'decode_median_ms': statistics.median(times[9:]), 'token_ids': output.token_ids,
                  'text':tokenizer.decode(output.token_ids), 'profile_metric':metric}
        if args.audit:
            import numpy as np
            routes = output.routed_experts
            assert routes is not None and routes.shape == (2048+count-1, 40, 6), getattr(routes, 'shape', None)
            valid=np.all((routes>=0)&(routes<384),axis=-1)&np.all(np.diff(np.sort(routes,axis=-1),axis=-1)>0,axis=-1)
            receipt={'arm':arm,'tag':tag,'shape':list(routes.shape),'min':int(routes.min()),'max':int(routes.max()),
                     'valid_prefill_pairs':int(valid[:2048].sum()),'valid_decode_pairs':int(valid[2048:].sum()),
                     'unique_ids':np.unique(routes).tolist()[:30]}
            (out/f'route_diagnostic_{counter}.json').write_text(json.dumps(receipt,indent=2)+'\n')
            print('REAL_TP8_ROUTE_DIAGNOSTIC',json.dumps(receipt),flush=True)
            import route_capture_patch
            route_status=llm.collective_rpc(route_capture_patch.status)
            (out/f'route_capture_status_{counter}.json').write_text(json.dumps(route_status,indent=2)+'\n')
            assert routes.min() >= 0 and routes.max() < 384 and np.all(valid)
            record['routes'] = routes.tolist()
            record['logprobs'] = [{str(k):v.logprob for k,v in row.items()} for row in output.logprobs]
            record['rank_audit'] = llm.collective_rpc(patches.audit, args=(arm,))
            import route_capture_patch
            record['route_capture_status'] = llm.collective_rpc(route_capture_patch.status)
        records.append(record)
        (out/'requests.json').write_text(json.dumps(records, indent=2)+'\n')
        print('REAL_TP8_REQUEST', json.dumps({k:v for k,v in record.items() if k not in ['step_ms','routes','token_ids','logprobs','rank_audit','route_capture_status']}), flush=True)
        return record

    request('tp8base', 'warmup', 0); request('tp8base', 'warmup', 1)
    print('REAL_TP8_NATIVE_READY', json.dumps({'formal_weights':True,'engram':True,
        'tp':8,'physical_chips':chips,'audit':args.audit}), flush=True)
    native_controls = []
    def check_native(reference, actual, phase):
        from formal_comparison import compare_requests
        comparison = compare_requests(reference, actual, phase)
        native_controls.append(comparison)
        (out/'native_controls.json').write_text(json.dumps(native_controls,indent=2)+'\n')
        print('REAL_TP8_NATIVE_CONTROL',json.dumps(comparison),flush=True)
        assert comparison['passed'], ('Native stability gate failed',phase,comparison)
    if args.audit:
        native_reference = request('tp8base','native-control-reference',0)
        check_native(native_reference,request('tp8base','native-control-repeat',0),'before-candidates')
    # Establish a valid native request before compiling candidate banks. All
    # variants remain in these same eight workers and use the same checkpoint.
    for arm in arms[1:]:
        if args.audit:
            native_before = request('tp8base','native-before-'+arm,0)
        banks[arm] = llm.collective_rpc(patches.create, args=(arm,))
        record_banks()
        request(arm, 'warmup', 0); request(arm, 'warmup', 1)
        if args.audit:
            check_native(native_before,request('tp8base','native-after-'+arm,0),'capture-'+arm)
    comparisons = []
    for pair in range(args.pairs):
        order = arms if pair % 2 == 0 else list(reversed(arms))
        rows = {arm: request(arm, f'pair-{pair}', pair % 3) for arm in order}
        for arm in arms[1:]:
            ref = rows[arms[0]]; actual = rows[arm]
            assert ref['token_ids'] == actual['token_ids'], (pair, arm, 'token')
            comparison = {'pair':pair,'arm':arm,'baseline_ms':ref['decode_median_ms'],
                          'candidate_ms':actual['decode_median_ms'],
                          'speedup':ref['decode_median_ms']/actual['decode_median_ms'],'tokens_equal':True}
            if args.audit:
                from formal_comparison import compare_requests
                diagnostic=compare_requests(ref,actual,f'pair-{pair}-{arm}')
                (out/f'comparison_diagnostic_{pair}_{arm}.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
                assert ref['routes'] == actual['routes'], (pair, arm, 'routes')
                assert all(set(r)==set(a) for r,a in zip(ref['logprobs'], actual['logprobs']))
                delta=max(abs(v-actual['logprobs'][i][k]) for i,row in enumerate(ref['logprobs']) for k,v in row.items())
                assert delta < 1e-3, (pair, arm, delta)
                comparison.update(actual_routes_equal=True, max_logprob_delta=delta)
            comparisons.append(comparison)
        (out/'comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    result = {'tensor_parallel_size':8,'physical_chips':chips, 'formal_weights':True, 'requested_load_format':'auto',
              'model_path':args.model, 'checkpoint_config_sha256':contract['config_sha256'],
              'safetensors_load_strategy':'lazy','multithread_loader_threads':128,
              'engram_enabled':True, 'speculative_decoding':False,
              'engram_storage':'int8', 'cpu_binding':args.cpu_bind,
              'fp32_decode_reduction':args.fp32_decode_reduction,
              'hccl_deterministic_env':os.getenv('HCCL_DETERMINISTIC'),
              'reduction_evidence':str(args.reduction_evidence) if args.reduction_evidence else None,
              'vision_enabled':True,'limit_mm_per_prompt':{'image':4},'request_modality':'text',
              'same_model_instance':True,'same_processes_per_rank':True,'audit':args.audit,
              'profiler':'OFF','A':1,'arms':{},'pairs':comparisons}
    for arm in arms:
        ms=statistics.median(r['decode_median_ms'] for r in records if r['arm']==arm and r['tag'].startswith('pair-'))
        result['arms'][arm]={'ms_per_step':ms,'A':1,'tokens_per_second':1000/ms}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('REAL_TP8_COMPLETE', json.dumps(result), flush=True)
    if args.profile:
        for arm in dict.fromkeys([arms[0], arms[-1]]):
            for metric in profile_metrics:
                request(arm, 'profile-'+metric, 0, metric)
        print('REAL_TP8_PROFILE_COMPLETE', json.dumps({'metrics':profile_metrics,
              'ranks':8,'warmup':5,'active':10,'timing_result_excludes_profile':True}),flush=True)


if __name__ == '__main__': main()
