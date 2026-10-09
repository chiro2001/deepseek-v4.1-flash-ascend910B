"""Single-process graph A/B; correctness audit is separate from timing."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['TINY_PERF_ARM'] = 'both'
os.environ['TINY_PERF_HC_ENABLE'] = '1'
os.environ['VLLM_DISABLE_COMPILE_CACHE'] = '1'
os.environ.setdefault('V41_DUMMY_WO_A_FIX', '1')


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--candidate', choices=['sparse_gmm','smla','indexer','epilogue','projection'], required=True)
    parser.add_argument('--arms', default='baseline,sparse')
    parser.add_argument('--output', required=True)
    parser.add_argument('--pairs', type=int, default=6)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    arms = args.arms.split(',')
    assert arms[0] == 'baseline' and len(set(arms)) == len(arms)
    os.environ['UP950_CANDIDATE'] = args.candidate
    os.environ['UP950_ARM'] = arms[0]
    if args.candidate == 'smla':
        vendor = '/work/build/smla/opp/vendors/up950_transformer'
        previous = os.environ.get('ASCEND_CUSTOM_OPP_PATH','')
        os.environ['ASCEND_CUSTOM_OPP_PATH'] = vendor+(':'+previous if previous else '')
    if args.audit:
        os.environ['TINY_PERF_RANDOM_VALIDATION'] = '1'
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    os.environ['VLLM_CACHE_ROOT'] = str(out / 'vllm_cache')
    import lane_patches as lane
    from vllm import LLM, SamplingParams

    llm = LLM(model='/model', tokenizer='/model', load_format='dummy', dtype='bfloat16',
              worker_cls='lane_worker.Upstream950Worker', tensor_parallel_size=1,
              enable_expert_parallel=True, seed=0, trust_remote_code=True,
              async_scheduling=False, limit_mm_per_prompt={'image': 0},
              max_model_len=8192, max_num_seqs=1, max_num_batched_tokens=2048,
              gpu_memory_utilization=.70, kv_cache_memory_bytes=4*1024**3,
              block_size=128, enable_prefix_caching=False,
              enable_return_routed_experts=args.audit,
              compilation_config={'cudagraph_mode': 'FULL_DECODE_ONLY',
                                  'cudagraph_capture_sizes': [1]},
              additional_config={'enable_engram': False, 'enable_cpu_binding': True,
                  'ascend_compilation_config': {'enable_npugraph_ex': True,
                                               'enable_static_kernel': False},
                  'multistream_dsv4_dsa_overlap': False})
    banks = [llm.collective_rpc(lane.save_bank, args=(arms[0],))[0]]
    for arm in arms[1:]:
        banks.append(llm.collective_rpc(lane.create_bank, args=(arm,))[0])
    (out / 'banks.json').write_text(json.dumps(banks, indent=2))
    print('BANKS', json.dumps(banks), flush=True)
    engine = llm.llm_engine
    records = []
    counter = 0

    def request(arm, tag, seed):
        nonlocal counter
        started_utc=time.time()
        llm.collective_rpc(lane.switch_bank, args=(arm,))
        counter += 1
        prompt = [100 + (i*17 + seed*19) % 97 for i in range(2048)]
        output_tokens=47+(seed%2) if args.candidate=='indexer' and args.audit else 48
        engine.add_request(f'up950-{counter}', {'prompt_token_ids': prompt},
                           SamplingParams(temperature=0, max_tokens=output_tokens, ignore_eos=True,
                                          detokenize=False, logprobs=5 if args.audit else None))
        timings, final = [], None
        while engine.has_unfinished_requests():
            start = time.perf_counter()
            outputs = engine.step()
            timings.append((time.perf_counter() - start)*1000)
            for item in outputs:
                if item.finished:
                    final = item
        assert final is not None and len(final.outputs[0].token_ids) == output_tokens
        output = final.outputs[0]
        record = {'arm': arm, 'tag': tag, 'seed': seed, 'step_ms': timings,
                  'started_utc':started_utc,'finished_utc':time.time(),
                  'decode_median_ms': statistics.median(timings[9:]),
                  'prefill_ms': timings[0], 'token_ids': output.token_ids}
        if args.audit:
            import numpy as np
            routes = output.routed_experts
            assert routes is not None and routes.shape == (2048+output_tokens-1, 40, 2)
            assert routes.min() >= 0 and routes.max() < 8
            assert np.all(routes[..., 0] != routes[..., 1])
            record['routes'] = routes.tolist()
            record['logprobs'] = [{str(key): value.logprob for key, value in row.items()}
                                  for row in output.logprobs]
            record['candidate_audit'] = llm.collective_rpc(lane.audit, args=(arm,))[0]
        records.append(record)
        (out / 'requests.json').write_text(json.dumps(records, indent=2))
        print('REQUEST', json.dumps({key: value for key, value in record.items()
                                     if key not in ['step_ms', 'token_ids', 'routes', 'logprobs']}), flush=True)
        return record

    for arm in arms:
        request(arm, 'warmup', 0)
        request(arm, 'warmup', 1)
    comparisons = []
    for pair in range(args.pairs):
        order = arms if pair % 2 == 0 else list(reversed(arms))
        rows = {arm: request(arm, f'pair-{pair}', pair % 3) for arm in order}
        reference = rows['baseline']
        for arm in arms[1:]:
            actual = rows[arm]
            assert reference['token_ids'] == actual['token_ids'], (pair, arm, 'tokens')
            row = {'pair': pair, 'arm': arm,
                   'baseline_ms': reference['decode_median_ms'],
                   'candidate_ms': actual['decode_median_ms'],
                   'speedup': reference['decode_median_ms']/actual['decode_median_ms'],
                   'tokens_equal': True}
            if args.audit:
                assert reference['routes'] == actual['routes'], (pair, arm, 'routes')
                assert all(set(r) == set(a) for r, a in zip(reference['logprobs'], actual['logprobs']))
                delta = max(abs(value - actual['logprobs'][i][key])
                            for i, values in enumerate(reference['logprobs'])
                            for key, value in values.items())
                assert delta < 1e-3, (pair, arm, delta)
                row.update({'actual_routes_equal': True, 'max_logprob_delta': delta})
            comparisons.append(row)
        (out / 'comparisons.json').write_text(json.dumps(comparisons, indent=2))
    baseline_ms = statistics.median(row['baseline_ms'] for row in comparisons)
    result = {'candidate': args.candidate, 'same_model_process': True, 'profiler': 'OFF',
              'audit': args.audit, 'baseline': {'ms_per_step': baseline_ms, 'A': 1,
                                               'tokens_per_second': 1000/baseline_ms},
              'pairs': comparisons, 'arms': {}}
    for arm in arms[1:]:
        rows = [row for row in comparisons if row['arm'] == arm]
        ms = statistics.median(row['candidate_ms'] for row in rows)
        result['arms'][arm] = {'ms_per_step': ms, 'A': 1, 'tokens_per_second': 1000/ms,
                              'paired_speedup_median': statistics.median(row['speedup'] for row in rows),
                              'all_pairs_faster': all(row['speedup'] > 1 for row in rows)}
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
