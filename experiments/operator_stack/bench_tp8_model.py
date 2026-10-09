"""Matched native/core/stack banks in the same eight worker processes."""
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
os.environ['V41_DUMMY_WO_A_FIX'] = '1'
os.environ['STACK_TP8_ARM'] = 'tp8base'


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--output', required=True)
    p.add_argument('--pairs', type=int, default=10)
    p.add_argument('--audit', action='store_true')
    args = p.parse_args()
    if args.audit: os.environ['TINY_PERF_RANDOM_VALIDATION'] = '1'
    os.environ['OPT_BLOCKMAP_VERIFY'] = '1' if args.audit else '0'
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    os.environ['VLLM_CACHE_ROOT'] = str(out / 'cache')
    import tp8_patches as patches
    from vllm import LLM, SamplingParams
    llm = LLM(model='/model', tokenizer='/model', load_format='dummy', dtype='bfloat16',
              worker_cls='tp8_worker.TP8StackWorker', tensor_parallel_size=8,
              distributed_executor_backend='mp', enable_expert_parallel=True, seed=0,
              trust_remote_code=True, async_scheduling=False, limit_mm_per_prompt={'image': 0},
              max_model_len=8192, max_num_seqs=1, max_num_batched_tokens=2048,
              gpu_memory_utilization=.70, kv_cache_memory_bytes=4*1024**3,
              block_size=128, enable_prefix_caching=False, enable_return_routed_experts=args.audit,
              compilation_config={'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': [1]},
              additional_config={'enable_engram': False, 'enable_cpu_binding': True,
                  'ascend_compilation_config': {'enable_npugraph_ex': True, 'enable_static_kernel': True},
                  'multistream_dsv4_dsa_overlap': False})
    arms = ['tp8base', 'tp8core', 'tp8act', 'tp8stack']; banks = {}
    banks[arms[0]] = llm.collective_rpc(patches.save, args=(arms[0],))
    for arm in arms[1:]: banks[arm] = llm.collective_rpc(patches.create, args=(arm,))
    for rows in banks.values():
        assert len(rows) == 8 and {r['rank'] for r in rows} == set(range(8)), rows
    (out/'banks.json').write_text(json.dumps(banks, indent=2)+'\n')
    engine = llm.llm_engine; records = []; counter = 0

    def request(arm, tag, seed):
        nonlocal counter
        llm.collective_rpc(patches.switch, args=(arm,)); counter += 1
        count = 47 + seed % 2 if args.audit else 48
        prompt = [100+(i*17+seed*19)%97 for i in range(2048)]
        engine.add_request(f'stack8-{counter}', {'prompt_token_ids': prompt},
            SamplingParams(temperature=0, max_tokens=count, ignore_eos=True,
                           detokenize=False, logprobs=5 if args.audit else None))
        times = []; final = None
        while engine.has_unfinished_requests():
            start = time.perf_counter(); outputs = engine.step()
            times.append((time.perf_counter()-start)*1000)
            for value in outputs:
                if value.finished: final = value
        assert final is not None and len(final.outputs[0].token_ids) == count
        output = final.outputs[0]
        record = {'arm': arm, 'tag': tag, 'seed': seed, 'step_ms': times,
                  'decode_median_ms': statistics.median(times[9:]), 'token_ids': output.token_ids}
        if args.audit:
            import numpy as np
            routes = output.routed_experts
            assert routes is not None and routes.shape == (2048+count-1, 40, 2), getattr(routes, 'shape', None)
            valid=(routes[...,0]>=0)&(routes[...,1]>=0)&(routes[...,0]<8)&(routes[...,1]<8)&(routes[...,0]!=routes[...,1])
            receipt={'arm':arm,'tag':tag,'shape':list(routes.shape),'min':int(routes.min()),'max':int(routes.max()),
                     'valid_prefill_pairs':int(valid[:2048].sum()),'valid_decode_pairs':int(valid[2048:].sum()),
                     'unique_ids':np.unique(routes).tolist()[:30]}
            (out/f'route_diagnostic_{counter}.json').write_text(json.dumps(receipt,indent=2)+'\n')
            print('TP8_ROUTE_DIAGNOSTIC',json.dumps(receipt),flush=True)
            import route_capture_patch
            route_status=llm.collective_rpc(route_capture_patch.status)
            (out/f'route_capture_status_{counter}.json').write_text(json.dumps(route_status,indent=2)+'\n')
            assert routes.min() >= 0 and routes.max() < 8 and np.all(routes[..., 0] != routes[..., 1])
            record['routes'] = routes.tolist()
            record['logprobs'] = [{str(k):v.logprob for k,v in row.items()} for row in output.logprobs]
            record['rank_audit'] = llm.collective_rpc(patches.audit, args=(arm,))
            import route_capture_patch
            record['route_capture_status'] = llm.collective_rpc(route_capture_patch.status)
        records.append(record)
        (out/'requests.json').write_text(json.dumps(records, indent=2)+'\n')
        print('TP8_REQUEST', json.dumps({k:v for k,v in record.items() if k not in ['step_ms','routes','token_ids','logprobs','rank_audit']}), flush=True)
        return record

    for arm in arms:
        request(arm, 'warmup', 0); request(arm, 'warmup', 1)
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
                assert ref['routes'] == actual['routes'], (pair, arm, 'routes')
                assert all(set(r)==set(a) for r,a in zip(ref['logprobs'], actual['logprobs']))
                delta=max(abs(v-actual['logprobs'][i][k]) for i,row in enumerate(ref['logprobs']) for k,v in row.items())
                assert delta < 1e-3, (pair, arm, delta)
                comparison.update(actual_routes_equal=True, max_logprob_delta=delta)
            comparisons.append(comparison)
        (out/'comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    result = {'tensor_parallel_size':8,'physical_chips':list(range(8,16)),
              'same_model_instance':True,'same_processes_per_rank':True,'audit':args.audit,
              'profiler':'OFF','A':1,'arms':{},'pairs':comparisons}
    for arm in arms:
        ms=statistics.median(r['decode_median_ms'] for r in records if r['arm']==arm and r['tag'].startswith('pair-'))
        result['arms'][arm]={'ms_per_step':ms,'A':1,'tokens_per_second':1000/ms}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('TP8_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__': main()
