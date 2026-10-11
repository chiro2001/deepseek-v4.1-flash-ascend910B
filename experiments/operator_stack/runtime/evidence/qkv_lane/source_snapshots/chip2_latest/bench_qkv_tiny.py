"""Full TP1 synthetic tiny A/A first, then numerical-gated native/merged banks."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_DISABLE_COMPILE_CACHE'] = '1'
os.environ['V41_DUMMY_WO_A_FIX'] = '1'


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=6)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--accuracy-evidence', type=Path)
    args = parser.parse_args()
    if not args.audit:
        assert args.accuracy_evidence
        gate = json.loads(args.accuracy_evidence.read_text())
        assert gate['passed'] and gate['max_bf16_ulp'] <= 1
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    os.environ['QKV_LANE_AUDIT'] = '1' if args.audit else '0'
    os.environ['VLLM_CACHE_ROOT'] = str(out/'cache')
    import qkv_patch as patches
    from vllm import LLM, SamplingParams
    import numpy as np
    llm = LLM(model='/model', tokenizer='/model', load_format='dummy', dtype='bfloat16',
              worker_cls='qkv_worker.QKVWorker', tensor_parallel_size=1,
              enable_expert_parallel=True, seed=0, trust_remote_code=True,
              async_scheduling=False, limit_mm_per_prompt={'image': 0}, max_model_len=8192,
              max_num_seqs=1, max_num_batched_tokens=2048, gpu_memory_utilization=.70,
              kv_cache_memory_bytes=4*1024**3, block_size=128, enable_prefix_caching=False,
              enable_return_routed_experts=args.audit,
              compilation_config={'cudagraph_mode':'FULL_DECODE_ONLY', 'cudagraph_capture_sizes':[1]},
              additional_config={'enable_engram':False, 'enable_cpu_binding':False,
                  'ascend_compilation_config':{'enable_npugraph_ex':True, 'enable_static_kernel':True},
                  'multistream_dsv4_dsa_overlap':True})
    banks = {'native_a': llm.collective_rpc(patches.save, args=('native_a',))[0]}
    banks['native_b'] = llm.collective_rpc(patches.create, args=('native_b',))[0]
    (out/'banks.json').write_text(json.dumps(banks, indent=2)+'\n')
    engine = llm.llm_engine
    records, comparisons = [], []
    counter = 0

    def request(arm, tag, seed):
        nonlocal counter
        llm.collective_rpc(patches.switch, args=(arm,))
        counter += 1
        prompt = [100+(i*17+seed*19)%97 for i in range(2048)]
        count = 32
        engine.add_request(f'qkv-lane-{counter}', {'prompt_token_ids':prompt},
                           SamplingParams(temperature=0, max_tokens=count, ignore_eos=True,
                                          detokenize=False, logprobs=5 if args.audit else None))
        times, final = [], None
        while engine.has_unfinished_requests():
            begin = time.perf_counter()
            outputs = engine.step()
            times.append((time.perf_counter()-begin)*1000)
            for item in outputs:
                if item.finished:
                    final = item
        assert final is not None and len(final.outputs[0].token_ids) == count
        value = final.outputs[0]
        row = {'arm':arm, 'tag':tag, 'seed':seed, 'step_ms':times,
               'decode_median_ms':statistics.median(times[9:]), 'token_ids':value.token_ids}
        if args.audit:
            row['logprobs'] = [{str(k):v.logprob for k,v in part.items()} for part in value.logprobs]
            routes = value.routed_experts
            assert routes is not None and routes.shape == (2048+count-1, 40, 2), routes.shape
            assert routes.min() >= 0 and routes.max() < 8
            assert np.all(routes[..., 0] != routes[..., 1])
            # The digest avoids transferring this large native-route trace over SSH.
            import hashlib
            row['route_sha256'] = hashlib.sha256(routes.tobytes()).hexdigest()
            row['route_shape'] = list(routes.shape)
        records.append(row)
        (out/'requests.json').write_text(json.dumps(records, indent=2)+'\n')
        print('REQUEST', json.dumps({k:v for k,v in row.items() if k not in ('step_ms','token_ids','logprobs')}), flush=True)
        return row

    def compare(reference, actual, tag):
        assert reference['token_ids'] == actual['token_ids'], ('Token divergence',tag)
        row = {'tag':tag, 'baseline_ms':reference['decode_median_ms'],
               'candidate_ms':actual['decode_median_ms'], 'tokens_equal':True,
               'speedup':reference['decode_median_ms']/actual['decode_median_ms']}
        if args.audit:
            assert reference['route_sha256'] == actual['route_sha256'], ('Route divergence',tag)
            assert all(set(a) == set(b) for a,b in zip(reference['logprobs'],actual['logprobs']))
            delta = max(abs(v-actual['logprobs'][i][k]) for i,part in enumerate(reference['logprobs']) for k,v in part.items())
            assert delta < 1e-3, ('Logprob divergence',tag,delta)
            row.update(routes_equal=True, max_logprob_delta=delta)
        comparisons.append(row)
        (out/'comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
        return row

    for arm in ('native_a','native_b'):
        request(arm,'aa-warmup',0)
    for pair in range(2):
        order = ('native_a','native_b') if pair%2 == 0 else ('native_b','native_a')
        values = {arm:request(arm,f'aa-{pair}',pair) for arm in order}
        compare(values['native_a'], values['native_b'], f'aa-{pair}')
    banks['merged'] = llm.collective_rpc(patches.create, args=('merged',))[0]
    (out/'banks.json').write_text(json.dumps(banks, indent=2)+'\n')
    request('merged','candidate-warmup',0)
    if args.audit:
        gate = llm.collective_rpc(patches.audit)[0]
        (out/'qkv_accuracy.json').write_text(json.dumps(gate, indent=2)+'\n')
        print('QKV_ACCURACY', json.dumps({k:v for k,v in gate.items() if k != 'rows'}), flush=True)
        assert gate['passed'], ('BF16 QKV exceeds 1 ULP',gate['max_bf16_ulp'])
    for pair in range(args.pairs):
        order = ('native_a','merged') if pair%2 == 0 else ('merged','native_a')
        values = {arm:request(arm,f'ab-{pair}',pair%3) for arm in order}
        compare(values['native_a'],values['merged'],f'ab-{pair}')
        if args.audit:
            gate = llm.collective_rpc(patches.audit)[0]
            assert gate['passed'], gate['max_bf16_ulp']
    ab = [r for r in comparisons if r['tag'].startswith('ab-')]
    base_ms = statistics.median(r['baseline_ms'] for r in ab)
    candidate_ms = statistics.median(r['candidate_ms'] for r in ab)
    result = {'scope':'Full synthetic BF16 TP1 tiny; cannot extrapolate formal TP8',
              'model':{'layers':40,'hidden':5120,'experts':8,'topk':2,'tp':1,'dtype':'BF16',
                       'quantization':None,'load_format':'dummy','q_lora_rank':512,
                       'engram':False,'vision':False,'hardware':'a3-21 Ascend910C physical2'},
              'same_model_process':True,'strict':os.environ['HCCL_DETERMINISTIC'],
              'audit':args.audit,'profiler_during_timing':'OFF','A':1.0,
              'native':{'ms_per_step':base_ms,'A':1.0,'tokens_per_second':1000/base_ms},
              'merged':{'ms_per_step':candidate_ms,'A':1.0,'tokens_per_second':1000/candidate_ms},
              'paired_speedup_median':statistics.median(r['speedup'] for r in ab),
              'pairs_faster':sum(r['speedup']>1 for r in ab),'pairs':len(ab),
              'a_a_comparisons':[r for r in comparisons if r['tag'].startswith('aa-')],
              'passed':True,'banks':banks}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('COMPLETE',json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
