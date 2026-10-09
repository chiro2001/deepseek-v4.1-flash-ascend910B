"""Matched next-round model A/B; activation and Q/KV overlap enabled in every arm."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
os.environ['TINY_PERF_ARM']='both'
os.environ['TINY_PERF_HC_ENABLE']='1'
os.environ['VLLM_DISABLE_COMPILE_CACHE']='1'
os.environ.setdefault('V41_DUMMY_WO_A_FIX','1')
os.environ['OPT_ACT_ARM']='overlap'
os.environ['OPT_TEST_OVERLAP']='1'
os.environ['GOAL20_ARM']='baseline'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--pairs',type=int,default=6)
    parser.add_argument('--arms')
    parser.add_argument('--test-overlap',action='store_true')
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--static-kernel',action='store_true')
    args=parser.parse_args()
    arms=(args.arms or 'baseline,hcstatic,hcpost,route,gmm1,combo').split(',')
    assert arms[0]=='baseline' and set(arms)<=set(['baseline','hcstatic','hcpost','route','gmm1','gmm2','combo','all_candidates'])
    if args.test_overlap:os.environ['OPT_TEST_OVERLAP']='1'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    os.environ['VLLM_CACHE_ROOT']=str(out/'vllm_cache')
    os.environ['OPT_ACT_ARM']='overlap'
    if args.audit:os.environ['TINY_PERF_RANDOM_VALIDATION']='1'
    import goal20_patches as patches
    from vllm import LLM,SamplingParams
    import numpy as np

    llm=LLM(model='/model',tokenizer='/model',load_format='dummy',dtype='bfloat16',
            worker_cls='goal20_worker.Goal20Worker',tensor_parallel_size=1,
            enable_expert_parallel=True,seed=0,trust_remote_code=True,
            async_scheduling=False,limit_mm_per_prompt={'image':0},max_model_len=8192,
            max_num_seqs=1,max_num_batched_tokens=2048,gpu_memory_utilization=.70,
            kv_cache_memory_bytes=4*1024**3,block_size=128,enable_prefix_caching=False,
            enable_return_routed_experts=args.audit,
            profiler_config={'profiler':'torch','torch_profiler_dir':str(out/'prof'),'torch_profiler_with_stack':False}
                            if args.profile else None,
            compilation_config={'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1]},
            additional_config={'enable_engram':False,'enable_cpu_binding':True,
                'ascend_compilation_config':{'enable_npugraph_ex':True,'enable_static_kernel':args.static_kernel},
                'multistream_dsv4_dsa_overlap':True})
    banks=[]
    banks.append(llm.collective_rpc(patches.save_bank,args=(arms[0],))[0])
    print('BANK',json.dumps(banks[-1]),flush=True)
    print('WARM_CANDIDATES',llm.collective_rpc(patches.warm),flush=True)
    for arm in arms[1:]:
        banks.append(llm.collective_rpc(patches.create_bank,args=(arm,))[0])
        print('BANK',json.dumps(banks[-1]),flush=True)
    (out/'banks.json').write_text(json.dumps(banks,indent=2)+'\n')
    engine=llm.llm_engine;records=[];counter=0

    def request(arm,tag,seed,profile=False):
        nonlocal counter
        llm.collective_rpc(patches.switch_bank,args=(arm,))
        counter+=1
        prompt=[100+(i*17+seed*19)%97 for i in range(2048)]
        engine.add_request(f'goal20-{counter}',{'prompt_token_ids':prompt},
            SamplingParams(temperature=0,max_tokens=48,ignore_eos=True,detokenize=False,
                           logprobs=5 if args.audit else None))
        times=[];final=None
        while engine.has_unfinished_requests():
            if profile and len(times)==9:
                from tiny_profile import configure_profiler
                llm.collective_rpc(configure_profiler,args=('PipeUtilization',str(out/'prof'/arm),20))
                llm.start_profile()
            start=time.perf_counter();outputs=engine.step();times.append((time.perf_counter()-start)*1000)
            if profile and 9<len(times)<=29:
                from tiny_profile import advance_profiler
                llm.collective_rpc(advance_profiler)
            if profile and len(times)==29:llm.stop_profile()
            for item in outputs:
                if item.finished:final=item
        assert final is not None and len(final.outputs[0].token_ids)==48
        output=final.outputs[0]
        record={'arm':arm,'tag':tag,'prompt_seed':seed,'step_ms':times,
                'decode_median_ms':statistics.median(times[9:]),'prefill_ms':times[0],
                'token_ids':output.token_ids,'profiled':profile}
        if args.audit:
            record['logprobs']=[{str(key):value.logprob for key,value in row.items()} for row in output.logprobs]
            routes=output.routed_experts
            assert routes is not None and routes.shape==(2095,40,2)
            assert routes.min()>=0 and routes.max()<8 and np.all(routes[...,0]!=routes[...,1])
            record['routes']=routes.tolist()
            record['route_shape']=list(routes.shape)
            record['unique_experts']=np.unique(routes).tolist()
            record['activation_audit']=llm.collective_rpc(patches.audit,args=(arm,))[0]
        records.append(record)
        (out/'requests.json').write_text(json.dumps(records,indent=2)+'\n')
        print('REQUEST',json.dumps({key:value for key,value in record.items()
                                   if key not in ['step_ms','token_ids','routes','logprobs']}),flush=True)
        return record

    for arm in arms:
        request(arm,'warmup',0);request(arm,'warmup',1)
    comparisons=[]
    for pair in range(args.pairs):
        order=arms if pair%2==0 else list(reversed(arms))
        results={arm:request(arm,f'pair-{pair}',pair%3) for arm in order}
        reference=results[arms[0]]
        for arm in arms[1:]:
            actual=results[arm]
            assert reference['token_ids']==actual['token_ids'],('Token divergence',pair,arm)
            comparison={'pair':pair,'arm':arm,'baseline_ms':reference['decode_median_ms'],
                        'candidate_ms':actual['decode_median_ms'],
                        'speedup':reference['decode_median_ms']/actual['decode_median_ms'],
                        'tokens_equal':True}
            if args.audit:
                assert reference['routes']==actual['routes'],('Route divergence',pair,arm)
                delta=max(abs(value-actual['logprobs'][i][key]) for i,row in enumerate(reference['logprobs']) for key,value in row.items())
                assert delta<1e-3,(pair,arm,delta)
                comparison.update(actual_routes_equal=True,max_logprob_delta=delta)
            comparisons.append(comparison)
        (out/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    baseline_ms=statistics.median(item['baseline_ms'] for item in comparisons)
    result={'same_model_process':True,'profiler_during_timing':'OFF','audit':args.audit,
            'static_kernel_requested':args.static_kernel,
            'randomized_gate_hc_and_mlp':args.audit,'A':1.0,'baseline_ms':baseline_ms,
            'baseline_tokens_per_second':1000/baseline_ms,'pairs':comparisons,'arms':{}}
    for arm in arms[1:]:
        rows=[item for item in comparisons if item['arm']==arm]
        ms=statistics.median(item['candidate_ms'] for item in rows)
        result['arms'][arm]={'ms_per_step':ms,'A':1.0,'tokens_per_second':1000/ms,
                             'paired_speedup_median':statistics.median(item['speedup'] for item in rows),
                             'all_pairs_faster':all(item['speedup']>1 for item in rows)}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(result),flush=True)
    if args.profile:
        for arm in [arms[0],arms[-1]]:request(arm,'profile',0,True)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
