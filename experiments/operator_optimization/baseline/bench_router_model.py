"""Interleaved native/vector graph replay in the same initialized tiny model."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
os.environ.setdefault('V41_DUMMY_WO_A_FIX','1')
import runtime_patches as patches


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='/work/results/router_model')
    p.add_argument('--pairs',type=int,default=6)
    p.add_argument('--audit-routes',action='store_true')
    p.add_argument('--arms',default='native,router')
    p.add_argument('--randomize-validation',action='store_true')
    p.add_argument('--profile',action='store_true')
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    os.environ['VLLM_CACHE_ROOT']=str(out/'vllm_cache')
    os.environ['VLLM_DISABLE_COMPILE_CACHE']='1'
    patches.install_router()
    arms=args.arms.split(',')
    assert arms[0]=='native' and all(a in ['native','router','hc','both'] for a in arms)
    if any(a in ['hc','both'] for a in arms):os.environ['TINY_PERF_HC_ENABLE']='1'
    if args.randomize_validation:os.environ['TINY_PERF_RANDOM_VALIDATION']='1'
    from vllm import LLM,SamplingParams
    llm=LLM(model='/model',tokenizer='/model',load_format='dummy',dtype='bfloat16',
            worker_cls='tiny_perf_worker.TinyPerfWorker',
            tensor_parallel_size=1,enable_expert_parallel=True,seed=0,trust_remote_code=True,
            async_scheduling=False,limit_mm_per_prompt={'image':0},max_model_len=8192,
            max_num_seqs=1,max_num_batched_tokens=2048,gpu_memory_utilization=.70,
            kv_cache_memory_bytes=4*1024**3,block_size=128,enable_prefix_caching=False,
            enable_return_routed_experts=args.audit_routes,
            profiler_config={'profiler':'torch','torch_profiler_dir':str(out/'prof'),'torch_profiler_with_stack':False}
                            if args.profile else None,
            compilation_config={'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1]},
            additional_config={'enable_engram':False,'enable_cpu_binding':True,
                'ascend_compilation_config':{'enable_npugraph_ex':True,'enable_static_kernel':False},
                'multistream_dsv4_dsa_overlap':False})
    print('BANK',llm.collective_rpc(patches.save_graph_bank,args=('native',)),flush=True)
    for arm in arms[1:]:
        print('BANK',llm.collective_rpc(patches.create_bank,args=(arm,)),flush=True)
    engine=llm.llm_engine
    records=[]
    counter=0
    def request(arm,tag,seed=0,profile=False):
        nonlocal counter
        llm.collective_rpc(patches.switch_bank,args=(arm,))
        counter+=1
        prompt=[100+(i*17+seed*19)%97 for i in range(2048)]
        engine.add_request(f'router-{counter}',{'prompt_token_ids':prompt},
            SamplingParams(temperature=0,max_tokens=48,ignore_eos=True,detokenize=False,
                           logprobs=5 if args.audit_routes else None))
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
                'token_ids':output.token_ids,
                'logprobs':[{str(k):v.logprob for k,v in x.items()} for x in output.logprobs] if output.logprobs else None,
                'router_audit':llm.collective_rpc(patches.audit_router,args=(arm,))[0]}
        if args.audit_routes:
            import numpy as np
            assert output.routed_experts is not None,'Actual routed experts were not captured'
            routes=output.routed_experts
            assert routes.shape==(2048+47,40,2),routes.shape
            assert routes.min()>=0 and routes.max()<8
            assert np.all(routes[...,0]!=routes[...,1]),'Degenerate or unpopulated route audit'
            record['routed_experts']=output.routed_experts.tolist()
            record['route_coverage']={'shape':list(routes.shape),'unique_experts':np.unique(routes).tolist(),
                                      'actual_top2_pairs':int(routes.shape[0]*routes.shape[1])}
            if any(a in ['hc','both'] for a in arms):
                record['hc_audit']=llm.collective_rpc(patches.audit_hc,args=(arm,))[0]
        records.append(record)
        (out/'requests.json').write_text(json.dumps(records,indent=2))
        print('REQUEST',json.dumps({k:v for k,v in record.items() if k not in ['step_ms','token_ids','logprobs','routed_experts']}),flush=True)
        return record
    for arm in arms:
        request(arm,'warmup',0)
        request(arm,'warmup',1)
    comparisons=[]
    for pair in range(args.pairs):
        order=arms if pair%2==0 else list(reversed(arms))
        results={arm:request(arm,f'pair-{pair}',pair%3) for arm in order}
        for arm in arms[1:]:
            a,b=results['native'],results[arm]
            assert a['token_ids']==b['token_ids'],f'Token divergence pair {pair}/{arm}'
            delta=None
            if args.audit_routes:
                delta=max(abs(v-b['logprobs'][i][key]) for i,row in enumerate(a['logprobs']) for key,v in row.items())
                assert delta<1e-3,(pair,arm,delta)
            cmp={'pair':pair,'arm':arm,'native_ms':a['decode_median_ms'],'candidate_ms':b['decode_median_ms'],
                 'speedup':a['decode_median_ms']/b['decode_median_ms'],'max_logprob_delta':delta,'tokens_equal':True}
            if args.audit_routes:
                assert a['routed_experts']==b['routed_experts'],f'Actual route divergence pair {pair}/{arm}'
                cmp['actual_routes_equal']=True
            comparisons.append(cmp)
        (out/'comparisons.json').write_text(json.dumps(comparisons,indent=2))
    result={'same_model_process':True,'profiler':'OFF','pairs':comparisons,'arms':{},
            'native_ms':statistics.median(x['native_ms'] for x in comparisons),
            'audit_actual_routes':args.audit_routes,'logprobs_enabled':args.audit_routes,
            'randomized_validation_weights':args.randomize_validation}
    for arm in arms[1:]:
        data=[c for c in comparisons if c['arm']==arm]
        ms=statistics.median(c['candidate_ms'] for c in data)
        result['arms'][arm]={'ms':ms,'speedup':statistics.median(c['speedup'] for c in data),'tokens_per_second':1000/ms}
    result['native_tokens_per_second']=1000/result['native_ms']
    (out/'result.json').write_text(json.dumps(result,indent=2))
    if args.profile:
        for arm in ['native',arms[-1]]:request(arm,'profile',0,True)
    print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
