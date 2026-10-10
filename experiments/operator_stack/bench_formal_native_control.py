"""Strict same-worker A/A control of the production formal graph or eager path."""
import argparse
import json
import os
from pathlib import Path

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
os.environ['VLLM_DISABLE_COMPILE_CACHE'] = '1'


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--model', required=True)
    p.add_argument('--physical-chips', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=3)
    p.add_argument('--audit', action='store_true', required=True)
    p.add_argument('--eager', action='store_true')
    p.add_argument('--probe-native-ops', action='store_true')
    args = p.parse_args()
    assert os.getenv('TINY_PERF_RANDOM_VALIDATION') != '1'
    assert os.environ['STACK_PHYSICAL_CHIPS'] == args.physical_chips
    assert len(set(args.physical_chips.split(','))) == 8 and args.pairs >= 2
    assert not args.probe_native_ops or args.eager
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['VLLM_CACHE_ROOT'] = str(args.output/'cache')
    from formal_model_contract import inspect_checkpoint
    contract = inspect_checkpoint(args.model)
    (args.output/'checkpoint.json').write_text(json.dumps(contract, indent=2)+'\n')
    from vllm import LLM, SamplingParams
    import numpy as np
    llm = LLM(model=args.model, tokenizer=args.model, load_format='auto', dtype='bfloat16',
        safetensors_load_strategy='lazy',
        model_loader_extra_config={'enable_multithread_load':True,'num_threads':128},
        worker_cls='formal_control_worker.FormalControlWorker', tensor_parallel_size=8,
        distributed_executor_backend='mp', enable_expert_parallel=True, seed=0,
        trust_remote_code=True, async_scheduling=False, limit_mm_per_prompt={'image':4},
        max_model_len=8192, max_num_seqs=1, max_num_batched_tokens=2048,
        gpu_memory_utilization=.70, kv_cache_memory_bytes=4*1024**3,
        block_size=128, enable_prefix_caching=False, enable_return_routed_experts=True,
        enforce_eager=args.eager,
        compilation_config={'cudagraph_mode':'NONE' if args.eager else 'FULL_DECODE_ONLY',
                            'cudagraph_capture_sizes':[] if args.eager else [1]},
        additional_config={'enable_engram':True,'engram_storage':'int8','enable_cpu_binding':False,
            'ascend_compilation_config':{'enable_npugraph_ex':True,'enable_static_kernel':not args.eager},
            'multistream_dsv4_dsa_overlap':False})
    tokenizer = llm.get_tokenizer()
    topics = ['请分析延迟受限计算的优化原理，给出判断依据。',
              '请解释缓存命中、数据搬运和流水线重叠的关系。']
    prompts = [tokenizer.encode((s+'这是供分析的上下文，保持事实准确。')*512,
                               add_special_tokens=False)[:2048] for s in topics]
    assert all(len(s)==2048 for s in prompts)
    records = []
    def full_request(seed, tag):
        result = llm.generate([{'prompt_token_ids':prompts[seed]}],
            SamplingParams(temperature=0,max_tokens=47,ignore_eos=True,detokenize=False,logprobs=5),
            use_tqdm=False)[0]
        output = result.outputs[0]; routes = output.routed_experts
        assert len(output.token_ids)==47 and routes is not None and routes.shape==(2094,40,6), getattr(routes,'shape',None)
        assert np.all((routes>=0)&(routes<384)) and np.all(np.diff(np.sort(routes,axis=-1),axis=-1)>0)
        row={'tag':tag,'seed':seed,'token_ids':output.token_ids,'routes':routes.tolist(),
             'logprobs':[{str(k):v.logprob for k,v in step.items()} for step in output.logprobs]}
        records.append(row)
        (args.output/'requests.json').write_text(json.dumps(records)+'\n')
        print('FORMAL_NATIVE_CONTROL_REQUEST',tag,flush=True)
        return row
    full_request(0,'warmup-0'); full_request(1,'warmup-1')
    reference = full_request(0,'reference')
    comparisons = []
    from formal_comparison import compare_requests
    for i in range(args.pairs):
        actual = full_request(0,'repeat-'+str(i))
        comparison=compare_requests(reference,actual,'repeat-'+str(i))
        comparison['repeat']=i
        comparisons.append(comparison)
        (args.output/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
        print('FORMAL_NATIVE_CONTROL_COMPARISON',json.dumps(comparison),flush=True)
    result={'formal_weights':True,'physical_chips':args.physical_chips,'eager':args.eager,
        'operator_bank_patches_installed':False,'one_native_graph_only':not args.eager,
        'checkpoint_config_sha256':contract['config_sha256'],'comparisons':comparisons,
        'passed':all(r['passed'] for r in comparisons),'performance_claim':None}
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    if args.probe_native_ops:
        from native_repeatability_probe import (install,repeat,disable_engram_subgraph,
                                               begin_capture,finish_capture,compare_captures)
        disabled=llm.collective_rpc(disable_engram_subgraph)
        (args.output/'engram_subgraph_disabled.json').write_text(json.dumps(disabled,indent=2)+'\n')
        graph_off_ref=full_request(0,'engram-subgraph-off-reference')
        graph_off=[]
        for i in range(args.pairs):
            actual=full_request(0,'engram-subgraph-off-repeat-'+str(i))
            diagnostic=compare_requests(graph_off_ref,actual,'engram-subgraph-off-'+str(i))
            graph_off.append(diagnostic)
            (args.output/'engram_subgraph_off_comparisons.json').write_text(json.dumps(graph_off,indent=2)+'\n')
            print('FORMAL_ENGRAM_SUBGRAPH_OFF_COMPARISON',json.dumps(diagnostic),flush=True)
        installed=llm.collective_rpc(install)
        (args.output/'native_probe_install.json').write_text(json.dumps(installed,indent=2)+'\n')
        full_request(0,'native-operator-probe')
        probe=llm.collective_rpc(repeat,args=(5,))
        (args.output/'native_operator_repeatability.json').write_text(json.dumps(probe,indent=2)+'\n')
        print('FORMAL_NATIVE_OPERATOR_PROBE_COMPLETE',json.dumps({'ranks':len(probe),
              'records':sum(len(r['records']) for r in probe)}),flush=True)
        assert all(r['expected_coverage_reached'] for r in probe), 'Native operator probe coverage incomplete'
        for tag in ('trace-reference','trace-repeat'):
            llm.collective_rpc(begin_capture,args=(tag,))
            full_request(0,tag)
            llm.collective_rpc(finish_capture,args=(tag,))
        trace=llm.collective_rpc(compare_captures,args=('trace-reference','trace-repeat'))
        (args.output/'cross_request_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
        print('FORMAL_CROSS_REQUEST_TRACE_COMPLETE',json.dumps([
            {'rank':r['rank'],'first_difference':r['first_difference']} for r in trace]),flush=True)
    assert result['passed'], 'Formal production A/A failed original route/logprob gate'


if __name__ == '__main__': main()
