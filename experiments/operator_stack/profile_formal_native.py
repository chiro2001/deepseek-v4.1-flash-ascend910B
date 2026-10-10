"""Diagnostic production TP8 metrics, without route audit or operator patches."""
import argparse
import json
import os
from pathlib import Path

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
os.environ['STACK_REAL_WEIGHTS']='1'
os.environ['STACK_REAL_AUDIT']='0'


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--model',required=True)
    p.add_argument('--physical-chips',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--metrics',default='PipeUtilization,ArithmeticUtilization,Memory,MemoryL0,MemoryUB,L2Cache,ResourceConflictRatio')
    args=p.parse_args()
    assert os.environ['STACK_PHYSICAL_CHIPS']==args.physical_chips
    assert len(set(args.physical_chips.split(',')))==8
    assert os.getenv('TINY_PERF_RANDOM_VALIDATION')!='1'
    from formal_model_contract import inspect_checkpoint
    from formal_tp8_profile import METRICS,configure,advance
    metrics=args.metrics.split(',')
    assert len(set(metrics))==len(metrics) and set(metrics)<=set(METRICS)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    contract=inspect_checkpoint(args.model)
    (out/'checkpoint.json').write_text(json.dumps(contract,indent=2)+'\n')
    from vllm import LLM,SamplingParams
    llm=LLM(model=args.model,tokenizer=args.model,load_format='auto',dtype='bfloat16',
        safetensors_load_strategy='lazy',model_loader_extra_config={'enable_multithread_load':True,'num_threads':128},
        tensor_parallel_size=8,distributed_executor_backend='mp',enable_expert_parallel=True,
        seed=0,trust_remote_code=True,async_scheduling=False,limit_mm_per_prompt={'image':4},
        max_model_len=8192,max_num_seqs=1,max_num_batched_tokens=2048,
        gpu_memory_utilization=.70,kv_cache_memory_bytes=4*1024**3,block_size=128,
        enable_prefix_caching=False,enable_return_routed_experts=False,
        profiler_config={'profiler':'torch','torch_profiler_dir':str(out/'prof'),'torch_profiler_with_stack':False},
        compilation_config={'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1]},
        additional_config={'enable_engram':True,'engram_storage':'int8','enable_cpu_binding':False,
            'ascend_compilation_config':{'enable_npugraph_ex':True,'enable_static_kernel':True},
            'multistream_dsv4_dsa_overlap':False})
    tokenizer=llm.get_tokenizer()
    prompt=tokenizer.encode(('请分析延迟受限计算的优化原理，给出判断依据。'
                            '这是供分析的上下文，保持事实准确。')*512,add_special_tokens=False)[:2048]
    assert len(prompt)==2048
    (out/'prompt.json').write_text(json.dumps(prompt)+'\n')
    engine=llm.llm_engine;records=[]
    def request(tag,metric=None):
        engine.add_request('native-prof-'+str(len(records)),{'prompt_token_ids':prompt},
                          SamplingParams(temperature=0,max_tokens=48,ignore_eos=True,detokenize=False))
        steps=0;profiling=False;final=None
        try:
            while engine.has_unfinished_requests():
                if metric and steps==9:
                    receipts=llm.collective_rpc(configure,args=(metric,str(out/'prof'/metric),5,10))
                    (out/f'receipt_{metric}.json').write_text(json.dumps(receipts,indent=2)+'\n')
                    llm.start_profile();profiling=True
                outputs=engine.step();steps+=1
                if profiling:
                    llm.collective_rpc(advance)
                    if steps==24:llm.stop_profile();profiling=False
                for item in outputs:
                    if item.finished:final=item
        finally:
            if profiling:llm.stop_profile()
        assert final is not None and len(final.outputs[0].token_ids)==48
        row={'tag':tag,'metric':metric,'engine_steps':steps,'token_ids':final.outputs[0].token_ids}
        records.append(row)
        (out/'requests.json').write_text(json.dumps(records,indent=2)+'\n')
        print('FORMAL_NATIVE_PROFILE_REQUEST',json.dumps({k:v for k,v in row.items() if k!='token_ids'}),flush=True)
    request('warmup-0');request('warmup-1')
    for metric in metrics:request(metric,metric)
    result={'formal_weights':True,'audit':False,'operator_patches_installed':False,
        'metrics':metrics,'ranks':8,'warmup_steps':5,'active_steps':10,
        'precision_validated':False,'performance_claim':None,
        'scope':'Diagnostic native CPU/NPU/hardware counters; formal A/A still failing'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FORMAL_NATIVE_PROFILE_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
