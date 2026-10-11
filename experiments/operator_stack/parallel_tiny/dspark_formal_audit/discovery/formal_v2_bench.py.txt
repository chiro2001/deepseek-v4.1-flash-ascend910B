"""Prepare/run a new native full40 TP8 DSpark A/A job.

Default mode is CPU-only preparation. NPU runs are performed by the authorized
main thread, never by the preparation agent. Consumer-device gates stay closed
until independently obtained actual main/draft/verifier/cache evidence exists.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

from parallel_tiny.dspark_formal_prep.contract import (
    PHYSICAL_CHIPS, TARGET_MS, inspect_dspark_checkpoint, model_kwargs,
    startup_env, validate_hot_flags)
import formal_dspark_counters as counters


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def freeze_sources():
    root=Path(__file__).parent
    files=['bench_formal_dspark.py','formal_dspark_counters.py','formal_model_contract.py',
           'formal_comparison.py','route_capture_patch.py',
           'parallel_tiny/dspark_formal_prep/contract.py','parallel_tiny/dspark_formal_prep/worker.py',
           'parallel_tiny/dspark_formal_prep/sdk_gate.py','parallel_tiny/dspark_formal_audit/stock_graph_observer.py']
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}


def validate_perf_gate(args, checkpoint, sources):
    if args.audit_root is None:
        raise ValueError('Perf needs the independent complete formal DSpark audit receipt')
    prior=json.loads((args.audit_root/'result.json').read_text())
    if not prior.get('native_aa_passed') or prior.get('full_consumer_audit_passed') is not True:
        raise ValueError('Full main/draft/verifier/cache device consumer audit is still unproved')
    if prior.get('source_hashes')!=sources or prior.get('checkpoint_config_sha256')!=checkpoint['config_sha256']:
        raise ValueError('Audit sources/checkpoint differ from this performance job')
    if prior.get('speculative_config')!=model_kwargs(args.model,args.speculative_tokens,
            audit=False,draft_graph=bool(args.draft_graph))['speculative_config']:
        raise ValueError('Audit DSpark configuration differs')


def run_native(args, result, checkpoint, sources):
    # The allocation is owned by the main thread. This script never creates a
    # container or selects a replacement chip if these declarations disagree.
    if os.environ.get('STACK_PHYSICAL_CHIPS')!=args.physical_chips or args.physical_chips!=PHYSICAL_CHIPS:
        raise ValueError('Run requires the main-thread physical8–15 allocation declaration')
    if args.mode=='perf':
        validate_perf_gate(args,checkpoint,sources)
    hot=Path('/tmp/v41_dspark_flags')
    hot_flags=validate_hot_flags(hot.read_text() if hot.exists() else '')
    audit=args.mode=='audit'
    os.environ.update(startup_env(args.output,audit=audit,draft_graph=bool(args.draft_graph)))
    result['hot_flags']=hot_flags
    from vllm import LLM, SamplingParams
    import numpy as np
    kwargs=model_kwargs(args.model,args.speculative_tokens,audit=audit,draft_graph=bool(args.draft_graph))
    result['phase']='loading-native-dspark'
    write_json(args.output/'progress.json',result)
    llm=LLM(**kwargs)
    tokenizer=llm.get_tokenizer()
    topics=['请分析延迟受限计算的优化原理，给出判断依据。',
            '请解释缓存命中、数据搬运和流水线重叠的关系。']
    prompts=[tokenizer.encode((s+'这是供分析的上下文，保持事实准确。')*512,
                             add_special_tokens=False)[:args.prompt_tokens] for s in topics]
    if any(len(p)!=args.prompt_tokens for p in prompts):
        raise ValueError('Exact prompt token budget was not reached')
    sampling=SamplingParams(temperature=0,max_tokens=args.max_tokens,ignore_eos=True,
                            detokenize=False,logprobs=5 if audit else None)
    records=[]
    def request(seed,tag,count):
        if count:
            llm.collective_rpc(counters.begin,args=(tag,args.max_tokens))
        output=llm.generate([{'prompt_token_ids':prompts[seed]}],sampling,use_tqdm=False)[0]
        # Save before collective RPC: counter retrieval latency is excluded.
        ended=time.monotonic_ns()
        generated=output.outputs[0]
        tokens=list(generated.token_ids)
        if len(tokens)!=args.max_tokens:
            raise ValueError('Output token budget mismatch')
        metrics=None
        round_receipts=None
        if count:
            round_receipts=llm.collective_rpc(counters.finish)
            metrics=counters.summarize(round_receipts,tokens,ended,counters.clock_scope(),
                                       draft_graph=bool(args.draft_graph))
            write_json(args.output/(tag+'_counters.json'),round_receipts)
        row={'tag':tag,'seed':seed,'token_ids':tokens,'metrics':metrics,
             'request_id':output.request_id,'client_end_ns':ended}
        comparison_input={'token_ids':tokens}
        if audit:
            routes=generated.routed_experts
            if routes is None:
                raise ValueError('Native complete target route export unavailable under DSpark')
            routes=np.asarray(routes)
            expected=(args.prompt_tokens+len(tokens)-1,40,6)
            if routes.shape!=expected:
                raise ValueError(f'Complete accepted target route shape {routes.shape} != {expected}')
            route_file=args.output/(tag+'_routes.npz')
            np.savez_compressed(route_file,routes=routes)
            logprobs=[{str(k):v.logprob for k,v in step.items()} for step in generated.logprobs]
            row.update(routes_path=route_file.name,route_shape=list(routes.shape),
                route_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),logprobs=logprobs)
            comparison_input.update(routes=routes.copy(),logprobs=logprobs)
        records.append(row)
        write_json(args.output/'requests.json',records)
        print('FORMAL_DSPARK_REQUEST',tag,json.dumps(metrics,ensure_ascii=False) if metrics else 'warmup',flush=True)
        return row,comparison_input,round_receipts
    # Counter tag remains inactive throughout load/profile/capture and warmup.
    request(0,'warmup-0',False)
    request(1,'warmup-1',False)
    banks=llm.collective_rpc(counters.bank)
    write_json(args.output/'banks.json',banks)
    if len(banks)!=8 or {r['rank'] for r in banks}!=set(range(8)):
        raise ValueError('Incomplete native TP8 bank')
    for bank in banks:
        if bank['main_layers']!=40 or bank['routed_experts']!=384 or bank['experts_per_token']!=6:
            raise ValueError('Native full40/384/top6 contract changed')
        if bank['draft_layer_ids']!=[40,41,42] or bank['method']!='dspark':
            raise ValueError('Native three-layer DSpark model not actually loaded')
        if bank['engram_layers']!=[1,14] or bank['async_scheduling']:
            raise ValueError('Formal Engram/synchronous counter contract changed')
        if args.draft_graph and not bank['draft_graph_configured']:
            raise ValueError('Draft graph is not configured; actual replay is checked for every real proposal')
    result.update(phase='native-aa' if audit else 'performance',bank_complete=True,
                  native_dspark_loaded=True,bank_path='banks.json')
    comparisons=[]
    for pair in range(args.pairs):
        seed=pair%2
        left,a,left_counters=request(seed,f'pair-{pair}-a',True)
        right,b,right_counters=request(seed,f'pair-{pair}-b',True)
        if audit:
            from formal_comparison import compare_requests
            comparison=compare_requests(a,b,f'pair-{pair}',prompt_len=args.prompt_tokens)
            # Verify complete raw verifier proposal/token streams as well as
            # accepted target routes/top5/logprobs. Timing is not compared.
            def streams(receipts):
                r=next(iter(receipts[0]['requests'].values()))
                return [(x['scheduled_draft_token_ids'],x['token_ids'],[{k:c[k] for k in ('k','output_shape','runtime_graph','method')} for c in x['draft_calls']]) for x in r['rounds']]
            comparison['actual_round_streams_equal']=streams(left_counters)==streams(right_counters)
            comparison['passed']=comparison['passed'] and comparison['actual_round_streams_equal']
            comparisons.append(comparison)
            write_json(args.output/'comparisons.json',comparisons)
        write_json(args.output/'progress.json',result)
    paired=[r for r in records if r['metrics'] is not None]
    zero_accept=any(r['metrics']['acceptance_zero_diagnostic_required'] for r in paired)
    result.update(native_aa_passed=all(r['passed'] for r in comparisons) if audit else None,
        full_consumer_audit_passed=False if audit else True,
        acceptance_zero_diagnostic_required=zero_accept,
        counter_triplets=[{'tag':r['tag'],**{k:r['metrics'][k] for k in ('ms_per_step','A','tokens_per_second','decoded_tokens','rounds')}} for r in paired],
        median_ms_per_step=statistics.median(r['metrics']['ms_per_step'] for r in paired),
        target_ms_per_step=TARGET_MS,
        pending_consumer_proofs=banks[0]['pending_consumer_proofs'],
        passed=False,phase='native-aa-device-proof-pending' if audit else 'complete',
        performance_claim=None,
        meaning='A/A and counters are diagnostic until complete actual main/draft/verifier/cache consumer proof passes')
    result['actual_npu_requests_executed']=len(records)
    result['actual_target_verifier_rounds_observed']=sum(r['metrics']['rounds'] for r in paired)
    elapsed=sum(r['metrics']['decode_elapsed_s'] for r in paired)
    decoded=sum(r['metrics']['decoded_tokens'] for r in paired)
    rounds=sum(r['metrics']['rounds'] for r in paired)
    result['aggregate_triplet']={'ms_per_step':1000*elapsed/rounds,'A':decoded/rounds,
                                 'tokens_per_second':decoded/elapsed,
                                 'decoded_tokens':decoded,'rounds':rounds,'decode_elapsed_s':elapsed}
    if not audit:
        result.update(passed=not zero_accept and result['median_ms_per_step']<=TARGET_MS,
                      performance_claim={'scope':'complete synchronous DSpark decode wall clock',
                                         'target_ms_per_step':TARGET_MS})
    write_json(args.output/'result.json',result)
    return 0 if (result.get('native_aa_passed') if audit else result['passed']) else 1


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--physical-chips',default=PHYSICAL_CHIPS)
    parser.add_argument('--mode',choices=('prepare','audit','perf'),default='prepare')
    parser.add_argument('--speculative-tokens',type=int,default=5)
    parser.add_argument('--draft-graph',type=int,choices=(0,1),default=1)
    parser.add_argument('--pairs',type=int,default=3)
    parser.add_argument('--prompt-tokens',type=int,default=2048)
    parser.add_argument('--max-tokens',type=int,default=47)
    parser.add_argument('--audit-root',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    # The host launcher may already have put launch.log/exit metadata here.
    # Never overwrite a previous program run's evidence.
    if any((args.output/name).exists() for name in ('result.json','checkpoint.json','prepared_plan.json','requests.json')):
        raise FileExistsError('Use a new DSpark output directory; prior run evidence exists')
    result={'scope':'formal full40 TP8 native DSpark preparation/control',
            'mode':args.mode,'target_ms_per_step':TARGET_MS,'physical_chips':args.physical_chips,
            'cpu_prepared':False,'npu_validated':False,'passed':False,
            'full_consumer_audit_passed':False,'formal_model_validated':False,
            'safe_formal_receipt':False,'performance_claim':None,'phase':'checkpoint-preflight'}
    try:
        if args.physical_chips!=PHYSICAL_CHIPS or args.pairs<2 or args.speculative_tokens<=0:
            raise ValueError('Exact formal physical8–15/paired/K contract required')
        if not 1<=args.prompt_tokens<=2048 or args.max_tokens<2:
            raise ValueError('Invalid exact request budget')
        if args.mode=='perf' and args.pairs<10:
            raise ValueError('Formal performance requires at least ten paired samples')
        checkpoint=inspect_dspark_checkpoint(args.model,args.speculative_tokens)
        sources=freeze_sources()
        result.update(cpu_prepared=True,checkpoint_config_sha256=checkpoint['config_sha256'],
                      source_hashes=sources,speculative_config=model_kwargs(args.model,args.speculative_tokens,
                      audit=args.mode=='audit',draft_graph=bool(args.draft_graph))['speculative_config'])
        write_json(args.output/'checkpoint.json',checkpoint)
        plan={'kwargs':model_kwargs(args.model,args.speculative_tokens,audit=args.mode!='perf',draft_graph=bool(args.draft_graph)),
              'environment':startup_env(args.output,audit=args.mode!='perf',draft_graph=bool(args.draft_graph)),
              'sources':sources,'target_ms_per_step':TARGET_MS,
              'step_denominator':'actual synchronous target/verifier execute->sample completion rounds',
              'timing_scope':'post-prefill seed bookkeeping before initial draft -> final client RequestOutput',
              'consumer_device_proof':'pending; no CPU preparation can grant this receipt'}
        write_json(args.output/'prepared_plan.json',plan)
        if args.mode=='prepare':
            result['phase']='cpu-prepared-no-npu-executed'
            write_json(args.output/'result.json',result)
            print('FORMAL_DSPARK_CPU_PREPARED',json.dumps(result,ensure_ascii=False),flush=True)
            return 0
        return run_native(args,result,checkpoint,sources)
    except BaseException as exc:
        result.update(passed=False,phase='terminal-'+result['phase'],exception_type=type(exc).__name__,
                      exception=str(exc),traceback=traceback.format_exc(),
                      action='terminal receipt; no retry or active-source repair')
        write_json(args.output/'result.json',result)
        print('FORMAL_DSPARK_TERMINAL_REJECTION',str(exc),flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
