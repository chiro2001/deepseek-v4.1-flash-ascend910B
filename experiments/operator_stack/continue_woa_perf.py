"""Launch uninstrumented timing only after complete formal consumer audit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--container',required=True)
    p.add_argument('--audit-job',required=True)
    p.add_argument('--perf-job',required=True)
    p.add_argument('--probe-job',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--candidate',choices=('tp8woa','tp8mask'),default='tp8woa')
    args=p.parse_args()
    for job in (args.audit_job,args.perf_job,args.probe_job):
        assert job.replace('_','').replace('-','').isalnum()
    assert args.source_dir.is_relative_to('/work') and '..' not in args.source_dir.parts
    out=args.root/'results'/args.audit_job
    deadline=time.monotonic()+3600
    while not (out/'run.exit').exists():
        assert time.monotonic()<deadline,'Audit wait timed out'
        time.sleep(10)
    assert (out/'run.exit').read_text().strip()=='0','Audit failed; do not run timing'
    result=json.loads((out/'result.json').read_text())
    arms=['tp8base','tp8metastack',args.candidate]
    assert set(result['arms'])==set(arms) and result['model_path']==args.model
    assert result['formal_weights'] and result['audit'] and result['async_scheduling']
    assert result['hccl_deterministic_env']=='strict' and result['profiler']=='OFF'
    assert result['physical_chips']==list(range(8,16)) and result['tensor_parallel_size']==8
    assert result['same_model_instance'] and result['same_processes_per_rank']
    assert not result['fp32_decode_reduction'] and not result['speculative_decoding']
    controls=json.loads((out/'native_controls.json').read_text())
    assert len(controls)>=3 and all(r['passed'] for r in controls)
    for arm in arms[1:]:
        pairs=[r for r in result['pairs'] if r['arm']==arm]
        assert len(pairs)>=3 and all(r['tokens_equal'] and r['actual_routes_equal'] and
                                    r['max_logprob_delta']<1e-3 for r in pairs)
    feature='woa_cube' if args.candidate=='tp8woa' else 'moe_mask'
    unique='captured_unique_layers' if args.candidate=='tp8woa' else 'unique_layers'
    banks=json.loads((out/'banks.json').read_text())[args.candidate]
    assert len(banks)==8 and {r['rank'] for r in banks}==set(range(8))
    assert all(r[feature]['selected_calls']==40 and r[feature][unique]==40 for r in banks)
    requests=json.loads((out/'requests.json').read_text())
    selected=[r for r in requests if r['arm']==args.candidate and r['tag'].startswith('pair-')]
    assert len(selected)>=3
    for request in selected:
        assert len(request['rank_audit'])==8
        for rank in request['rank_audit']:
            assert rank[feature]['passed'] and rank[feature]['consumer_calls']==40
            assert rank['metadata_slots']['all_exact']
            if args.candidate=='tp8woa':assert rank[feature]['max_bf16_ulp']<=1
            else:assert rank[feature]['all_bitwise_equal'] and rank[feature]['unique_layers']==40
    source=args.root/args.source_dir.relative_to('/work')
    command=[sys.executable,str(source/'stack/launch_real_tp8_job.py'),
             '--root',str(args.root),'--source-dir',str(args.source_dir),'--container',args.container,
             '--job',args.perf_job,'--model',args.model,'--chips','8,9,10,11,12,13,14,15',
             '--arms',','.join(arms),'--pairs','12','--async-scheduling',
             '--woa-evidence-job' if args.candidate=='tp8woa' else '--mask-evidence-job',args.probe_job,
             '--hccl-deterministic','strict',
             '--hccl-npu-socket-port-range','auto','--allow-alarm','--wait-seconds','600']
    marker='validated_for_woa_perf.json' if args.candidate=='tp8woa' else 'validated_for_mask_perf.json'
    (out/marker).write_text(json.dumps({'passed':True,'argv':command,'candidate':args.candidate,
        'paired_requests':len(selected),'ranks':8,'unique_consumer_layers_per_rank':40},indent=2)+'\n')
    print('FORMAL_CONSUMER_FULL_AUDIT_VALIDATED',args.candidate,flush=True)
    subprocess.run(command,check=True)


if __name__=='__main__':main()
