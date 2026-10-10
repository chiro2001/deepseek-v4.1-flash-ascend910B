"""Deploy and validate only a formally audited, stably faster exact-mask bank."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ('container','audit-job','perf-job','probe-job','service-job','served-model','model','encoder','images'):
        p.add_argument('--'+name,required=True)
    for name in ('root','source-dir','controller-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--client-root',default='/work/client')
    p.add_argument('--data-root',default='/work/client_data/gsm8k')
    p.add_argument('--target-ms',type=float,default=17.)
    args=p.parse_args()
    assert args.source_dir.is_relative_to('/work') and '..' not in args.source_dir.parts
    assert args.served_model.startswith('dsv41-a321-formal-') and args.target_ms==17.
    for name in (args.audit_job,args.perf_job,args.probe_job,args.service_job):
        assert name.replace('_','').replace('-','').isalnum()
    audit=args.root/'results'/args.audit_job;perf=args.root/'results'/args.perf_job
    deadline=time.monotonic()+7200
    while not (perf/'run.exit').exists():
        if (audit/'run.exit').exists():assert (audit/'run.exit').read_text().strip()=='0','Audit failed; refuse deployment'
        assert time.monotonic()<deadline,'Performance wait timed out'
        time.sleep(10)
    assert (audit/'run.exit').read_text().strip()==(perf/'run.exit').read_text().strip()=='0'
    assert (audit/'validated_for_mask_perf.json').exists()
    result=json.loads((perf/'result.json').read_text())
    assert not result['audit'] and result['profiler']=='OFF' and result['async_scheduling']
    assert result['formal_weights'] and result['tensor_parallel_size']==8
    assert result['hccl_deterministic_env']=='strict' and result['model_path']==args.model
    assert result['physical_chips']==list(range(8,16))
    paired={arm:{r['pair']:r['candidate_ms'] for r in result['pairs'] if r['arm']==arm}
            for arm in ('tp8metastack','tp8mask')}
    assert set(paired['tp8mask'])==set(paired['tp8metastack']) and len(paired['tp8mask'])>=12
    savings=[paired['tp8metastack'][i]-paired['tp8mask'][i] for i in sorted(paired['tp8mask'])]
    best=min(result['arms'],key=lambda arm:result['arms'][arm]['ms_per_step'])
    selection={'best_arm':best,'paired_median_saved_ms':statistics.median(savings),
               'faster_pairs':sum(s>0 for s in savings),'total_pairs':len(savings)}
    args.controller_root.mkdir(parents=True,exist_ok=True)
    if best!='tp8mask' or selection['faster_pairs']<8 or selection['paired_median_saved_ms']<=.05:
        (args.controller_root/'delivery_skipped.json').write_text(json.dumps({**selection,
            'reason':'No stable material incremental benefit; do not deploy mask candidate'},indent=2)+'\n')
        print('MASK_DELIVERY_SKIPPED',json.dumps(selection),flush=True);return
    source=args.root/args.source_dir.relative_to('/work')
    command=[sys.executable,str(source/'stack/launch_real_tp8_job.py'),
        '--root',str(args.root),'--source-dir',str(args.source_dir),'--container',args.container,
        '--job',args.service_job,'--model',args.model,'--chips','8,9,10,11,12,13,14,15',
        '--serve','--service-arm','tp8mask','--audit-job',args.audit_job,'--perf-job',args.perf_job,
        '--mask-evidence-job',args.probe_job,'--port',str(args.port),'--served-model',args.served_model,
        '--hccl-deterministic','strict','--hccl-npu-socket-port-range','auto','--allow-alarm','--wait-seconds','600']
    (args.controller_root/'validated_for_delivery.json').write_text(json.dumps({**selection,'argv':command},indent=2)+'\n')
    subprocess.run(command,check=True)
    client=['docker','exec','-e','STACK_SRC_ROOT='+str(args.source_dir),args.container,'bash',
            str(args.source_dir/'stack/run_stack.sh'),'run_formal_client_acceptance.py',
            '--service-root=/work/results/'+args.service_job,'--client-root='+args.client_root,
            '--data-root='+args.data_root,'--encoder='+args.encoder,'--images='+args.images,
            '--target-ms='+str(args.target_ms)]
    proc=subprocess.run(client)
    out=args.root/'results'/args.service_job/'client'
    receipt={**selection,'client_returncode':proc.returncode,'service_job':args.service_job,
             'target_ms':args.target_ms,'client_argv':client}
    if (out/'acceptance.json').exists():
        acceptance=json.loads((out/'acceptance.json').read_text())
        receipt.update(client_quality_passed=acceptance['client_quality_passed'],
                       performance_target_passed=acceptance['performance_target_passed'],
                       timing=json.loads((out/'timing.json').read_text()))
    (args.controller_root/'delivery_completed.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert proc.returncode==0 and receipt['client_quality_passed'],'Client validation failed'
    print('MASK_CLIENT_DELIVERY_COMPLETE',json.dumps(receipt),flush=True)


if __name__=='__main__':main()
