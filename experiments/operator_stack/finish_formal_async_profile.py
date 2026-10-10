"""Owned postprocessor: validate real worker windows, parse and summarize."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

METRICS = ('PipeUtilization', 'ArithmeticUtilization', 'Memory', 'MemoryL0',
           'MemoryUB', 'L2Cache', 'ResourceConflictRatio')
ARMS = ('tp8base', 'tp8metastack')


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--controller-job', required=True)
    args = parser.parse_args()
    for name in (args.job,args.controller_job):
        assert name.replace('_','').replace('-','').isalnum()
    root = args.root.resolve()
    job = root/'results'/args.job
    out = root/'results'/args.controller_job
    out.mkdir(parents=True,exist_ok=True)
    launched=json.loads((job/'launched.json').read_text())
    assert launched['container']==args.container and '--async-scheduling' in launched['argv']
    assert launched['env']['STACK_TINY_PROFILE']=='0' and launched['env']['HCCL_DETERMINISTIC']=='strict'
    deadline=time.monotonic()+3600
    while not (job/'run.exit').exists():
        code=("from pathlib import Path; r=Path('/work/results/"+args.job+
              "'); pid=(r/'runner.pid').read_text().strip(); assert Path('/proc/'+pid).exists(),'Profile runner vanished'; print(pid)")
        pid=subprocess.check_output(['docker','exec',args.container,'python','-c',code],text=True).strip()
        print('FORMAL_PROFILE_VERIFIED_LIVE',pid,flush=True)
        assert time.monotonic()<deadline,'Observation timed out; do not restart a live job'
        time.sleep(20)
    assert (job/'run.exit').read_text().strip()=='0','Profile failed; inspect evidence'
    result=json.loads((job/'result.json').read_text())
    assert result['formal_weights'] and result['async_scheduling'] and not result['audit']
    assert result['profiler']=='OFF' and set(result['arms'])==set(ARMS)
    windows=[]
    for arm in ARMS:
        for metric in METRICS:
            path=job/f'profile_steps_{arm}_{metric}.json'
            rows=json.loads(path.read_text())
            assert len(rows)==8 and {r['rank'] for r in rows}==set(range(8))
            assert all(r['driver']=='worker.execute_model' and r['manual_calls']==0 and
                       r['schedule_steps']==r['required_schedule_steps']==16 and
                       r['worker_calls']>=16 and r['complete'] for r in rows)
            windows.append({'arm':arm,'metric':metric,'receipt_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})

    def run(script,*parameters):
        subprocess.run(['docker','exec','-e','STACK_SRC_ROOT='+str(args.source_dir),
                        '-e','STACK_TINY_PROFILE=0',args.container,'bash',
                        str(args.source_dir/'stack/run_stack.sh'),script,*parameters],check=True)

    run('parse_formal_profile.py','--root=/work/results/'+args.job,'--expected-directories=112','--jobs=4')
    for arm in ARMS:
        code=("from pathlib import Path; base=Path('/work/results/"+args.job+
              "'); target=base/'analysis'/"+repr(arm)+"; target.mkdir(parents=True,exist_ok=True); "
              "p=target/'prof'; expected=base/'prof'/"+repr(arm)+
              "; p.symlink_to(expected,target_is_directory=True) if not p.exists() else None; assert p.resolve()==expected.resolve()")
        subprocess.run(['docker','exec',args.container,'python','-c',code],check=True)
        run('analyse_formal_microarch.py','--root=/work/results/'+args.job+'/analysis/'+arm)
    run('analyse_formal_timeline.py','--root=/work/results/'+args.job,'--arms='+','.join(ARMS))
    receipt={'passed':True,'formal_result_sha256':hashlib.sha256((job/'result.json').read_bytes()).hexdigest(),
             'windows':windows,'expected_directories':112,
             'scope':'Actual worker-step schedule and all-rank offline parse; collection timing is diagnostic, not17ms client acceptance'}
    (out/'completed.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('FORMAL_ASYNC_ANALYSIS_COMPLETE',json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
