"""Host launcher for a small owned probe; verify8–15 even if probe uses onlychip8."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from launch_healthy_probe import resources


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--container',required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--probe',choices=('woa','moe-mask','quant-gemv','qkv-merge'),default='woa')
    args=p.parse_args()
    assert args.job.replace('_','').replace('-','').isalnum()
    root=args.root.resolve();assert args.source_dir.is_relative_to('/work')
    source=root/args.source_dir.relative_to('/work')
    script={'woa':'bench_formal_oprojection.py','moe-mask':'bench_formal_moe_mask.py',
            'quant-gemv':'bench_formal_quant_gemv.py',
            'qkv-merge':'bench_formal_qkv_merge.py'}[args.probe]
    assert (source/'stack'/script).is_file()
    cfg=json.loads(subprocess.check_output(['docker','inspect',args.container],text=True))[0]
    env=dict(x.split('=',1) for x in cfg['Config']['Env'])
    chips=list(range(8,16));visible=','.join(map(str,chips))
    assert cfg['State']['Running'] and env['ASCEND_RT_VISIBLE_DEVICES']==visible
    assert any(m['Destination']=='/work' and Path(m['Source']).resolve()==root for m in cfg['Mounts'])
    setup="from pathlib import Path; import os; p=Path('/work/results/"+args.job+"'); p.mkdir(exist_ok=True); os.chown(p,"+str(os.getuid())+","+str(os.getgid())+")"
    subprocess.run(['docker','exec',args.container,'python','-c',setup],check=True)
    out=root/'results'/args.job
    with (out/'launch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (out/'launched.json').exists()
        raw=subprocess.check_output(['npu-smi','info'],text=True);parsed=resources(raw);alarms={}
        for chip in chips:
            assert parsed[chip]['health'] in ['OK','Alarm'] and not parsed[chip]['processes'],(chip,parsed[chip])
            if parsed[chip]['health']=='Alarm':
                s=subprocess.check_output(['npu-smi','info','-t','health','-i',str(chip//2),'-c',str(chip%2)],text=True)
                codes=[x.split(':',1)[1].strip() for x in s.splitlines() if 'Error Code' in x and ':' in x]
                assert codes==['80C98001'],(chip,s)
                alarms[chip]=s
        command=['bash',str(args.source_dir/'stack/run_stack.sh'),script,
                 '--output=/work/results/'+args.job,'--physical-chips='+visible,'--model='+args.model]
        body='#!/usr/bin/env bash\nset -uo pipefail\nexport STACK_SRC_ROOT='+shlex.quote(str(args.source_dir))+'\nexport STACK_TINY_PROFILE=0\n'
        body+='echo $$ > /work/results/'+args.job+'/runner.pid\n'+shlex.join(command)+' > /work/results/'+args.job+'/run.log 2>&1\n'
        body+='status=$?\nprintf "%s\\n" "$status" > /work/results/'+args.job+'/run.exit\nexit "$status"\n'
        path=root/(args.job+'.sh');path.write_text(body)
        receipt={'container':args.container,'source_dir':str(args.source_dir),'argv':command,
                 'resources':parsed,'alarm_details':alarms,'chips':chips,'executing_chip':8,
                 'source_sha256':hashlib.sha256((source/'stack'/script).read_bytes()).hexdigest(),
                 'script_sha256':hashlib.sha256(body.encode()).hexdigest()}
        (out/'resource_prelaunch.txt').write_text(raw)
        (out/'launched.json').write_text(json.dumps(receipt,indent=2)+'\n')
        subprocess.run(['docker','exec','-d',args.container,'bash','/work/'+path.name],check=True)
        print('FORMAL_OPERATOR_PROBE_LAUNCHED',json.dumps(receipt),flush=True)


if __name__=='__main__':main()
