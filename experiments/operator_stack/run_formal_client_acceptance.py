"""Run serial clients against the selected service with explicit ownership.

Launch in the private container. Raw requests stay under /work; only small
receipts are fetched. No profiler or route audit is enabled during timing.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--service-root',required=True,type=Path)
    parser.add_argument('--client-root',required=True,type=Path)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--encoder',required=True)
    parser.add_argument('--images',required=True)
    args=parser.parse_args()
    selection=json.loads((args.service_root/'service_command.json').read_text())
    model=selection['served_model'];port=selection['port'];base=f'http://127.0.0.1:{port}'
    assert model.startswith('dsv41-a321-formal-') and selection['client_validation']=='pending'
    assert selection['selected_arm']=='tp8core' and selection['measured']['A']==1
    out=args.service_root/'client';out.mkdir(exist_ok=True)
    assert not (out/'acceptance.json').exists(),'Do not overwrite previous acceptance'
    deadline=time.monotonic()+1800
    while True:
        try:
            with urllib.request.urlopen(base+'/v1/models',timeout=3) as response:
                models=json.load(response)
            break
        except (OSError,ValueError):
            assert time.monotonic()<deadline,'Formal service readiness timed out'
            assert not (args.service_root/'run.exit').exists(),'Formal service exited during startup'
            time.sleep(10)
    assert [r['id'] for r in models['data']]==[model],('Wrong service',models)
    candidates=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:cmd=p.joinpath('cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        text=[word.decode(errors='replace') for word in cmd if word]
        if 'vllm.entrypoints.openai.api_server' in text and model in text and str(port) in text:
            candidates.append({'pid':int(p.name),'argv':text})
    assert len(candidates)==1,('Service process ownership mismatch',candidates)
    ownership={'base_url':base,'served_model':model,'selected_arm':selection['selected_arm'],
               'models':models,'api_process':candidates[0],'A':1,'speculative_decoding':False,
               'checkpoint':selection['argv'][selection['argv'].index('--model')+1]}
    (out/'ownership.json').write_text(json.dumps(ownership,indent=2)+'\n')
    receipts=[]
    def run(label,command):
        log=out/(label+'.log')
        with log.open('w') as stream:
            proc=subprocess.run(command,cwd=args.client_root,stdout=stream,stderr=subprocess.STDOUT)
        receipt={'label':label,'argv':command,'returncode':proc.returncode,
                 'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
        receipts.append(receipt)
        (out/'commands.json').write_text(json.dumps(receipts,indent=2)+'\n')
        print('FORMAL_CLIENT_DONE',json.dumps(receipt),flush=True)
        assert proc.returncode==0,('Client failed',label,log)
    python=sys.executable
    run('bench8',[python,'tools/bench_concurrency.py','--base-url',base,'--model',model,
        '--require-model','--concurrency','1','--prompt-count','8','--prompt-tokens','2048',
        '--output-tokens','256','--repeats','1','--json-out',str(out/'bench8.json'),
        '--label','formal-core-strict-a321'])
    bench=json.loads((out/'bench8.json').read_text())
    assert bench['model']==model and len(bench['rows'])==1
    timing=bench['rows'][0]
    assert timing['conc']==1 and timing['ok']==8 and timing['fail']==0
    assert timing['prompt_tokens']==2048 and timing['out_per_req']==256
    assert math.isfinite(timing['per_stream_med']) and timing['per_stream_med']>0
    (out/'timing.json').write_text(json.dumps({'ms_per_step':1000/timing['per_stream_med'],
        'A':1,'tokens_per_second':timing['per_stream_med'],'samples':8,
        'raw_accept_length':timing['accept_len'],'speculative_decoding':False,
        'scope':'Client decode single-stream median; 2K input, 256 output; profiler/audit off',
        'passed_19ms':1000/timing['per_stream_med']<=19},indent=2)+'\n')
    run('vision',[python,'tests/t_vision.py','--server',base,'--model',model,
        '--images-dir',args.images,'--out',str(out/'vision.json')])
    deadline=time.monotonic()+240
    while not (args.data_root/'source.json').exists():
        assert time.monotonic()<deadline,'Canonical GSM8K split download is incomplete'
        time.sleep(10)
    run('gsm100',[python,'tests/acc_eval.py','--task','gsm8k','--limit','100',
        '--base-url',base,'--model',model,'--conc','1','--mode','chat','--max-tokens','512',
        '--enc-dir',args.encoder,'--gsm8k-data-dir',str(args.data_root),
        '--out',str(out/'gsm100.json'),'--tag','formal-core-strict-a321'])
    gsm=json.loads((out/'gsm100.json').read_text())['summary']
    vision=json.loads((out/'vision.json').read_text())
    report={'ownership_validated':True,'commands':receipts,'gsm_summary':gsm,'vision':vision,
            'client_quality_passed':gsm['n']==100 and gsm['correct']>=98 and
                gsm['empty']==gsm['errors']==0 and vision['threshold_19_of_23'],
            'target_19ms_claim':False,'bench_result':str(out/'bench8.json'),
            'timing_requires_separate_review':True}
    (out/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FORMAL_CLIENT_ACCEPTANCE',json.dumps(report),flush=True)
    assert report['client_quality_passed'],'Formal client quality gate failed'


if __name__=='__main__':main()
