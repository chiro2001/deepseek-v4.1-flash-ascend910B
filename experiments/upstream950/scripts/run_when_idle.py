"""Wait for chip4 and this lane, then run owned performance commands serially."""
import argparse
import fcntl
import json
import os
import subprocess
import threading
import time
from pathlib import Path

parser=argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--tag',required=True)
parser.add_argument('--candidate',choices=['indexer','epilogue','projection'],required=True)
parser.add_argument('--confirm',action='store_true')
parser.add_argument('--micro-only',action='store_true')
parser.add_argument('--arms')
args=parser.parse_args()
assert args.tag.replace('_','').isalnum()
events=[]
idle_since=None
root=Path('/work/results')
report=root/(args.tag+'_idle_guard.json')
gpu_lock=(root/'gpu_experiment.lock').open('a')
while True:
    smi=subprocess.check_output(['npu-smi','info'],text=True).splitlines()
    row=next(line for line in smi if '0000:95:00.0' in line)
    fields=[s.strip() for s in row.split('|')]
    chip4_util=int(fields[3].split()[0])
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:command=(p/'cmdline').read_bytes().split(bytes([0]))
        except OSError:continue
        scripts=[s.decode(errors='replace') for s in command if s.startswith(b'/work/scripts/')]
        if any(Path(s).name.startswith(('bench_','verify_')) for s in scripts):
            active.append({'pid':p.name,'scripts':scripts})
    state={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           'chip4_util':chip4_util,'chip4_smi':row,'other_lane_experiments':active}
    ready=chip4_util==0 and not active
    if ready:
        if idle_since is None:idle_since=time.monotonic()
    else:idle_since=None
    if not events or state['chip4_util']!=events[-1]['chip4_util'] or active!=events[-1]['other_lane_experiments']:
        events.append(state)
        report.write_text(json.dumps({'candidate':args.candidate,'events':events,'started':False},indent=2))
        print('WAIT',json.dumps(state),flush=True)
    if ready and time.monotonic()-idle_since>=30:
        try:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            idle_since=None
            continue
        events.append(state)
        report.write_text(json.dumps({'candidate':args.candidate,'events':events,'started':True},indent=2))
        break
    time.sleep(5)

if args.candidate=='indexer':
    commands=[['python','-u','/work/scripts/verify_indexer_post.py','--stress',
               '--output','/work/results/indexer_post_final_102.json'],
              ['python','-u','/work/scripts/bench_indexer_post.py','--output',
               '/work/results/indexer_post_perf.json'],
              ['python','-u','/work/scripts/bench_lane_model.py','--candidate','indexer',
               '--arms','baseline,fused','--pairs','12','--output','/work/results/'+args.tag]]
    if args.confirm:commands=commands[-1:]
elif args.candidate=='epilogue':
    arms=args.arms or 'baseline,packed,inplace'
    commands=[['python','-u','/work/scripts/bench_epilogue.py','--arms',arms,'--output','/work/results/'+args.tag+'_micro.json'],
              ['python','-u','/work/scripts/bench_lane_model.py','--candidate','epilogue',
               '--arms',arms,'--pairs','12','--output','/work/results/'+args.tag]]
    if args.micro_only:commands=commands[:1]
elif args.candidate=='projection':
    arms=args.arms or 'baseline,joint,nz,panel'
    commands=[['python','-u','/work/scripts/bench_projection.py','--arms',arms,
               '--output','/work/results/'+args.tag+'_micro.json'],
              ['python','-u','/work/scripts/bench_lane_model.py','--candidate','projection',
               '--arms',arms,'--pairs','12','--output','/work/results/'+args.tag]]
    if args.micro_only:commands=commands[:1]
stop_monitor=threading.Event()
load_samples=[]
monitor_path=root/(args.tag+'_card_load.json')
def monitor():
    while not stop_monitor.is_set():
        lines=subprocess.check_output(['npu-smi','info'],text=True).splitlines()
        row=next(line for line in lines if '0000:95:00.0' in line)
        utilization=int(row.split('|')[3].split()[0])
        load_samples.append({'utc':time.time(),'chip4_util':utilization,'smi':row})
        monitor_path.write_text(json.dumps(load_samples,indent=2))
        stop_monitor.wait(2)
thread=threading.Thread(target=monitor,daemon=True)
thread.start()
try:
    for i,command in enumerate(commands):
        log=Path('/work/logs')/(args.tag+'_'+str(i)+'.log')
        print('START',json.dumps({'command':command,'log':str(log)}),flush=True)
        with log.open('w') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
finally:
    stop_monitor.set();thread.join()
print('COMPLETE',args.tag,flush=True)
