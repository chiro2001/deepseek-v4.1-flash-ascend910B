"""Start one owned experiment; stop only this lane's API process when asked."""
import argparse
import json
import shlex
import subprocess

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--tag', required=True)
parser.add_argument('--script', required=True)
parser.add_argument('--stop-service', action='store_true')
parser.add_argument('script_args', nargs=argparse.REMAINDER)
args = parser.parse_args()
assert args.tag.replace('_','').replace('-','').isalnum()
assert args.script.endswith('.py') and '/' not in args.script
script_args = args.script_args[1:] if args.script_args[:1] == ['--'] else args.script_args
remote = '/home/l00886679/projects/dsv41-tiny-upstream950-20261009'
name = 'dsv41-tiny-upstream950-20261009-c5'
command = ['python','-u','/work/scripts/'+args.script,*script_args]
shell = ('source /usr/local/Ascend/ascend-toolkit/set_env.sh; '
         'source /usr/local/Ascend/nnal/atb/set_env.sh; '
         'export PYTHONPATH=/work/scripts${PYTHONPATH:+:$PYTHONPATH}; ' + shlex.join(command))
lock_prefix = '' if args.script == 'run_when_idle.py' else (
    'exec 9>/work/results/gpu_experiment.lock; '
    'flock -n 9 || { echo "BLOCKED: an owned experiment holds the lane GPU lock"; exit 73; }; ')
shell = ('exec > '+shlex.quote('/work/logs/'+args.tag+'.log')+' 2>&1; '+lock_prefix+shell)
code = f'''import json,os,subprocess,time
from pathlib import Path
root=Path({remote!r});name={name!r}
assert (root/'results/launch.json').exists()
manifest=json.loads((root/'results/launch.json').read_text())
assert manifest['container']==name and manifest['physical_chip']==5
if {args.stop_service!r}:
    stop="""import os,signal,time
from pathlib import Path
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try: command=(p/'cmdline').read_bytes().split(bytes([0]))
    except OSError: continue
    if b'vllm.entrypoints.openai.api_server' in command and b'dsv41-tiny-upstream950-20261009' in command:
        print('Stopping own API pid',p.name,flush=True);os.kill(int(p.name),signal.SIGTERM)
time.sleep(5)
"""
    subprocess.run(['sudo','-n','docker','exec',name,'python','-c',stop],check=True)
inspect=subprocess.check_output(['sudo','-n','docker','top',name,'-eo','pid,args'],text=True)
if {args.script!r} != 'run_when_idle.py':
    assert '/work/scripts/bench_lane_model.py' not in inspect, 'An owned model benchmark is already running'
subprocess.run(['sudo','-n','docker','exec','-d',name,'bash','-lc',{shell!r}],check=True)
print(json.dumps({{'tag':{args.tag!r},'command':{command!r},'log':str(root/'logs'/{(args.tag+'.log')!r})}}))
'''
subprocess.run(['ssh','-o','BatchMode=yes','a3-21','python3','-'],input=code,text=True,check=True)
