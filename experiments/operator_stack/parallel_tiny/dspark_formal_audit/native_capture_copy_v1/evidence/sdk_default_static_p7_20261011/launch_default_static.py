from pathlib import Path
import json,re,subprocess,traceback
root=Path('/home/l00886679/projects/dsv41-real-stack-tp8-20261010/native_copy_gate_20261011_c7_v1');case='default_static';code=99
try:
    smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
    health=subprocess.run(['npu-smi','info','-t','health','-i','3','-c','1'],capture_output=True,text=True,check=True).stdout
    (root/'evidence'/(case+'_prelaunch.txt')).write_text(smi)
    (root/'evidence'/(case+'_health.txt')).write_text(health)
    assert re.search(r'Health Status\s*:\s*OK',health)
    assert re.search(r'Error Code\s*:\s*(?:NA|0)\s*$',health,re.M),health
    proc=smi.split('Process id in container',1)[1]
    assert not re.search(r'^\|\s*3\s+1\s+\|\s*\d+',proc,re.M),'physical7 occupied'
    command=['docker', 'exec', '-e', 'ASCEND_RT_VISIBLE_DEVICES=7', '-e', 'ASCEND_VISIBLE_DEVICES=7', '-e', 'SDK_NATIVE_COPY_PHYSICAL_CHIP=7', '-e', 'HCCL_DETERMINISTIC=strict', '-e', 'LOCAL_WORLD_SIZE=1', '-e', 'OMP_NUM_THREADS=1', '-e', 'STACK_TINY_PROFILE=0', '-e', 'PYTHONPATH=/work/native_copy_gate_20261011_c7_v1/source_v1', '-e', 'NATIVE_COPY_RUNTIME_ROOT=/work/native_copy_gate_20261011_c7_v1/runtime/default_static', 'dsv41-real-stack-tp8-20261010-a321', 'bash', '/work/native_copy_gate_20261011_c7_v1/source_v1/run_gate.sh', '--stage', 'graphex', '--static-kernel', '--physical-chip', '7', '--generations', '32', '--output', '/work/native_copy_gate_20261011_c7_v1/results/default_static']
    (root/'evidence'/(case+'_argv.json')).write_text(json.dumps({'argv':command,'physical_chip':7,'fresh_case':True},indent=2)+'\n')
    with (root/'evidence'/(case+'.log')).open('xb') as log:
        code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
except BaseException:
    (root/'evidence'/(case+'_launch_error.txt')).write_text(traceback.format_exc())
finally:
    (root/'evidence'/(case+'.exit')).write_text(str(code)+'\n')
