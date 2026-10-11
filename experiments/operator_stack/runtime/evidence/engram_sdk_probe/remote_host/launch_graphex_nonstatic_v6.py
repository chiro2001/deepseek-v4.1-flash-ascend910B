from pathlib import Path
import json, re, subprocess, traceback
root=Path('/home/l00886679/projects/dsv41-engram-sdk-chip4-20261011-v1')
stage='graphex_nonstatic_v6'
evidence=root/'evidence'
exit_code=99
try:
    assert (evidence/'graphex_static_v6.exit').read_text().strip()=='1'
    pre=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
    (evidence/(stage+'_prelaunch.txt')).write_text(pre)
    assert '| 2     Ascend910           | OK' in pre, 'physical4 health not OK'
    proc=pre.split('Process id in container',1)[1]
    assert not re.search(r'^\|\s*2\s+0\s+\|\s*\d+',proc,re.M), 'physical4 occupied'
    command=['docker', 'exec', '-e', 'ASCEND_RT_VISIBLE_DEVICES=4', '-e', 'ASCEND_VISIBLE_DEVICES=4', '-e', 'SDK_PROBE_PHYSICAL_CHIP=4', '-e', 'HCCL_DETERMINISTIC=strict', '-e', 'LOCAL_WORLD_SIZE=1', '-e', 'OMP_NUM_THREADS=1', 'dsv41-tiny-goal20-20261010-c6', 'bash', '/work/engram_sdk_probe_20261011_c4_v1/run_v6.sh', '--stage', 'graphex', '--generations', '32', '--enqueue-order', 'consumer-first', '--output', '/work/engram_sdk_probe_20261011_c4_v1/evidence/graphex_nonstatic_v6']
    (evidence/(stage+'_launch.json')).write_text(json.dumps({'command':command,'physical_chip':4,'source_sha256':'bd5d995abc63dde901b558c0e44d31266d3cc14ac990e9ecd3b9007f900cc167','diagnostic_only':'static kernel disabled; not the formal static configuration'},indent=2)+'\n')
    with (evidence/(stage+'.log')).open('xb') as output:
        exit_code=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT).returncode
except BaseException:
    (evidence/(stage+'_launch_error.txt')).write_text(traceback.format_exc())
finally:
    (evidence/(stage+'.exit')).write_text(str(exit_code)+'\n')
