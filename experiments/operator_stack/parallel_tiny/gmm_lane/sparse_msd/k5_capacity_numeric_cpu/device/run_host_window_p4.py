"""Host-owned fresh preflight/run/release wrapper for one physical4 window.

Run on the a3-21 host (not inside the container). It snapshots live NPU
health/occupancy for NPU 2 chip 0 (physical 4), confirms our container really
owns the /dev/davinci4 node with the container-global 8-15 job untouched,
then executes the one-shot container gate. Must not touch any other process,
device, container or cache.
"""
import datetime
import json
import stat
import subprocess
import sys
from pathlib import Path

CONTAINER = 'dsv41-real-stack-tp8-20261010-a321'
HOST_PROJECTS = Path('/home/l00886679/projects/dsv41-real-stack-tp8-20261010')
ROOT_NAME = sys.argv[1] if len(sys.argv) > 1 else 'gmm_lane_sparse_msd_k5_capacity_p4_20261012_v1'
PHASE = sys.argv[2] if len(sys.argv) > 2 else 'numeric_then_timing'
NUMERIC_ROOT_NAME = sys.argv[3] if len(sys.argv) > 3 else ROOT_NAME
HOST_ROOT = HOST_PROJECTS / ROOT_NAME
ROOT = '/work/' + ROOT_NAME
PHYSICAL = 4
NPU_CARD = PHYSICAL // 2
NPU_CHIP = PHYSICAL % 2


def run(command):
    return subprocess.check_output(command, text=True)


def snapshot():
    smi = run(['npu-smi', 'info'])
    health = run(['npu-smi', 'info', '-t', 'health', '-i', str(NPU_CARD), '-c', str(NPU_CHIP)])
    processes = []
    assert 'Process id' in smi
    section = smi[smi.index('Process id'):]
    for line in section.splitlines():
        cells = [x.strip() for x in line.split('|')[1:-1]]
        if len(cells) >= 2 and cells[0].split() == [str(NPU_CARD), str(NPU_CHIP)] and cells[1].isdigit():
            processes.append(line)
    inspect = json.loads(run(['docker', 'inspect', CONTAINER]))[0]
    env = [x for x in inspect['Config']['Env'] if x.startswith('ASCEND')]
    node_program = ("import os,stat,json; s=os.stat('/dev/davinci" + str(PHYSICAL) + "'); "
                    "print(json.dumps({'char':stat.S_ISCHR(s.st_mode),'major':os.major(s.st_rdev),"
                    "'minor':os.minor(s.st_rdev),'rw':os.access('/dev/davinci" + str(PHYSICAL) + "',os.R_OK|os.W_OK)}))")
    node = json.loads(run(['docker', 'exec', '-e', 'ASCEND_RT_VISIBLE_DEVICES=', '-e', 'ASCEND_VISIBLE_DEVICES=',
                           CONTAINER, 'python3', '-c', node_program]))
    permission = bool(inspect['HostConfig']['Privileged'] and node['char'] and node['major'] == 509 and
                      node['minor'] == PHYSICAL and node['rw'])
    healthy = 'Health Status' in health and ': OK' in health and 'Error Code' in health and ': NA' in health
    return {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'physical': PHYSICAL,
            'npu_card': NPU_CARD, 'npu_chip': NPU_CHIP, 'physical_processes': processes,
            'health_allowed': healthy, 'health_output': health, 'smi_output': smi,
            'container_has_dev_permission': permission, 'device_node': node,
            'container_id': inspect['Id'], 'container_PID': inspect['State']['Pid'],
            'container_global_ASCEND_env': env,
            'host_root_matches_container_root': HOST_ROOT.is_dir(),
            'container_global_8_15_env_expected': 'ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15'}


def main():
    assert HOST_ROOT.is_dir(), HOST_ROOT
    assert not (HOST_ROOT / 'runtime_attempt.json').exists(), 'No retry'
    assert not (HOST_ROOT / 'run_receipt.json').exists(), 'No retry'
    pre = snapshot()
    (HOST_ROOT / 'preflight.json').write_text(json.dumps(pre, indent=2) + '\n')
    print(json.dumps({k: pre[k] for k in ['UTC', 'physical', 'physical_processes', 'health_allowed',
                                          'container_has_dev_permission', 'container_PID']}), flush=True)
    assert pre['physical_processes'] == [] and pre['health_allowed'] and pre['container_has_dev_permission']
    assert 'ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15' in pre['container_global_ASCEND_env']
    assert PHASE in ('numeric_then_timing', 'timing_only')
    command = ['docker', 'exec', '-e', f'ASCEND_RT_VISIBLE_DEVICES={PHYSICAL}',
               '-e', f'ASCEND_VISIBLE_DEVICES={PHYSICAL}', '-e', f'CAPACITY_EXPECTED_PHYSICAL={PHYSICAL}',
               '-e', 'TASK_QUEUE_ENABLE=0', '-e', 'ASCEND_GLOBAL_LOG_LEVEL=0',
               '-e', 'ASCEND_SLOG_PRINT_TO_STDOUT=1', '-e', f'K5_CAPACITY_PHASE={PHASE}',
               '-e', 'K5_CAPACITY_NUMERIC_ROOT=/work/' + NUMERIC_ROOT_NAME, CONTAINER, 'python3',
               ROOT + '/k5_capacity_numeric_cpu/device/run_gate_device.py']
    code = 1
    try:
        code = subprocess.run(command, timeout=540).returncode
    finally:
        release = snapshot()
        release['physical_released'] = release['physical_processes'] == []
        release['container_global_env_unchanged'] = (release['container_global_ASCEND_env'] ==
                                                     pre['container_global_ASCEND_env'])
        release['runtime_wrapper_exit'] = code
        (HOST_ROOT / 'release_fresh_smi.json').write_text(json.dumps(release, indent=2) + '\n')
        print(json.dumps({k: release[k] for k in ['UTC', 'physical', 'physical_released', 'health_allowed',
                                                 'container_global_env_unchanged', 'runtime_wrapper_exit']}),
              flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
