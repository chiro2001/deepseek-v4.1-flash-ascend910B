"""Host-owned fresh preflight/run/release wrapper; one physical6 launch only."""
import datetime
import json
from pathlib import Path
import subprocess

CONTAINER = 'dsv41-real-stack-tp8-20261010-a321'
HOST_ROOT = Path('/home/l00886679/projects/dsv41-real-stack-tp8-20261010/gmm_lane_sparse_msd_m1_numeric_p6_20261011_v4')
ROOT = '/work/gmm_lane_sparse_msd_m1_numeric_p6_20261011_v4'


def run(command):
    return subprocess.check_output(command, text=True)


def snapshot():
    smi = run(['npu-smi', 'info'])
    health = run(['npu-smi', 'info', '-t', 'health', '-i', '3', '-c', '0'])
    processes = []
    assert 'Process id' in smi
    section = smi[smi.index('Process id'):]
    for line in section.splitlines():
        cells = [x.strip() for x in line.split('|')[1:-1]]
        if len(cells) >= 2 and cells[0].split() == ['3', '0'] and cells[1].isdigit():
            processes.append(line)
    inspect = json.loads(run(['docker', 'inspect', CONTAINER]))[0]
    env = [x for x in inspect['Config']['Env'] if x.startswith('ASCEND')]
    node_program = "import os,stat,json; s=os.stat('/dev/davinci6'); print(json.dumps({'char':stat.S_ISCHR(s.st_mode),'major':os.major(s.st_rdev),'minor':os.minor(s.st_rdev),'rw':os.access('/dev/davinci6',os.R_OK|os.W_OK)}))"
    node = json.loads(run(['docker', 'exec', '-e', 'ASCEND_RT_VISIBLE_DEVICES=', '-e', 'ASCEND_VISIBLE_DEVICES=', CONTAINER, 'python3', '-c', node_program]))
    permission = bool(inspect['HostConfig']['Privileged'] and node['char'] and node['major'] == 509 and node['minor'] == 6 and node['rw'])
    healthy = 'Health Status' in health and ': OK' in health and 'Error Code' in health and ': NA' in health
    return {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'physical': 6,
            'physical6_processes': processes, 'health_allowed': healthy, 'health_output': health,
            'smi_output': smi, 'container_has_dev6_permission': permission, 'device_node': node,
            'container_id': inspect['Id'], 'container_PID': inspect['State']['Pid'],
            'container_global_ASCEND_env': env}


def write_container(name, data):
    command = ['docker', 'exec', '-i', '-e', 'ASCEND_RT_VISIBLE_DEVICES=', '-e', 'ASCEND_VISIBLE_DEVICES=',
               CONTAINER, 'python3', '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())', ROOT + '/' + name]
    subprocess.run(command, input=(json.dumps(data, indent=2) + '\n').encode(), check=True)


def main():
    assert not (HOST_ROOT / 'runtime_attempt.json').exists() and not (HOST_ROOT / 'run_receipt.json').exists(), 'No retry'
    pre = snapshot()
    write_container('preflight.json', pre)
    assert pre['physical6_processes'] == [] and pre['health_allowed'] and pre['container_has_dev6_permission']
    command = ['docker', 'exec', '-e', 'ASCEND_RT_VISIBLE_DEVICES=6', '-e', 'ASCEND_VISIBLE_DEVICES=6',
               '-e', 'TASK_QUEUE_ENABLE=0', '-e', 'ASCEND_GLOBAL_LOG_LEVEL=0', '-e', 'ASCEND_SLOG_PRINT_TO_STDOUT=1',
               CONTAINER, 'python3', ROOT + '/source/run_gate.py']
    code = 1
    try:
        code = subprocess.run(command, timeout=170).returncode
    finally:
        release = snapshot()
        release['physical6_released'] = release['physical6_processes'] == []
        release['container_global_env_unchanged'] = release['container_global_ASCEND_env'] == pre['container_global_ASCEND_env']
        release['runtime_wrapper_exit'] = code
        write_container('release_fresh_smi.json', release)
        print(json.dumps({k: release[k] for k in ['UTC','physical','physical6_released','health_allowed','container_global_env_unchanged','runtime_wrapper_exit']}), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
