"""Run on the chosen host; refuse unhealthy/busy devices or existing names."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def resources(raw):
    devices = {}
    addresses = {}
    npu = health = None
    for line in raw.splitlines():
        match = re.match(r'^\|\s*(\d+)\s+Ascend\S*\s*\|\s*(\S+)', line)
        if match:
            npu, health = int(match[1]), match[2]
        physical = re.match(r'^\|\s*(\d+)\s+(\d+)\s*\|\s*[0-9A-Fa-f]{4}:', line)
        if physical:
            chip, physical_id = map(int, physical.groups())
            assert npu is not None and health is not None
            devices[physical_id] = {'health': health, 'processes': []}
            addresses[npu, chip] = physical_id
        process = re.match(r'^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*(\S+)', line)
        if process:
            board, chip, pid = map(int, process.groups()[:3])
            physical_id = addresses[board, chip]
            devices[physical_id]['processes'].append({'pid': pid, 'name': process[4]})
    assert devices, 'Cannot parse npu-smi device table; refusing allocation'
    return devices


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--chips', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--models-root', default='/home/l00886679/models')
    parser.add_argument('--model-mount-dir', action='append', default=[],
                        help='Additional literal symlink-hop directories from tools/model_mount_args.sh')
    parser.add_argument('--privileged-profile', action='store_true',
                        help='Repository standard NPU profiling runtime; selected physical chips still filtered by ACL')
    parser.add_argument('--allow-alarm', action='store_true',
                        help='Explicit authorization to use Alarm chips; occupancy is still rejected')
    parser.add_argument('--model-rw-dir', action='append', default=[],
                        help='Engram table directories requiring writable host-registration mappings')
    args = parser.parse_args()
    chips = [int(c) for c in args.chips.split(',')]
    assert chips and len(chips) == len(set(chips)) and min(chips) >= 0
    raw = subprocess.check_output(['npu-smi', 'info'], text=True)
    parsed = resources(raw)
    alarm_details = {}
    for chip in chips:
        allowed = ['OK', 'Alarm'] if args.allow_alarm else ['OK']
        assert chip in parsed and parsed[chip]['health'] in allowed, (chip, parsed.get(chip))
        assert not parsed[chip]['processes'], (chip, parsed[chip])
        if parsed[chip]['health'] == 'Alarm':
            details = subprocess.check_output(['npu-smi','info','-t','health','-i',str(chip//2),'-c',str(chip%2)],text=True)
            codes = [line.split(':', 1)[1].strip() for line in details.splitlines()
                     if 'Error Code' in line and ':' in line]
            assert codes == ['80C98001'], ('Unexpected Alarm requires investigation', chip, details)
            alarm_details[chip] = details
    existing = subprocess.run(['docker', 'container', 'inspect', args.name], capture_output=True)
    assert existing.returncode != 0, 'Container exists; refusing to replace it'
    digest = subprocess.check_output(['docker', 'image', 'inspect', args.image,
                                     '--format', '{{.Id}}'], text=True).strip()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'resource_prelaunch.txt').write_text(raw)
    command = ['docker', 'run', '-d', '--name', args.name, '--network', 'host',
               '--shm-size', '16g', '--ulimit', 'memlock=-1',
               '--label', 'task=' + args.name, '--label', 'chips=' + args.chips]
    if args.privileged_profile:
        command += ['--privileged=true']
    for chip in chips:
        command += ['--device', '/dev/davinci' + str(chip)]
    for path in ['/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc']:
        assert Path(path).exists(), path
        command += ['--device', path]
    for path in ['/usr/local/Ascend/driver', '/usr/local/Ascend/firmware', '/usr/local/dcmi', '/usr/local/bin/npu-smi',
                 '/etc/ascend_install.info', '/etc/hccn.conf']:
        if Path(path).exists():
            command += ['-v', path + ':' + path + ':ro']
    for directory in args.model_mount_dir:
        assert Path(directory).is_absolute() and Path(directory).is_dir(), directory
        if directory not in args.model_rw_dir:
            command += ['-v', directory + ':' + directory + ':ro']
    # With device cgroups restricting the host nodes, ACL enumerates the
    # accessible chips as logical 0..N-1. Physical IDs stay in a separate map.
    logical = args.chips if args.privileged_profile else ','.join(str(i) for i in range(len(chips)))
    command += ['-v', str(root) + ':/work', '-v', args.models_root + ':' + args.models_root + ':ro',
                '-e', 'ASCEND_RT_VISIBLE_DEVICES=' + logical,
                '-e', 'STACK_EXPECTED_CHIPS=' + logical,
                '-e', 'STACK_PHYSICAL_CHIPS=' + args.chips,
                '-e', 'HCCL_CONNECT_TIMEOUT=120']
    for directory in args.model_rw_dir:
        assert Path(directory).is_absolute() and Path(directory).is_dir(), directory
        command += ['-v', directory + ':' + directory + ':rw']
    command += ['--entrypoint', '/bin/bash', args.image, '-lc', 'sleep infinity']
    container_id = subprocess.check_output(command, text=True).strip()
    result = {'host': subprocess.check_output(['hostname'], text=True).strip(),
              'timestamp_unix': time.time(), 'chips': chips, 'resources': parsed,
              'name': args.name, 'image': args.image, 'image_digest': digest,
              'container_id': container_id, 'command': command,
              'alarm_explicitly_allowed': args.allow_alarm, 'alarm_details': alarm_details,
              'claim': 'isolated operator probe; not a TP8 deployment'}
    (root / 'allocation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
