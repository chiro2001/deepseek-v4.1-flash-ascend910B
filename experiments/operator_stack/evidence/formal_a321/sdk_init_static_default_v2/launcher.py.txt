"""Launch one new TP8 SDK case after normal worker initialization, without weights."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from launch_healthy_probe import resources


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ('root', 'source-dir', 'container', 'job', 'model', 'driver'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--variant', choices=('static_default', 'nonstatic_default',
                                             'static_keep', 'nonstatic_keep'),
                        default='static_default')
    args = parser.parse_args()
    assert args.job.replace('_', '').replace('-', '').isalnum()
    root, source = Path(args.root).resolve(), Path(args.source_dir)
    assert source.is_absolute() and source.is_relative_to('/work') and '..' not in source.parts
    driver = Path(args.driver)
    assert not driver.is_absolute() and '..' not in driver.parts
    host_source = root / source.relative_to('/work')
    manifest_path = host_source / 'initialized_sdk_snapshot.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['scope'] == 'normal-init-only SDK capability; no model or performance receipt'
    assert manifest['driver'] == str(driver)
    assert manifest['formal_checkpoint_config_sha256'] == '40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424'
    assert manifest['files'] and ('stack/' + str(driver)) in manifest['files']
    for name, expected in manifest['files'].items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert digest(host_source / relative) == expected, name
    config_program = ('from pathlib import Path; import hashlib,sys; '
                      'print(hashlib.sha256((Path(sys.argv[1])/"config.json").read_bytes()).hexdigest())')
    actual_config = subprocess.check_output(['docker', 'exec', args.container, 'python3',
                                            '-c', config_program, args.model], text=True).strip()
    assert actual_config == manifest['formal_checkpoint_config_sha256']
    container = json.loads(subprocess.check_output(['docker', 'inspect', args.container], text=True))[0]
    container_env = dict(item.split('=', 1) for item in container['Config']['Env'])
    chips = list(range(8, 16))
    visible = ','.join(map(str, chips))
    assert container['State']['Running']
    assert container_env['STACK_PHYSICAL_CHIPS'] == container_env['ASCEND_RT_VISIBLE_DEVICES'] == visible
    assert any(m['Destination'] == '/work' and Path(m['Source']).resolve() == root for m in container['Mounts'])
    out = root / 'results' / args.job
    setup = ('from pathlib import Path; import os; p=Path(' + repr('/work/results/' + args.job)
             + '); p.mkdir(exist_ok=False); os.chown(p,' + str(os.getuid()) + ',' + str(os.getgid()) + ')')
    subprocess.run(['docker', 'exec', args.container, 'python3', '-c', setup], check=True)
    with (out / 'launch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = subprocess.check_output(['npu-smi', 'info'], text=True)
        parsed = resources(raw)
        health = {}
        for chip in chips:
            assert parsed[chip]['health'] in ('OK', 'Alarm') and not parsed[chip]['processes'], (chip, parsed[chip])
            info = subprocess.check_output(['npu-smi', 'info', '-t', 'health',
                                            '-i', str(chip // 2), '-c', str(chip % 2)], text=True)
            codes = [line.split(':', 1)[1].strip() for line in info.splitlines() if 'Error Code' in line]
            assert codes == (['80C98001'] if parsed[chip]['health'] == 'Alarm' else ['NA']), (chip, info)
            health[chip] = info
        flags = {
            'STACK_SRC_ROOT': str(source), 'STACK_PHYSICAL_CHIPS': visible,
            'ASCEND_RT_VISIBLE_DEVICES': visible, 'ASCEND_VISIBLE_DEVICES': visible,
            'HCCL_DETERMINISTIC': 'strict', 'HCCL_NPU_SOCKET_PORT_RANGE': 'auto',
            'LOCAL_WORLD_SIZE': '8', 'TASK_QUEUE_ENABLE': '1', 'HCCL_OP_EXPANSION_MODE': 'AIV',
            'HCCL_BUFFSIZE': '1024', 'ASCEND_MAX_OP_CACHE_SIZE': '-1',
            'PYTORCH_NPU_ALLOC_CONF': 'expandable_segments:True',
            'V41_ENGRAM_HOST_RESIDENT': '1', 'V41_ENGRAM_REUSE_EP_GROUP': '1',
            'V41_ENGRAM_GATE_CHUNK': '0', 'V41_ENGRAM_GATE_MAX_TOKENS': '4096',
            'V41_ENGRAM_GATE_HOIST': '0', 'V41_ENGRAM_JIT': '1',
            'V41_ENGRAM_DEVICE_INDEX': '1', 'V41_ENGRAM_DEVICE_FALLBACK': '0',
            'V41_QLI_NO_CANDIDATE': '1', 'V41_MOE_COMM_ALLGATHER': '1',
            'V41_MOE_MASK_RANGE': '1', 'V41_ROPE_IDXSEL': '1', 'V41_O_PROJ_2D': '1',
            'V41_MOE_ZERO_INVALID': '0', 'V41_MOE_ZERO_NONFINITE': '0',
            'V41_ENGRAM_ROUTE_PROBE': '0', 'NUMBA_CACHE_DIR': '/work/cache/numba',
        }
        job_root = '/work/results/' + args.job
        command = ['bash', str(source / 'stack/run_stack.sh'), str(driver),
                   '--model', args.model, '--physical-chips', visible,
                   '--output', job_root + '/gate', '--variant', args.variant]
        body = '#!/usr/bin/env bash\nset -uo pipefail\numask 022\n'
        body += '\n'.join('export ' + key + '=' + shlex.quote(value) for key, value in flags.items()) + '\n'
        body += 'echo $$ > ' + job_root + '/runner.pid\ndate -u +%FT%TZ > ' + job_root + '/start.utc\n'
        body += shlex.join(command) + ' > ' + job_root + '/run.log 2>&1\n'
        body += 'sdk_status=$?\nprintf "%s\\n" "$sdk_status" > ' + job_root + '/run.exit\n'
        body += 'date -u +%FT%TZ > ' + job_root + '/end.utc\nexit "$sdk_status"\n'
        script = root / (args.job + '.sh')
        assert not script.exists()
        script.write_text(body)
        receipt = {'timestamp_unix': time.time(), 'container': args.container,
                   'container_id': container['Id'], 'image_digest': container['Image'],
                   'chips': chips, 'resources': parsed, 'health_details': health,
                   'env': flags, 'argv': command, 'source_dir': str(source),
                   'source_hashes': manifest['files'], 'snapshot_manifest_sha256': digest(manifest_path),
                   'script_sha256': hashlib.sha256(body.encode()).hexdigest(),
                   'scope': manifest['scope'], 'variant': args.variant,
                   'model_weights_requested': False, 'full_consumer_audit_passed': False,
                   'performance_receipt_granted': False, 'formal_performance_claim': None,
                   'launcher_sha256': digest(Path(__file__))}
        (out / 'resource_prelaunch.txt').write_text(raw)
        # Recheck occupancy after the per-chip health reads, immediately before submission.
        fresh = subprocess.check_output(['npu-smi', 'info'], text=True)
        fresh_parsed = resources(fresh)
        assert all(not fresh_parsed[c]['processes'] and
                   fresh_parsed[c]['health'] == parsed[c]['health'] for c in chips), fresh_parsed
        (out / 'resource_presubmit.txt').write_text(fresh)
        (out / 'launched.json').write_text(json.dumps(receipt, indent=2) + '\n')
        subprocess.run(['docker', 'exec', '-d', args.container, 'bash', '/work/' + script.name], check=True)
        print('NATIVE_INITIALIZED_SDK_CASE_LAUNCHED', json.dumps({'job': args.job,
              'chips': chips, 'variant': args.variant, 'source_dir': str(source)}), flush=True)


if __name__ == '__main__':
    main()
