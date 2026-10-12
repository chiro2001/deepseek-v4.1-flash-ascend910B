"""Launch a source-guarded native DSpark API integration diagnostic, TP8 only.

v3 adds an explicit ``--draft-graph`` switch: the frozen service entry
(``dspark_api_native_diagnostic_v2`` + its bridge) already accepts the flag and
threads it into ``model_kwargs``/``startup_env`` and the counter bridge, so the
source snapshot stays byte-identical while the launch records the arm honestly.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from launch_healthy_probe import resources


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def docker_python(container, code, *args):
    return subprocess.check_output(['docker', 'exec', container, 'python3', '-c', code, *args], text=True)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    for key in ('root', 'source-dir', 'container', 'job', 'model', 'workload'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--draft-graph', action='store_true')
    p.add_argument('--extra-env', action='append', default=[],
                   help='KEY=VALUE exported in the job script (repeatable; recorded in the receipt)')
    a = p.parse_args()
    extra = {}
    for item in a.extra_env:
        assert '=' in item, item
        key, value = item.split('=', 1)
        assert key and key.replace('_', '').isalnum() and value, item
        extra[key] = value
    assert a.job.replace('_', '').isalnum() and 1024 <= a.port <= 65535
    root, source = Path(a.root).resolve(), Path(a.source_dir)
    assert source.is_absolute() and source.is_relative_to('/work') and '..' not in source.parts
    host_source = root / source.relative_to('/work')
    manifest_path = host_source / 'native_api_snapshot.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['scope'] == 'native DSpark real API integration diagnostic; no full-consumer or performance admission'
    for rel, expected in manifest['files'].items():
        assert sha(host_source / rel) == expected, rel
    actual = json.loads(docker_python(a.container,
        'import hashlib,json,sys; from pathlib import Path; '
        'print(json.dumps([hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sys.argv[1:]]))',
        a.model + '/config.json', a.workload))
    assert actual == [manifest['checkpoint_config_sha256'],
                      '8c6a1a4ea2acbb5c72706a21870ac25fd71770d2af4c26b81238ccf043d82572']
    container = json.loads(subprocess.check_output(['docker', 'inspect', a.container], text=True))[0]
    env = dict(item.split('=', 1) for item in container['Config']['Env'])
    visible, chips = '8,9,10,11,12,13,14,15', list(range(8, 16))
    assert container['State']['Running'] and env['ASCEND_RT_VISIBLE_DEVICES'] == env['STACK_PHYSICAL_CHIPS'] == visible
    assert any(m['Destination'] == '/work' and Path(m['Source']).resolve() == root for m in container['Mounts'])
    docker_python(a.container, 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()', str(a.port))
    job = '/work/results/' + a.job
    docker_python(a.container, 'from pathlib import Path; import os,sys; p=Path(sys.argv[1]); '
        'p.mkdir(exist_ok=False); os.chown(p,int(sys.argv[2]),int(sys.argv[3]))', job, str(os.getuid()), str(os.getgid()))
    out = root / 'results' / a.job
    raw = subprocess.check_output(['npu-smi', 'info'], text=True)
    parsed = resources(raw)
    health = {}
    for chip in chips:
        assert parsed[chip]['health'] in ('OK', 'Alarm') and not parsed[chip]['processes'], (chip, parsed[chip])
        text = subprocess.check_output(['npu-smi', 'info', '-t', 'health', '-i', str(chip // 2), '-c', str(chip % 2)], text=True)
        codes = [line.split(':', 1)[1].strip() for line in text.splitlines() if 'Error Code' in line]
        assert codes == (['80C98001'] if parsed[chip]['health'] == 'Alarm' else ['NA']), (chip, codes)
        health[chip] = text
    flags = {'STACK_SRC_ROOT': str(source), 'STACK_PHYSICAL_CHIPS': visible,
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
        'V41_ENGRAM_ROUTE_PROBE': '0', 'NUMBA_CACHE_DIR': '/work/cache/numba'}
    served = 'dsv41-dspark-private-' + a.job
    flags.update(extra)
    command = ['bash', str(source / 'stack/run_stack.sh'),
        manifest['entry_file'], '--model', a.model,
        '--served-model', served, '--port', str(a.port), '--output', job + '/service',
        '--counter-file', str(source / 'stack/formal_dspark_counters.py')]
    if a.draft_graph:
        command.append('--draft-graph')
    command.append('--execute')
    body = '#!/usr/bin/env bash\nset -uo pipefail\numask 022\n'
    body += '\n'.join('export ' + key + '=' + shlex.quote(value) for key, value in flags.items()) + '\n'
    body += 'echo $$ > ' + job + '/runner.pid\ndate -u +%FT%TZ > ' + job + '/start.utc\n'
    body += shlex.join(command) + ' > ' + job + '/run.log 2>&1\n'
    body += 'api_status=$?\nprintf "%s\\n" "$api_status" > ' + job + '/run.exit\nexit "$api_status"\n'
    script = root / (a.job + '.sh')
    assert not script.exists()
    script.write_text(body)
    fresh = resources(subprocess.check_output(['npu-smi', 'info'], text=True))
    assert all(not fresh[c]['processes'] and fresh[c]['health'] == parsed[c]['health'] for c in chips)
    receipt = {'scope': manifest['scope'], 'timestamp_unix': time.time(), 'chips': chips,
        'container': a.container, 'container_id': container['Id'], 'source_dir': str(source),
        'snapshot_manifest_sha256': sha(manifest_path), 'source_hashes': manifest['files'],
        'resources': parsed, 'health_details': health, 'argv': command, 'env': flags,
        'port': a.port, 'served_model': served, 'workload': a.workload,
        'audit': False, 'profiler': False, 'draft_graph': bool(a.draft_graph), 'dspark': True,
        'full_consumer_audit_passed': False, 'performance_receipt_granted': False,
        'formal_receipt_granted': False, 'script_sha256': hashlib.sha256(body.encode()).hexdigest()}
    (out / 'launched.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (out / 'resource_prelaunch.txt').write_text(raw)
    subprocess.run(['docker', 'exec', '-d', a.container, 'bash', '/work/' + script.name], check=True)
    print(json.dumps({'job': a.job, 'port': a.port, 'served_model': served,
                      'draft_graph': bool(a.draft_graph), 'scope': manifest['scope']}), flush=True)


if __name__ == '__main__':
    main()
