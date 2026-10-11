"""Main-thread launch gate for the native full40 DSpark A/A baseline."""
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


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ('root', 'source-dir', 'container', 'job', 'model'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--draft-graph', choices=('0', '1'), default='1')
    parser.add_argument('--native-profile-aa-job')
    args = parser.parse_args()
    assert args.job.replace('_', '').replace('-', '').isalnum()
    root, source = Path(args.root).resolve(), Path(args.source_dir)
    assert source.is_absolute() and source.is_relative_to('/work') and '..' not in source.parts
    host_source = root/source.relative_to('/work')
    freeze = json.loads((host_source/'stack/parallel_tiny/dspark_formal_prep/freeze.json').read_text())
    for name, expected in freeze['files'].items():
        assert hashlib.sha256((host_source/'stack'/name).read_bytes()).hexdigest() == expected, name
    if args.native_profile_aa_job:
        assert args.draft_graph == '0'
        assert args.native_profile_aa_job.replace('_', '').replace('-', '').isalnum()
        native_aa = root/'results'/args.native_profile_aa_job
        control = json.loads((native_aa/'result.json').read_text())
        assert (native_aa/'run.exit').read_text().strip() == '0'
        assert control['mode'] == 'audit' and control['native_aa_passed']
        for name, expected in control['source_hashes'].items():
            assert freeze['files'][name] == expected, name
        for name in ('bench.py', 'worker.py', 'phase.py'):
            assert 'parallel_tiny/dspark_native_profile/'+name in freeze['files']
    config = json.loads(subprocess.check_output(['docker', 'inspect', args.container], text=True))[0]
    env = dict(item.split('=', 1) for item in config['Config']['Env'])
    chips = list(range(8, 16))
    visible = ','.join(map(str, chips))
    assert config['State']['Running']
    assert env['STACK_PHYSICAL_CHIPS'] == env['ASCEND_RT_VISIBLE_DEVICES'] == visible
    assert any(m['Destination'] == '/work' and Path(m['Source']).resolve() == root for m in config['Mounts'])
    out = root/'results'/args.job
    setup = ("from pathlib import Path; import os; p=Path("+repr('/work/results/'+args.job)+
             "); p.mkdir(exist_ok=False); os.chown(p,"+str(os.getuid())+","+str(os.getgid())+")")
    subprocess.run(['docker', 'exec', args.container, 'python', '-c', setup], check=True)
    with (out/'launch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = subprocess.check_output(['npu-smi', 'info'], text=True)
        parsed = resources(raw)
        health = {}
        for chip in chips:
            assert parsed[chip]['health'] in ('OK', 'Alarm') and not parsed[chip]['processes'], (chip, parsed[chip])
            if parsed[chip]['health'] == 'Alarm':
                info = subprocess.check_output(['npu-smi', 'info', '-t', 'health', '-i', str(chip//2), '-c', str(chip%2)], text=True)
                assert [line.split(':', 1)[1].strip() for line in info.splitlines() if 'Error Code' in line] == ['80C98001'], info
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
        command = ['bash', str(source/'stack/run_stack.sh'), 'bench_formal_dspark.py',
                   '--model', args.model, '--physical-chips', visible,
                   '--output', '/work/results/'+args.job, '--mode', 'audit',
                   '--pairs', '3', '--speculative-tokens', '5', '--draft-graph', args.draft_graph]
        if args.native_profile_aa_job:
            command = ['bash', str(source/'stack/run_stack.sh'),
                       'parallel_tiny/dspark_native_profile/bench.py', '--model', args.model,
                       '--physical-chips', visible, '--output', '/work/results/'+args.job,
                       '--native-aa-root', '/work/results/'+args.native_profile_aa_job]
        job = '/work/results/'+args.job
        body = '#!/usr/bin/env bash\nset -uo pipefail\numask 022\n'
        body += '\n'.join('export '+key+'='+shlex.quote(value) for key, value in flags.items())+'\n'
        body += 'echo $$ > '+job+'/runner.pid\ndate -u +%FT%TZ > '+job+'/start.utc\n'
        body += shlex.join(command)+' > '+job+'/run.log 2>&1\n'
        body += 'dspark_status=$?\nprintf "%s\\n" "$dspark_status" > '+job+'/run.exit\n'
        body += 'date -u +%FT%TZ > '+job+'/end.utc\nexit "$dspark_status"\n'
        script = root/(args.job+'.sh')
        script.write_text(body)
        receipt = {'timestamp_unix': time.time(), 'container': args.container, 'container_id': config['Id'],
                   'image_digest': config['Image'], 'chips': chips, 'resources': parsed,
                   'alarm_details': health, 'env': flags, 'argv': command, 'source_dir': str(source),
                   'source_hashes': freeze['files'], 'script_sha256': hashlib.sha256(body.encode()).hexdigest(),
                   'scope': 'Native DSpark ON A/A control; no formal performance/consumer pass inferred',
                   'draft_graph_requested': args.draft_graph == '1',
                   'native_profile_aa_job': args.native_profile_aa_job,
                   'target_ms_per_step': 20, 'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (out/'resource_prelaunch.txt').write_text(raw)
        if args.native_profile_aa_job:
            receipt['scope'] = 'Completed native DSpark AA control intrusive CPU/NPU timeline; no formal performance/consumer pass inferred'
        (out/'launched.json').write_text(json.dumps(receipt, indent=2)+'\n')
        subprocess.run(['docker', 'exec', '-d', args.container, 'bash', '/work/'+script.name], check=True)
        marker = 'FORMAL_DSPARK_NATIVE_PROFILE_LAUNCHED' if args.native_profile_aa_job else 'FORMAL_DSPARK_NATIVE_AA_LAUNCHED'
        print(marker, json.dumps({'job': args.job, 'chips': chips, 'source_dir': str(source)}), flush=True)


if __name__ == '__main__':
    main()
