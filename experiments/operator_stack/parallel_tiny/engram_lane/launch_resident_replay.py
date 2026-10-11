"""Prepare a remote-local real Engram replay, then guard any chip7 NPU launch."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fixture-root', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--physical-chip', type=int, choices=(4, 7), required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root, fixtures = args.root.resolve(), args.fixture_root.resolve()
    chip = args.physical_chip
    assert root.name == 'engram_lane_20261011_c' + str(chip)
    assert fixtures.name == 'cpu_fixture_v1'
    assert '/' not in args.job and args.job
    name = 'dsv41-tiny-goal20-20261010-c6'
    source_container = 'dsv41-engram-lane-20261011-c3'
    container = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
    assert container['HostConfig']['Privileged']
    source = json.loads(subprocess.check_output(['docker', 'inspect', source_container]))[0]
    source_mount = next(m for m in source['Mounts'] if m['Destination'] == '/work')
    assert fixtures.parent == Path(source_mount['Source'])
    fixture_receipt = json.loads((fixtures / 'fixture_receipt.json').read_text())
    assert fixture_receipt['cpu_only'] and not fixture_receipt['npu_initialized']
    assert file_digest(fixtures / 'lookup_fixture.pt') == fixture_receipt['fixture_sha256']
    target_fixture = root / 'lookup_fixture.pt'
    if target_fixture.exists():
        assert file_digest(target_fixture) == fixture_receipt['fixture_sha256']
    else:
        shutil.copyfile(fixtures / 'lookup_fixture.pt', target_fixture)
    assert file_digest(target_fixture) == fixture_receipt['fixture_sha256']
    model = root / 'model'
    model.mkdir(exist_ok=True)
    for filename in ['config.json', 'quant_model_weights.safetensors.index.json']:
        target = model / filename
        data = (fixtures / filename).read_bytes()
        if target.exists():
            assert target.read_bytes() == data
        else:
            target.write_bytes(data)
    sources = root / 'native_sources'
    sources.mkdir(exist_ok=True)
    for entry in fixture_receipt['native_sources']:
        filename = Path(entry['path']).name
        data = (fixtures / 'native_sources' / filename).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
        target = sources / filename
        if target.exists():
            assert target.read_bytes() == data
        else:
            target.write_bytes(data)
    weight = model / 'engram_extra.safetensors'
    # docker cp preserves source symlinks unless -L is requested. The formal
    # target tree is not mounted in this existing fallback container.
    if weight.is_symlink():
        weight.unlink()  # Remove only this lane's private copied link.
    if not weight.exists():
        source_file = fixture_receipt['source_model'] + '/engram_extra.safetensors'
        subprocess.run(['docker', 'cp', '-L', source_container + ':' + source_file, str(weight)], check=True)
    assert not weight.is_symlink()
    weight_receipt = json.loads((fixtures / 'weight_receipt.json').read_text())
    assert file_digest(weight) == weight_receipt['file_sha256']
    shutil.copyfile(fixtures / 'fixture_receipt.json', root / 'fixture_receipt.json')
    shutil.copyfile(fixtures / 'weight_receipt.json', root / 'weight_receipt.json')
    prepared = {'timestamp_unix': time.time(), 'container': name, 'container_id': container['Id'],
                'image_digest': container['Image'], 'physical_chip': chip,
                'fixture_sha256': fixture_receipt['fixture_sha256'],
                'weight_file_sha256': weight_receipt['file_sha256'],
                'weight_bytes': weight.stat().st_size, 'model': str(model),
                'scope': 'Pre-read real table rows from original CPU hashes; real BF16 Engram weights; synthetic residual',
                'source_container_used_read_only_cpu': source_container,
                'no_ssh_weight_or_fixture_transfer': True, 'npu_started': False,
                'replay_script_sha256': file_digest(root / 'resident_engram_replay.py')}
    (root / 'prepared.json').write_text(json.dumps(prepared, indent=2) + '\n')
    if args.prepare_only:
        print('ENGRAM_PREPARED', json.dumps(prepared), flush=True)
        return
    # Failed SDK probes cannot be silently promoted to a real replay.
    assert (root / 'init_exit.txt').read_text().strip() == '0', 'A successful assigned-chip SDK init is required'
    node = subprocess.check_output(['docker', 'exec', name, 'ls', '-l', '/dev/davinci' + str(chip)], text=True)
    enumeration = subprocess.check_output(['docker', 'exec', '-e', 'ASCEND_RT_VISIBLE_DEVICES=' + str(chip), name,
                                          'python', '-c', 'import torch,torch_npu; print(torch.npu.device_count())'], text=True)
    assert enumeration.strip() == '1'
    spec = importlib.util.spec_from_file_location('guard', fixtures.parent / 'launch_healthy_probe.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    raw = subprocess.check_output(['npu-smi', 'info'], text=True)
    resources = guard.resources(raw)
    assert resources[chip]['health'] == 'OK' and not resources[chip]['processes'], resources[chip]
    job = root / args.job
    job.mkdir(exist_ok=False)
    (job / 'resource_prelaunch.txt').write_text(raw)
    prefix = '/work/' + root.name
    command = ['python', prefix + '/resident_engram_replay.py', '--model', prefix + '/model',
               '--lookup-fixture', prefix + '/lookup_fixture.pt', '--native-sources', prefix + '/native_sources',
               '--physical-chip', str(chip), '--output', prefix + '/' + args.job + '/replay']
    script = '\n'.join(['#!/usr/bin/env bash', 'set -e', 'set -o pipefail',
        'source /usr/local/Ascend/ascend-toolkit/set_env.sh', 'source /usr/local/Ascend/nnal/atb/set_env.sh',
        'export ASCEND_RT_VISIBLE_DEVICES=' + str(chip) + ' STACK_PHYSICAL_CHIPS=' + str(chip) + ' HCCL_DETERMINISTIC=strict',
        'export V41_ENGRAM_GATE_CHUNK=0 V41_ENGRAM_GATE_HOIST=0 V41_ENGRAM_JIT=1 OMP_NUM_THREADS=1',
        'export NUMBA_CACHE_DIR=' + prefix + '/cache/numba TMPDIR=' + prefix + '/tmp',
        'mkdir -p "$NUMBA_CACHE_DIR" "$TMPDIR"', 'set +e',
        shlex.join(command) + ' > ' + shlex.quote(prefix + '/' + args.job + '/run.log') + ' 2>&1',
        'rc=$?', 'printf "%s\\n" "$rc" > ' + shlex.quote(prefix + '/' + args.job + '/exit_code'),
        'exit "$rc"', ''])
    (job / 'run.sh').write_text(script)
    prepared.update(device_node=node, enumeration=enumeration, assigned_chip_resource=resources[chip],
                    command=command, job=str(job), npu_started=True)
    (job / 'launch_receipt.json').write_text(json.dumps(prepared, indent=2) + '\n')
    subprocess.run(['docker', 'exec', '-d', '-e', 'ASCEND_RT_VISIBLE_DEVICES=' + str(chip), name,
                    'bash', prefix + '/' + args.job + '/run.sh'], check=True)
    print('ENGRAM_REPLAY_LAUNCHED', json.dumps(prepared), flush=True)


if __name__ == '__main__':
    main()
