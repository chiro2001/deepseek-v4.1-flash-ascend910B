"""Compile/link the isolated bridge and freeze CPU preparation, no NPU calls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/work/gmm_lane_sparse_msd_m1_numeric_p6_20261011_v2')
FROZEN = Path('/work/gmm_lane_sparse_msd_build_20261011_v3_repro')
V2 = Path('/work/gmm_lane_sparse_msd_runtime_loader_v2_cpu_20261011')
V3 = Path('/work/gmm_lane_sparse_msd_launch_args_v3_cpu_20261011')
SDK = Path('/usr/local/Ascend/cann-9.1.0')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == os.environ.get('ASCEND_VISIBLE_DEVICES') == ''
    assert not (ROOT / 'build_receipt.json').exists()
    command = ['g++', '-std=c++17', '-O2', '-shared', '-fPIC', str(ROOT / 'source/probe_bridge.cpp'),
               '-I', str(V2 / 'source/runtime_loader_v2'), '-I', str(V3 / 'source'),
               '-I', str(SDK / 'include'), '-I', str(SDK / 'aarch64-linux/pkg_inc/runtime'),
               '-I', str(SDK / 'aarch64-linux/pkg_inc'),
               '-L', str(FROZEN), '-lgmm2_sparse_msd_probe',
               '-L', str(V2 / 'build'), '-lgmm2_sparse_msd_loader_v2',
               '-L', str(V3 / 'build'), '-lgmm2_launch_args_v3',
               '-L', str(SDK / 'lib64'), '-lascendcl', '-lruntime', '-Wl,--no-undefined',
               '-Wl,-rpath,' + ':'.join(map(str, [FROZEN, V2 / 'build', V3 / 'build'])),
               '-o', str(ROOT / 'libm1_numeric_bridge.so')]
    with (ROOT / 'compile.log').open('wb') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    receipt = {'CPU_only': True, 'NPU_calls': 0, 'command': command, 'exit_code': result.returncode,
               'source_sha256': sha(ROOT / 'source/probe_bridge.cpp')}
    if result.returncode == 0:
        receipt['shared_sha256'] = sha(ROOT / 'libm1_numeric_bridge.so')
        receipt['shared_bytes'] = (ROOT / 'libm1_numeric_bridge.so').stat().st_size
    (ROOT / 'build_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    assert result.returncode == 0
    old = Path('/work/gmm_lane_sparse_msd_launch_args_v3_npu_p6_20261011_v1/pre_runtime_sha.json')
    (ROOT / 'frozen_dependencies.json').write_bytes(old.read_bytes())
    sources = {p.name: sha(p) for p in sorted((ROOT / 'source').glob('*')) if p.is_file()}
    (ROOT / 'source_freeze.json').write_text(json.dumps(sources, indent=2) + '\n')
    with (ROOT / 'prepare_cpu.log').open('wb') as log:
        result = subprocess.run(['python3', str(ROOT / 'source/prepare_cpu.py')], stdout=log, stderr=subprocess.STDOUT)
    (ROOT / 'prepare_cpu_receipt.json').write_text(json.dumps({'CPU_only': True, 'NPU_calls': 0,
                                                              'exit_code': result.returncode}, indent=2) + '\n')
    assert result.returncode == 0
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
