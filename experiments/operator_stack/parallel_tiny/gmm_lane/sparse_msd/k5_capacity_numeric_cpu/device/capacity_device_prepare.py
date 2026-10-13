"""K5 capacity device-stage source preparation and CPU-side kernel compilation.

Runs inside the CANN 9.1.0 container with an empty visible-device guard. For
each capacity profile (target36/E48, draft15/E16) it builds a private candidate
kernel tree (frozen M1 sparse insertion plus capacity geometry), a native
control tree (unmodified original sources), the capacity planner source and the
OPC compile inputs derived from the frozen M1 templates, then compiles planner
(ccec), candidate/native (OPC) and reads the real workspace bytes from the new
kernel metadata. No NPU launch happens here; OPC itself performs a read-only
visible-device enumeration with both guards empty.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parent
SPARSE = LINE.parent
sys.path.insert(0, str(SPARSE))

from k5_capacity_numeric_cpu import profile as KP  # noqa: E402
from k5_capacity_numeric_cpu import source_factory as SF  # noqa: E402

SDK = Path('/usr/local/Ascend/cann-9.1.0')
NATIVE = SDK / 'opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/grouped_matmul'
CCEC = Path('/usr/local/Ascend/ascend-toolkit/latest/bin/ccec')
CCEC_INCLUDES = ['compiler/tikcpp/tikcfw', 'compiler/tikcpp/tikcfw/interface',
                 'compiler/tikcpp/tikcfw/impl', 'compiler/tikcpp/tikcfw/lib',
                 'aarch64-linux/asc/include', 'aarch64-linux/ascendc/include/highlevel_api']
FLAVORS = [('api', 'grouped_matmul_antiquant_a8w4_msd.h'),
           ('vector', 'grouped_matmul_antiquant_a8w4_msd_new.h')]
TEMPLATES = {'candidate': ('compile_input_candidate.json', 'Gmm2SparseMsdDown_candidate_v6'),
             'native': ('compile_input_native.json', 'Gmm2NativeMsdDown_control_v6')}
FRAME_NEEDLE = 'user1 += 128; // extra private workspace prefix; native workspace/tiling is unchanged'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tiling_bytes(header):
    match = re.search(r"const uint8_t __ascendc_arr_##tiling_data\[(\d+)\]\s*=\s*\{([^}]+)\}", header.read_text())
    assert match, 'Native generated constant tiling array missing'
    data = bytes(int(x.strip()) for x in match[2].split(',') if x.strip())
    assert len(data) == int(match[1])
    return data


def run(name, command, cwd, env, receipts):
    log_path = cwd / 'logs' / (name + '.log')
    assert not log_path.exists(), 'Refuse to overwrite prior build log: ' + str(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('wb') as log:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipts[name] = {'command': command, 'exit_code': result.returncode, 'log': str(log_path)}
    print(json.dumps({'stage': name, 'exit_code': result.returncode}), flush=True)
    if result.returncode:
        raise RuntimeError(name + ' failed; see ' + str(log_path))
    return result


def profile_env():
    env = os.environ.copy()
    assert env.get('ASCEND_RT_VISIBLE_DEVICES') == env.get('ASCEND_VISIBLE_DEVICES') == ''
    env.update(ASCEND_HOME_PATH=str(SDK), ASCEND_OPP_PATH=str(SDK / 'opp'), ASCEND_AICPU_PATH=str(SDK))
    env['PYTHONPATH'] = ':'.join([str(SDK / 'python/site-packages'),
                                  str(SDK / 'opp/built-in/op_impl/ai_core/tbe'), env.get('PYTHONPATH', '')])
    env['LD_LIBRARY_PATH'] = ':'.join([str(SDK / 'lib64'),
                                       str(SDK / 'opp/built-in/op_impl/ai_core/tbe/op_tiling'),
                                       env.get('LD_LIBRARY_PATH', '')])
    return env


def candidate_source(role_root, profile, receipts):
    """Frozen M1 prepare_source copy, then capacity geometry on top."""
    assert not (role_root / 'private_impl').exists(), 'Candidate source tree exists'
    env = profile_env()
    run('prepare_source', [sys.executable, str(LINE / 'borrowed/prepare_source.py'), '--root', str(role_root)],
        role_root, env, receipts)
    frozen = SF.frozen_factory()
    geometry = {}
    for flavor, filename in FLAVORS:
        native_text = (NATIVE / filename).read_text()
        expected, math = frozen.change_header(native_text, flavor)
        current = (role_root / 'private_impl/op_kernel' / filename).read_text()
        assert current == expected, 'Frozen prepare_source output is not the M1 insertion for ' + filename
        changed, receipt = SF.change_header(native_text, flavor, profile)
        (role_root / 'private_impl/op_kernel' / filename).write_text(changed)
        geometry[flavor] = {'native_file': str(NATIVE / filename),
                            'native_sha256': hashlib.sha256(native_text.encode()).hexdigest(),
                            'frozen_m1_candidate_sha256': hashlib.sha256(expected.encode()).hexdigest(),
                            'capacity_candidate_sha256': hashlib.sha256(changed.encode()).hexdigest(),
                            'm1_function_body_sha256': math,
                            'capacity_native_function_body_sha256': receipt['native_function_body_sha256_unchanged'],
                            'only_AIC_schedule_geometry_and_frame_extent_changed': True}
    cpp = role_root / 'private_impl/op_kernel/grouped_matmul.cpp'
    text = cpp.read_text()
    assert text.count(FRAME_NEEDLE) == 1, 'Frozen frame prefix is not unique'
    cpp.write_text(text.replace(FRAME_NEEDLE,
                                f'user1 += {profile.frame_bytes}; // capacity frame prefix; native workspace/tiling unchanged'))
    for name in ['grouped_matmul_antiquant_a8w4_msd_pre.h', 'grouped_matmul_antiquant_a8w4_msd_pre_nz.h']:
        assert (role_root / 'private_impl/op_kernel' / name).read_bytes() == (NATIVE / name).read_bytes()
    native = role_root / 'native_control'
    native.mkdir(exist_ok=False)
    shutil.copytree(NATIVE, native / 'op_kernel')
    frontend = (role_root / 'private_impl/grouped_matmul_cpu_frontend.py').read_text()
    needle = 'origin_func_name = "gmm2_sparse_msd"'
    assert frontend.count(needle) == 1
    (native / 'grouped_matmul.py').write_text(frontend.replace(needle, 'origin_func_name = "grouped_matmul"'))
    receipts['capacity_geometry'] = geometry
    receipts['candidate_cpp_sha256'] = sha(cpp)
    receipts['candidate_frame_bytes'] = profile.frame_bytes
    return geometry


def compile_inputs(role_root, profile, receipts):
    for flavor, (template_name, bin_name) in TEMPLATES.items():
        template = json.loads((LINE / 'borrowed' / template_name).read_text())
        assert template['op_list'][0]['bin_filename'] == bin_name
        parameter = KP.compile_input(template, profile)
        (role_root / f'compile_input_{flavor}.json').write_text(json.dumps(parameter, indent=2) + '\n')
        receipts[f'compile_input_{flavor}'] = {'bytes': (role_root / f'compile_input_{flavor}.json').stat().st_size,
                                              'bin_filename': bin_name,
                                              'x_shape': [profile.capacity, profile.k],
                                              'weight_shape': [profile.experts, profile.n // 64, profile.k // 16, 16, 64]}
    return


def compile_all(role_root, profile, receipts):
    env = profile_env()
    planner_source = role_root / 'planner_capacity.cce'
    planner_source.write_text(SF.planner_source(profile))
    command = [str(CCEC), '-x', 'cce', '-std=c++17', '--cce-aicore-only', '--cce-aicore-arch=dav-c220-vec',
               '-O2', '-c', str(planner_source), '-o', str(role_root / 'planner_cpu_rebuild.o')]
    for rel in CCEC_INCLUDES:
        command += ['-I', str(SDK / rel)]
    run('planner_cpu_rebuild', command, role_root, env, receipts)
    receipts['planner_cpu_rebuild'].update(
        source_sha256=sha(planner_source), object_bytes=(role_root / 'planner_cpu_rebuild.o').stat().st_size,
        object_sha256=sha(role_root / 'planner_cpu_rebuild.o'), target='dav-c220-vec', expected_NPU_ARCH=2201)
    tiling = {}
    for flavor, (_, bin_name) in TEMPLATES.items():
        frontend = role_root / ('private_impl/grouped_matmul_cpu_frontend.py' if flavor == 'candidate'
                                else 'native_control/grouped_matmul.py')
        assert frontend.exists(), frontend
        parameter = role_root / f'compile_input_{flavor}.json'
        command = [sys.executable, str(LINE / 'borrowed/opc_diagnostics.py'), str(frontend),
                   '--input_param=' + str(parameter), '--main_func=grouped_matmul',
                   '--output=' + str(role_root / ('op_build_' + flavor)),
                   '--soc_version=Ascend910_9382', '--op_mode=dynamic', '--deterministic=true',
                   '--op_debug_config=dump_cce', '--log=debug']
        run('compile_' + flavor, command, role_root, env, receipts)
        object_path = role_root / ('op_build_' + flavor) / (bin_name + '.o')
        meta_dir = role_root / ('kernel_meta_' + bin_name) / 'kernel_meta'
        metadata = json.loads((meta_dir / (bin_name + '.json')).read_text())
        workspace = metadata['workspace']
        assert workspace['num'] == 1 and len(workspace['size']) == 1
        header = next(meta_dir.glob('*_tiling_data.h'))
        tiling[flavor] = tiling_bytes(header)
        receipts['compile_' + flavor].update(
            object_bytes=object_path.stat().st_size, object_sha256=sha(object_path),
            tiling_bytes=len(tiling[flavor]), tiling_sha256=hashlib.sha256(tiling[flavor]).hexdigest(),
            tiling_header=str(header), kernel_meta=str(meta_dir / (bin_name + '.json')),
            kernel_name=metadata['kernelName'], block_dim=metadata['blockDim'],
            magic=metadata['magic'], task_ration=metadata.get('taskRation'),
            schedule_mode=metadata.get('schedule_mode'),
            native_workspace_bytes=workspace['size'][0], workspace_type=workspace['type'][0])
    assert tiling['native'] == tiling['candidate'], 'Native/candidate tiling changed'
    receipts['tiling_byte_equality'] = {'passed': True, 'native_candidate_byte_exact': True,
                                        'bytes': len(tiling['candidate']),
                                        'sha256': hashlib.sha256(tiling['candidate']).hexdigest()}
    assert receipts['compile_native']['native_workspace_bytes'] == receipts['compile_candidate']['native_workspace_bytes']
    return receipts['compile_candidate']['native_workspace_bytes']


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--family', choices=['target', 'draft'], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert 'gmm_lane_sparse_msd' in str(root) and SDK.exists() and not root.exists()
    root.mkdir(parents=True)
    (root / 'logs').mkdir()
    profile = KP.bind(args.family, (48 if args.family == 'target' else 16, 2304, 2560),
                      'original_CPU_transposed_INT8', 2304,
                      'k5_capacity_device:' + str(LINE))
    receipts = {'schema': 'k5_capacity_device_prepare_v1', 'CPU_only': True, 'NPU_launches': 0,
                'SDK': str(SDK), 'family': args.family, 'profile': profile.to_dict(),
                'visible_guard': 'ASCEND_RT_VISIBLE_DEVICES=ASCEND_VISIBLE_DEVICES=',
                'actual_numeric_passed': False, 'admission_granted': False,
                'performance_receipt_granted': False, 'formal_receipt_granted': False}
    candidate_source(root, profile, receipts)
    compile_inputs(root, profile, receipts)
    receipts['native_workspace_bytes'] = compile_all(root, profile, receipts)
    receipts['capacity_workspace_bytes'] = receipts['native_workspace_bytes'] + profile.frame_bytes
    (root / 'prepare_receipt.json').write_text(json.dumps(receipts, indent=2) + '\n')
    print(json.dumps({'family': args.family, 'native_workspace_bytes': receipts['native_workspace_bytes'],
                      'capacity_workspace_bytes': receipts['capacity_workspace_bytes'],
                      'object_sha256': {k: receipts['compile_' + k]['object_sha256']
                                        for k in ('native', 'candidate')}}), flush=True)


if __name__ == '__main__':
    main()
