"""physical6-only minimal SDK/raw/candidate numeric gate, no timing or retry."""
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

from numeric_contract import cases, activation, array_sha, array_stats, oracle, bf16_stats, compare_bf16
from weights_cpu import load_weights, nz_expert_bytes

ROOT = Path('/work/gmm_lane_sparse_msd_m1_numeric_p6_20261011_v5')
FROZEN = Path('/work/gmm_lane_sparse_msd_build_20261011_v3_repro')
V2 = Path('/work/gmm_lane_sparse_msd_runtime_loader_v2_cpu_20261011')
V3 = Path('/work/gmm_lane_sparse_msd_launch_args_v3_cpu_20261011')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def checked(stage, code):
    emit({'stage': stage, 'code': int(code)})
    if code: raise RuntimeError(f'{stage}: SDK/bridge returned {code}')


def frozen_guard():
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == os.environ.get('ASCEND_VISIBLE_DEVICES') == '6'
    assert os.environ.get('TASK_QUEUE_ENABLE') == '0'
    preflight = json.loads((ROOT / 'preflight.json').read_text())
    assert preflight['physical'] == 6 and preflight['physical6_processes'] == []
    assert preflight['health_allowed'] and preflight['container_has_dev6_permission']
    age = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(preflight['UTC'])).total_seconds()
    assert 0 <= age <= 30
    assert not (ROOT / 'runtime_terminal.json').exists(), 'No runtime retry'
    build = json.loads((ROOT / 'build_receipt.json').read_text())
    assert build['exit_code'] == 0 and build['shared_sha256'] == digest(ROOT / 'libm1_numeric_bridge.so')
    frozen = json.loads((ROOT / 'frozen_dependencies.json').read_text())
    for record in frozen['frozen_v3']['objects'].values(): assert digest(record['path']) == record['sha256']
    for name, expected in frozen['frozen_v3']['source_sha256'].items(): assert digest(FROZEN / 'source' / name) == expected
    for label, library, sources in [
        ('loader_v2', V2 / 'build/libgmm2_sparse_msd_loader_v2.so', V2 / 'source/runtime_loader_v2'),
        ('launch_v3', V3 / 'build/libgmm2_launch_args_v3.so', V3 / 'source')]:
        assert digest(library) == frozen[label]['shared_sha256']
        for name, expected in frozen[label]['source_sha256'].items(): assert digest(sources / name) == expected
    for name, expected in json.loads((ROOT / 'source_freeze.json').read_text()).items():
        assert digest(ROOT / 'source' / name) == expected
    for path, record in json.loads((ROOT / 'consumer_source_manifest.json').read_text())['sources'].items():
        assert digest(path) == record['sha256']
    emit({'frozen_dependencies_guard': True, 'preflight_age_seconds': age,
          'physical': 6, 'synthetic_activation': True, 'full_model_and_timing': False})


def bridge_api():
    lib = ctypes.CDLL(str(ROOT / 'libm1_numeric_bridge.so'))
    u64, u32, ptr = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p
    lib.m1_probe_mapping.argtypes = []; lib.m1_probe_mapping.restype = ctypes.c_int
    lib.m1_probe_open.argtypes = [ctypes.c_char_p, u64, ctypes.POINTER(ptr)]; lib.m1_probe_open.restype = ctypes.c_int
    lib.m1_probe_read.argtypes = [u64, ptr, u64]; lib.m1_probe_read.restype = ctypes.c_int
    lib.m1_probe_run.argtypes = [ptr, u64, u64, u64, u64, u64, u64, u32, ptr, ptr]; lib.m1_probe_run.restype = ctypes.c_int
    lib.m1_probe_close.argtypes = [ctypes.POINTER(ptr)]; lib.m1_probe_close.restype = ctypes.c_int
    return lib


def parameter_storage(lib, weight, scale, assist, cpu_weight, manifest):
    full = hashlib.sha256(); expert_sha = []
    for expert in range(48):
        expected = nz_expert_bytes(cpu_weight, expert)
        host = ctypes.create_string_buffer(len(expected))
        checked('D2H:weight_NZ_expert', lib.m1_probe_read(weight.data_ptr() + expert * len(expected), host, len(expected)))
        actual = host.raw
        assert actual == expected, f'NZ physical byte layout mismatch expert{expert}'
        full.update(actual); expert_sha.append(hashlib.sha256(actual).hexdigest())
    assert full.hexdigest() == manifest['processed']['expected_physical_NZ']['sha256']
    other = {}
    for name, tensor in [('scale', scale), ('assist', assist)]:
        size = manifest['processed'][name]['bytes']
        host = ctypes.create_string_buffer(size)
        checked('D2H:' + name, lib.m1_probe_read(tensor.data_ptr(), host, size))
        other[name] = hashlib.sha256(host.raw).hexdigest()
        assert other[name] == manifest['processed'][name]['sha256']
    result = {'weight_physical_NZ_byte_exact': True, 'weight_physical_NZ_sha256': full.hexdigest(),
              'weight_expert_sha256': expert_sha, 'scale_and_assist_physical_sha256': other,
              'weight_shape': list(weight.shape), 'weight_stride': list(weight.stride()),
              'weight_dtype': str(weight.dtype), 'gm_addresses': {'weight': weight.data_ptr(),
              'scale': scale.data_ptr(), 'assist': assist.data_ptr()},
              'all_parameters_original_loader': True}
    (ROOT / 'parameter_storage_result.json').write_text(json.dumps(result, indent=2) + '\n')
    emit(result)


def run():
    frozen_guard()
    cpu_weight, cpu_scale, cpu_assist, manifest = load_weights()
    expected = json.loads((ROOT / 'weights_source_manifest.json').read_text())
    assert manifest['raw_tensor_records'] == expected['raw_tensor_records']
    assert manifest['processed'] == expected['processed']
    for path, source in manifest['source_files'].items():
        assert all(expected['source_files'][path][k] == v for k, v in source.items())
    import torch
    import torch_npu
    lib = bridge_api(); owner = ctypes.c_void_p()
    done = []; current_entry = None
    sdk_attempts = native_attempts = candidate_attempts = planner_attempts = 0
    terminal = {'passed': False, 'scope': 'synthetic INT8/FP32 activation, real original-loader E48 W2, M1 capacity6',
                'physical': 6, 'logical': 0, 'performance_measured': False, 'full_model_executed': False,
                'K5_verifier_coverage': False, 'explicit_reset_or_empty_cache_calls': 0,
                'Torch_context_and_stream_borrowed': True, 'container_PID': os.getpid()}
    try:
        assert torch.npu.device_count() == 1
        # Exact production model_runner_v1.py module setting. This isolated
        # process omitted it in v1; that default disables requested NZ casts.
        torch.npu.config.allow_internal_format = True
        getter = getattr(torch_npu._C, '_npu_getOption', None)
        option = getter('ALLOW_INTERNAL_FORMAT') if getter else None
        if isinstance(option, bytes): option = option.decode()
        emit({'production_allow_internal_format': True, 'SDK_option_observed': option,
              'production_setting_source': '/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'})
        torch.npu.set_device(0)
        checked('mapping_before_weight_device_allocation', lib.m1_probe_mapping())
        stream = torch.npu.current_stream()
        emit({'torch_version': torch.__version__, 'torch_npu_version': torch_npu.__version__,
              'borrowed_stream': int(stream.npu_stream), 'TASK_QUEUE_ENABLE': '0'})
        # Original modelslim loader: transpose on CPU was frozen above; NZ cast
        # and INT32 view are the exact original loading-time operations.
        stages = []
        def layout_snapshot(stage, tensor):
            value = {'stage': stage, 'shape': list(tensor.shape), 'stride': list(tensor.stride()),
                     'dtype': str(tensor.dtype), 'format': torch_npu.get_npu_format(tensor),
                     'storage_bytes': tensor.untyped_storage().nbytes(),
                     'storage_offset': tensor.storage_offset(), 'GM_data_ptr': tensor.data_ptr(),
                     'is_contiguous': tensor.is_contiguous()}
            stages.append(value); emit({'original_loader_layout_stage': value})
            (ROOT / 'loader_layout_stages.json').write_text(json.dumps(stages, indent=2) + '\n')
        weight_nd_device = cpu_weight.to('npu:0')
        layout_snapshot('CPU_transposed_to_NPU_ND', weight_nd_device)
        weight_nz_device = torch_npu.npu_format_cast(weight_nd_device, 29)
        layout_snapshot('original_single_NZ_cast', weight_nz_device)
        weight_int32_view = weight_nz_device.view(torch.int32)
        layout_snapshot('original_view_dtype_INT32', weight_int32_view)
        weight = weight_int32_view.contiguous()
        layout_snapshot('original_contiguous', weight)
        del weight_nd_device, weight_nz_device, weight_int32_view
        scale = cpu_scale.to('npu:0'); assist = cpu_assist.to('npu:0')
        torch.npu.synchronize()
        assert list(weight.shape) == [48, 2304, 640]
        assert torch_npu.get_npu_format(weight) == 29, stages
        assert weight.untyped_storage().nbytes() >= expected['processed']['weight_ND']['bytes']
        checked('open_raw_probe_on_original_SDK_stream', lib.m1_probe_open(str(FROZEN).encode(), stream.npu_stream, ctypes.byref(owner)))
        parameter_storage(lib, weight, scale, assist, cpu_weight, expected)
        del cpu_weight, cpu_scale, cpu_assist
        emit({'live_weight_bytes': weight.numel() * 4, 'live_scale_bytes': scale.numel() * 8,
              'live_assist_bytes': assist.numel() * 4,
              'torch_memory_allocated': torch.npu.memory_allocated(), 'torch_memory_reserved': torch.npu.memory_reserved(),
              'no_weight_duplicate_retained_by_probe': True})
        for index, (name, cpu_counts) in enumerate(cases()):
            cpu_x, cpu_token = activation(index)
            rows = int(cpu_counts.sum())
            x = torch.from_numpy(cpu_x).to('npu:0')
            token = torch.from_numpy(cpu_token).to('npu:0')
            counts = torch.from_numpy(cpu_counts).to('npu:0')
            torch.npu.synchronize()
            entry = {'case_index': index, 'name': name, 'R': rows, 'row_capacity': 6, 'group_list_type': 1,
                     'counts': cpu_counts.tolist(), 'active_experts': list(map(int, cpu_counts.nonzero()[0])),
                     'synthetic_activation': True, 'x_sha256': array_sha(cpu_x), 'x_stats': array_stats(cpu_x),
                     'token_scale_sha256': array_sha(cpu_token), 'token_scale_stats': array_stats(cpu_token),
                     'counts_sha256': array_sha(cpu_counts), 'weight_physical_NZ_sha256': expected['processed']['expected_physical_NZ']['sha256'],
                     'scale_sha256': expected['processed']['scale']['sha256'], 'assist_sha256': expected['processed']['assist']['sha256'],
                     'SDK_gm_addresses': {'x': x.data_ptr(), 'weight': weight.data_ptr(), 'assist': assist.data_ptr(),
                                          'scale': scale.data_ptr(), 'counts': counts.data_ptr(), 'token_scale': token.data_ptr()},
                     'borrowed_stream': int(stream.npu_stream)}
            current_entry = entry
            lifecycle = []; entry['x_input_lifecycle'] = lifecycle
            same_x_ptr = x.data_ptr()
            def capture_x(stage, restored=False):
                host = ctypes.create_string_buffer(cpu_x.nbytes)
                checked('D2H:x:' + stage, lib.m1_probe_read(x.data_ptr(), host, cpu_x.nbytes))
                actual = np.frombuffer(host.raw, dtype=np.int8).reshape(6, 2304).copy()
                item = {'stage': stage, 'GM_ptr': x.data_ptr(), 'same_original_GM_ptr': x.data_ptr() == same_x_ptr,
                        'full_capacity_sha256': array_sha(actual), 'valid_rows_sha256': array_sha(actual[:rows]),
                        'unused_rows_sha256': array_sha(actual[rows:]), 'valid_rows': rows, 'unused_rows': 6 - rows,
                        'valid_stats': array_stats(actual[:rows]),
                        'full_capacity_original_byte_exact': bool(np.array_equal(actual, cpu_x)),
                        'unused_original_byte_exact': bool(np.array_equal(actual[rows:], cpu_x[rows:]))}
                assert item['same_original_GM_ptr'] and item['unused_original_byte_exact']
                if restored: assert item['full_capacity_original_byte_exact']
                lifecycle.append(item); emit({'x_lifecycle': item})
                (ROOT / (name + '_input_lifecycle.json')).write_text(json.dumps(lifecycle, indent=2) + '\n')
            def restore_x(stage):
                x.copy_(torch.from_numpy(cpu_x))
                torch.npu.synchronize()
                capture_x(stage + ':before_original_restored', restored=True)
            readonly_seals = []; entry['readonly_parameter_seals_after_each_call'] = readonly_seals
            def seal_readonly(stage):
                full = hashlib.sha256()
                expert_bytes = 2304 * 5120 // 2
                for expert in range(48):
                    host = ctypes.create_string_buffer(expert_bytes)
                    checked('D2H:readonly_weight_expert', lib.m1_probe_read(weight.data_ptr() + expert * expert_bytes, host, expert_bytes))
                    full.update(host.raw)
                record = {'stage': stage, 'all_E48_weight_sha256': full.hexdigest(), 'sha256': {}}
                assert record['all_E48_weight_sha256'] == expected['processed']['expected_physical_NZ']['sha256']
                for label, tensor, size, target_sha in [
                        ('scale', scale, expected['processed']['scale']['bytes'], expected['processed']['scale']['sha256']),
                        ('assist', assist, expected['processed']['assist']['bytes'], expected['processed']['assist']['sha256']),
                        ('counts', counts, cpu_counts.nbytes, array_sha(cpu_counts)),
                        ('token_scale', token, cpu_token.nbytes, array_sha(cpu_token))]:
                    host = ctypes.create_string_buffer(size)
                    checked('D2H:readonly_' + label, lib.m1_probe_read(tensor.data_ptr(), host, size))
                    record['sha256'][label] = hashlib.sha256(host.raw).hexdigest()
                    assert record['sha256'][label] == target_sha
                readonly_seals.append(record); emit({'readonly_parameter_seal': record})
            emit({'begin_case': entry})
            restore_x('SDK_native')
            sdk_attempts += 1
            reference = torch_npu.npu_grouped_matmul(
                x=[x], weight=[weight], scale=[scale], bias=[assist], per_token_scale=[token],
                split_item=2, group_list_type=1, group_type=0, group_list=counts,
                output_dtype=torch.bfloat16)[0]
            torch.npu.synchronize()
            assert reference.dtype == torch.bfloat16 and list(reference.shape) == [6, 5120]
            ref_bits = reference.cpu().view(torch.uint16).numpy().copy()
            entry['SDK_gm_addresses']['output'] = reference.data_ptr()
            entry['SDK_output'] = bf16_stats(ref_bits[:rows])
            assert entry['SDK_output']['finite']
            ref_bits.tofile(ROOT / f'{name}_sdk_native.bf16.bin')
            capture_x('SDK_native:after_sync')
            seal_readonly('SDK_native:after_sync')
            outputs = []
            for label, sparse in [('raw_native_A1', 0), ('raw_native_A2', 0), ('candidate', 1)]:
                bits = np.empty((6, 5120), dtype=np.uint16); frame = np.zeros(32, dtype=np.uint32)
                if sparse: candidate_attempts += 1; planner_attempts += 1
                else: native_attempts += 1
                restore_x(label)
                checked(label, lib.m1_probe_run(owner, x.data_ptr(), weight.data_ptr(), assist.data_ptr(),
                        scale.data_ptr(), counts.data_ptr(), token.data_ptr(), sparse,
                        bits.ctypes.data, frame.ctypes.data))
                capture_x(label + ':after_sync')
                seal_readonly(label + ':after_sync')
                bits.tofile(ROOT / f'{name}_{label}.bf16.bin')
                entry[label + '_output'] = bf16_stats(bits[:rows])
                assert entry[label + '_output']['finite']
                entry[label + '_unused_rows_sentinel_intact'] = bool((bits[rows:] == 0xa5a5).all())
                assert entry[label + '_unused_rows_sentinel_intact']
                if label == 'raw_native_A1':
                    entry['SDK_native_vs_raw_native'] = compare_bf16(ref_bits[:rows], bits[:rows], 1)
                    assert entry['SDK_native_vs_raw_native']['passed'], entry['SDK_native_vs_raw_native']
                elif label == 'raw_native_A2':
                    entry['raw_native_AA'] = compare_bf16(outputs[0][:rows], bits[:rows], 0)
                    assert entry['raw_native_AA']['byte_exact'], entry['raw_native_AA']
                else:
                    expected_frame = oracle(cpu_counts)
                    entry['planner_frame_words'] = frame.tolist()
                    entry['planner_frame_byte_exact'] = bool(np.array_equal(frame, expected_frame))
                    assert entry['planner_frame_byte_exact']
                    entry['candidate_vs_raw_native'] = compare_bf16(outputs[0][:rows], bits[:rows], 1)
                    entry['candidate_vs_SDK_native'] = compare_bf16(ref_bits[:rows], bits[:rows], 1)
                    assert entry['candidate_vs_raw_native']['passed'] and entry['candidate_vs_SDK_native']['passed']
                    entry['planner_included'] = True
                outputs.append(bits)
            entry['passed'] = True
            (ROOT / f'{name}_result.json').write_text(json.dumps(entry, indent=2) + '\n')
            done.append(entry); emit({'case_passed': name, 'results': entry})
            current_entry = None
        terminal['passed'] = True
    except BaseException as error:
        terminal['failure_type'] = type(error).__name__; terminal['failure'] = str(error)
        terminal['traceback'] = traceback.format_exc()
        terminal['partial_case_result'] = current_entry
        if current_entry is not None:
            (ROOT / (current_entry['name'] + '_partial_failure.json')).write_text(json.dumps(current_entry, indent=2) + '\n')
        emit({'numeric_gate_failure': terminal['failure'], 'stop_all_following_cases': True})
    finally:
        close_code = lib.m1_probe_close(ctypes.byref(owner)) if owner.value else 0
        terminal.update({'bridge_cleanup_code': close_code, 'owner_retained_after_close': bool(owner.value),
                         'SDK_math_launch_attempts': sdk_attempts, 'raw_native_entry_attempts': native_attempts,
                         'candidate_entry_attempts': candidate_attempts, 'planner_expected_entry_attempts': planner_attempts,
                         'cases_passed': len(done), 'case_names_passed': [x['name'] for x in done],
                         'all_case_results': done, 'Torch_resources_release': 'Original SDK owner cleanup at this process exit; no device reset/cache clear'})
        terminal['passed'] &= close_code == 0 and not owner.value
        (ROOT / 'runtime_terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
        emit({'terminal': True, **{k: v for k, v in terminal.items() if k != 'all_case_results'}})
    return 0 if terminal['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(run())
