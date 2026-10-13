"""K5 capacity device numeric gate on the authorized physical4 window.

Runs one borrowed-context process: the original SDK operator
(torch_npu.npu_grouped_matmul, exactly the production A8W4 down path) as the
device reference, the raw native control kernel, and the capacity candidate
(device planner + sparse kernel) for both frozen profiles target36/E48 and
draft15/E16. Fixed inputs come from numeric_contract; the planner frame is
compared byte-exactly against the frozen CPU oracle SHAs of the CPU line, the
BF16 outputs are compared device-internally (SDK vs raw, raw A/A, candidate vs
raw) with the retained <=1 ULP threshold. Illegal-count negative cases run the
planner only (no GMM math). A minimal interleaved event benchmark closes the
run with the same operator-window definition as the M1 event template.
"""
import ctypes
import datetime
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LINE = HERE.parent
ROOT = LINE.parent
sys.path.insert(0, str(ROOT))

from k5_capacity_numeric_cpu import numeric_contract as NC  # noqa: E402
from k5_capacity_numeric_cpu import weights_cpu as WC  # noqa: E402
from k5_capacity_numeric_cpu import profile as KP  # noqa: E402

PHYSICAL = 4
FAMILIES = ['target', 'draft']
FROZEN_EVIDENCE_SHA = 'eb6bf966331f97d5f1f0414929a6ef0cf9bc519a9c71597d4fbe3d68e45dc9a2'
M1_MANIFEST = Path('/work/gmm_lane_sparse_msd_m1_numeric_p6_20261011_v5/weights_source_manifest.json')
TIMING_WARMUP = 3
TIMING_SAMPLES = 10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def load_v5():
    spec = importlib.util.spec_from_file_location('v5_numeric_contract', LINE / 'borrowed/v5_numeric_contract.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V5 = load_v5()
FROZEN = json.loads((LINE / 'evidence/k5_capacity_numeric_cpu_verification_20261012.json').read_text())


def checked(stage, code):
    emit({'stage': stage, 'code': int(code)})
    if code:
        raise RuntimeError(f'{stage}: SDK/bridge returned {code}')


def frozen_guard():
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == os.environ.get('ASCEND_VISIBLE_DEVICES') == str(PHYSICAL)
    assert os.environ.get('TASK_QUEUE_ENABLE') == '0'
    assert os.environ.get('CAPACITY_EXPECTED_PHYSICAL') == str(PHYSICAL)
    assert digest(LINE / 'evidence/k5_capacity_numeric_cpu_verification_20261012.json') == FROZEN_EVIDENCE_SHA
    preflight = json.loads((ROOT / 'preflight.json').read_text())
    assert preflight['physical'] == PHYSICAL and preflight['physical_processes'] == []
    assert preflight['health_allowed'] and preflight['container_has_dev_permission']
    age = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(preflight['UTC'])).total_seconds()
    assert 0 <= age <= 30
    assert not (ROOT / 'runtime_terminal.json').exists(), 'No runtime retry'
    evidence = {'preflight_age_seconds': age, 'frozen_cpu_oracle_sha256': FROZEN_EVIDENCE_SHA,
                'families': {}}
    for family in FAMILIES:
        prepare = json.loads((ROOT / 'build' / family / 'prepare_receipt.json').read_text())
        bridge = json.loads((ROOT / 'build' / family / 'bridge_receipt.json').read_text())
        assert prepare['family'] == bridge['family'] == family
        assert prepare['CPU_only'] and prepare['NPU_launches'] == 0
        assert bridge['exit_code'] == 0
        assert bridge['shared_sha256'] == digest(ROOT / 'build' / family / f'libk5_capacity_{family}.so')
        assert bridge['source_sha256']['probe_bridge.cpp'] == digest(ROOT / 'build' / family / 'bridge/probe_bridge.cpp')
        assert prepare['capacity_workspace_bytes'] == bridge['capacity_workspace_bytes']
        assert bridge['actual_numeric_passed'] is False and bridge['performance_receipt_granted'] is False
        assert bridge['formal_receipt_granted'] is False
        objects = {'planner': (ROOT / 'build' / family / 'planner_cpu_rebuild.o',
                               prepare['planner_cpu_rebuild']['object_sha256']),
                   'candidate': (ROOT / 'build' / family / 'op_build_candidate/Gmm2SparseMsdDown_candidate_v6.o',
                                 prepare['compile_candidate']['object_sha256']),
                   'native': (ROOT / 'build' / family / 'op_build_native/Gmm2NativeMsdDown_control_v6.o',
                              prepare['compile_native']['object_sha256'])}
        for name, (path, expected) in objects.items():
            assert digest(path) == expected, f'{family} {name} object changed after the CPU-stage freeze'
        evidence['families'][family] = {
            'native_workspace_bytes': prepare['native_workspace_bytes'],
            'capacity_workspace_bytes': prepare['capacity_workspace_bytes'],
            'frame_bytes': prepare['profile']['frame_bytes'],
            'object_sha256': {k: prepare['compile_' + k]['object_sha256'] for k in ('native', 'candidate')},
            'planner_object_sha256': prepare['planner_cpu_rebuild']['object_sha256'],
            'tiling_sha256': prepare['tiling_byte_equality']['sha256'],
            'bridge_shared_sha256': bridge['shared_sha256']}
    for name, expected in [('source/capacity_adapter.cpp', digest(LINE / 'capacity_adapter.cpp')),
                           ('source/borrowed/v5_probe_bridge.cpp', digest(LINE / 'borrowed/v5_probe_bridge.cpp'))]:
        evidence[name] = expected
    emit({'frozen_dependencies_guard': True, **evidence})
    return evidence


def bridge_api(family):
    lib = ctypes.CDLL(str(ROOT / 'build' / family / f'libk5_capacity_{family}.so'))
    u64, u32, ptr = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p
    lib.m1_probe_mapping.argtypes = []; lib.m1_probe_mapping.restype = ctypes.c_int
    lib.m1_probe_open.argtypes = [ctypes.c_char_p, u64, ctypes.POINTER(ptr)]; lib.m1_probe_open.restype = ctypes.c_int
    lib.m1_probe_read.argtypes = [u64, ptr, u64]; lib.m1_probe_read.restype = ctypes.c_int
    lib.m1_probe_run.argtypes = [ptr, u64, u64, u64, u64, u64, u64, u32, ptr, ptr]; lib.m1_probe_run.restype = ctypes.c_int
    lib.m1_probe_close.argtypes = [ctypes.POINTER(ptr)]; lib.m1_probe_close.restype = ctypes.c_int
    lib.k5_probe_planner_only.argtypes = [ptr, u64, ptr]; lib.k5_probe_planner_only.restype = ctypes.c_int
    lib.k5_probe_bind.argtypes = [ptr] + [u64] * 7 + [u32]; lib.k5_probe_bind.restype = ctypes.c_int
    lib.k5_probe_launch_only.argtypes = [ptr]; lib.k5_probe_launch_only.restype = ctypes.c_int
    lib.k5_probe_batch_finish.argtypes = [ptr]; lib.k5_probe_batch_finish.restype = ctypes.c_int
    return lib


def frozen_case(name):
    family, suffix = name.split('_', 1)
    for record in FROZEN['profiles'][family]['cases']:
        if record['case'] == name:
            return record
    raise KeyError(name)


def device_cases(profile):
    """The frozen oracle case list, including the all-zero counts case."""
    return list(NC.cases(profile)) + [(profile.family + '_zero_rows',
                                       np.zeros(profile.experts, dtype=np.int64))]


def device_family(family, lib, torch, torch_npu, owner_holder):
    profile = KP.bind(family, (48 if family == 'target' else 16, 2304, 2560),
                      'original_CPU_transposed_INT8', 2304, 'k5_capacity_device_gate:' + str(LINE))
    cpu_weight, cpu_scale, cpu_assist, manifest = WC.load_weights(family)
    result = {'family': family, 'profile': profile.to_dict(), 'cases': [], 'negatives': [], 'timing': {},
              'device_physical': PHYSICAL, 'device_logical': 0, 'planner_included': True,
              'performance_receipt_granted': False, 'admission_granted': False, 'formal_receipt_granted': False}
    if family == 'target' and M1_MANIFEST.exists():
        m1 = json.loads(M1_MANIFEST.read_text())
        result['M1_frozen_physical_NZ_sha256'] = m1['processed']['expected_physical_NZ']['sha256']
        result['K5_loader_physical_NZ_sha256'] = manifest['processed']['expected_physical_NZ']['sha256']
        result['K5_loader_reproduces_M1_frozen_NZ'] = (
            result['M1_frozen_physical_NZ_sha256'] == result['K5_loader_physical_NZ_sha256'])
        assert result['K5_loader_reproduces_M1_frozen_NZ'], 'K5 W2 loader does not reproduce the frozen M1 NZ bytes'
        for label, key in [('scale', 'scale'), ('assist', 'assist')]:
            assert manifest['processed'][key]['sha256'] == m1['processed'][key]['sha256'], label
    stages = []
    def layout_snapshot(stage, tensor):
        stages.append({'stage': stage, 'shape': list(tensor.shape), 'stride': list(tensor.stride()),
                       'dtype': str(tensor.dtype), 'format': torch_npu.get_npu_format(tensor),
                       'storage_bytes': tensor.untyped_storage().nbytes(), 'GM_data_ptr': tensor.data_ptr()})
        emit({'original_loader_layout_stage': stages[-1]})
    weight_nd_device = cpu_weight.to('npu:0')
    layout_snapshot('CPU_transposed_to_NPU_ND', weight_nd_device)
    weight_nz_device = torch_npu.npu_format_cast(weight_nd_device, 29)
    layout_snapshot('original_single_NZ_cast', weight_nz_device)
    weight = weight_nz_device.view(torch.int32).contiguous()
    layout_snapshot('original_view_dtype_INT32_contiguous', weight)
    del weight_nd_device, weight_nz_device
    scale = cpu_scale.to('npu:0'); assist = cpu_assist.to('npu:0')
    torch.npu.synchronize()
    assert list(weight.shape) == [profile.experts, profile.k, profile.n // 8], weight.shape
    assert torch_npu.get_npu_format(weight) == 29
    result['loader_layout_stages'] = stages
    result['weight_physical_sha256'] = {'K5_loader_expected': manifest['processed']['expected_physical_NZ']['sha256']}
    checked(f'open_bridge:{family}',
            lib.m1_probe_open(str(ROOT / 'build' / family).encode(), torch.npu.current_stream().npu_stream,
                              ctypes.byref(owner_holder[0])))
    weight_bytes = profile.experts * profile.k * profile.n // 2
    full = hashlib.sha256(); expert_sha = []
    for expert in range(profile.experts):
        host = ctypes.create_string_buffer(weight_bytes // profile.experts)
        checked(f'D2H:weight_NZ_expert:{family}',
                lib.m1_probe_read(weight.data_ptr() + expert * (weight_bytes // profile.experts), host,
                                  weight_bytes // profile.experts))
        actual = host.raw
        assert actual == WC.nz_expert_bytes(cpu_weight, expert), f'NZ physical byte layout mismatch expert{expert}'
        full.update(actual); expert_sha.append(hashlib.sha256(actual).hexdigest())
    result['weight_physical_sha256'].update({'device_readback_sha256': full.hexdigest(),
                                             'expert_sha256': expert_sha,
                                             'device_matches_original_loader_bytes': True})
    assert full.hexdigest() == manifest['processed']['expected_physical_NZ']['sha256']
    emit({'family': family, 'weight_physical_NZ_byte_exact': True, 'sha256': full.hexdigest()})
    for name, tensor, size, expected in [('scale', scale, manifest['processed']['scale']['bytes'],
                                          manifest['processed']['scale']['sha256']),
                                         ('assist', assist, manifest['processed']['assist']['bytes'],
                                          manifest['processed']['assist']['sha256'])]:
        host = ctypes.create_string_buffer(size)
        checked(f'D2H:{name}:{family}', lib.m1_probe_read(tensor.data_ptr(), host, size))
        assert hashlib.sha256(host.raw).hexdigest() == expected, name
    del cpu_weight
    emit({'family': family, 'live_device_parameter_bytes': weight_bytes + profile.experts * profile.n * 12,
          'torch_memory_allocated': torch.npu.memory_allocated()})
    for index, (name, cpu_counts) in enumerate(device_cases(profile)):
        case = frozen_case(name)
        cpu_x, cpu_token = NC.activation(profile, index)
        rows = int(cpu_counts.sum())
        assert NC.array_sha(cpu_counts) == case['counts_sha256']
        assert NC.array_sha(cpu_x) == case['x_sha256'] and NC.array_sha(cpu_token) == case['token_sha256']
        assert rows == case['counts_sum'] and case['activation_index'] == index
        expected_frame = NC.oracle(profile, cpu_counts)
        entry = {'case_index': index, 'name': name, 'R': rows, 'row_capacity': profile.capacity,
                 'experts': profile.experts, 'group_list_type': 1, 'counts': cpu_counts.tolist(),
                 'active_experts': list(map(int, cpu_counts.nonzero()[0])), 'synthetic_activation': True,
                 'frozen_oracle_frame_sha256': case['oracle_frame_sha256'],
                 'x_sha256': case['x_sha256'], 'token_scale_sha256': case['token_sha256'],
                 'counts_sha256': case['counts_sha256'],
                 'weight_physical_NZ_sha256': manifest['processed']['expected_physical_NZ']['sha256']}
        x = torch.from_numpy(cpu_x).to('npu:0')
        token = torch.from_numpy(cpu_token).to('npu:0')
        counts = torch.from_numpy(cpu_counts).to('npu:0')
        torch.npu.synchronize()
        entry['SDK_gm_addresses'] = {'x': x.data_ptr(), 'weight': weight.data_ptr(), 'assist': assist.data_ptr(),
                                     'scale': scale.data_ptr(), 'counts': counts.data_ptr(),
                                     'token_scale': token.data_ptr()}
        lifecycle = []; entry['x_input_lifecycle'] = lifecycle
        same_x_ptr = x.data_ptr()
        def capture_x(stage, restored=False):
            host = ctypes.create_string_buffer(cpu_x.nbytes)
            checked('D2H:x:' + stage, lib.m1_probe_read(x.data_ptr(), host, cpu_x.nbytes))
            actual = np.frombuffer(host.raw, dtype=np.int8).reshape(profile.capacity, profile.k).copy()
            item = {'stage': stage, 'same_original_GM_ptr': x.data_ptr() == same_x_ptr,
                    'full_capacity_sha256': NC.array_sha(actual), 'valid_rows_sha256': NC.array_sha(actual[:rows]),
                    'unused_rows_sha256': NC.array_sha(actual[rows:]),
                    'unused_original_byte_exact': bool(np.array_equal(actual[rows:], cpu_x[rows:])),
                    'valid_rows': rows, 'unused_rows': profile.capacity - rows}
            if restored:
                item['full_capacity_original_byte_exact'] = bool(np.array_equal(actual, cpu_x))
            lifecycle.append(item); emit({'x_lifecycle': item})
            if restored:
                assert item['full_capacity_original_byte_exact']
            assert item['same_original_GM_ptr'] and item['unused_original_byte_exact']
        def restore_x(stage):
            x.copy_(torch.from_numpy(cpu_x)); torch.npu.synchronize()
            capture_x(stage + ':original_restored', restored=True)
        seals = []; entry['readonly_parameter_seals'] = seals
        def seal_readonly(stage):
            total = hashlib.sha256()
            for expert in range(profile.experts):
                host = ctypes.create_string_buffer(weight_bytes // profile.experts)
                checked('D2H:readonly_weight_expert',
                        lib.m1_probe_read(weight.data_ptr() + expert * (weight_bytes // profile.experts), host,
                                          weight_bytes // profile.experts))
                total.update(host.raw)
            record = {'stage': stage, 'all_weight_sha256': total.hexdigest(), 'sha256': {}}
            assert record['all_weight_sha256'] == manifest['processed']['expected_physical_NZ']['sha256']
            for label, tensor, size, target in [
                    ('scale', scale, manifest['processed']['scale']['bytes'], manifest['processed']['scale']['sha256']),
                    ('assist', assist, manifest['processed']['assist']['bytes'], manifest['processed']['assist']['sha256']),
                    ('counts', counts, cpu_counts.nbytes, NC.array_sha(cpu_counts)),
                    ('token_scale', token, cpu_token.nbytes, NC.array_sha(cpu_token))]:
                host = ctypes.create_string_buffer(size)
                checked('D2H:readonly_' + label, lib.m1_probe_read(tensor.data_ptr(), host, size))
                record['sha256'][label] = hashlib.sha256(host.raw).hexdigest()
                assert record['sha256'][label] == target, label
            seals.append(record); emit({'readonly_parameter_seal': record})
        emit({'begin_case': name, 'family': family})
        restore_x('SDK_native')
        reference = None
        sdk_error = None
        try:
            reference = torch_npu.npu_grouped_matmul(
                x=[x], weight=[weight], scale=[scale], bias=[assist], per_token_scale=[token],
                split_item=2, group_list_type=1, group_type=0, group_list=counts,
                output_dtype=torch.bfloat16)[0]
            torch.npu.synchronize()
            assert reference.dtype == torch.bfloat16 and list(reference.shape) == [profile.capacity, profile.n]
        except BaseException as error:
            sdk_error = {'type': type(error).__name__, 'message': str(error), 'rows': rows}
            emit({'SDK_reference_unavailable': {**sdk_error, 'case': name, 'family': family}})
            if rows != 0:
                raise
        entry['SDK_reference_unavailable'] = sdk_error
        if reference is not None:
            ref_bits = reference.cpu().view(torch.uint16).numpy().copy()
            if rows:
                entry['SDK_output'] = V5.bf16_stats(ref_bits[:rows])
                assert entry['SDK_output']['finite']
        else:
            ref_bits = None
        capture_x('SDK_native:after_sync')
        outputs = []
        for label, sparse in [('raw_native_A1', 0), ('raw_native_A2', 0), ('candidate', 1)]:
            bits = np.empty((profile.capacity, profile.n), dtype=np.uint16)
            frame = np.zeros(profile.frame_words, dtype=np.uint32)
            restore_x(label)
            checked(f'{label}:{family}', lib.m1_probe_run(owner_holder[0], x.data_ptr(), weight.data_ptr(),
                                                          assist.data_ptr(), scale.data_ptr(), counts.data_ptr(),
                                                          token.data_ptr(), sparse, bits.ctypes.data,
                                                          frame.ctypes.data))
            capture_x(label + ':after_sync')
            entry[label + '_unused_rows_sentinel_intact'] = bool((bits[rows:] == 0xa5a5).all())
            assert entry[label + '_unused_rows_sentinel_intact']
            if rows:
                entry[label + '_output'] = V5.bf16_stats(bits[:rows])
                assert entry[label + '_output']['finite']
            if label == 'raw_native_A1':
                if ref_bits is not None and rows:
                    entry['SDK_native_vs_raw_native'] = V5.compare_bf16(ref_bits[:rows], bits[:rows], 1)
                    assert entry['SDK_native_vs_raw_native']['passed'], entry['SDK_native_vs_raw_native']
            elif label == 'raw_native_A2':
                if rows:
                    entry['raw_native_AA'] = V5.compare_bf16(outputs[0][:rows], bits[:rows], 0)
                    assert entry['raw_native_AA']['byte_exact'], entry['raw_native_AA']
            else:
                if rows:
                    entry['candidate_vs_raw_native'] = V5.compare_bf16(outputs[0][:rows], bits[:rows], 1)
                    assert entry['candidate_vs_raw_native']['passed']
                    if ref_bits is not None:
                        entry['candidate_vs_SDK_native'] = V5.compare_bf16(ref_bits[:rows], bits[:rows], 1)
                        assert entry['candidate_vs_SDK_native']['passed']
                entry['planner_frame_words'] = frame.tolist()
                entry['planner_frame_sha256'] = hashlib.sha256(frame.tobytes()).hexdigest()
                entry['planner_frame_byte_exact_vs_frozen_oracle'] = bool(np.array_equal(frame, expected_frame))
                entry['planner_frame_matches_frozen_cpu_oracle'] = (
                    entry['planner_frame_sha256'] == case['oracle_frame_sha256'])
                assert entry['planner_frame_byte_exact_vs_frozen_oracle'] and entry['planner_frame_matches_frozen_cpu_oracle']
            outputs.append(bits)
        seal_readonly('after_candidate')
        entry['passed'] = True
        (ROOT / 'evidence' / f'{family}_{name}_result.json').write_text(json.dumps(entry, indent=2) + '\n')
        result['cases'].append(entry); emit({'case_passed': name, 'family': family})
        del x, token, counts, reference, ref_bits, outputs
        torch.npu.synchronize()
    for label, bad in [('individual_overflow', {0: profile.capacity + 1}),
                       ('cumulative_overflow', {0: 1, profile.experts - 1: profile.capacity})]:
        cpu_counts = np.zeros(profile.experts, dtype=np.int64)
        for expert, rows in bad.items(): cpu_counts[expert] = rows
        counts = torch.from_numpy(cpu_counts).to('npu:0'); torch.npu.synchronize()
        frame = np.zeros(profile.frame_words, dtype=np.uint32)
        checked(f'planner_only:{family}:{label}', lib.k5_probe_planner_only(owner_holder[0], counts.data_ptr(),
                                                                           frame.ctypes.data))
        expected = NC.plan_frame(cpu_counts, profile)
        record = {'negative': label, 'counts': cpu_counts.tolist(), 'status_word': int(frame[2]),
                  'frame_sha256': hashlib.sha256(frame.tobytes()).hexdigest(),
                  'expected_plan_frame_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
                  'byte_exact_vs_cpu_twin': bool(np.array_equal(frame, expected)),
                  'math_launched': False}
        assert record['status_word'] == 1 and record['byte_exact_vs_cpu_twin'], record
        result['negatives'].append(record); emit({'device_negative_passed': record})
        del counts
        torch.npu.synchronize()
    try:
        result['timing'] = timing_family(family, lib, torch, torch_npu, owner_holder, profile, weight, scale,
                                         assist)
        result['timing_passed'] = bool(result['timing']['cases'])
    except BaseException as timing_error:
        result['timing'] = {'error_type': type(timing_error).__name__, 'error': str(timing_error),
                            'performance_receipt_granted': False}
        result['timing_passed'] = False
        emit({'timing_failure': {'family': family, 'error_type': type(timing_error).__name__,
                                 'error': str(timing_error)}})
    del weight, scale, assist
    torch.npu.synchronize()
    if owner_holder[0].value:
        checked(f'close:{family}', lib.m1_probe_close(ctypes.byref(owner_holder[0])))
    result['closed_owner_null'] = not owner_holder[0].value
    assert result['closed_owner_null'], 'bridge owner retained after close'
    return result


def timing_family(family, lib, torch, torch_npu, owner_holder, profile, weight, scale, assist):
    """Minimal interleaved operator-window benchmark, M1 event-template wording.

    bind (validation, list H2D, canary and workspace resets) and the x restore
    happen outside the window; the window is a torch event pair around the
    planner+GMM launch only; the finish sync plus D2H-free canary read happen
    after the end event record.
    """
    cases = NC.cases(profile)
    result = {'warmup_rounds': TIMING_WARMUP, 'measured_samples': TIMING_SAMPLES,
              'operator_window': 'torch.npu.Event(enable_timing) pair around the planner+GMM launch',
              'restore_and_bind_outside_window': True, 'D2H_outside_window': True,
              'host_workspace_reset_outside_window': True, 'planner_included_in_candidate': True,
              'interleaved_order': ['raw_native', 'candidate'], 'cases': {},
              'same_instance_paired': True, 'performance_receipt_granted': False,
              'note': 'queue/submit idle inside the window is not pure core time'}
    for case_index in [0, 2]:
        name, cpu_counts = cases[case_index]
        cpu_x, cpu_token = NC.activation(profile, case_index)
        x = torch.from_numpy(cpu_x).to('npu:0')
        token = torch.from_numpy(cpu_token).to('npu:0')
        counts = torch.from_numpy(cpu_counts).to('npu:0')
        torch.npu.synchronize()
        entry = {'case': name, 'R': int(cpu_counts.sum()),
                 'active_slots': int((cpu_counts != 0).sum()), 'methods': {}}
        for method, sparse in [('raw_native', 0), ('candidate', 1)]:
            checked(f'timing_bind:{family}:{name}:{method}',
                    lib.k5_probe_bind(owner_holder[0], x.data_ptr(), weight.data_ptr(), assist.data_ptr(),
                                      scale.data_ptr(), counts.data_ptr(), token.data_ptr(), sparse))
            for _ in range(TIMING_WARMUP):
                x.copy_(torch.from_numpy(cpu_x)); torch.npu.synchronize()
                checked(f'timing_warmup:{family}:{method}', lib.k5_probe_launch_only(owner_holder[0]))
            checked(f'timing_warmup_finish:{family}:{method}', lib.k5_probe_batch_finish(owner_holder[0]))
        elapsed = {'raw_native': [], 'candidate': []}
        for _ in range(TIMING_SAMPLES):
            for method, sparse in [('raw_native', 0), ('candidate', 1)]:
                checked(f'timing_rebind:{family}:{name}:{method}',
                        lib.k5_probe_bind(owner_holder[0], x.data_ptr(), weight.data_ptr(), assist.data_ptr(),
                                          scale.data_ptr(), counts.data_ptr(), token.data_ptr(), sparse))
                x.copy_(torch.from_numpy(cpu_x)); torch.npu.synchronize()
                begin = torch.npu.Event(enable_timing=True); end = torch.npu.Event(enable_timing=True)
                begin.record()
                code = lib.k5_probe_launch_only(owner_holder[0])
                end.record()
                torch.npu.synchronize()
                if code: raise RuntimeError(f'timing launch failed: {family}/{name}/{method}: {code}')
                elapsed[method].append(begin.elapsed_time(end))
                checked(f'timing_finish:{family}:{name}:{method}', lib.k5_probe_batch_finish(owner_holder[0]))
        for method in ('raw_native', 'candidate'):
            values = np.asarray(elapsed[method], dtype=np.float64)
            entry['methods'][method] = {'ms': elapsed[method], 'median_ms': float(np.median(values)),
                                        'min_ms': float(values.min()), 'max_ms': float(values.max()),
                                        'mean_ms': float(values.mean())}
        entry['candidate_over_raw_median_ratio'] = (entry['methods']['candidate']['median_ms'] /
                                                    entry['methods']['raw_native']['median_ms'])
        entry['candidate_planner_included'] = True
        result['cases'][name] = entry
        emit({'timing_case': {'family': family, **{k: entry[k] for k in
              ('case', 'R', 'active_slots', 'candidate_over_raw_median_ratio')}}})
        del x, token, counts
        torch.npu.synchronize()
    return result


def run():
    frozen_guard()
    (ROOT / 'evidence').mkdir(parents=True, exist_ok=True)
    import torch
    import torch_npu
    terminal = {'passed': False, 'physical': PHYSICAL, 'logical': 0, 'families': {}, 'timing': {},
                'scope': 'K5 capacity numeric + minimal same-instance operator-window timing',
                'explicit_reset_or_empty_cache_calls': 0, 'Torch_context_and_stream_borrowed': True,
                'container_PID': os.getpid(), 'performance_receipt_granted': False,
                'admission_granted': False, 'formal_receipt_granted': False, 'retries': 0,
                'frozen_dependencies_guard': True}
    owner_holder = [ctypes.c_void_p()]
    libs = {}
    try:
        assert torch.npu.device_count() == 1
        torch.npu.config.allow_internal_format = True
        getter = getattr(torch_npu._C, '_npu_getOption', None)
        option = getter('ALLOW_INTERNAL_FORMAT') if getter else None
        if isinstance(option, bytes): option = option.decode()
        emit({'production_allow_internal_format': True, 'SDK_option_observed': option,
              'production_setting_source': '/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'})
        torch.npu.set_device(0)
        for family in FAMILIES:
            lib = bridge_api(family); libs[family] = lib
            checked(f'mapping:{family}', lib.m1_probe_mapping())
        emit({'torch_version': torch.__version__, 'torch_npu_version': torch_npu.__version__,
              'borrowed_stream': int(torch.npu.current_stream().npu_stream), 'TASK_QUEUE_ENABLE': '0'})
        for family in FAMILIES:
            result = device_family(family, libs[family], torch, torch_npu, owner_holder)
            terminal['families'][family] = result
            terminal['timing'][family] = result['timing']
        terminal['numeric_passed'] = (len(terminal['families']) == len(FAMILIES) and
                                      all(len(v['cases']) == len(FROZEN['profiles'][f]['cases'])
                                          for f, v in terminal['families'].items()) and
                                      all(v['closed_owner_null'] for v in terminal['families'].values()) and
                                      all(len(v['negatives']) == 2 for v in terminal['families'].values()))
        assert terminal['numeric_passed'], 'Incomplete family results'
        terminal['timing_passed'] = all(v.get('timing_passed') for v in terminal['families'].values())
        terminal['passed'] = terminal['numeric_passed']
    except BaseException as error:
        terminal['failure_type'] = type(error).__name__; terminal['failure'] = str(error)
        terminal['traceback'] = traceback.format_exc()
        emit({'numeric_gate_failure': terminal['failure'], 'stop_all_following_cases': True})
        if owner_holder[0].value:
            for family in FAMILIES:
                if not owner_holder[0].value:
                    break
                if family in libs:
                    try:
                        terminal.setdefault('failure_cleanup_codes', {})[family] = \
                            libs[family].m1_probe_close(ctypes.byref(owner_holder[0]))
                    except Exception as cleanup_error:  # pragma: no cover - diagnostic path
                        terminal.setdefault('failure_cleanup_codes', {})[family] = repr(cleanup_error)
        terminal['passed'] = False
    (ROOT / 'runtime_terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
    emit({'terminal': True, 'passed': terminal['passed'],
          'failure': terminal.get('failure'), 'families': list(terminal['families'])})
    return 0 if terminal['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(run())
