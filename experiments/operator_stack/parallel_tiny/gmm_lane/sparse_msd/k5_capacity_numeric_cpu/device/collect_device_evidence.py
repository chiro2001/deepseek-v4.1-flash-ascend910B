"""Compact the two device-phase roots into result JSONs (large files stay remote).

Numeric root:  runtime_terminal.json + run_receipt.json of the authorized
physical4 numeric window (5 cases per profile, 2 illegal-count planner-only
negatives per profile).
Timing root:   the minimal operator-window benchmark window that cites the
numeric root and SHA.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

HOST_PROJECTS = Path(sys.argv[3] if len(sys.argv) > 3 else
                     '/home/l00886679/projects/dsv41-real-stack-tp8-20261010')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_compact(case):
    return {k: case[k] for k in ('case_index', 'name', 'R', 'row_capacity', 'experts', 'counts',
                                 'active_experts', 'counts_sha256', 'x_sha256', 'token_scale_sha256',
                                 'frozen_oracle_frame_sha256', 'planner_frame_sha256',
                                 'planner_frame_byte_exact_vs_frozen_oracle',
                                 'planner_frame_matches_frozen_cpu_oracle', 'SDK_output',
                                 'raw_native_A1_output', 'raw_native_A2_output', 'candidate_output',
                                 'SDK_native_vs_raw_native', 'raw_native_AA', 'candidate_vs_raw_native',
                                 'candidate_vs_SDK_native', 'raw_native_A1_unused_rows_sentinel_intact',
                                 'raw_native_A2_unused_rows_sentinel_intact',
                                 'candidate_unused_rows_sentinel_intact', 'SDK_reference_unavailable',
                                 'passed') if k in case}


def main():
    numeric_name = sys.argv[1]
    timing_name = sys.argv[2] if len(sys.argv) > 2 else None
    numeric_root = HOST_PROJECTS / numeric_name
    terminal = json.loads((numeric_root / 'runtime_terminal.json').read_text())
    attempt = json.loads((numeric_root / 'runtime_attempt.json').read_text())
    preflight = json.loads((numeric_root / 'preflight.json').read_text())
    release = json.loads((numeric_root / 'release_fresh_smi.json').read_text())
    receipt = json.loads((numeric_root / 'run_receipt.json').read_text())
    counters = {k: v for k, v in receipt['accepted_stage_counters'].items()}
    families = {}
    for family, value in terminal['families'].items():
        build = numeric_root / 'build' / family
        prepare = json.loads((build / 'prepare_receipt.json').read_text())
        bridge = json.loads((build / 'bridge_receipt.json').read_text())
        families[family] = {
            'profile': value['profile'],
            'workspace': {'native_workspace_bytes': prepare['native_workspace_bytes'],
                          'frame_bytes': prepare['profile']['frame_bytes'],
                          'capacity_workspace_bytes': prepare['capacity_workspace_bytes']},
            'compiled': {'planner_object_sha256': prepare['planner_cpu_rebuild']['object_sha256'],
                         'planner_object_bytes': prepare['planner_cpu_rebuild']['object_bytes'],
                         'candidate_object_sha256': prepare['compile_candidate']['object_sha256'],
                         'candidate_object_bytes': prepare['compile_candidate']['object_bytes'],
                         'native_object_sha256': prepare['compile_native']['object_sha256'],
                         'native_object_bytes': prepare['compile_native']['object_bytes'],
                         'tiling_sha256': prepare['tiling_byte_equality']['sha256'],
                         'tiling_native_candidate_byte_exact': prepare['tiling_byte_equality']['native_candidate_byte_exact'],
                         'kernel_name': prepare['compile_candidate']['kernel_name'],
                         'block_dim': prepare['compile_candidate']['block_dim'],
                         'task_ration': prepare['compile_candidate']['task_ration'],
                         'schedule_mode': prepare['compile_candidate']['schedule_mode']},
            'bridge': {'shared_sha256': bridge['shared_sha256'], 'shared_bytes': bridge['shared_bytes'],
                       'probe_bridge_sha256': bridge['source_sha256']['probe_bridge.cpp'],
                       'capacity_geometry_sha256': bridge['source_sha256']['capacity_geometry.h'],
                       'workspace_layout_sha256': bridge['source_sha256']['workspace_layout.h'],
                       'physical_declaration_accepted': bridge['physical_declaration_accepted'],
                       'planner_only_entry': bridge['planner_only_entry']},
            'loader': {'layout_stages': value['loader_layout_stages'],
                       'weight_physical_sha256': value['weight_physical_sha256'],
                       'M1_frozen_physical_NZ_sha256': value.get('M1_frozen_physical_NZ_sha256'),
                       'K5_loader_reproduces_M1_frozen_NZ': value.get('K5_loader_reproduces_M1_frozen_NZ')},
            'cases': [case_compact(c) for c in value['cases']],
            'negatives': value['negatives'],
            'closed_owner_null': value['closed_owner_null']}
    compact = {'schema': 'k5_capacity_device_numeric_v1', 'date': '20261012',
               'root': str(numeric_root), 'attempt': attempt, 'passed': terminal['passed'],
               'numeric_passed': terminal['numeric_passed'], 'timing_passed_in_numeric_phase': terminal['timing_passed'],
               'phase': terminal.get('phase', 'numeric_then_timing'),
               'physical': terminal['physical'], 'logical': terminal['logical'],
               'scope': terminal['scope'], 'families': families,
               'preflight': {k: preflight[k] for k in ('UTC', 'physical', 'physical_processes', 'health_allowed',
                                                       'health_output', 'container_has_dev_permission',
                                                       'device_node', 'container_id', 'container_PID',
                                                       'container_global_ASCEND_env')},
               'release': {k: release[k] for k in ('UTC', 'physical', 'physical_released', 'health_allowed',
                                                   'container_global_env_unchanged', 'runtime_wrapper_exit')},
               'counters': counters, 'runtime_log_bytes': receipt['runtime_log_bytes'],
               'runtime_log_sha256': receipt['runtime_log_sha256'],
               'runtime_terminal_sha256': digest(numeric_root / 'runtime_terminal.json'),
               'run_receipt_sha256': digest(numeric_root / 'run_receipt.json'),
               'flags': {'explicit_reset_or_empty_cache_calls': terminal['explicit_reset_or_empty_cache_calls'],
                         'retries_in_root': terminal['retries'], 'admission_granted': terminal['admission_granted'],
                         'performance_receipt_granted': terminal['performance_receipt_granted'],
                         'formal_receipt_granted': terminal['formal_receipt_granted']},
               'attempt_history': json.loads((HOST_PROJECTS / 'gmm_lane_sparse_msd_k5_capacity_p4_20261012_v4'
                                              / 'attempt_history.json').read_text())
               if (HOST_PROJECTS / 'gmm_lane_sparse_msd_k5_capacity_p4_20261012_v4' / 'attempt_history.json').exists()
               else None}
    out_numeric = numeric_root / 'k5_capacity_device_numeric_20261012.json'
    out_numeric.write_text(json.dumps(compact, indent=2) + '\n')
    print(json.dumps({'numeric_compact': str(out_numeric), 'bytes': out_numeric.stat().st_size,
                      'sha256': digest(out_numeric)}), flush=True)
    if timing_name:
        timing_root = HOST_PROJECTS / timing_name
        timing_terminal = json.loads((timing_root / 'runtime_terminal.json').read_text())
        timing_attempt = json.loads((timing_root / 'runtime_attempt.json').read_text())
        timing_receipt = json.loads((timing_root / 'run_receipt.json').read_text())
        timing_release = json.loads((timing_root / 'release_fresh_smi.json').read_text())
        timing = {'schema': 'k5_capacity_device_timing_v1', 'date': '20261012', 'root': str(timing_root),
                  'attempt': timing_attempt, 'passed': timing_terminal['passed'],
                  'timing_passed': timing_terminal['timing_passed'],
                  'numeric_phase_evidence': timing_terminal['numeric_phase_evidence'],
                  'physical': timing_terminal['physical'], 'logical': timing_terminal['logical'],
                  'counters': timing_receipt['accepted_stage_counters'],
                  'runtime_log_bytes': timing_receipt['runtime_log_bytes'],
                  'runtime_log_sha256': timing_receipt['runtime_log_sha256'],
                  'runtime_terminal_sha256': digest(timing_root / 'runtime_terminal.json'),
                  'release': {k: timing_release[k] for k in ('UTC', 'physical', 'physical_released',
                                                             'health_allowed', 'container_global_env_unchanged',
                                                             'runtime_wrapper_exit')},
                  'families': {}}
        for family, value in timing_terminal['families'].items():
            build = timing_root / 'build' / family
            bridge = json.loads((build / 'bridge_receipt.json').read_text())
            timing['families'][family] = {
                'profile': value['profile'], 'bridge_shared_sha256': bridge['shared_sha256'],
                'timing': value['timing'], 'closed_owner_null': value['closed_owner_null']}
        out_timing = timing_root / 'k5_capacity_device_timing_20261012.json'
        out_timing.write_text(json.dumps(timing, indent=2) + '\n')
        print(json.dumps({'timing_compact': str(out_timing), 'bytes': out_timing.stat().st_size,
                          'sha256': digest(out_timing)}), flush=True)


if __name__ == '__main__':
    main()
