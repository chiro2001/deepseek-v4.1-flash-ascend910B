"""Validate complete matching decode windows and actual metadata replacement."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--candidate', default='mdall')
    parser.add_argument('--reference', default='gmmact')
    parser.add_argument('--device', default='4')
    args = parser.parse_args(); results = {}
    for arm in [args.reference, args.candidate]:
        paths = list((args.root / 'prof' / arm).rglob('kernel_details.csv'))
        assert len(paths) == 1, paths
        path = paths[0]; rows = list(csv.DictReader(path.open()))
        steps = collections.Counter(row['Step Id'] for row in rows)
        assert len(steps) == 20 and len(set(steps.values())) == 1, steps
        assert {row['Device_id'] for row in rows} == {args.device}
        json.loads((path.parent / 'trace_view.json').read_text())
        for name in ['op_statistic.csv', 'api_statistic.csv']:
            assert (path.parent / name).is_file(), name
        kinds = collections.Counter(row['Type'] for row in rows)
        expected = {'project_static': 80, 'finish_hc': 80, 'hc_post_kernel': 80,
                    'tiny_router_gemv': 40, 'route_init_kernel': 40, 'route_combine_kernel': 40,
                    'GroupedMatmul': 40, 'clamped_swiglu_kernel': 40, 'gmm1_activation_kernel': 40}
        for kind, count in expected.items():
            assert kinds[kind] == 20 * count, (arm, kind, kinds[kind], count)
        new = ['slot_mapping_kernel', 'ring_counts_kernel', 'ring_sources_kernel']
        if arm == 'gmmact':
            assert all(kinds[kind] == 0 for kind in new), (arm, kinds)
        else:
            assert kinds['slot_mapping_kernel'] > 0 and kinds['slot_mapping_kernel'] % 20 == 0, kinds
            for kind in ['ring_counts_kernel', 'ring_sources_kernel']:
                assert kinds[kind] == 20, (arm, kind, kinds[kind])
            for row in rows:
                if row['Type'] in new:
                    assert row['Accelerator Core'] == 'AI_VECTOR_CORE', row
        if arm == 'mdfull':
            assert kinds['_compute_slot_mappings_multi_kernel'] == 20, kinds
            assert kinds['_compute_slot_mapping_kernel'] == 0, kinds
        durations = collections.defaultdict(float)
        for row in rows: durations[row['Type']] += float(row['Duration(us)'])
        pipe_fields = ['aiv_vec_ratio', 'aiv_scalar_ratio', 'aiv_mte2_ratio', 'aiv_mte3_ratio']
        pipe = {}
        for kind in ['slot_mapping_kernel', 'ring_counts_kernel', 'ring_sources_kernel',
                     '_compute_slot_mapping_kernel', '_compute_slot_mappings_multi_kernel']:
            subset = [row for row in rows if row['Type'] == kind]
            if not subset: continue
            total = sum(float(row['Duration(us)']) for row in subset)
            pipe[kind] = {field: sum(float(row['Duration(us)']) * float(row[field]) for row in subset) / total
                          for field in pipe_fields}
        results[arm] = {'steps': 20, 'rows': len(rows), 'tasks_per_step': len(rows) / 20,
                        'device': args.device, 'csv': str(path),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'trace_valid': True, 'structural_checks_passed': True,
                        'kernel_us_per_step': sum(durations.values()) / 20,
                        'pipe_duration_weighted_task_ratios': pipe,
                        'pipe_note': 'Duration-weighted per-task ratios; overlapping counters, not whole-chip utilization',
                        'types': {kind: {'calls_per_step': count / 20, 'us_per_step': durations[kind] / 20}
                                  for kind, count in kinds.items()}}
    assert results[args.candidate]['tasks_per_step'] < results[args.reference]['tasks_per_step'], results
    (args.root / 'profile_validation.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({arm: {k:v for k,v in row.items() if k != 'types'} for arm,row in results.items()}, indent=2))


if __name__ == '__main__': main()
