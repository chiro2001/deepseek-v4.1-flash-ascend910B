"""Summarize measured GMM kernels without treating their sums as latency savings."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    ranks = data['ranks']
    assert data['actual_complete_steps'] == 80 and len(ranks) == 8
    assert {r['rank'] for r in ranks} == set(range(8))
    assert all(r['actual_complete_steps'] == 10 for r in ranks)
    operations = []
    for role, state, expected in (('target', 'static', 40), ('draft', 'dynamic', 3)):
        for kind in ('GroupedMatmulSwigluQuantV2', 'GroupedMatmul'):
            measurements = []
            for rank in ranks:
                entries = [o for o in rank['operators'] if o['type'] == kind and o['state'] == state]
                assert len(entries) == 1 and entries[0]['count_per_step'] == expected
                measurements.append({'rank': rank['rank'], 'physical_chip': rank['physical_chip'],
                                     'kernels_per_step': expected,
                                     'kernel_duration_sum_ms_per_step': entries[0]['duration_sum_us_per_step'] / 1000})
            values = [m['kernel_duration_sum_ms_per_step'] for m in measurements]
            operations.append({'role': role, 'state': state, 'operator': kind,
                               'kernels_per_rank_step': expected,
                               'median_rank_kernel_duration_sum_ms_per_step': statistics.median(values),
                               'min_rank_kernel_duration_sum_ms_per_step': min(values),
                               'max_rank_kernel_duration_sum_ms_per_step': max(values),
                               'ranks': measurements})
    return {'scope': 'same full40 TP8 DSpark profiled instance, kernel-cost diagnosis only',
            'actual_complete_rank_steps': 80,
            'input_timeline_sha256': hashlib.sha256(raw).hexdigest(),
            'input_timeline_bytes': len(raw),
            'parser_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_csv_sha256': {str(r['rank']): r['raw_sources'] for r in ranks},
            'operations': operations,
            'duration_sums_are_not_recoverable_e2e_budget': True,
            'HCCL_parent_child_durations_not_added': True,
            'profile_enabled': True, 'formal_performance_claim': None,
            'full_consumer_audit_passed': False,
            'M1_capacity6_gain_does_not_establish_K5_or_full_model_gain': True}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--timeline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.timeline)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'actual_complete_rank_steps': result['actual_complete_rank_steps'],
                      'operations': [{k: v for k, v in op.items() if k != 'ranks'}
                                     for op in result['operations']],
                      'formal_performance_claim': None}), flush=True)


if __name__ == '__main__':
    main()
