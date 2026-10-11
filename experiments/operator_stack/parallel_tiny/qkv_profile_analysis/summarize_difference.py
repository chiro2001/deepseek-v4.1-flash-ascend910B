"""Matched-rank diagnostics from frozen analysis; sums are not E2E budgets."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.analysis / 'full_analysis.json').read_text())
    records = report['records']
    banks = {arm: {r['rank']: r for r in records if r['arm'] == arm} for arm in ('tp8mask', 'tp8qkv')}
    assert set(banks['tp8mask']) == set(banks['tp8qkv']) == set(range(8))
    def comparison(fetch):
        left = [fetch(banks['tp8mask'][rank]) for rank in range(8)]
        right = [fetch(banks['tp8qkv'][rank]) for rank in range(8)]
        assert all(v is not None for v in left + right)
        return {'mask_rank_median': statistics.median(left), 'qkv_rank_median': statistics.median(right),
                'matched_rank_delta_median': statistics.median(b - a for a, b in zip(left, right)),
                'mask_by_rank': left, 'qkv_by_rank': right}
    summary = {'scope': 'Matched rank summaries under PipeUtilization; unpaired collection windows. '
                       'Kernel sums, event overlap and interval spans are not recoverable E2E budgets.',
               'collection_exit': 0, 'worker_receipts': 16, 'complete_device_windows': 16,
               'active_steps_per_window': 10, 'hc_pre_per_step': 80, 'matched_layer_windows': 6400,
               'families': {}, 'layer_intervals': {}, 'graph_phases': {}, 'events': {}}
    for kind in ('QuantMatmul', 'Slice', 'ViewCopy', 'Contiguous', 'LayoutTransData', 'Cast', 'RMSNorm', 'DynamicQuant'):
        summary['families'][kind] = {key: comparison(lambda r, kind=kind, key=key: r['families'].get(kind, {}).get(key, 0))
                                    for key in ('calls_per_step', 'sum_duration_us_per_step')}
    def projection_sum(r):
        keys = ('q_a', 'kv') if r['arm'] == 'tp8mask' else ('merged_qkv',)
        return sum(r['roles'][key]['sum_duration_us_per_step'] for key in keys)
    summary['projection_kernel_sum_us_per_step'] = comparison(projection_sum)
    for key in ('projection_span_us', 'projection_union_us', 'q_norm_us', 'kv_norm_us',
                'q_kv_norm_overlap_us', 'kv_projection_q_norm_overlap_us',
                'kv_norm_start_minus_q_norm_end_us', 'kv_norm_start_minus_q_quant_end_us',
                'kv_norm_start_minus_projection_end_us', 'projection_to_first_qb_end_us',
                'copies_1792_calls', 'copies_1792_sum_us'):
        summary['layer_intervals'][key] = comparison(lambda r, key=key: r['layer_window_summary'][key]['mean'])
    for phase in ('before_main_graph', 'main_graph', 'after_main_graph'):
        summary['graph_phases'][phase] = {key: comparison(lambda r, phase=phase, key=key: r['timeline']['phase_means'][phase][key]['mean'])
                                         for key in ('span_us', 'kernel_union_us', 'kernel_gap_us')}
    for kind in ('EVENT_RECORD', 'EVENT_WAIT', 'EVENT_RESET'):
        summary['events'][kind] = {phase: {key: comparison(lambda r, kind=kind, phase=phase, key=key:
                    r['event_phase_stats'][kind][phase][key]['mean'])
                    for key in ('calls_starting_in_phase', 'clipped_sum_us', 'interval_union_us')}
                    for phase in ('before_main_graph', 'main_graph', 'after_main_graph')}
    path = args.analysis / 'differential_summary.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'summary_bytes': path.stat().st_size, 'scope': summary['scope']}))


if __name__ == '__main__':
    main()
