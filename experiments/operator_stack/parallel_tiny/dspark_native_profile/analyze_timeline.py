"""Stdlib analysis of actual full40/MTP3 kernels; never produce a quote."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def union(intervals):
    total, end = 0.0, None
    for left, right in sorted(intervals):
        if end is None or left > end:
            total += right-left
        elif right > end:
            total += right-end
        end = max(right, end if end is not None else right)
    return total


def read_rank(path, rank):
    folder = path/'ASCEND_PROFILER_OUTPUT'
    kernel_path, task_path = folder/'kernel_details.csv', folder/'task_time.csv'
    raw = list(csv.DictReader(kernel_path.open()))
    assert raw and {int(row['Device_id']) for row in raw} == {8+rank}
    base = min(Decimal(row['Start Time(us)'].strip()) for row in raw)
    kernels = [row | {'start': float(Decimal(row['Start Time(us)'].strip())-base),
                     'duration': float(row['Duration(us)'])} for row in raw]
    for row in kernels:
        row['end'] = row['start']+row['duration']
    tasks = []
    if task_path.exists():
        for row in csv.DictReader(task_path.open()):
            start = float(Decimal(row['task_start(us)'].strip())-base)
            tasks.append(row | {'start': start, 'end': start+float(row['task_time(us)'])})
    grouped = defaultdict(list)
    unlabelled = []
    for row in kernels:
        if not row['Step Id'].strip():
            unlabelled.append(row)
            continue
        grouped[int(row['Step Id'])].append(row)
    assert sorted(grouped) == list(range(5,15)), sorted(grouped)
    labelled = [row for rows in grouped.values() for row in rows]
    labelled_begin = min(row['start'] for row in labelled)
    labelled_end = max(row['end'] for row in labelled)
    assert all(row['end'] <= labelled_begin or row['start'] >= labelled_end for row in unlabelled), 'Unlabelled kernel overlaps an actual decode window'
    assert not any(row['Type'] in ('HcPre','SparseAttnSharedkv') for row in unlabelled), 'Unlabelled model layer/cache reader'
    steps, types = [], defaultdict(lambda: {'count': 0, 'duration': 0.0})
    for step, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row['start'])
        hc = [row for row in rows if row['Type'] == 'HcPre']
        shape = lambda row: row['Input Shapes'].strip('"').replace(' ', '')
        target = [row for row in hc if shape(row).startswith('6,4,5120;')]
        draft = [row for row in hc if shape(row).startswith('5,4,5120;')]
        assert len(hc) == 86 and len(target) == 80 and len(draft) == 6, (rank,step,len(hc),len(target),len(draft))
        assert {row['OP State'] for row in target} == {'static'}
        assert {row['OP State'] for row in draft} == {'dynamic'}
        models = {row['Model ID'] for row in target}
        assert len(models) == 1, models
        graph = [row for row in rows if row['Model ID'] in models]
        # HCCL rows are labelled N/A while retaining the actual graph model
        # ID. They are part of the TP8 graph and must remain in its envelope.
        assert all(row['OP State'] == 'static' or
                   (row['OP State'] == 'N/A' and row['Type'].startswith('hcom_')) for row in graph)
        graph_start, graph_end = min(row['start'] for row in graph), max(row['end'] for row in graph)
        draft_start = min(row['start'] for row in draft)
        projection = [row for row in rows if 'matmul' in row['Type'].lower() and shape(row).startswith('6,15360;')]
        assert len(projection) == 1, (rank,step,[(row['Type'],shape(row)) for row in projection])
        proj = projection[0]
        attention = [row for row in rows if row['Type'] == 'SparseAttnSharedkv' and row['OP State'] == 'dynamic']
        assert len(attention) == 3, (rank,step,len(attention))
        for index, row in enumerate(attention):
            assert draft[2*index]['start'] <= row['start'] < draft[2*index+1]['start']
        scatter = [row for row in rows if row['Type'] == 'ScatterNdUpdateSk' and row['OP State'] == 'dynamic']
        context = [row for row in scatter if proj['start'] <= row['start'] < draft_start]
        query = [row for row in scatter if row['start'] >= draft_start]
        assert len(context) == len(query) == 3, (rank,step,len(context),len(query))
        first, last = min(row['start'] for row in rows), max(row['end'] for row in rows)
        assert first <= graph_start < graph_end <= proj['start'] < draft_start <= last
        windows = {'pre_target': (first,graph_start), 'target_graph': (graph_start,graph_end),
                   'verify_and_prepare': (graph_end,proj['start']),
                   'draft_context_projection_cache': (proj['start'],draft_start),
                   'draft_body_logits': (draft_start,last)}
        phases = {}
        for name,(left,right) in windows.items():
            entries = [row for row in rows if left <= row['start'] < right]
            waits = [row for row in tasks if row.get('kernel_type') == 'EVENT_WAIT' and row['start'] < right and row['end'] > left]
            phases[name] = {'span_us': right-left, 'kernel_count': len(entries),
                'kernel_duration_sum_us': sum(row['duration'] for row in entries),
                'kernel_union_us': union([(max(left,row['start']),min(right,row['end'])) for row in entries]),
                'wait_union_us': union([(max(left,row['start']),min(right,row['end'])) for row in waits]),
                'streams': dict(Counter(row['Stream ID'] for row in entries))}
        steps.append({'step': step, 'target_hc_pre': 80, 'draft_hc_pre': 6,
                      'draft_attention_readers': 3, 'context_cache_writes': 3, 'query_cache_writes': 3,
                      'target_graph_op_states': dict(Counter(row['OP State'] for row in graph)),
                      'phases': phases})
        for row in rows:
            value = types[(row['Type'],row['OP State'])]
            value['count'] += 1
            value['duration'] += row['duration']
    files = [kernel_path, task_path, folder/'api_statistic.csv', folder/'step_trace_time.csv']
    statistics_by_phase = {name: {key: {'median': statistics.median([step['phases'][name][key] for step in steps]),
                                      'mean': statistics.mean([step['phases'][name][key] for step in steps])}
                                for key in ('span_us','kernel_union_us','kernel_duration_sum_us','wait_union_us')}
                           for name in steps[0]['phases']}
    return {'rank': rank, 'physical_chip': 8+rank, 'actual_complete_steps': 10,
            'main_layers_per_step': 40, 'mtp_layers_per_step': 3,
            'phases': statistics_by_phase, 'steps': steps,
            'excluded_unlabelled_boundary_kernels': {'count': len(unlabelled),
                'types': dict(Counter(row['Type'] for row in unlabelled)),
                'outside_all_labelled_decode_windows': True,
                'model_layers_or_draft_cache_readers': 0},
            'operators': [{'type': key[0], 'state': key[1], 'count_per_step': value['count']/10,
                           'duration_sum_us_per_step': value['duration']/10}
                          for key,value in sorted(types.items(), key=lambda item:-item[1]['duration'])],
            'raw_sources': {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files if p.exists()}}


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    schedules = json.loads((args.root/'profile_workers.json').read_text())
    assert {row['rank'] for row in schedules} == set(range(8))
    assert all(row['completed'] and row['actual_decode_steps'] == 15 for row in schedules)
    results = []
    for rank in range(8):
        paths = list((args.root/'profile').glob('dspark_rank'+str(rank)+'_*_ascend_pt'))
        assert len(paths) == 1, paths
        results.append(read_rank(paths[0],rank))
    result = {'scope': 'actual native full40 TP8 DSpark profile; diagnostic spans/unions only',
              'actual_complete_steps': 80, 'total_hc_pre': 6880, 'draft_graph': False,
              'formal_performance_claim': None, 'full_consumer_audit_passed': False,
              'profile_intervals_are_not_recoverable_e2e_budget': True,
              'parser_sha256': sha(Path(__file__)), 'ranks': results}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('FORMAL_DSPARK_DEVICE_PROFILE_VALIDATED', result['actual_complete_steps'], flush=True)


if __name__ == '__main__':
    main()
