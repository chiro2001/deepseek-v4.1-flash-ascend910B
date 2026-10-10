"""Summarize existing formal profiles by shape; do not collect or time a model."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {'count': len(values), 'mean': statistics.mean(values),
            'min': values[0], 'median': statistics.median(values),
            'p95': values[math.ceil(.95 * len(values)) - 1], 'max': values[-1]}


def execution_rows(rows):
    # torch_npu exports a communication envelope and its AIV kernel as two
    # rows. Remove an envelope only with an exact corresponding execution.
    # Duration and timestamp alone are not sufficient across devices/steps.
    def identity(row):
        return tuple(row[k].strip() for k in
                     ('Step Id', 'Device_id', 'Type', 'Start Time(us)', 'Duration(us)'))
    kernels = Counter(identity(r) for r in rows if r['Name'] == 'AivKernel')
    excluded = []
    kept = []
    for row in rows:
        key = identity(row)
        if row['Name'].startswith('hcom_') and row['Block Num'] == '0' and kernels[key]:
            excluded.append(row)
            kernels[key] -= 1
        else:
            kept.append(row)
    return kept, excluded


def summarize(path, rank, metric):
    with path.open() as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        raw = list(reader)
    rows, excluded = execution_rows(raw)
    steps = sorted({r['Step Id'] for r in rows}, key=int)
    assert len(steps) == 10, (path, steps)
    assert {int(r['Device_id']) for r in rows} == {rank + 8}, path
    groups = defaultdict(list)
    for row in rows:
        assert number(row['Duration(us)']) is not None and float(row['Duration(us)']) >= 0
        key = tuple(row[k] for k in ('Type', 'Input Shapes', 'Input Data Types',
                                     'Block Num', 'Mix Block Num', 'Accelerator Core'))
        groups[key].append(row)
    counters = [f for f in fields if f.startswith(('aic_', 'aiv_', 'aicore_', 'cube_'))]
    details = []
    for key, selected in groups.items():
        durations = [float(r['Duration(us)']) for r in selected]
        summary = {'type': key[0], 'input_shapes': key[1], 'input_dtypes': key[2],
                   'block_num': key[3], 'mix_block_num': key[4], 'core': key[5],
                   'calls_per_step': len(selected) / len(steps),
                   'sum_execution_us_per_step': sum(durations) / len(steps),
                   'duration_us': distribution(durations), 'raw_counter_means': {}}
        for field in counters:
            values = [v for r in selected if (v := number(r[field])) is not None]
            if values and any(v != 0 for v in values):
                summary['raw_counter_means'][field] = statistics.mean(values)
        details.append(summary)
    details.sort(key=lambda r: r['sum_execution_us_per_step'], reverse=True)
    by_type = defaultdict(list)
    for row in rows:
        by_type[row['Type']].append(float(row['Duration(us)']))
    result = {'metric': metric, 'rank': rank, 'device_id': rank + 8,
              'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'raw_rows': len(raw), 'execution_rows': len(rows),
              'removed_exact_hccl_envelopes': dict(Counter(r['Type'] for r in excluded)),
              'steps': steps,
              'types': [{'type': kind, 'calls_per_step': len(values) / len(steps),
                         'sum_execution_us_per_step': sum(values) / len(steps),
                         'duration_us': distribution(values)}
                        for kind, values in sorted(by_type.items(), key=lambda item: sum(item[1]), reverse=True)[:20]],
              'top_shapes': details[:16]}
    return result, rows


def communication_timeline(rank_rows):
    # These are calibrated tool timestamps, not proof of the actual arrival
    # instant. Do not call start spread a pure network wait or add envelopes.
    streams = {}
    for rank, rows in rank_rows.items():
        for step in sorted({r['Step Id'] for r in rows}, key=int):
            selected = sorted((r for r in rows if r['Step Id'] == step and
                               r['Type'] == 'hcom_allReduce_' and r['Name'] == 'AivKernel'),
                              key=lambda r: float(r['Start Time(us)']))
            assert len(selected) == 82, (rank, step, len(selected))
            streams[rank, step] = selected
    steps = sorted({s for _, s in streams}, key=int)
    aligned = []
    for ordinal in range(82):
        starts, ends, durations = [], [], []
        for step in steps:
            group = [streams[rank, step][ordinal] for rank in range(8)]
            begin = [float(r['Start Time(us)']) for r in group]
            duration = [float(r['Duration(us)']) for r in group]
            end = [s + d for s, d in zip(begin, duration)]
            starts.append(max(begin) - min(begin))
            ends.append(max(end) - min(end))
            durations.extend(duration)
        aligned.append({'ordinal': ordinal, 'start_spread_us': distribution(starts),
                        'end_spread_us': distribution(ends), 'duration_us': distribution(durations)})
    return {'scope': 'AivKernel ordered by calibrated start timestamp within each profiler step; '
                     'start spread includes launch/arrival/clock effects; no pure wait attribution',
            'allreduce_executions_per_step': 82, 'ordinals': aligned}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted((args.root / 'prof').rglob('kernel_details.csv'))
    assert len(paths) == 56, len(paths)
    results, pipe_rows = [], {}
    for path in paths:
        metric = path.relative_to(args.root / 'prof').parts[0]
        match = re.search(r'formal_tp8_rank(\d+)_chip(\d+)', str(path))
        assert match and int(match[2]) == int(match[1]) + 8, path
        rank = int(match[1])
        result, rows = summarize(path, rank, metric)
        results.append(result)
        if metric == 'PipeUtilization':
            pipe_rows[rank] = rows
    assert set(pipe_rows) == set(range(8))
    report = {'scope': 'Diagnostic collection; sums may overlap and are not E2E timing. '
                       'Block Num is launch configuration, not measured active-core occupancy. '
                       'Raw counter means use profiler task/core normalization.',
              'precision_validated': False, 'performance_claim': None, 'records': results}
    output = args.root / 'shape_microarch.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    compact = dict(report)
    compact['full_report_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    compact['records'] = [dict(r, top_shapes=r['top_shapes'] if r['rank'] == 0 else []) for r in results]
    summary = args.root / 'shape_microarch_summary.json'
    summary.write_text(json.dumps(compact, indent=2) + '\n')
    assert summary.stat().st_size < 1000000, 'Keep full report remotely or use COS'
    timeline = args.root / 'communication_timeline.json'
    timeline.write_text(json.dumps(communication_timeline(pipe_rows), indent=2) + '\n')
    print(json.dumps({'records': len(results), 'shape_bytes': output.stat().st_size,
                      'summary_bytes': summary.stat().st_size,
                      'timeline_bytes': timeline.stat().st_size}))


if __name__ == '__main__':
    main()
