"""Small exact task-row excerpts around layer0; no profiler collection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--native-file', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.analysis / 'full_analysis.json').read_text())
    out = {'scope': 'Actual task_time event clusters; no causal or recoverable E2E budget claim', 'records': []}
    for arm in ('tp8mask', 'tp8qkv'):
        record = next(r for r in report['records'] if r['arm'] == arm and r['rank'] == 0)
        output = Path(record['output_path'])
        with (output / 'kernel_details.csv').open() as stream:
            kernels = list(csv.DictReader(stream))
        first = record['layer_windows'][0]
        projected = [r for r in kernels if r['Step Id'] == first['step'] and 'QuantBatchMatmul' in r['Type']
                     and '1,5120;' in r['Input Shapes']
                     and any(';' + str(width) + ';' in r['Input Shapes'] for width in (1280, 512, 1792))]
        start = min(float(r['Start Time(us)']) for r in projected)
        finish = start + first['kv_norm_task']['relative_start_us'] + first['kv_norm_task']['duration_us'] + 1
        with (output / 'task_time.csv').open() as stream:
            reader = csv.DictReader(stream)
            fields, tasks = reader.fieldnames, list(reader)
        events = []
        for row in tasks:
            left, right = float(row['task_start(us)']), float(row['task_stop(us)'])
            if 'EVENT' in row['kernel_type'].upper() and (start <= left < finish or start <= right < finish):
                events.append({k: v for k, v in row.items() if k not in ('task_start(us)', 'task_stop(us)')}
                              | {'relative_start_us': left - start, 'relative_stop_us': right - start})
        events.sort(key=lambda r: r['relative_start_us'])
        out['records'].append({'arm': arm, 'rank': 0, 'step': first['step'], 'layer': 0,
                               'task_fields': fields, 'layer_window': first, 'events': events})
    (args.analysis / 'event_clusters.json').write_text(json.dumps(out, indent=2) + '\n')
    paths = [args.snapshot / name for name in ('formal_qkv_merge_patches.py', 'bench_real_tp8_model.py',
                                             'formal_tp8_profile.py', 'profile_step_clock.py')]
    paths.append(args.native_file)
    shas = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.analysis / 'source_sha256.json').write_text(json.dumps(shas, indent=2) + '\n')
    print(json.dumps({'event_examples': len(out['records']), 'source_files': len(shas)}))


if __name__ == '__main__':
    main()
