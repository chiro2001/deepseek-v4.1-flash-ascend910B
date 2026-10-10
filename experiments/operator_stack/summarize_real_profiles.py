"""Validate actual device/rows and separate task totals from wall-clock timing."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def rows(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig')))


def micro(root, physical_chip, required):
    folders = list(root.glob('OPPROF_*'))
    assert len(folders) == 1, folders
    records = {}
    for path in folders[0].glob('*.csv'):
        data = rows(path)
        assert data, path
        records[path.stem] = {'rows': len(data), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                              'data': data}
    assert set(required) <= set(records), (root, list(records))
    basic = records['OpBasicInfo']['data']
    assert len(basic) == 1 and basic[0]['Device Id'] == str(physical_chip), basic
    assert basic[0]['Op Name'] == 'post_kernel' and basic[0]['Block Dim'] == '1', basic
    return records


def trace(root, arms):
    results = {}
    for arm in arms:
        folders = list((root/'profile'/arm).glob('*_ascend_pt/ASCEND_PROFILER_OUTPUT'))
        assert len(folders) == 1, (arm, folders)
        path = folders[0]/'kernel_details.csv'
        data = rows(path)
        assert data and all(float(r['Duration(us)']) >= 0 for r in data)
        results[arm] = {'tasks': len(data), 'names': dict(collections.Counter(r['Name'] for r in data)),
                        'device_ids': sorted({r['Device_id'] for r in data}),
                        'kernel_us_per_call': sum(float(r['Duration(us)']) for r in data)/5,
                        'warmup': 5, 'active': 5, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return results


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--physical-chip', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    required = ['OpBasicInfo','PipeUtilization','Memory','MemoryL0','MemoryUB',
                'L2Cache','ArithmeticUtilization','ResourceConflictRatio']
    result = {'Default': micro(args.root/'indexer_microarch_default_v2', args.physical_chip, required),
              'MemoryDetail': micro(args.root/'indexer_microarch_memory_v1', args.physical_chip, required),
              'two_way_trace': trace(args.root/'real_indexer_bench_v1', ['native','fused']),
              'three_way_trace': trace(args.root/'real_indexer_threeway_v1', ['native','fused','scalar']),
              'scope': 'formal operator tensors, synthetic inputs; not full-model profiling',
              'counter_note': 'Pipe/wait counters overlap. Traffic bandwidth is not capacity occupancy or whole-chip HBM utilization.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('REAL_PROFILE_VALIDATED', args.output, flush=True)


if __name__ == '__main__':
    main()
