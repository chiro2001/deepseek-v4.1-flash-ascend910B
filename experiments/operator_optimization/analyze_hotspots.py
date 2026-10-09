"""Aggregate already validated model profiles by actual operator shape."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arms', default='overlap')
    args = p.parse_args()
    result = {'timing_scope': 'profiled cumulative kernel time, not recoverable wall time', 'arms': {}}
    for arm in args.arms.split(','):
        paths = list((args.root/'prof'/arm).rglob('kernel_details.csv'))
        assert len(paths) == 1, paths
        path = paths[0]
        rows = list(csv.DictReader(path.open()))
        steps = collections.Counter(r['Step Id'] for r in rows)
        assert len(steps) == 20 and {r['Device_id'] for r in rows} == {'4'}
        groups = collections.defaultdict(list)
        for r in rows:
            groups[(r['Type'], r['Input Shapes'], r['Accelerator Core'])].append(r)
        items = []
        for (kind, shape, core), batch in groups.items():
            item = {'type': kind, 'shape': shape, 'core': core,
                    'calls_per_step': len(batch)/20,
                    'us_per_step': sum(float(r['Duration(us)']) for r in batch)/20,
                    'blocks': sorted({r['Block Num'] for r in batch}),
                    'formats': sorted({r['Input Formats'] for r in batch})}
            for key in ['aic_scalar_ratio', 'aic_mte2_ratio', 'aic_mte1_ratio', 'aic_mac_ratio',
                        'aiv_vec_ratio', 'aiv_scalar_ratio', 'aiv_mte2_ratio',
                        'aic_icache_miss_rate', 'aiv_icache_miss_rate']:
                values = []
                for r in batch:
                    try:
                        value = float(r[key]); weight = float(r['aiv_time(us)' if key.startswith('aiv_') else 'aicore_time(us)'])
                        values.append((value, weight))
                    except (ValueError, KeyError):
                        pass
                if sum(w for _, w in values):
                    item[key] = sum(v*w for v, w in values)/sum(w for _, w in values)
            items.append(item)
        items.sort(key=lambda r: -r['us_per_step'])
        result['arms'][arm] = {'csv': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'rows': len(rows), 'steps': 20, 'shapes': items}
        print('HOTSPOTS', arm, json.dumps(items[:18]), flush=True)
    (args.root/'hotspots.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
