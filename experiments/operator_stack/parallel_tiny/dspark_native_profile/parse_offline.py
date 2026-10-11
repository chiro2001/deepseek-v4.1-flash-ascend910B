"""Export eight collected profiles outside daemon workers, without NPU init."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=4)
    args = p.parse_args()
    assert 1 <= args.jobs <= 8
    paths = sorted((args.root/'profile').glob('*_ascend_pt'))
    assert len(paths) == 8, (len(paths), paths)
    logs = args.root/'offline_parse'
    logs.mkdir(exist_ok=True)
    def parse(path):
        csv = path/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
        if csv.exists() and csv.stat().st_size > 100:
            return {'path': str(path), 'existing_csv': True, 'csv_present': True}
        code = ('import torch,torch_npu\n'
                'assert not torch.npu.is_initialized()\n'
                'torch_npu.profiler.profiler.analyse('+repr(str(path))+',max_process_number=1,export_type="text")\n'
                'assert not torch.npu.is_initialized()\n'
                'print("OFFLINE_PARSE_NO_NPU_INIT")\n')
        env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0')
        log = logs/(path.name+'.log')
        with log.open('w') as stream:
            proc = subprocess.run([sys.executable, '-u', '-c', code], env=env,
                                  stdout=stream, stderr=subprocess.STDOUT)
        return {'path': str(path), 'returncode': proc.returncode, 'log': str(log),
                'no_npu_init': 'OFFLINE_PARSE_NO_NPU_INIT' in log.read_text(),
                'csv_present': csv.exists() and csv.stat().st_size > 100}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = list(pool.map(parse, paths))
    (args.root/'offline_parse_manifest.json').write_text(json.dumps(records, indent=2)+'\n')
    assert all(r['csv_present'] and r.get('returncode', 0) == 0 for r in records), records
    print('NATIVE_DSPARK_OFFLINE_PARSED', len(records), flush=True)


if __name__ == '__main__':
    main()
