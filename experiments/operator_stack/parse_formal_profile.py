"""Parse each collected rank directory in bounded, non-daemon subprocesses."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--expected-directories',type=int,default=56)
    args=p.parse_args();assert 1<=args.jobs<=8
    paths=sorted((args.root/'prof').rglob('*_ascend_pt'))
    assert args.expected_directories>0 and len(paths)==args.expected_directories,(len(paths),args.expected_directories)
    logdir=args.root/'offline_parse';logdir.mkdir(exist_ok=True)
    def parse(path):
        csv=path/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
        if csv.exists() and csv.stat().st_size>100:
            return {'path':str(path),'skipped_existing':True,'csv_present':True}
        relative=path.relative_to(args.root/'prof')
        log=logdir/('__'.join(relative.parts)+'.log')
        code='import torch_npu; torch_npu.profiler.profiler.analyse('+repr(str(path))+', max_process_number=1, export_type="text")'
        with log.open('w') as stream:
            proc=subprocess.run([sys.executable,'-u','-c',code],stdout=stream,stderr=subprocess.STDOUT)
        return {'path':str(path),'returncode':proc.returncode,'log':str(log),
                'csv_present':csv.exists() and csv.stat().st_size>100}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        receipts=list(pool.map(parse,paths))
    (args.root/'offline_parse_manifest.json').write_text(json.dumps(receipts,indent=2)+'\n')
    assert all(r['csv_present'] and r.get('returncode',0)==0 for r in receipts), 'Missing/failed CSV export'
    print('FORMAL_PROFILE_PARSED',json.dumps({'directories':len(receipts),'jobs':args.jobs}),flush=True)


if __name__=='__main__':main()
