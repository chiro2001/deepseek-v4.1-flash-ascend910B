"""Fetch individual small result files; raw traces/requests stay on the device host."""
import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('files', nargs='+')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for relative in args.files:
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        code = ("from pathlib import Path\nimport sys\np=Path('/work/operator_opt/results')/"+repr(relative)+
                "\nassert p.stat().st_size<1000000, 'Use COS for >=1 MB'\nsys.stdout.buffer.write(p.read_bytes())")
        command = shlex.join(['docker','exec','dsv41-tiny-prof-20261009-c4','python','-c',code])
        data = subprocess.check_output(['ssh','-o','BatchMode=yes','a3-21',command])
        assert len(data) < 1000000
        target = args.output/relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        records.append({'relative':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        print('FETCHED',json.dumps(records[-1]),flush=True)
    manifest=args.output/'fetch_manifest.json'
    old=json.loads(manifest.read_text()) if manifest.is_file() else []
    kept=[r for r in old if r['relative'] not in {r['relative'] for r in records}]
    manifest.write_text(json.dumps(kept+records,indent=2)+'\n')


if __name__ == '__main__':
    main()
