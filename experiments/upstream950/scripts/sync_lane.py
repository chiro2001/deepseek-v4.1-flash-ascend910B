"""Sync small source/docs only into this lane; refuse large SSH transfers."""
import ast
import argparse
import subprocess
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--prepare-only',action='store_true')
args=parser.parse_args()
runtime = root / 'runtime'
runtime.mkdir(exist_ok=True)
bundle = runtime / 'lane_source.tgz'
paths = sorted((root / 'scripts').glob('*.py')) + sorted((root / 'scripts').glob('*.sh'))
paths += sorted(root.glob('*.md'))
for path in paths:
    if path.suffix == '.py':
        ast.parse(path.read_text(), filename=str(path))
with tarfile.open(bundle, 'w:gz') as archive:
    for path in paths:
        archive.add(path, arcname=str(path.relative_to(root)))
assert bundle.stat().st_size < 1_000_000, 'Use COS for source bundles >= 1 MB'
if args.prepare_only:
    print('Prepared source bundle:',str(bundle),bundle.stat().st_size)
    raise SystemExit(0)
remote = '/home/l00886679/projects/dsv41-tiny-upstream950-20261009'
subprocess.run(['scp','-q',str(bundle),'a3-21:'+remote+'/lane_source.tgz'],check=True)
code = f'''from pathlib import Path
import tarfile
root=Path({remote!r})
with tarfile.open(root/'lane_source.tgz') as archive:archive.extractall(root,filter='data')
print('Synced independent lane source:',str(root))
'''
subprocess.run(['ssh','-o','BatchMode=yes','a3-21','python3','-'],input=code,text=True,check=True)
