"""Export compact evidence with hashes of the complete remote request records."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--run', action='append', required=True)
parser.add_argument('--file', action='append', default=[])
parser.add_argument('--output', required=True)
args = parser.parse_args()
root = Path('/work/results')
target = Path(args.output)
target.parent.mkdir(parents=True, exist_ok=True)
files = []
standalone_manifest={}
for name in args.file:
    assert '/' not in name and '..' not in name and name.endswith('.json')
    path=root/name
    standalone_manifest[name]={'bytes':path.stat().st_size,
                               'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    files.append(path)
if standalone_manifest:
    manifest_path=root/'standalone_manifest.json'
    manifest_path.write_text(json.dumps(standalone_manifest,indent=2))
    files.append(manifest_path)
for run in args.run:
    assert '/' not in run and '..' not in run
    folder = root / run
    rows = json.loads((folder/'requests.json').read_text())
    summaries = []
    for row in rows:
        summary = {key: value for key, value in row.items() if key not in ['routes','logprobs']}
        if 'routes' in row:
            summary['route_sha256'] = hashlib.sha256(json.dumps(row['routes'],separators=(',',':')).encode()).hexdigest()
        if 'logprobs' in row:
            summary['logprobs_sha256'] = hashlib.sha256(json.dumps(row['logprobs'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        summaries.append(summary)
    (folder/'request_summary.json').write_text(json.dumps(summaries, indent=2))
    manifest = {}
    for path in sorted(folder.glob('*.json')):
        if path.name=='raw_manifest.json':
            continue  # A manifest cannot include its previous generation's hash.
        manifest[path.name] = {'bytes': path.stat().st_size,
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.name != 'requests.json':
            files.append(path)
    (folder/'raw_manifest.json').write_text(json.dumps(manifest,indent=2))
    files.append(folder/'raw_manifest.json')
with tarfile.open(target,'w:gz') as archive:
    for path in files:
        archive.add(path,arcname=str(path.relative_to(root)))
assert target.stat().st_size < 1_000_000, 'Use COS if evidence archive reaches 1 MB'
print(json.dumps({'path':str(target),'bytes':target.stat().st_size,'runs':args.run}))
