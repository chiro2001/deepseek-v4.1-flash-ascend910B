"""Fetch immutable, terminal init-only SDK receipts in one small SSH archive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


REMOTE = r'''
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1])
assert (root/'run.exit').is_file(), 'SDK case is not terminal'
files={}; oversized={}
for p in sorted(root.rglob('*')):
    if not p.is_file() or p.is_symlink(): continue
    rel=p.relative_to(root)
    if any(part in ('cache','tmp','__pycache__') for part in rel.parts): continue
    if rel.parts[0]!='gate' and str(rel) not in (
        'launched.json','run.exit','run.log','runner.pid','start.utc','end.utc',
        'resource_prelaunch.txt','resource_presubmit.txt','release_resource.json'):
        continue
    data=p.read_bytes()
    if len(data)>=1000000:
        oversized[str(rel)]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                             'retained_remotely':True,'raw_bytes_transferred':False}
        if str(rel)=='gate/result.json':
            result=json.loads(data)
            # The original contains the same full receipts saved per rank.
            compact={key:value for key,value in result.items() if key!='rank_results'}
            compact['rank_results']=[{key:row.get(key) for key in (
                'process_identity','phase','passed','stage_semantics_passed',
                'actual_copy_alias_gate_passed','alias_error','worker_lifecycle')}
                for row in result.get('rank_results',[])]
            compact['original_result']=dict(oversized[str(rel)],remote_path=str(p))
            content=(json.dumps(compact,indent=2)+'\n').encode()
            assert len(content)<1000000
            files['gate/result_compact.json']=content
        continue
    files[str(rel)]=data
assert 'run.exit' in files and 'launched.json' in files
if oversized:
    files['oversized_source_files.json']=(json.dumps(oversized,indent=2)+'\n').encode()
manifest={rel:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
          for rel,data in files.items()}
files['fetch_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as archive:
    for rel,data in files.items():
        item=tarfile.TarInfo(rel); item.size=len(data); item.mode=0o644
        archive.addfile(item,io.BytesIO(data))
data=buf.getvalue()
assert len(data)<1000000, 'Use COS for compact archive >=1 MB'
sys.stdout.buffer.write(data)
'''


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--host', default='a3-21')
    parser.add_argument('--container', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.job and args.job.replace('_', '').replace('-', '').isalnum()
    assert not args.output.exists(), 'Keep prior fetched evidence immutable'
    command = shlex.join(['docker', 'exec', args.container, 'python3', '-c',
                         REMOTE, '/work/results/' + args.job])
    data = subprocess.check_output(['ssh', '-o', 'ControlPath=none', '-o',
                                    'BatchMode=yes', args.host, command])
    assert len(data) < 1000000
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        members = archive.getmembers()
        assert len({item.name for item in members}) == len(members)
        assert all(item.isfile() and item.size < 1000000
                   and not Path(item.name).is_absolute()
                   and '..' not in Path(item.name).parts for item in members)
        payload = {item.name: archive.extractfile(item).read() for item in members}
    manifest = json.loads(payload['fetch_manifest.json'])
    assert set(payload) == set(manifest) | {'fetch_manifest.json'}
    for rel, expected in manifest.items():
        assert len(payload[rel]) == expected['bytes'], rel
        assert hashlib.sha256(payload[rel]).hexdigest() == expected['sha256'], rel
    args.output.mkdir(parents=True)
    for rel, content in payload.items():
        target = args.output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    receipt = {'job': args.job, 'container': args.container, 'host': args.host,
               'files': len(manifest), 'archive_bytes': len(data),
               'archive_sha256': hashlib.sha256(data).hexdigest(),
               'fetch_manifest_sha256': hashlib.sha256(payload['fetch_manifest.json']).hexdigest(),
               'run_exit': int(payload['run.exit'].strip()),
               'oversized_files_retained_remotely': json.loads(payload.get('oversized_source_files.json', b'{}')),
               'files_sha256_verified': True, 'scope': 'init-only SDK evidence; no performance claim'}
    (args.output / 'archive_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
