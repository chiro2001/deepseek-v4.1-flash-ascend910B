"""Prepare an immutable SDK case snapshot on the host; never allocate an NPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, base_name, source_name, bundle_name, package, freeze_sha):
    root = Path(root).resolve()
    for name in (base_name, source_name, bundle_name, package):
        assert name and name.replace('_', '').replace('-', '').replace('.', '').isalnum()
    base, source, bundle = root / base_name, root / source_name, root / bundle_name
    assert base.is_dir() and not source.exists() and bundle.is_file()
    old = json.loads((base / 'stack/parallel_tiny/dspark_formal_prep/freeze.json').read_text())
    for rel, expected in old['files'].items():
        assert sha(base / 'stack' / rel) == expected, rel
    with tarfile.open(bundle) as archive:
        entries = archive.getmembers()
        assert entries and all(e.isfile() and e.size < 1000000 for e in entries)
        names = [Path(e.name) for e in entries]
        assert len(set(names)) == len(names)
        assert all(not n.is_absolute() and '..' not in n.parts and len(n.parts) >= 2
                   and n.parts[0] == 'gate' for n in names)
        payload = {str(Path(*n.parts[1:])): archive.extractfile(e).read()
                   for n, e in zip(names, entries)}
    assert hashlib.sha256(payload['freeze.json']).hexdigest() == freeze_sha
    freeze = json.loads(payload['freeze.json'])
    assert set(payload) == set(freeze['files']) | {'freeze.json', 'SHA256SUMS'}
    for rel, expected in freeze['files'].items():
        assert hashlib.sha256(payload[rel]).hexdigest() == expected, rel
    source.mkdir()
    for name in ('stack', 'operator', 'up950'):
        shutil.copytree(base / name, source / name, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (source / 'runtime/tmp').mkdir(parents=True)
    gate = source / 'stack/parallel_tiny/dspark_formal_audit' / package
    gate.mkdir(exist_ok=False)
    for rel, data in payload.items():
        target = gate / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    paths = ({source / 'stack' / rel for rel in old['files']}
             | {gate / rel for rel in payload}
             | {source / 'stack' / rel for rel in ('run_stack.sh', 'formal_model_contract.py', 'sitecustomize.py')})
    manifest = {
        'scope': 'normal-init-only SDK capability; no model or performance receipt',
        'driver': f'parallel_tiny/dspark_formal_audit/{package}/init_only_driver.py',
        'formal_checkpoint_config_sha256': '40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424',
        'base': str(base), 'gate_freeze_sha256': freeze_sha,
        'bundle_sha256': sha(bundle), 'NPU_calls': 0,
        'files': {str(p.relative_to(source)): sha(p) for p in sorted(paths)},
    }
    (source / 'initialized_sdk_snapshot.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return {'source': str(source), 'files': len(paths), 'NPU_calls': 0,
            'snapshot_manifest_sha256': sha(source / 'initialized_sdk_snapshot.json')}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ('root', 'base-name', 'source-name', 'bundle-name', 'package', 'freeze-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.base_name, args.source_name, args.bundle_name,
                             args.package, args.freeze_sha256)), flush=True)


if __name__ == '__main__':
    main()
