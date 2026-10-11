"""Prepare one immutable real-model native API diagnostic snapshot on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    for key in ('root', 'base-name', 'source-name', 'bundle-name', 'api-freeze-sha256'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--entry-package', choices=('dspark_api_native_diagnostic_v1', 'dspark_api_native_diagnostic_v2'),
                   default='dspark_api_native_diagnostic_v1')
    a = p.parse_args()
    for name in (a.base_name, a.source_name, a.bundle_name):
        assert name and name.replace('_', '').replace('-', '').replace('.', '').isalnum()
    root = Path(a.root).resolve()
    base, source, bundle = root / a.base_name, root / a.source_name, root / a.bundle_name
    assert base.is_dir() and not source.exists()
    old = json.loads((base / 'stack/parallel_tiny/dspark_formal_prep/freeze.json').read_text())
    for rel, expected in old['files'].items():
        assert sha(base / 'stack' / rel) == expected, rel
    with tarfile.open(bundle) as archive:
        members = archive.getmembers()
        assert members and len({m.name for m in members}) == len(members)
        assert all(m.isfile() and m.size < 1000000 and not Path(m.name).is_absolute()
                   and '..' not in Path(m.name).parts and m.name.startswith('stack/')
                   for m in members)
        payload = {m.name: archive.extractfile(m).read() for m in members}
    api = 'stack/parallel_tiny/dspark_api_counter_bridge_cpu/'
    assert hashlib.sha256(payload[api + 'frozen_files.json']).hexdigest() == a.api_freeze_sha256
    frozen = json.loads(payload[api + 'frozen_files.json'])
    api_names = set(frozen['text_files']) | {'frozen_files.json', 'delivery.json'}
    entry_file = 'stack/parallel_tiny/' + a.entry_package + '/entry.py'
    expected_names = ({api + rel for rel in api_names}
                      | {'stack/bench_formal_dspark.py', 'stack/formal_dspark_counters.py', entry_file})
    if a.entry_package == 'dspark_api_native_diagnostic_v2':
        expected_names.add('stack/parallel_tiny/' + a.entry_package + '/http_facade.py')
    assert set(payload) == expected_names
    for rel, spec in frozen['text_files'].items():
        assert hashlib.sha256(payload[api + rel]).hexdigest() == spec['sha256'], rel
    counter = 'da845a5891557f21e80eac06d96aec1955263d1ae93afbc4d84e86dd9585dbec'
    assert hashlib.sha256(payload['stack/formal_dspark_counters.py']).hexdigest() == counter
    assert hashlib.sha256(payload['stack/bench_formal_dspark.py']).hexdigest() == 'e93582adf78029b092baacd9f7f9ae80fc451f13554bcc70209fa4721c527888'
    source.mkdir()
    for name in ('stack', 'operator', 'up950'):
        shutil.copytree(base / name, source / name, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (source / 'runtime/tmp').mkdir(parents=True)
    for rel, raw in payload.items():
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    paths = {source / 'stack' / rel for rel in old['files']}
    paths |= {source / rel for rel in payload}
    paths |= {source / 'stack' / rel for rel in ('run_stack.sh', 'sitecustomize.py', 'formal_model_contract.py')}
    manifest = {'scope': 'native DSpark real API integration diagnostic; no full-consumer or performance admission',
                'base': str(base), 'api_freeze_sha256': a.api_freeze_sha256,
                'bundle_sha256': sha(bundle), 'frozen_counter_sha256': counter,
                'entry_file': entry_file.removeprefix('stack/'),
                'checkpoint_config_sha256': '40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424',
                'files': {str(path.relative_to(source)): sha(path) for path in sorted(paths)},
                'NPU_calls': 0, 'full_consumer_audit_passed': False, 'performance_claim': None}
    path = source / 'native_api_snapshot.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'source': str(source), 'files': len(paths), 'manifest_sha256': sha(path), 'NPU_calls': 0}))


if __name__ == '__main__':
    main()
