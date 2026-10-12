"""One CPU-only entry from the final qkv seed to an immutable quality snapshot."""
import argparse
import json
from pathlib import Path

from .common import false_claims, read_pinned, reject_fixture, sha
from .export_candidate import export
from .prepare_snapshot import prepare

HERE = Path(__file__).resolve().parent


def _sibling(name, local_name):
    """Resolve a pinned sibling package from the deployment root or this tree."""
    for candidate in (HERE.parent / name, HERE.parent / local_name,
                      HERE.parent / (local_name + '_v2')):
        if (candidate / 'frozen_files.json').is_file():
            return candidate
    return HERE.parent / name


# [async arm v2] The v2 delivery pins the async-gated general/bridge packages
# and the async-gated counter file of the frozen candidate source snapshot.
GENERAL = _sibling('dspark_general_quality_api_cpu_20261012_v2', 'dspark_general_quality_api_cpu')
BRIDGE = Path('/work/dspark_api_counter_bridge_cpu_20261012_v2')
ASYNC_COUNTER = Path('/work/src_dspark_native_mls_v1/stack/formal_dspark_counters.py')


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--candidate-seed', required=True, type=Path)
    parser.add_argument('--candidate-seed-sha256', required=True)
    parser.add_argument('--sdk-manifest', type=Path)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--source-manifest-sha256')
    parser.add_argument('--general-package', type=Path, default=GENERAL)
    parser.add_argument('--bridge-package', type=Path, default=BRIDGE)
    parser.add_argument('--counter-file', type=Path, default=ASYNC_COUNTER)
    parser.add_argument('--output', type=Path, default=Path('/work/src_dspark_general_quality_v1'))
    parser.add_argument('--execute', action='store_true', help='CPU metadata export and snapshot copy only')
    args = parser.parse_args(argv)
    seed = read_pinned(args.candidate_seed, args.candidate_seed_sha256)
    reject_fixture(seed)
    source_ref = seed.get('frozen_source_manifest', {})
    source_path = args.source_manifest or Path(source_ref.get('path', args.candidate_seed))
    source_sha = args.source_manifest_sha256 or source_ref.get('sha256')
    if source_path == args.candidate_seed:
        source_sha = args.candidate_seed_sha256
    if not source_sha:
        raise ValueError('Final source manifest must have an explicit SHA pin')
    source_manifest = read_pinned(source_path, source_sha)
    sdk_ref = seed.get('sdk_source_manifest', {})
    if not sdk_ref:
        sdk_ref = seed.get('sdk_pair_provenance', {}).get('static', {}).get('source_manifest', {})
    sdk_path = args.sdk_manifest or (Path(sdk_ref['path']) if sdk_ref.get('path') else None)
    sdk_sha = seed.get('sdk_source_manifest_sha256')
    if sdk_path is None or not sdk_sha:
        raise ValueError('The final actual seed must bind its SDK manifest; supply --sdk-manifest when only its SHA is embedded')
    candidate = export(seed, seed['source_root'], source_manifest, source_path, sdk_path, sdk_sha)
    package_roots = {'dspark_api_counter_bridge_cpu': args.bridge_package,
                     'dspark_general_quality_api_cpu': args.general_package,
                     'dspark_general_quality_launcher_cpu': HERE}
    packages = {name: (root, sha(root / 'frozen_files.json')) for name, root in package_roots.items()}
    asset_manifest = json.loads((HERE / 'evidence/actual_asset_paths_CPU.json').read_text())
    if args.execute:
        exported = args.output.with_name(args.output.name + '_candidate_exported.json')
        if exported.exists():
            raise FileExistsError('Use a fresh actual contract/snapshot output')
        exported.parent.mkdir(parents=True, exist_ok=True)
        exported.write_text(json.dumps(candidate, indent=2) + '\n')
    result = prepare(candidate, args.output, packages, args.counter_file, asset_manifest, args.execute)
    result['actual_final_seed_used'] = True
    result['launch_or_HTTP_or_NPU_executed'] = False
    if args.execute:
        result['exported_candidate_contract'] = str(exported)
        result['exported_candidate_contract_sha256'] = sha(exported)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
