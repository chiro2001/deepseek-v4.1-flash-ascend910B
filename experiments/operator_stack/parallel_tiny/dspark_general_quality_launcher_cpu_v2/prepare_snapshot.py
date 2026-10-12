"""Copy a verified real-source candidate into a new CPU-only quality snapshot."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil

from . import assets
from .common import (COUNTER_SHA, checkpoint_metadata, counter_sha, false_claims, fingerprint,
                     read_pinned, relocate, sha, source_map, validate_real_candidate, verify_map)

PACKAGES = ('dspark_api_counter_bridge_cpu', 'dspark_general_quality_api_cpu',
            'dspark_general_quality_launcher_cpu')


def package_files(root, expected_manifest_sha):
    root = Path(root).resolve(strict=True)
    manifest = read_pinned(root / 'frozen_files.json', expected_manifest_sha)
    mapping = source_map(manifest)
    verify_map(root, mapping)
    return {**{name: (root / name).read_bytes() for name in mapping},
            'frozen_files.json': (root / 'frozen_files.json').read_bytes()}


def source_files(root):
    """Close source symlinks into file bytes; never copy a checkpoint tree."""
    root = Path(root).resolve(strict=True)
    result = []
    for name in ('stack', 'operator', 'up950'):
        directory = root / name
        if not directory.is_dir():
            raise ValueError('Original candidate component tree is missing: ' + name)
        for current, directories, names in os.walk(directory, followlinks=False):
            directories[:] = sorted(value for value in directories if value != '__pycache__')
            for value in directories:
                if (Path(current) / value).is_symlink():
                    raise ValueError('Source directory symlinks need an explicit closed snapshot: ' + value)
            for value in sorted(names):
                path = Path(current) / value
                if path.suffix == '.pyc':
                    continue
                if not path.is_file() or path.suffix == '.safetensors':
                    raise ValueError('Only source/component files may enter a quality snapshot: ' + str(path))
                result.append(path)
    return result


def overlay(target, payload):
    for name, raw in payload.items():
        path = target / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('Never overwrite a different existing candidate source: ' + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(raw)


def prepare(candidate, output, packages, counter_file, asset_manifest, execute=False):
    validate_real_candidate(candidate)
    before = copy.deepcopy(candidate)
    checkpoint = checkpoint_metadata(candidate)
    assets.verify(asset_manifest)
    if sha(counter_file) != counter_sha():
        raise ValueError('Exact frozen counter for the selected scheduler arm is required')
    base = Path(candidate['source_root']).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(base) or base.is_relative_to(output):
        raise FileExistsError('Use a new independent quality source snapshot')
    payloads = {}
    for name in PACKAGES:
        root, expected = packages[name]
        payloads[name] = package_files(root, expected)
    files = source_files(base)
    proposal = {'schema': 'dspark_general_quality_snapshot_plan_v1', 'CPU_only': True,
                'source': str(base), 'output': str(output), 'candidate_id': candidate['candidate_id'],
                'original_candidate_fingerprint': fingerprint(before), 'source_files_to_copy': len(files),
                'packages': {name: {'manifest_sha256': value[1], 'texts': len(payloads[name]) - 1}
                             for name, value in packages.items()},
                'counter_sha256': counter_sha(), 'NPU_calls': 0, 'HTTP_requests': 0,
                'services_started': 0, **false_claims()}
    if not execute:
        return proposal
    output.mkdir(parents=True)
    try:
        for path in files:
            destination = output / path.relative_to(base)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination, follow_symlinks=True)
        for name, payload in payloads.items():
            overlay(output / 'stack/parallel_tiny' / name, payload)
        counter = output / 'stack/parallel_tiny/dspark_general_quality_launcher_cpu/assets/frozen_v4_counter.py'
        overlay(counter.parent, {counter.name: Path(counter_file).read_bytes()})
        original_eval = json.loads((Path(packages['dspark_general_quality_api_cpu'][0]) /
                                    'evidence/evaluator_sources.json').read_text())['files']
        client = output / 'client/tests'
        overlay(client, {Path(item['path']).name: item['source'].encode() for item in original_eval.values()})
        verify_map(output, before['source_sha256'])
        after = copy.deepcopy(before)
        for key in ('source_root', 'startup_env', 'startup_env_basis', 'pythonpath'):
            after[key] = relocate(after[key], base, output)
        after['startup_env_fingerprint'] = fingerprint(after['startup_env'])
        (output / 'runtime/tmp').mkdir(parents=True)
        asset_manifest = copy.deepcopy(asset_manifest)
        asset_manifest['original_client_root'] = asset_manifest['paths']['client_root']
        asset_manifest['paths']['client_root'] = str(output / 'client')
        for key, item in asset_manifest['files'].items():
            if key.startswith('evaluator/'):
                item['original_path'] = item['path']
                item['path'] = str(client / key.split('/', 1)[1])
        asset_path = output / 'quality_assets.json'
        asset_path.write_text(json.dumps(asset_manifest, indent=2) + '\n')
        mapping = {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob('*'))
                   if path.is_file() and 'runtime' not in path.relative_to(output).parts}
        after['source_sha256'] = mapping
        after['original_candidate_source_provenance'] = {
            'candidate_fingerprint': fingerprint(before), 'source_root': str(base),
            'source_sha256': before['source_sha256'], 'original_model_kwargs': before['model_kwargs'],
            'original_startup_env': before['startup_env'], 'source_manifest': before.get('frozen_source_manifest'),
        }
        after['checkpoint_metadata'] = checkpoint
        after.update(actual_flags=false_claims(), CPU_metadata_preparation_only=True,
                     actual_runtime_candidate_loaded=False)
        validate_real_candidate(after)
        contract_path = output / 'candidate_contract.json'
        contract_path.write_text(json.dumps(after, indent=2) + '\n')
        manifest = {**proposal, 'schema': 'dspark_general_quality_source_snapshot_v1',
                    'files': mapping, 'candidate_contract': {'path': str(contract_path), 'sha256': sha(contract_path)},
                    'candidate_fingerprint': fingerprint(after), 'asset_manifest_sha256': sha(asset_path),
                    'counter_file': str(counter), 'actual_snapshot_files_written': True,
                    'source_weights_or_original_candidate_code_modified': False,
                    'original_source_sha256_all_unchanged': True, 'checkpoint': checkpoint}
        snapshot_path = output / 'quality_source_snapshot.json'
        snapshot_path.write_text(json.dumps(manifest, indent=2) + '\n')
        return {**proposal, 'actual_snapshot_files_written': True, 'snapshot_manifest': str(snapshot_path),
                'snapshot_manifest_sha256': sha(snapshot_path), 'candidate_contract_sha256': sha(contract_path),
                'snapshot_source_files': len(mapping)}
    except BaseException as exc:
        (output / 'INCOMPLETE_snapshot.json').write_text(json.dumps({
            'error_type': type(exc).__name__, 'error': str(exc), **false_claims()}, indent=2) + '\n')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--counter-file', type=Path, required=True)
    parser.add_argument('--asset-manifest', type=Path, required=True)
    parser.add_argument('--asset-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    for name in PACKAGES:
        parser.add_argument('--' + name.replace('_', '-') + '-root', type=Path, required=True)
        parser.add_argument('--' + name.replace('_', '-') + '-manifest-sha256', required=True)
    parser.add_argument('--execute', action='store_true', help='CPU snapshot copy only; never launches a service')
    args = parser.parse_args(argv)
    candidate = read_pinned(args.candidate, args.candidate_sha256)
    manifest = read_pinned(args.asset_manifest, args.asset_manifest_sha256)
    packages = {name: (getattr(args, name + '_root'), getattr(args, name + '_manifest_sha256')) for name in PACKAGES}
    print(json.dumps(prepare(candidate, args.output, packages, args.counter_file, manifest, args.execute), indent=2))


if __name__ == '__main__':
    main()
