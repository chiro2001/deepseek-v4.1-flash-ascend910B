"""Export a real-source candidate from pinned manifests; default is CPU plan."""
import argparse
import copy
import json
from pathlib import Path

from .common import (checkpoint_metadata, false_claims, fingerprint, read_pinned,
                     reject_fixture, sha, source_map, validate_real_candidate, verify_map, verify_sdk_manifest)


def export(seed, source_root, manifest, manifest_path, sdk_path, sdk_sha):
    reject_fixture(seed)
    candidate = copy.deepcopy(seed)
    root = Path(source_root).resolve(strict=True)
    mapping = verify_map(root, source_map(manifest))
    seed_mapping = candidate.get('source_sha256')
    if seed_mapping is not None:
        # qkv freezes its owned package; the root snapshot also guards inherited
        # formal worker/checker files. Every seed entry must agree with that map.
        seed_mapping = source_map({'source_sha256': seed_mapping})
        if any(mapping.get(name) != value for name, value in seed_mapping.items()):
            raise ValueError('Seed candidate differs from its pinned frozen source manifest')
        candidate['original_seed_package_source_sha256'] = seed_mapping
    if candidate.get('source_root') not in (None, str(root)):
        raise ValueError('Do not silently rebind another candidate source owner')
    if candidate.get('sdk_source_manifest_sha256') not in (None, sdk_sha):
        raise ValueError('SDK provenance belongs to another candidate')
    sdk = read_pinned(sdk_path, sdk_sha)
    candidate.update(schema='dspark_general_quality_actual_source_candidate_v1', source_root=str(root),
                     source_sha256=mapping, sdk_source_manifest_sha256=sdk_sha,
                     startup_env_fingerprint=fingerprint(candidate['startup_env']),
                     actual_flags=false_claims())
    candidate['frozen_source_manifest'] = {'path': str(Path(manifest_path).resolve()),
                                           'sha256': sha(manifest_path),
                                           'scope': 'exact frozen listed candidate/base stack files; not all installed SDK or weight contents'}
    candidate['sdk_source_manifest'] = {'path': str(Path(sdk_path).resolve()), 'sha256': sdk_sha}
    candidate['SDK_provenance_readonly_verification'] = verify_sdk_manifest(sdk)
    validate_real_candidate(candidate)
    candidate['checkpoint_metadata'] = checkpoint_metadata(candidate)
    candidate.update(source_freeze_and_checkpoint_metadata_verified=True,
                     CPU_metadata_preparation_only=True, actual_runtime_candidate_loaded=False)
    return candidate


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ('seed', 'source-root', 'source-manifest', 'sdk-manifest', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('source-manifest-sha256', 'sdk-manifest-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--execute', action='store_true', help='Only write the CPU-exported actual-source contract; no backend calls')
    args = parser.parse_args(argv)
    seed = json.loads(args.seed.read_text())
    manifest = read_pinned(args.source_manifest, args.source_manifest_sha256)
    candidate = export(seed, args.source_root, manifest, args.source_manifest,
                       args.sdk_manifest, args.sdk_manifest_sha256)
    if args.execute:
        if args.output.exists():
            raise FileExistsError('Actual candidate contract output must be new')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(candidate, indent=2) + '\n')
    print(json.dumps({'CPU_metadata_preparation_only': True, 'candidate_id': candidate['candidate_id'],
                      'candidate_fingerprint': fingerprint(candidate), 'source_files': len(candidate['source_sha256']),
                      'output_written': bool(args.execute), 'NPU_calls': 0, 'HTTP_requests': 0,
                      'services_started': 0, **false_claims()}, indent=2))


if __name__ == '__main__':
    main()
