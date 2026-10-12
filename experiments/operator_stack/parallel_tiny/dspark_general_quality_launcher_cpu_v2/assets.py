"""Read-only original evaluator/encoder/image/canonical-data provenance."""
import argparse
import json
from pathlib import Path

from parallel_tiny.dspark_general_quality_api_cpu.scoring import canonical_gold
from .common import false_claims, sha

HERE = Path(__file__).resolve().parent


def _api_package_root():
    """[async arm v2] The general package is delivered under its v1 name or the
    versioned deployment name; both are checked so the same code runs unchanged."""
    for root in (HERE.parent / 'dspark_general_quality_api_cpu',
                 HERE.parent / 'dspark_general_quality_api_cpu_20261012_v2'):
        if (root / 'evidence' / 'evaluator_sources.json').is_file():
            return root
    return HERE.parent / 'dspark_general_quality_api_cpu'


EVALUATOR_GUARD = _api_package_root() / 'evidence/evaluator_sources.json'


def capture(client_root, encoder, images, official_dir, gsm_data):
    roots = {name: Path(value).resolve(strict=True) for name, value in {
        'client_root': client_root, 'encoder': encoder, 'images': images,
        'official_dir': official_dir, 'gsm_data': gsm_data,
    }.items()}
    if roots['encoder'] != (roots['official_dir'] / 'encoding').resolve(strict=True):
        raise ValueError('Use the actual official checkpoint encoder owner')
    if roots['images'] != (roots['official_dir'] / 'inference/examples/images').resolve(strict=True):
        raise ValueError('Use the unchanged original official image directory')
    if not (roots['encoder'] / 'encoding.py').is_file():
        raise ValueError('Official encode_messages source is missing')
    files = {}
    frozen = json.loads(EVALUATOR_GUARD.read_text())['files']
    for label, name in [('gsm', 'acc_eval.py'), ('vision', 'vision_accuracy_check.py'),
                        ('vision_wrapper', 't_vision.py')]:
        path = roots['client_root'] / 'tests' / name
        if sha(path) != frozen[label]['sha256']:
            raise ValueError('Original evaluator changed: ' + name)
        files['evaluator/' + name] = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
    for path in sorted(roots['encoder'].rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            files['encoder/' + str(path.relative_to(roots['encoder']))] = {
                'path': str(path), 'resolved': str(path.resolve(strict=True)),
                'sha256': sha(path), 'bytes': path.stat().st_size}
    for name in ('carrots.jpeg', 'corn.jpeg'):
        path = roots['images'] / name
        files['images/' + name] = {'path': str(path), 'resolved': str(path.resolve(strict=True)),
                                  'sha256': sha(path), 'bytes': path.stat().st_size}
    _, splits = canonical_gold(roots['gsm_data'])
    for name, value in splits.items():
        path = roots['gsm_data'] / (name + '.jsonl')
        files['gsm8k/' + name + '.jsonl'] = {'path': str(path), 'sha256': value,
                                           'bytes': path.stat().st_size}
    return {'schema': 'dspark_general_quality_asset_manifest_v1',
            'CPU_readonly_real_file_provenance': True, 'paths': {key: str(value) for key, value in roots.items()},
            'files': files, 'canonical_GSM_split_sha256': splits,
            'evaluator_scripts_executed': False, 'encoder_modules_imported': False,
            'image_pixels_decoded': False, 'NPU_calls': 0, 'HTTP_requests': 0,
            'services_started': 0, **false_claims()}


def verify(manifest):
    if manifest.get('schema') != 'dspark_general_quality_asset_manifest_v1':
        raise ValueError('Real original asset manifest is required')
    reference = json.loads((HERE / 'evidence/actual_asset_paths_CPU.json').read_text())
    if set(manifest['files']) != set(reference['files']):
        raise ValueError('Original evaluator/encoder/image/canonical-data asset set is incomplete')
    for name, item in manifest['files'].items():
        original = reference['files'][name]
        if item['sha256'] != original['sha256'] or item['bytes'] != original['bytes']:
            raise ValueError('Quality assets differ from the observed original source: ' + name)
        path = Path(item['path'])
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Quality asset source changed: ' + str(path))
    current = capture(**manifest['paths'])
    if set(current['files']) != set(manifest['files']):
        raise ValueError('Actual original asset directory membership changed')
    for name, item in current['files'].items():
        if any(item.get(key) != manifest['files'][name].get(key) for key in ('path', 'sha256', 'bytes')):
            raise ValueError('Asset paths no longer refer to the actual official owners: ' + name)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ('client-root', 'encoder', 'images', 'official-dir', 'gsm-data', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Write CPU file-provenance manifest only')
    args = parser.parse_args(argv)
    result = capture(args.client_root, args.encoder, args.images, args.official_dir, args.gsm_data)
    if args.execute:
        if args.output.exists():
            raise FileExistsError('Asset manifest must be new')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
