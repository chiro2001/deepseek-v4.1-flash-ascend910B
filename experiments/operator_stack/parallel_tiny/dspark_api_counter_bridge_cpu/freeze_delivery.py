"""Freeze only this owned text package and make a deterministic <1MB archive."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / 'dspark_api_counter_bridge_cpu_v1_compact.tgz'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    validation = json.loads((HERE / 'evidence/cpu_validation.json').read_text())
    if (not validation['passed'] or validation['tests_run'] != 21 or
            validation['negative_case_count'] != 90 or validation['unhandled_async_errors'] != 0 or
            validation['actual_npu_calls'] != 0 or validation['actual_http_inference_requests'] != 0 or
            any(validation['claims'].values())):
        raise ValueError('The frozen CPU21 validation must pass and all real claims remain false')
    prior = {}
    parent = HERE.parent
    for rel in ('gmm_lane/sparse_msd/m1_event_perf_v1_failure/frozen_files.json',
            'gmm_lane/sparse_msd/m1_event_perf_v2/runtime_frozen_files.json',
            'gmm_lane/sparse_msd/k5_route_feasibility_cpu/frozen_files.json',
            'dspark_context_graph_design_cpu/frozen_files.json'):
        manifest = json.loads((parent / rel).read_text())
        root = Path(manifest['root'])
        for name, receipt in manifest['text_files'].items():
            data = (root / name).read_bytes()
            if len(data) != receipt['bytes'] or digest(data) != receipt['sha256']:
                raise ValueError(f'Previous frozen artifact changed: {root / name}')
        prior[rel] = {'checked_text_files': len(manifest['text_files']),
            'unchanged': True, 'manifest_sha256': digest((parent / rel).read_bytes())}
    texts = {}
    for path in sorted(HERE.rglob('*')):
        if (not path.is_file() or '__pycache__' in path.parts or
                path.name in ('frozen_files.json', 'delivery.json') or
                path.suffix not in ('.py', '.md', '.json', '.log')):
            continue
        data = path.read_bytes()
        data.decode('utf-8')
        if len(data) >= 1_000_000:
            raise ValueError('Every transferred text file must be smaller than 1MB')
        texts[str(path.relative_to(HERE))] = {'sha256': digest(data), 'bytes': len(data)}
    frozen = {'root': str(HERE), 'CPU_only': True, 'NPU_permission': 0,
        'actual_NPU_calls': 0, 'actual_HTTP_inference_requests': 0, 'services_started': 0,
        'text_files': texts, 'CPU_tests_passed': 21, 'negative_cases_rejected': 90,
        'actual_AST_functions_executed_with_CPU_fakes': 9, 'installed_API_source_files_guarded': 16,
        'previous_frozen_packages': prior, 'actual_claims': validation['claims'],
        'actual_frozen_v4_counter_sha256': validation['unchanged_frozen_counter_sha256']}
    manifest_path = HERE / 'frozen_files.json'
    manifest_path.write_text(json.dumps(frozen, indent=2) + '\n')
    with ARCHIVE.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w') as tar:
                for name in [*texts, 'frozen_files.json']:
                    data = (HERE / name).read_bytes()
                    info = tarfile.TarInfo(HERE.name + '/' + name)
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))
    data = ARCHIVE.read_bytes()
    if len(data) >= 1_000_000:
        raise ValueError('Compact archive must remain smaller than 1MB for SSH transfer')
    with tarfile.open(ARCHIVE, 'r:gz') as tar:
        for name, receipt in texts.items():
            member = tar.extractfile(HERE.name + '/' + name).read()
            if digest(member) != receipt['sha256']:
                raise ValueError('Archive text differs from the frozen package')
    delivery = {'owned_root': str(HERE), 'compact': str(ARCHIVE),
        'compact_bytes': len(data), 'compact_sha256': digest(data),
        'frozen_manifest_sha256': digest(manifest_path.read_bytes()),
        'frozen_text_file_count': len(texts), 'CPU_tests': 21, 'negative_cases': 90,
        'NPU_permission': 0, 'actual_NPU_calls': 0, 'actual_HTTP_inference_requests': 0,
        'prior_frozen_texts_checked_unchanged': sum(item['checked_text_files'] for item in prior.values()),
        'claims': validation['claims']}
    (HERE / 'delivery.json').write_text(json.dumps(delivery, indent=2) + '\n')
    print(json.dumps(delivery, indent=2))


if __name__ == '__main__':
    main()
