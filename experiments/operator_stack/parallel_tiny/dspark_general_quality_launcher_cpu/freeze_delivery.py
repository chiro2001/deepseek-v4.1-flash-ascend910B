"""Freeze independent launcher CPU preparation; do not mutate prior packages."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    cpu = json.loads((HERE / 'evidence/CPU_validation.json').read_text())
    assert cpu['passed'] and cpu['tests'] == 14 and cpu['negative_case_count'] == 59
    assert cpu['NPU_calls'] == cpu['HTTP_requests'] == cpu['services_started'] == cpu['real_signals_sent'] == 0
    assert cpu['backend_modules_imported'] == [] and not any(cpu['actual_flags'].values())
    assets = json.loads((HERE / 'evidence/actual_asset_paths_CPU.json').read_text())
    assert len(assets['files']) == 20 and not assets['evaluator_scripts_executed'] and not assets['encoder_modules_imported']
    assert assets['NPU_calls'] == assets['HTTP_requests'] == assets['services_started'] == 0
    assert assets['backend_modules_imported'] == []
    checkpoint = json.loads((HERE / 'evidence/actual_checkpoint_metadata_CPU_reference.json').read_text())
    assert checkpoint['original_checkpoint_inspection_passed'] and checkpoint['shards'] == 90
    assert checkpoint['vision_tensor_headers'] == 266 and checkpoint['MTP_tensor_headers'] == 4718
    assert checkpoint['indexed_MTP_and_vision_tensor_headers_verified'] == 4984
    assert checkpoint['actual_candidate_contract_exported'] is False and not checkpoint['tensor_payloads_read']
    assert checkpoint['backend_modules_imported'] == [] and not any(checkpoint['actual_flags'].values())
    plan = HERE.parent / 'dspark_formal_audit/consumer_source_plan.json'
    assert sha(plan.read_bytes()) == '5d6cd23dd434662932b4446435e8d632a18136e2785db861aabec3145bc0455c'
    prior = {}
    for rel in (
        'dspark_api_counter_bridge_cpu/frozen_files.json',
        'dspark_api_workload_cpu/frozen_files.json',
        'gmm_lane/sparse_msd/m1_event_perf_v1_failure/frozen_files.json',
        'gmm_lane/sparse_msd/m1_event_perf_v2/runtime_frozen_files.json',
        'gmm_lane/sparse_msd/k5_route_feasibility_cpu/frozen_files.json',
        'dspark_context_graph_design_cpu/frozen_files.json',
        'dspark_prepack_audit_cpu/frozen_files.json',
        'dspark_general_quality_api_cpu/frozen_files.json',
    ):
        path = HERE.parent / rel
        manifest = json.loads(path.read_text())
        root = Path(manifest['root'])
        for name, item in manifest['text_files'].items():
            data = (root / name).read_bytes()
            assert sha(data) == item['sha256'] and len(data) == item['bytes']
        prior[rel] = {'checked_unchanged_texts': len(manifest['text_files']), 'manifest_sha256': sha(path.read_bytes())}
    assert sum(item['checked_unchanged_texts'] for item in prior.values()) == 286
    texts = {}
    for path in sorted(HERE.rglob('*')):
        if (not path.is_file() or '__pycache__' in path.parts or path.name in ('frozen_files.json', 'delivery.json') or
                path.suffix not in ('.py', '.sh', '.json', '.md', '.txt', '.log')):
            continue
        data = path.read_bytes();data.decode('utf-8');assert len(data) < 1_000_000
        texts[str(path.relative_to(HERE))] = {'bytes': len(data), 'sha256': sha(data)}
    manifest = {'root': str(HERE), 'CPU_only': True, 'NPU_permission': 0, 'NPU_calls': 0,
                'HTTP_requests': 0, 'services_started': 0, 'real_signals_sent': 0,
                'CPU_tests': 14, 'negative_cases': 59, 'text_files': texts,
                'actual_asset_files_readonly_verified': 20, 'actual_checkpoint_shards_readonly_verified': 90,
                'actual_MTP_and_vision_headers_readonly_verified': 4984,
                'prior_frozen_packages': prior, 'frozen17_plan_unchanged': True, 'actual_flags': cpu['actual_flags']}
    manifest_path = HERE / 'frozen_files.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    archive = HERE / 'dspark_general_quality_launcher_cpu_v1_compact.tgz'
    with archive.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w') as tar:
                for name in [*texts, 'frozen_files.json']:
                    data = (HERE / name).read_bytes()
                    info = tarfile.TarInfo(HERE.name + '/' + name)
                    info.size, info.mode, info.mtime = len(data), 0o644, 0
                    tar.addfile(info, io.BytesIO(data))
    data = archive.read_bytes();assert len(data) < 1_000_000
    with tarfile.open(archive) as tar:
        for name, item in texts.items():
            assert sha(tar.extractfile(HERE.name + '/' + name).read()) == item['sha256']
    delivery = {'owned_root': str(HERE), 'compact': str(archive), 'compact_bytes': len(data),
                'compact_sha256': sha(data), 'frozen_manifest_sha256': sha(manifest_path.read_bytes()),
                'frozen_text_file_count': len(texts), 'CPU_tests': 14, 'negative_cases': 59,
                'prior_frozen_texts_checked_unchanged': 286, 'frozen17_plan_sha256': sha(plan.read_bytes()),
                'actual_asset_files_readonly_verified': 20, 'actual_checkpoint_shards_readonly_verified': 90,
                'actual_MTP_and_vision_headers_readonly_verified': 4984,
                'NPU_calls': 0, 'HTTP_requests': 0, 'services_started': 0, 'real_signals_sent': 0,
                'actual_flags': cpu['actual_flags']}
    (HERE / 'delivery.json').write_text(json.dumps(delivery, indent=2) + '\n')
    print(json.dumps(delivery, indent=2))


if __name__ == '__main__':
    main()
