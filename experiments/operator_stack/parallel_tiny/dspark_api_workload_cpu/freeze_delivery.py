"""Freeze this CPU workload package; validate previous owned frozen packages."""
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
    validation = json.loads((HERE / 'evidence/CPU_final_validation.json').read_text())
    assert validation['passed'] and validation['checkpoint_AutoTokenizer_parity_checks'] == 10
    assert validation['NPU_calls'] == validation['actual_HTTP_inference_requests'] == validation['services_started'] == 0
    assert not any(validation['actual_flags'].values())
    prior = {}
    for rel in ('dspark_api_counter_bridge_cpu/frozen_files.json',
            'gmm_lane/sparse_msd/m1_event_perf_v1_failure/frozen_files.json',
            'gmm_lane/sparse_msd/m1_event_perf_v2/runtime_frozen_files.json',
            'gmm_lane/sparse_msd/k5_route_feasibility_cpu/frozen_files.json',
            'dspark_context_graph_design_cpu/frozen_files.json'):
        path = HERE.parent / rel
        manifest = json.loads(path.read_text())
        original = Path(manifest['root'])
        for name, expected in manifest['text_files'].items():
            raw = (original / name).read_bytes()
            assert sha(raw) == expected['sha256'] and len(raw) == expected['bytes'], str(original / name)
        prior[rel] = {'frozen_text_files_unchanged': len(manifest['text_files']),
            'manifest_sha256': sha(path.read_bytes())}
    texts = {}
    for path in sorted(HERE.rglob('*')):
        if (not path.is_file() or '__pycache__' in path.parts or
                path.name in ('frozen_files.json', 'delivery.json') or
                path.suffix not in ('.py', '.md', '.json', '.txt', '.log')):
            continue
        raw = path.read_bytes()
        raw.decode('utf-8')
        assert len(raw) < 1_000_000
        texts[str(path.relative_to(HERE))] = {'sha256': sha(raw), 'bytes': len(raw)}
    manifest = {'root': str(HERE), 'CPU_only': True, 'NPU_permission': 0,
        'NPU_calls': 0, 'actual_HTTP_inference_requests': 0, 'services_started': 0,
        'warmup_prompts': 2, 'measured_prompts': 8, 'prompt_tokens_each': 2048,
        'output_budget_each': 256, 'AutoTokenizer_parity_checks_passed': 10,
        'frozen_client_contract_checks_passed': 10,
        'source_excerpt_bytes': 78527, 'full_tokenizer_or_corpus_exported': False,
        'text_files': texts, 'previous_frozen_packages': prior,
        'actual_flags': validation['actual_flags'],
        'workload_sha256': validation['workload_sha256']}
    manifest_path = HERE / 'frozen_files.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    archive = HERE / 'dspark_api_workload_cpu_v1_compact.tgz'
    with archive.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w') as tar:
                for name in [*texts, 'frozen_files.json']:
                    data = (HERE / name).read_bytes()
                    info = tarfile.TarInfo(HERE.name + '/' + name)
                    info.size, info.mode, info.mtime = len(data), 0o644, 0
                    tar.addfile(info, io.BytesIO(data))
    packed = archive.read_bytes()
    assert len(packed) < 1_000_000
    with tarfile.open(archive) as tar:
        for name, expected in texts.items():
            assert sha(tar.extractfile(HERE.name + '/' + name).read()) == expected['sha256']
    delivery = {'owned_root': str(HERE), 'compact': str(archive),
        'compact_bytes': len(packed), 'compact_sha256': sha(packed),
        'frozen_manifest_sha256': sha(manifest_path.read_bytes()),
        'frozen_text_file_count': len(texts), 'workload_sha256': validation['workload_sha256'],
        'warmup': 2, 'measured': 8, 'prompt_tokens': 2048, 'output_budget': 256,
        'real_checkpoint_AutoTokenizer_parity_passed': True,
        'previous_frozen_texts_checked_unchanged': sum(x['frozen_text_files_unchanged'] for x in prior.values()),
        'NPU_calls': 0, 'HTTP_inference_requests': 0, 'services_started': 0,
        'actual_flags': validation['actual_flags']}
    (HERE / 'delivery.json').write_text(json.dumps(delivery, indent=2) + '\n')
    print(json.dumps(delivery, indent=2))


if __name__ == '__main__':
    main()
