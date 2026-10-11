"""Freeze real W2 sources and synthetic activation cases without NPU calls."""
import ast
import hashlib
import json
import os
from pathlib import Path

from numeric_contract import cases, activation, array_sha, array_stats
from weights_cpu import load_weights

ROOT = Path('/work/gmm_lane_sparse_msd_m1_numeric_p6_20261011_v5')


def main():
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == os.environ.get('ASCEND_VISIBLE_DEVICES') == ''
    output = ROOT / 'weights_source_manifest.json'
    assert not output.exists(), 'CPU source freeze is immutable'
    previous = Path('/work/gmm_lane_sparse_msd_m1_numeric_p6_20261011_v1/weights_source_manifest.json')
    manifest = json.loads(previous.read_text())
    assert manifest['loader_source_sha256'] == hashlib.sha256(Path(manifest['loader_source']).read_bytes()).hexdigest()
    for path, record in manifest['source_files'].items():
        source = Path(path)
        assert source.stat().st_size == record['bytes'] and source.stat().st_mtime_ns == record['mtime_ns']
    output.write_bytes(previous.read_bytes())
    source_paths = [
        Path('/vllm-workspace/vllm-ascend/vllm_ascend/quantization/methods/w4a8/w4a8.py'),
        Path('/vllm-workspace/vllm-ascend/vllm_ascend/utils.py'),
        Path('/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py'),
        Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/npu/streams.py'),
        Path('/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'),
        Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/npu/npu_config.py')]
    consumer = {'scope': 'Original source consumer evidence; model consumer execution is not covered',
                'CPU_only': True, 'NPU_calls': 0, 'sources': {}}
    for path in source_paths:
        data = path.read_bytes()
        tree = ast.parse(data)
        names = {'apply_gmm2', '_process_scale', '_pack_to_int32',
                 'process_weights_after_loading_modelslim', 'maybe_trans_nz'}
        excerpts = {node.name: ast.get_source_segment(data.decode(), node)
                    for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names}
        consumer['sources'][str(path)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                                          'function_excerpts': excerpts,
                                          'internal_format_assignment_lines': [l for l in data.decode().splitlines() if 'allow_internal_format' in l or 'ALLOW_INTERNAL_FORMAT' in l]}
    (ROOT / 'consumer_source_manifest.json').write_text(json.dumps(consumer, indent=2) + '\n')
    entries = []
    for i, (name, counts) in enumerate(cases()):
        x, token = activation(i)
        entries.append({'name': name, 'counts': counts.tolist(), 'R': int(counts.sum()),
                        'active_experts': list(map(int, counts.nonzero()[0])),
                        'synthetic_activation': True, 'x_sha256': array_sha(x),
                        'token_scale_sha256': array_sha(token), 'counts_sha256': array_sha(counts),
                        'x_stats': array_stats(x), 'token_scale_stats': array_stats(token)})
    (ROOT / 'case_manifest.json').write_text(json.dumps(entries, indent=2) + '\n')
    print(json.dumps({'CPU_only': True, 'NPU_calls': 0, 'source_tensors': len(manifest['raw_tensor_records']),
                      'source_files': len(manifest['source_files']), 'cases': len(entries),
                      'weight_bytes': manifest['processed']['weight_ND']['bytes'],
                      'processed_weight_sha256': manifest['processed']['weight_ND']['sha256']}))


if __name__ == '__main__':
    main()
