"""Read-only checkpoint preflight, including shard headers and symlink closure."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_checkpoint(model):
    model = Path(model).absolute()
    config_path = model / 'config.json'
    index_path = model / 'quant_model_weights.safetensors.index.json'
    config = json.loads(config_path.read_text())
    text = config.get('text_config', config)
    quant = config.get('quantization_config', {})
    assert config.get('model_type') == 'deepseek_v41', config.get('model_type')
    expected = dict(hidden_size=5120, num_hidden_layers=40,
                    n_routed_experts=384, num_experts_per_tok=6,
                    moe_intermediate_size=2304)
    assert all(text.get(k) == v for k, v in expected.items()), text
    assert quant.get('quant_method') == 'ascend' and quant.get('model_quant_type') == 'W4A8_DYNAMIC', quant
    assert text.get('engram_layer_ids') == [1, 14], text.get('engram_layer_ids')
    weight_map = json.loads(index_path.read_text())['weight_map']
    selected = [k for k in weight_map if re.match(r'^layers\.(?:[0-9]|[1-3][0-9])\.', k)
                and (('.hc_' in k and k.endswith(('_fn', '_scale', '_base')))
                     or ('indexer.' in k and k.endswith(('wk.weight', 'k_norm.weight'))))]
    shards = []
    headers = {}
    for name in sorted(set(weight_map.values())):
        path = model / name
        assert path.is_file(), f'Missing shard or broken link: {path}'
        resolved = path.resolve(strict=True)
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            assert 0 < length <= 100_000_000, (name, length)
            header = json.loads(stream.read(length))
        for key in selected:
            if weight_map[key] == name:
                assert key in header, (key, name)
                headers[key] = {k: header[key][k] for k in ['dtype', 'shape']}
                headers[key]['shard'] = name
        shards.append({'name': name, 'resolved': str(resolved), 'bytes': resolved.stat().st_size})
    assert len(shards) == 90, len(shards)
    assert sum(k.endswith('wk.weight') for k in headers) == 4
    assert sum(k.endswith('k_norm.weight') for k in headers) == 4
    assert sum(k.endswith('_fn') for k in headers) == 80, list(headers)
    for key, value in headers.items():
        if key.endswith('wk.weight'):
            assert value['shape'] == [128, 512] and value['dtype'] in ['BF16', 'F32'], (key, value)
            value['runtime_dtype'] = 'BF16'  # nn.Linear(..., dtype=torch.bfloat16)
        if key.endswith('k_norm.weight'):
            assert value['shape'] == [128] and value['dtype'] in ['BF16', 'F32'], (key, value)
            value['runtime_dtype'] = 'BF16'  # DeepseekV41RMSNorm parameter allocation
        if key.endswith('_fn'):
            assert value['shape'] == [24, 20480] and value['dtype'] == 'F32', (key, value)
    # Check every reachable auxiliary symlink without reading Engram tables.
    auxiliary = {}
    for name in ['optional', 'engram_int8', 'engram_extra.safetensors']:
        path = model / name
        assert path.exists(), f'Missing auxiliary: {path}'
        files = [path] if path.is_file() else sorted(path.rglob('*'))
        broken = [str(p) for p in files if p.is_symlink() and not p.exists()]
        assert not broken, broken
        auxiliary[name] = {'resolved': str(path.resolve(strict=True)), 'entries': len(files)}
    assert (model / 'optional' / 'quarot.safetensors').is_file()
    return {'model_path': str(model), 'config_sha256': sha256(config_path),
            'index_sha256': sha256(index_path), 'topology': expected,
            'quantization': quant, 'engram_layer_ids': text['engram_layer_ids'],
            'shard_count': len(shards), 'shards': shards, 'auxiliary': auxiliary,
            'operator_tensors': headers, 'status': 'passed',
            'weight_content_hash_scope': 'selected operator tensors are hashed by the operator verifier; full shard content not hashed'}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = inspect_checkpoint(args.model)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print('FORMAL_CHECKPOINT', json.dumps({k: v for k, v in result.items()
                                         if k not in ['shards', 'operator_tensors']}), flush=True)


if __name__ == '__main__':
    main()
