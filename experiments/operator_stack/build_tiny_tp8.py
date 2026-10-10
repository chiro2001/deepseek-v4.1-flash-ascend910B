"""Build a diagnostic TP8 checkpoint using byte-exact formal tensor payloads.

Run beside the source checkpoint. Large tensor data never crosses SSH.
Engram host tables retain their original inodes; marker tensors are the same
ones used by the formal host-mapped loader. This is not a quality checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import time

LAYERS = (0, 1, 2, 3, 14, 20, 21, 24)
LAYER_MAP = dict(zip(LAYERS, range(len(LAYERS))))


def rename(key):
    match = re.match(r'^layers\.(\d+)\.(.*)', key)
    if not match:
        return key
    source = int(match[1])
    return f'layers.{LAYER_MAP[source]}.{match[2]}' if source in LAYER_MAP else None


def header(path):
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        assert 0 < size < 100_000_000
        return json.loads(stream.read(size)), 8 + size


def write_shard(source, destination, entries):
    """Copy selected ranges with bounded RAM and record per-tensor SHA256."""
    old, base = header(source)
    packed = {'__metadata__': {'format': 'pt'}}
    offset = 0
    for original, target in entries:
        item = old[original]
        length = item['data_offsets'][1] - item['data_offsets'][0]
        packed[target] = dict(item, data_offsets=[offset, offset + length])
        offset += length
    encoded = json.dumps(packed, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    records = []
    with source.open('rb') as reader, destination.open('xb') as writer:
        writer.write(struct.pack('<Q', len(encoded)))
        writer.write(encoded)
        for original, target in entries:
            item = old[original]
            begin, end = item['data_offsets']
            reader.seek(base + begin)
            remaining = end - begin
            digest = hashlib.sha256()
            while remaining:
                chunk = reader.read(min(8 * 1024**2, remaining))
                assert chunk, (source, original, remaining)
                writer.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            records.append({'source_key': original, 'key': target,
                            'dtype': item['dtype'], 'shape': item['shape'],
                            'bytes': end - begin, 'payload_sha256': digest.hexdigest()})
    assert destination.stat().st_size == 8 + len(encoded) + offset
    # A second, independent read proves that the written payloads match.
    new, new_base = header(destination)
    with destination.open('rb') as reader:
        for record in records:
            begin, end = new[record['key']]['data_offsets']
            reader.seek(new_base + begin)
            digest = hashlib.sha256()
            remaining = end - begin
            while remaining:
                chunk = reader.read(min(8 * 1024**2, remaining))
                assert chunk
                digest.update(chunk)
                remaining -= len(chunk)
            assert digest.hexdigest() == record['payload_sha256'], record['key']
    return records


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    model, out = args.model.resolve(), args.output.absolute()
    assert not out.exists(), 'Use a new destination; never replace an existing model'
    source_config = (model / 'config.json').read_bytes()
    config = json.loads(source_config)
    text = config['text_config']
    assert (text['num_hidden_layers'], text['hidden_size'], text['n_routed_experts'],
            text['num_experts_per_tok']) == (40, 5120, 384, 6)
    assert config['quantization_config']['model_quant_type'] == 'W4A8_DYNAMIC'
    assert text['engram_layer_ids'] == [1, 14]
    original_text = dict(text)
    text['num_hidden_layers'] = len(LAYERS)
    text['compress_ratios'] = [text['compress_ratios'][i] for i in LAYERS]
    for field in ('kv_source_layer_ids', 'index_source_layer_ids', 'engram_layer_ids'):
        text[field] = [LAYER_MAP[i] for i in text[field] if i in LAYER_MAP]
    text['candidate_source_layer_id'] = LAYER_MAP[text['candidate_source_layer_id']]
    text['num_nextn_predict_layers'] = 0
    text['dspark_target_layer_ids'] = []
    config['diagnostic_tiny_tp8'] = True
    out.mkdir(parents=True)
    for name in ('tokenizer.json', 'tokenizer_config.json', 'configuration.json'):
        (out / name).symlink_to(model / name)
    (out / 'optional').symlink_to(model / 'optional', target_is_directory=True)
    (out / 'engram_int8').mkdir()
    tables = []
    for original in original_text['engram_layer_ids']:
        for kind in ('weight', 'scale'):
            source = model / 'engram_int8' / f'layers_{original}_engram_embed.{kind}.safetensors'
            target = out / 'engram_int8' / f'layers_{LAYER_MAP[original]}_engram_embed.{kind}.safetensors'
            target.symlink_to(source.resolve(strict=True))
            stat = source.stat()
            assert (target.stat().st_dev, target.stat().st_ino) == (stat.st_dev, stat.st_ino)
            tables.append({'source_layer': original, 'layer': LAYER_MAP[original],
                           'kind': kind, 'source': str(source.resolve()),
                           'bytes': stat.st_size, 'same_inode': True})
    weight_map = json.loads((model / 'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    grouped = {}
    for key, shard in weight_map.items():
        target = rename(key)
        if target is None or '.engram.embed.' in key:
            continue
        grouped.setdefault(shard, []).append((key, target))
    # Formal engram_extra contains tiny placeholders for the host table loader.
    extra, _ = header(model / 'engram_extra.safetensors')
    for key in extra:
        if '.engram.embed.' in key:
            grouped.setdefault('engram_extra.safetensors', []).append((key, rename(key)))
    output_map, tensors, shards = {}, [], []
    for index, (source_name, entries) in enumerate(sorted(grouped.items())):
        destination_name = f'quant_model_weights-{index + 1:05d}-of-{len(grouped):05d}.safetensors'
        records = write_shard(model / source_name, out / destination_name, sorted(entries))
        for record in records:
            key = record['key']
            assert key not in output_map
            output_map[key] = destination_name
            record['shard'] = destination_name
        tensors.extend(records)
        shards.append({'name': destination_name, 'bytes': (out / destination_name).stat().st_size})
        print('TINY_SHARD_COPIED', json.dumps(shards[-1]), flush=True)
    description = json.loads((model / 'quant_model_description.json').read_text())
    description = {target: value for key, value in description.items() if (target := rename(key)) is not None}
    for name, value in (
        ('config.json', config), ('quant_model_description.json', description),
        ('quant_model_weights.safetensors.index.json',
         {'metadata': {'total_size': sum(r['bytes'] for r in tensors)}, 'weight_map': output_map}),
    ):
        (out / name).write_text(json.dumps(value, indent=2) + '\n')
    receipt = {'diagnostic_only': True, 'formal_quality_checkpoint': False,
               'source_model': str(model), 'model': str(out),
               'source_config_sha256': hashlib.sha256(source_config).hexdigest(),
               'config_sha256': hashlib.sha256((out / 'config.json').read_bytes()).hexdigest(),
               'layer_map': {str(k): v for k, v in LAYER_MAP.items()},
               'unchanged_fields': {k: v for k, v in original_text.items() if text.get(k) == v},
               'changed_fields': {k: {'source': v, 'tiny': text[k]} for k, v in original_text.items() if text.get(k) != v},
               'prefix_layers_exact': [0, 1, 2, 3], 'engram_tables': tables,
               'shards': shards, 'tensors': tensors, 'copied_tensors_byte_exact': True,
               'seconds': time.monotonic() - started}
    (out / 'tiny_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    compact = {k: v for k, v in receipt.items() if k not in ('tensors', 'unchanged_fields')}
    (out / 'tiny_summary.json').write_text(json.dumps(compact, indent=2) + '\n')
    print('TINY_TP8_BUILD_COMPLETE', json.dumps(compact), flush=True)


if __name__ == '__main__':
    main()
