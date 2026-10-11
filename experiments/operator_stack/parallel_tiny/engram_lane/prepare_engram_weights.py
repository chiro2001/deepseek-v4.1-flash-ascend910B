"""CPU-only source/partition byte receipts; no arithmetic precision claim."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import torch
import torch_npu
from safetensors import safe_open


def digest(tensor):
    return hashlib.sha256(memoryview(tensor.contiguous().view(torch.uint8).numpy())).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not torch.npu.is_initialized()
    source = args.model / 'engram_extra.safetensors'
    with source.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(length))
    weights = []
    with safe_open(str(source), framework='pt', device='cpu') as file:
        for layer in (1, 14):
            key = f'layers.{layer}.engram.wkv.weight'
            weight = file.get_tensor(key)
            assert weight.shape == (25600, 6144) and weight.dtype == torch.bfloat16
            shards = weight.chunk(8, dim=0)
            restored = hashlib.sha256()
            for shard in shards:
                restored.update(memoryview(shard.contiguous().view(torch.uint8).numpy()))
            full = digest(weight)
            assert restored.hexdigest() == full
            weights.append({'layer': layer, 'key': key, 'shape': list(weight.shape),
                            'dtype': str(weight.dtype), 'payload_sha256': full,
                            'shard_sha256': [digest(s) for s in shards], 'shard_shape': [3200, 6144],
                            'stride': list(weight.stride()), 'contiguous_shards': all(s.is_contiguous() for s in shards),
                            'reconstructed_payload_sha256': restored.hexdigest(),
                            'output_slices_reconstruct_original_bytes': True, 'header': header[key]})
    digest_file = hashlib.sha256()
    with source.open('rb') as stream:
        while chunk := stream.read(8 * 1024**2):
            digest_file.update(chunk)
    receipt = {'cpu_only': True, 'npu_initialized': torch.npu.is_initialized(),
               'source': str(source), 'resolved_source': str(source.resolve()),
               'file_bytes': source.stat().st_size, 'file_sha256': digest_file.hexdigest(), 'weights': weights}
    assert not receipt['npu_initialized']
    args.output.mkdir(exist_ok=True, parents=True)
    path = args.output / 'weight_receipt.json'
    if path.exists():
        assert json.loads(path.read_text()) == receipt
    else:
        path.write_text(json.dumps(receipt, indent=2) + '\n')
    print('ENGRAM_WEIGHT_BYTES_VERIFIED', json.dumps({'file_sha256': receipt['file_sha256'],
                                                   'bytes': receipt['file_bytes'], 'layers': [1, 14]}), flush=True)


if __name__ == '__main__':
    main()
