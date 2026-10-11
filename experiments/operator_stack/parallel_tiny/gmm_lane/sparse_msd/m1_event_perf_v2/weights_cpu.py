"""CPU-only original modelslim W2 loading; never imports torch_npu.

Only layer0/rank0 local experts0..47 are read. No full-model allocation.
Raw source tensor hashes identify exact safetensors byte ranges. The transforms
match w4a8.process_weights_after_loading_modelslim/_process_scale.
"""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

MODEL = Path('/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq')
SOURCE = Path('/vllm-workspace/vllm-ascend/vllm_ascend/quantization/methods/w4a8/w4a8.py')
SOURCE_SHA = 'f3a8cdd205719453606fa106d47c1e556ef40d2ee4046869a029d5f12a3b6de9'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def nz_expert_bytes(weight_nd, expert):
    # Original packed INT8 [K,N/2] -> NZ INT32 [N/64,K/16,16,8].
    a = weight_nd[expert].numpy().reshape(144, 16, 80, 32)
    return np.ascontiguousarray(a.transpose(2, 0, 1, 3)).tobytes()


def load_weights(full_source_file_hash=False):
    assert sha(SOURCE.read_bytes()) == SOURCE_SHA
    index_path = MODEL / 'quant_model_weights.safetensors.index.json'
    description_path = MODEL / 'quant_model_description.json'
    mapping = json.loads(index_path.read_text())['weight_map']
    quant = json.loads(description_path.read_text())
    assert quant['group_size'] == 0
    headers = {}
    raw_weight = np.empty((48, 2560, 2304), dtype=np.int8)
    raw_scale = np.empty((48, 5120, 1), dtype=np.float32)
    raw_assist = np.empty((48, 5120, 16), dtype=np.float32)
    records = []
    offsets_all_zero = True
    specs = {'weight': ('I8', [2560, 2304], np.int8),
             'weight_scale': ('F32', [5120, 1], np.float32),
             'scale_bias': ('F32', [5120, 16], np.float32),
             'weight_offset': ('F32', [5120, 1], np.float32)}
    for expert in range(48):
        for suffix, (dtype_name, shape, dtype) in specs.items():
            name = f'layers.0.ffn.experts.{expert}.w2.{suffix}'
            path = MODEL / mapping[name]
            if path not in headers:
                with path.open('rb') as f:
                    header_bytes = f.read(8)
                    length = struct.unpack('<Q', header_bytes)[0]
                    encoded = f.read(length)
                headers[path] = (8 + length, json.loads(encoded), sha(header_bytes + encoded))
            data_start, header, header_sha = headers[path]
            spec = header[name]
            assert spec['dtype'] == dtype_name and spec['shape'] == shape
            assert quant[name] == 'W4A8_DYNAMIC'
            begin, end = spec['data_offsets']
            with path.open('rb') as f:
                f.seek(data_start + begin)
                encoded = f.read(end - begin)
            assert len(encoded) == int(np.prod(shape)) * np.dtype(dtype).itemsize
            array = np.frombuffer(encoded, dtype=dtype).reshape(shape)
            records.append({'name': name, 'global_expert': expert, 'suffix': suffix,
                            'file': str(path), 'file_bytes': path.stat().st_size,
                            'file_mtime_ns': path.stat().st_mtime_ns, 'header_sha256': header_sha,
                            'absolute_byte_range': [data_start + begin, data_start + end],
                            'dtype': dtype_name, 'shape': shape, 'bytes': len(encoded),
                            'source_tensor_sha256': sha(encoded),
                            'finite': bool(np.isfinite(array).all()),
                            'min': float(array.min()), 'max': float(array.max())})
            assert records[-1]['finite']
            if suffix == 'weight': raw_weight[expert] = array
            elif suffix == 'weight_scale': raw_scale[expert] = array
            elif suffix == 'scale_bias': raw_assist[expert] = array
            else: offsets_all_zero &= bool((array == 0).all())
    # Original Torch CPU operations preserve the original FP32 sum order.
    weight_nd = torch.from_numpy(raw_weight).transpose(1, 2).contiguous()
    scale_transposed = torch.from_numpy(raw_scale).transpose(1, 2).contiguous().numpy()
    scale_np = scale_transposed.copy()
    scale_np.dtype = np.uint32
    scale = torch.from_numpy(scale_np.astype(np.int64))
    assist = torch.from_numpy(raw_assist).transpose(1, 2).contiguous().sum(axis=1)
    assert weight_nd.shape == (48, 2304, 2560)
    assert scale.shape == (48, 1, 5120) and assist.shape == (48, 5120)
    nz_hash = hashlib.sha256()
    nz_experts = []
    for expert in range(48):
        encoded = nz_expert_bytes(weight_nd, expert)
        nz_hash.update(encoded)
        nz_experts.append(sha(encoded))
    manifest = {
        'CPU_only': True, 'NPU_launches': 0, 'synthetic_weights': False,
        'checkpoint': str(MODEL), 'layer': 0, 'rank': 0,
        'local_to_global_experts': list(range(48)),
        'loader_source': str(SOURCE), 'loader_source_sha256': SOURCE_SHA,
        'config_sha256': file_sha(MODEL / 'config.json'),
        'index_sha256': file_sha(index_path), 'description_sha256': file_sha(description_path),
        'raw_tensor_records': records, 'all_E48_weight_offsets_zero': offsets_all_zero,
        'loader_transforms': [
            'raw packed INT8 [E,N/2,K].transpose(1,2).contiguous()',
            'scale FP32.transpose(1,2).contiguous() then numpy dtype UINT32 -> INT64 zero extension',
            'assist FP32.transpose(1,2).contiguous().sum(axis=1)',
            'runtime original torch_npu.npu_format_cast(weight,29), weight.view(torch.int32).contiguous()'],
        'processed': {
            'weight_ND': {'shape': list(weight_nd.shape), 'dtype': 'INT8', 'bytes': weight_nd.numel(),
                          'sha256': sha(weight_nd.numpy().tobytes())},
            'scale': {'shape': list(scale.shape), 'dtype': 'INT64', 'bytes': scale.numel() * 8,
                      'sha256': sha(scale.numpy().tobytes()),
                      'original_FP32_min': float(raw_scale.min()), 'original_FP32_max': float(raw_scale.max())},
            'assist': {'shape': list(assist.shape), 'dtype': 'FP32', 'bytes': assist.numel() * 4,
                       'sha256': sha(assist.numpy().tobytes()), 'finite': bool(torch.isfinite(assist).all()),
                       'min': float(assist.min()), 'max': float(assist.max())},
            'expected_physical_NZ': {'INT32_storage': [48, 80, 144, 16, 8],
                                     'sha256': nz_hash.hexdigest(), 'expert_sha256': nz_experts}},
        'source_files': {str(path): {'bytes': path.stat().st_size,
                                    'mtime_ns': path.stat().st_mtime_ns,
                                    'header_sha256': headers[path][2]}
                         for path in headers}}
    if full_source_file_hash:
        for path in headers:
            manifest['source_files'][str(path)]['full_file_sha256'] = file_sha(path)
    return weight_nd, scale, assist, manifest
