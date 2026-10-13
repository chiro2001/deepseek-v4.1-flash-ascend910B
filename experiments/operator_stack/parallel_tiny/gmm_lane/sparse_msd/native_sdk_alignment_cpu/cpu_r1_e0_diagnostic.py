"""Read-only CPU direction check of frozen R1/e0, never imports torch_npu.

Preserves the separate high/low INT4 integer dots, their FP16 rounding, FP32
merge/assist/token order. This diagnostic does not replace or alter the frozen
actual SDK-vs-raw gate, and is not permission to run another NPU case.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/work/gmm_lane_sparse_msd_m1_numeric_p6_20261011_v4')


def sha(data): return hashlib.sha256(data).hexdigest()


def read_source(record):
    with Path(record['file']).open('rb') as f:
        begin, end = record['absolute_byte_range']; f.seek(begin); data = f.read(end - begin)
    assert sha(data) == record['source_tensor_sha256']
    dtype = np.int8 if record['dtype'] == 'I8' else np.float32
    return np.frombuffer(data, dtype=dtype).reshape(record['shape']).copy()


def bf16(values):
    words = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((words + 0x7fff + ((words >> 16) & 1)) >> 16).astype(np.uint16)


def metric(prediction, actual):
    p = (prediction.astype(np.uint32) << 16).view(np.float32)
    a = (actual.astype(np.uint32) << 16).view(np.float32)
    def ordered(bits):
        b = bits.astype(np.int32)
        return np.where(b & 0x8000, 0x8000 - (b & 0x7fff), 0x8000 + b)
    delta = np.abs(ordered(prediction) - ordered(actual))
    return {'byte_exact': bool(np.array_equal(prediction, actual)), 'differing': int((prediction != actual).sum()),
            'max_ULP': int(delta.max()), 'elements_over_1ULP': int((delta > 1).sum()),
            'max_abs_difference': float(np.max(np.abs(p - a))),
            'correlation': float(np.corrcoef(p, a)[0, 1])}


def main():
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == os.environ.get('ASCEND_VISIBLE_DEVICES') == ''
    manifest = json.loads((ROOT / 'weights_source_manifest.json').read_text())
    records = {r['suffix']: r for r in manifest['raw_tensor_records'] if r['global_expert'] == 0}
    packed = read_source(records['weight']).view(np.uint8)
    weight = np.empty((2304, 5120), dtype=np.int32)
    for lane in [0, 1]:
        nibble = (packed >> (4 * lane)) & 15
        signed = nibble.astype(np.int32); signed[signed >= 8] -= 16
        weight[:, lane::2] = signed.T
    scale = read_source(records['weight_scale']).reshape(5120)
    assist_raw = read_source(records['scale_bias'])
    assist = torch.from_numpy(assist_raw).transpose(0, 1).contiguous().sum(axis=0).numpy()
    levels = np.asarray([-128,-127,-65,-64,-17,-16,-15,-9,-8,-1,0,1,7,8,15,16,17,63,64,126,127], dtype=np.int8)
    x = levels[(np.arange(2304) * 13) % len(levels)]
    high = np.floor_divide(x.astype(np.int32), 16)
    low = (x.astype(np.int32) & 15) - 8
    dot_high = high @ weight; dot_low = low @ weight
    high_half = (dot_high.astype(np.float32) * scale).astype(np.float16)
    low_half = (dot_low.astype(np.float32) * scale).astype(np.float16)
    merged = high_half.astype(np.float32) * np.float32(16)
    merged = (merged + low_half.astype(np.float32)).astype(np.float32)
    merged = (merged + assist).astype(np.float32)
    predicted = bf16((merged * np.float32(1 / 256)).astype(np.float32))
    # The native ND preprocessor's source op1.Init(x,x,...) writes packed
    # high/low INT4 bytes back into the same INT8 x GM tensor.
    def pack_pair(values):
        return ((values[0::2] & 15) | ((values[1::2] & 15) << 4)).astype(np.uint8).view(np.int8)
    x_after_one_pre = np.concatenate([pack_pair(high), pack_pair(low)])
    high2 = np.floor_divide(x_after_one_pre.astype(np.int32), 16)
    low2 = (x_after_one_pre.astype(np.int32) & 15) - 8
    dot_h2 = high2 @ weight; dot_l2 = low2 @ weight
    h2 = (dot_h2.astype(np.float32) * scale).astype(np.float16).astype(np.float32)
    l2 = (dot_l2.astype(np.float32) * scale).astype(np.float16).astype(np.float32)
    double_pre = bf16((((h2 * np.float32(16) + l2).astype(np.float32) + assist).astype(np.float32) * np.float32(1/256)).astype(np.float32))
    sdk = np.fromfile(ROOT / 'R1_e0_sdk_native.bf16.bin', dtype=np.uint16)[:5120]
    raw = np.fromfile(ROOT / 'R1_e0_raw_native_A1.bf16.bin', dtype=np.uint16)[:5120]
    weight_sum = weight.sum(axis=0, dtype=np.int32)
    assist_from_weight = (weight_sum.astype(np.float32) * scale * np.float32(8)).astype(np.float32)
    result = {'CPU_only': True, 'NPU_calls': 0, 'frozen_outputs_unchanged': True, 'case': 'R1_e0',
              'scope': 'Read-only direction check; hardware dequant details still require ABI/tiling audit',
              'source_tensor_sha256': {k: v['source_tensor_sha256'] for k, v in records.items()},
              'x_row_sha256': sha(x.tobytes()), 'logical_weight_int4_min': int(weight.min()), 'logical_weight_int4_max': int(weight.max()),
              'separate_INT4_dot_FP16_dequant_FP32_merge_used': True,
              'CPU_diagnostic_vs_actual_SDK': metric(predicted, sdk),
              'CPU_diagnostic_vs_actual_raw': metric(predicted, raw),
              'CPU_double_inplace_preprocess_vs_actual_raw': metric(double_pre, raw),
              'emulated_x_after_one_pre_sha256': sha(x_after_one_pre.tobytes()),
              'preprocess_source_calls': 'GMM_CV_SPLIT_IMP_A8W4_MSD ND path op1.Init(x,x,groupList,user1); GMMA8W4PreProcess::Init yGm=GetTensorAddr(0,y) then writes packed high/low into same GM x',
              'assist_relation_max_abs_delta_from_8_sum_weight_scale': float(np.max(np.abs(assist - assist_from_weight))),
              'first_16': {'CPU_predicted_BF16': ((predicted.astype(np.uint32) << 16).view(np.float32))[:16].tolist(),
                           'high_INT32_dot': dot_high[:16].tolist(), 'low_INT32_dot': dot_low[:16].tolist(),
                           'original_assist': assist[:16].tolist()}}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
