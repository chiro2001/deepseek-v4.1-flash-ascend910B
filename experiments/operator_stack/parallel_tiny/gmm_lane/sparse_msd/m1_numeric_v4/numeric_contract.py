"""Deterministic probe inputs and BF16 ULP comparison; CPU-only."""
import hashlib
import numpy as np


def cases():
    layouts = [('R1_e0', {0: 1}), ('R1_e47', {47: 1}),
               ('R3_e0', {0: 3}), ('R3_e47', {47: 3}),
               ('R3_three_slots', {0: 1, 23: 1, 47: 1}),
               ('R6_e0', {0: 6}), ('R6_e47', {47: 6}),
               ('R6_mixed_counts', {0: 1, 23: 2, 47: 3}),
               ('R6_six_slots', {0: 1, 7: 1, 15: 1, 23: 1, 31: 1, 47: 1})]
    result = []
    for name, distribution in layouts:
        counts = np.zeros(48, dtype=np.int64)
        for expert, rows in distribution.items(): counts[expert] = rows
        assert 1 <= counts.sum() <= 6
        result.append((name, counts))
    return result


def activation(case_index):
    levels = np.asarray([-128, -127, -65, -64, -17, -16, -15, -9, -8, -1,
                         0, 1, 7, 8, 15, 16, 17, 63, 64, 126, 127], dtype=np.int8)
    r = np.arange(6, dtype=np.int64)[:, None]
    k = np.arange(2304, dtype=np.int64)[None, :]
    x = levels[(k * 13 + r * 7 + case_index * 5) % len(levels)].copy()
    token = np.asarray([1 / 256, 1 / 128, 1 / 64, 0.0125, 0.015625, 0.03125], dtype=np.float32)
    return x, token


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def array_stats(array):
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'bytes': array.nbytes,
            'finite': bool(np.isfinite(array).all()), 'min': float(array.min()), 'max': float(array.max())}


def oracle(counts):
    frame = np.zeros(32, dtype=np.uint32)
    frame[3] = 48
    frame[4:28:4] = 0xffffffff
    rows = slots = 0
    for expert, count in enumerate(counts):
        assert 0 <= count <= 6 and rows + count <= 6
        if not count: continue
        frame[4 + slots * 4:8 + slots * 4] = [expert, rows, count, 0]
        rows += int(count); slots += 1
    frame[0:2] = [rows, slots]
    return frame


def bf16_stats(bits):
    values = (bits.astype(np.uint32) << 16).view(np.float32)
    return {**array_stats(values), 'BF16_bytes_sha256': array_sha(bits)}


def compare_bf16(reference, actual, max_ulp):
    assert reference.dtype == actual.dtype == np.uint16
    ref_values = (reference.astype(np.uint32) << 16).view(np.float32)
    act_values = (actual.astype(np.uint32) << 16).view(np.float32)
    def ordered(bits):
        b = bits.astype(np.int32)
        return np.where(b & 0x8000, 0x8000 - (b & 0x7fff), 0x8000 + b)
    delta = np.abs(ordered(reference) - ordered(actual))
    finite = bool(np.isfinite(ref_values).all() and np.isfinite(act_values).all())
    result = {'passed': finite and bool((delta <= max_ulp).all()), 'threshold_ULP': max_ulp,
              'max_ULP': int(delta.max()), 'differing_elements': int((reference != actual).sum()),
              'elements_exceeding_threshold': int((delta > max_ulp).sum()), 'finite': finite,
              'byte_exact': bool(np.array_equal(reference, actual)),
              'max_abs_difference': float(np.max(np.abs(ref_values - act_values)))}
    mismatch = np.argwhere(delta > max_ulp)
    result['first_exceeding_coordinates'] = [list(map(int, c)) for c in mismatch[:8]]
    return result
