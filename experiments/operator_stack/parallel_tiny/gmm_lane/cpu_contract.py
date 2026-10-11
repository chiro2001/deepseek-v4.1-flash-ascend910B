"""Signed INT4/NZ/MSD and sparse scheduling checks; standard-library CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import sys


def signed4(bits):
    bits &= 15
    return bits - 16 if bits >= 8 else bits


def pack8(values):
    assert len(values) == 8 and all(-8 <= value <= 7 for value in values)
    return sum((value & 15) << (4 * i) for i, value in enumerate(values))


def nz_word(e, k, n, K, N):
    assert K % 16 == 0 and N % 64 == 0 and 0 <= k < K and 0 <= n < N
    return (((e * (N // 64) + n // 64) * (K // 16) + k // 16) * 16 + k % 16) * 8 + (n % 64) // 8


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f16(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def bf16_bits(value):
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((word + 0x7fff + ((word >> 16) & 1)) & 0xffffffff) >> 16


def memory_budget():
    w13, w2 = 48 * 5120 * 4608 // 2, 48 * 2304 * 5120 // 2
    scales = 48 * (4608 + 5120) * 8
    assists = 48 * (4608 + 5120) * 4
    return {"w13_bytes_per_layer_rank": w13, "w2_bytes_per_layer_rank": w2,
            "weights_MiB_per_layer_rank": (w13 + w2) / 2**20,
            "full_40layer_ND_duplicate_GiB_per_rank": 40 * (w13 + w2) / 2**30,
            "scale_bytes_per_layer_rank": scales, "assist_bytes_per_layer_rank": assists,
            "descriptor_bytes_per_layer_rank": 128, "descriptor_40layer_bytes_per_rank": 5120,
            "six_expert_weight_materialization_bytes_per_layer": 6 * (w13 + w2) // 48,
            "max_gmm1_msd_half_scratch_bytes": 2 * 6 * 4608 * 2,
            "max_gmm2_msd_half_scratch_bytes": 2 * 6 * 5120 * 2}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert sys.byteorder == "little"
    evidence = json.loads(args.evidence.read_text())
    for entry in evidence["sources"].values():
        assert hashlib.sha256(Path(entry["local_reference"]).read_bytes()).hexdigest() == entry["sha256"]
    adapters = json.loads((args.evidence.parent / "adapter_boundary_evidence.json").read_text())
    for entry in adapters["sources"].values():
        assert hashlib.sha256(Path(entry["local_reference"]).read_bytes()).hexdigest() == entry["sha256"]
    nz_api = Path(adapters["sources"]["gmm1_nz_alias_api"]["local_reference"]).read_text()
    packed_view = Path(adapters["sources"]["gmm1_packed_view_base"]["local_reference"]).read_text()
    assert "auto expertNum = groupListViewShape[0]" in nz_api
    assert "storageShape = {expertNum, n / 64, k / 16, 16, 8}" in nz_api
    assert "SetDataType(DataType::DT_INT4)" in packed_view
    assert "storageShape[storageShapeDim - 1] *= INT4_PER_INT32" in packed_view
    samples = evidence["checkpoint_samples"]
    assert len(samples) == 18 and evidence["quant_group_size"] == 0
    raw_checks = 0
    all_sampled_offsets_zero = True
    for sample in samples:
        expected_shape = [2560, 2304] if sample["projection"] == "w2" else [1152, 5120]
        tensors = sample["tensors"]
        weight = next(v for k, v in tensors.items() if k.endswith(".weight"))
        assert weight["dtype"] == "I8" and weight["shape"] == expected_shape
        raw = bytes.fromhex(weight["first_128_bytes_hex"])
        for coordinate in sample["coordinate_samples"]:
            byte, n = coordinate["raw_byte"], coordinate["n"]
            assert signed4(byte >> (4 * (n % 2))) == coordinate["signed_int4"]
            assert coordinate["raw_byte_offset"] == (n // 2) * sample["logical_k"] + coordinate["k"]
            if coordinate["raw_byte_offset"] < len(raw):
                assert raw[coordinate["raw_byte_offset"]] == byte
            raw_checks += 1
        for name, tensor in tensors.items():
            if name.endswith(".weight_scale"):
                for value in tensor["first_f32_values"]:
                    bits = struct.unpack("<I", struct.pack("<f", value))[0]
                    encoded = struct.pack("<Q", bits)
                    assert encoded[4:] == b"\0" * 4
                    assert struct.unpack("<f", encoded[:4])[0] == value
            if name.endswith(".weight_offset"):
                all_sampled_offsets_zero &= all(value == 0 for value in tensor["first_f32_values"])
    assert all_sampled_offsets_zero
    # Independent blocked-loop packing vs the address equation.
    E, K, N = 3, 32, 128
    logical = lambda e, k, n: (13 * e + 7 * k + 11 * n) % 16 - 8
    blocked = []
    for e in range(E):
        for n1 in range(N // 64):
            for k1 in range(K // 16):
                for k0 in range(16):
                    for n0word in range(8):
                        blocked.append(pack8([logical(e, k1 * 16 + k0, n1 * 64 + n0word * 8 + lane)
                                              for lane in range(8)]))
    layout_checks = 0
    for e in range(E):
        for k in range(K):
            for n in range(N):
                word = blocked[nz_word(e, k, n, K, N)]
                assert signed4(word >> (4 * (n % 8))) == logical(e, k, n)
                # modelslim transpose + INT32 view uses little-endian byte order.
                pair_base = n // 8 * 8
                pairs = bytes((logical(e, k, pair_base + j) & 15)
                              | ((logical(e, k, pair_base + j + 1) & 15) << 4) for j in range(0, 8, 2))
                assert struct.unpack("<I", pairs)[0] == pack8([logical(e, k, pair_base + j) for j in range(8)])
                layout_checks += 1
    # Exact A8->two signed A4 identity and integer-accumulation range.
    split_checks = 0
    for x in range(-128, 128):
        high, low = x // 16, (x & 15) - 8
        assert -8 <= high <= 7 and -8 <= low <= 7 and x == 16 * high + low + 8
        for weight in range(-8, 8):
            assert x * weight == 16 * high * weight + low * weight + 8 * weight
            split_checks += 1
    # Demonstrate why preserving INT32 alone is insufficient: native branches
    # dequantize to half independently before the FP32 merge and assist term.
    rng = random.Random(20261011)
    scale = next(v for k, v in samples[0]["tensors"].items() if k.endswith(".weight_scale"))["first_f32_values"][0]
    counterexample = None
    for trial in range(20000):
        x = [rng.randrange(-128, 128) for _ in range(64)]
        w = [rng.randrange(-8, 8) for _ in range(64)]
        hi = sum((a // 16) * b for a, b in zip(x, w))
        lo = sum(((a & 15) - 8) * b for a, b in zip(x, w))
        assist = f32(8 * sum(w) * scale)
        native_order = f32(f32(f32(f16(f32(hi * scale)) * 16) + f16(f32(lo * scale))) + assist)
        collapsed = f32(sum(a * b for a, b in zip(x, w)) * scale)
        if native_order >= 0 and collapsed >= 0 and abs(bf16_bits(native_order) - bf16_bits(collapsed)) > 1:
            counterexample = {"trial": trial, "scale": scale, "int32_high_sum": hi, "int32_low_sum": lo,
                              "assist": assist, "native_half_branch_order": native_order,
                              "collapsed_one_dot": collapsed,
                              "bf16_ulp_distance": abs(bf16_bits(native_order) - bf16_bits(collapsed)),
                              "scope": "CPU rounding counterexample, not NPU numerical validation"}
            break
    assert counterexample is not None
    sources = evidence["sources"]
    read = lambda key: Path(sources[key]["local_reference"]).read_text()
    assert "GroupListType must be 0 or 1" in read("gmm1_base_tiling")
    assert "GroupListType must be 0 or 1" in read("gmm1_fusion_tiling")
    assert "GROUP_LIST_TYPE_SPARSE = 2" in read("gmm2_generic_utils")
    assert "GetValue(groupIdx)" in read("gmm2_mid_post") and "groupListType" not in read("gmm2_mid_post")
    # This shows the unsafe layout mismatch without executing an operator:
    # sparse [[47,1],[0,0]] represents one row; dense MSD sum of first two
    # flattened elements would incorrectly calculate 48 rows.
    assert sum([47, 1, 0, 0][:2]) == 48
    result = {"passed": True, "CPU_only": True, "NPU_launches": 0,
              "real_parameter_sets": len(samples), "real_raw_coordinate_checks": raw_checks,
              "synthetic_signed_NZ_coordinate_checks": layout_checks,
              "signed_MSD_product_checks": split_checks, "sampled_weight_offsets_zero": True,
              "source_sha256_verified": len(sources) + len(adapters["sources"]), "GMM1_group_list_types": [0, 1],
              "old_NZ_adapter_derives_E_from_group_list_shape": True,
              "sparse_six_slot_descriptor_must_not_use_old_NZ_adapter": True,
              "generic_GMM_sparse_type2_exists": True,
              "formal_A8W4_MSD_type2_safe_to_enable": False,
              "INT32_max_A8W4_K5120_absolute_sum_bound": 128 * 8 * 5120,
              "INT32_max_A4W4_K5120_branch_absolute_sum_bound": 8 * 8 * 5120,
              "rounding_counterexample": counterexample, "memory_budget": memory_budget(),
              "original_1ULP_gate_relaxed": False, "NPU_accuracy_claim": False,
              "physical_NZ_roundtrip_claim": False, "formal_speedup_claim": False}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
