"""CPU-only end-to-end verification of the K5 capacity numeric probe.

Compiles the capacity adapter and the planner-frame CPU reference with the
host g++ (no device headers, no ACL, no NPU), then byte/SHA-compares:
  * numeric_contract oracle vs CPU plan_frame vs the compiled C++ frame twin,
  * the adapter accept/reject matrix (positives + negatives),
  * negative rejection for illegal counts / shapes / dtypes / spans,
  * activation INT8/FP32 samples and the BF16 ULP threshold tooling,
  * profile / compile_input / source_factory / bridge_factory text checks.

Writes machine-readable evidence JSON (main + compact) and never touches the
NPU, docker, ssh or the network. Trace: k5_capacity_cpu 2026-10-12.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPARSE = HERE.parent
sys.path.insert(0, str(SPARSE))

from k5_capacity_numeric_cpu import bridge_factory as BF  # noqa: E402
from k5_capacity_numeric_cpu import numeric_contract as NC  # noqa: E402
from k5_capacity_numeric_cpu import profile as KP  # noqa: E402
from k5_capacity_numeric_cpu import source_factory as SF  # noqa: E402
from k5_capacity_numeric_cpu import weights_cpu as WC  # noqa: E402

# CPU-side test input for the future bridge workspace size (112 MiB, matching
# verify_adapter_cpu.cpp). The real value must come from the new-geometry
# kernel compiler metadata; this constant is not that metadata.
TEST_NATIVE_WORKSPACE_BYTES = 112 * 1024 * 1024
TEST_ROLE_ROOT = "/work/gmm_lane_sparse_msd_build_capacity_cpu_20261012_v1"

failures = []
checks_passed = 0
checks_failed = 0


def check(condition, message):
    global checks_passed, checks_failed
    if condition:
        checks_passed += 1
    else:
        checks_failed += 1
        failures.append(message)
        print("FAIL:", message, file=sys.stderr)
    return bool(condition)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def jsonable(value):
    return json.loads(json.dumps(value, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def compile_unit(cxx, sources, binary, build_dir, name, log, object_only=False):
    command = [cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-I", str(HERE)]
    command += [str(HERE / source) for source in sources]
    command += ["-c", "-o", str(binary)] if object_only else ["-o", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True)
    (build_dir / (name + ".log")).write_text(
        "exit_code=" + str(result.returncode) + "\n" + result.stdout + result.stderr)
    log.append({"unit": name, "command": [str(x) for x in command], "exit_code": result.returncode,
                "warnings_or_errors": (result.stdout + result.stderr).strip()})
    if result.returncode != 0:
        raise SystemExit("compile failed for " + name + "; see " + str(build_dir))
    log[-1]["binary_sha256"] = sha_file(binary)
    log[-1]["binary_bytes"] = binary.stat().st_size
    return binary


def run_adapter(binary):
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    payload["exit_code"] = result.returncode
    payload["stdout_sha256"] = sha_bytes(result.stdout.encode())
    if result.stderr.strip():
        payload["stderr"] = result.stderr.strip().splitlines()[:20]
    check(result.returncode == 0 and payload.get("passed") is True, "adapter sweep passed")
    check(payload.get("NPU_calls") == 0 and payload.get("real_SDK_calls") == 0, "adapter made no device call")
    return payload


def run_frame_reference(binary, profile, counts, workdir, tag):
    counts_path = workdir / ("counts_" + tag + ".bin")
    frame_path = workdir / ("frame_" + tag + ".bin")
    counts.astype("<i8").tofile(counts_path)
    result = subprocess.run([str(binary), str(profile.experts), str(profile.capacity),
                             str(profile.frame_words), str(counts_path), str(frame_path)],
                            capture_output=True, text=True)
    info = json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else None
    frame = np.fromfile(frame_path, dtype="<u4") if frame_path.exists() else None
    return {"exit_code": result.returncode, "info": info, "frame": frame, "stderr": result.stderr.strip()}


def positive_cases(profile):
    cases = list(NC.cases(profile))
    cases.append((profile.family + "_zero_rows", np.zeros(profile.experts, dtype=np.int64)))
    return cases


def frame_record(profile, name, counts, index, oracle_frame, plan_frame, cpp, x, token):
    byte_exact = bool(np.array_equal(oracle_frame, plan_frame) and
                      cpp["frame"] is not None and np.array_equal(oracle_frame, cpp["frame"]))
    cpp_info = cpp["info"] or {}
    consistent_json = (cpp_info.get("rows") == int(oracle_frame[0]) and
                       cpp_info.get("slots") == int(oracle_frame[1]) and
                       cpp_info.get("error_flag") == int(oracle_frame[2]))
    check(byte_exact, profile.family + ":" + name + " oracle/plan_frame/C++ byte exact")
    check(consistent_json, profile.family + ":" + name + " C++ JSON counters match frame")
    return {"case": name, "counts_sum": int(counts.sum()), "active_experts": int((counts > 0).sum()),
            "counts_sha256": sha_bytes(counts.astype("<i8").tobytes()),
            "oracle_frame_sha256": sha_bytes(oracle_frame.astype("<u4").tobytes()),
            "plan_frame_sha256": sha_bytes(plan_frame.astype("<u4").tobytes()),
            "cpp_frame_sha256": sha_bytes(cpp["frame"].astype("<u4").tobytes()) if cpp["frame"] is not None else None,
            "byte_exact": byte_exact, "rows": int(oracle_frame[0]), "slots": int(oracle_frame[1]),
            "error_flag": int(oracle_frame[2]), "cpp_exit_code": cpp["exit_code"],
            "activation_index": index, "x_dtype": str(x.dtype), "x_shape": list(x.shape),
            "x_sha256": NC.array_sha(x), "token_dtype": str(token.dtype),
            "token_shape": list(token.shape), "token_sha256": NC.array_sha(token)}


def python_negatives(profile):
    experts, capacity = profile.experts, profile.capacity
    variants = [
        ("float_dtype", np.zeros(experts, dtype=np.float64)),
        ("wrong_shape", np.zeros(experts + 1, dtype=np.int64)),
        ("negative_count", np.concatenate([[np.int64(-1)], np.zeros(experts - 1, dtype=np.int64)])),
        ("sum_over_capacity", np.concatenate([[np.int64(capacity + 1)], np.zeros(experts - 1, dtype=np.int64)])),
        ("single_count_over_capacity", np.concatenate([np.zeros(experts // 2, dtype=np.int64),
                                                       [np.int64(capacity + 1)],
                                                       np.zeros(experts - experts // 2 - 1, dtype=np.int64)])),
    ]
    records = []
    for name, counts in variants:
        raised = None
        try:
            NC.oracle(profile, counts)
        except ValueError as error:
            raised = "ValueError:" + str(error)[:60]
        check(raised is not None, profile.family + ":oracle rejects " + name)
        plan = None
        try:
            plan = NC.plan_frame(counts, profile)
        except ValueError:
            plan = "raises"
        dtype_or_shape = name in ("float_dtype", "wrong_shape")
        if dtype_or_shape:
            check(plan == "raises", profile.family + ":plan_frame raises on " + name)
        else:
            check(isinstance(plan, np.ndarray) and int(plan[2]) == 1 and int(plan[0]) == 0 and int(plan[1]) == 0,
                  profile.family + ":plan_frame error flag on " + name)
        records.append({"profile": profile.family, "case": name, "oracle": raised or "accepted",
                        "plan_frame": "raises" if isinstance(plan, str) else "error_flag"})
    return records


def cpp_negatives(binary, profile, workdir):
    experts, capacity = profile.experts, profile.capacity
    first = np.zeros(experts, dtype=np.int64)
    first[0] = -1
    after_slot = np.zeros(experts, dtype=np.int64)
    after_slot[0] = 1
    after_slot[1] = -1
    row_overflow = np.zeros(experts, dtype=np.int64)
    row_overflow[0] = capacity - 1
    row_overflow[1] = 2
    records = []
    for name, counts in [("negative_first", first), ("negative_after_slot", after_slot),
                         ("row_sum_overflow", row_overflow)]:
        plan = NC.plan_frame(counts, profile)
        result = run_frame_reference(binary, profile, counts, workdir, profile.family + "_" + name)
        frame = result["frame"]
        check(result["exit_code"] == 0 and frame is not None and np.array_equal(plan, frame),
              profile.family + ":C++ error frame " + name + " byte exact")
        check(int(plan[2]) == 1 and int(plan[0]) == 0 and int(plan[1]) == 0,
              profile.family + ":error frame semantics " + name)
        records.append({"profile": profile.family, "case": name, "cpp_error_flag": int(plan[2]),
                        "byte_exact_vs_plan_frame": bool(frame is not None and np.array_equal(plan, frame)),
                        "frame_sha256": sha_bytes(plan.astype("<u4").tobytes())})
    short = np.zeros(experts - 1, dtype="<i8")
    counts_path = workdir / ("counts_" + profile.family + "_short.bin")
    short.tofile(counts_path)
    result = subprocess.run([str(binary), str(profile.experts), str(profile.capacity),
                             str(profile.frame_words), str(counts_path), str(workdir / "frame_short.bin")],
                            capture_output=True, text=True)
    check(result.returncode == 2 and "experts*8" in result.stderr,
          profile.family + ":C++ rejects short counts buffer")
    records.append({"profile": profile.family, "case": "short_counts_buffer",
                    "cpp_exit_code": result.returncode, "stderr": result.stderr.strip()})
    return records


def bf16_selfcheck():
    module = load_v5_module()
    rng = np.random.default_rng(20261012)
    reference = rng.integers(0, 0x7F00, size=64, dtype=np.uint16)  # positive finite BF16 patterns only
    reference[0] = 0x0100  # +1 LSB stays in the positive segment
    identical = module.compare_bf16(reference, reference.copy(), max_ulp=1)
    one_lsb = reference.copy()
    one_lsb[0] = np.uint16(int(reference[0]) + 1)
    within = module.compare_bf16(reference, one_lsb, max_ulp=1)
    strict = module.compare_bf16(reference, one_lsb, max_ulp=0)
    beyond = reference.copy()
    beyond[0] = np.uint16(int(reference[0]) + 0x0100)
    overflow = module.compare_bf16(reference, beyond, max_ulp=1)
    nonfinite = np.full(4, 0x7F80, dtype=np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # inf-inf in the retained tool's max_abs_difference
        infinite = module.compare_bf16(nonfinite, nonfinite.copy(), max_ulp=1)
    check(identical["passed"] and identical["max_ULP"] == 0, "bf16 identical inputs pass")
    check(within["passed"] and within["max_ULP"] == 1, "bf16 1 ULP within threshold")
    check(not strict["passed"], "bf16 1 ULP fails 0 ULP threshold")
    check(not overflow["passed"] and overflow["elements_exceeding_threshold"] == 1, "bf16 beyond threshold fails")
    check(not infinite["passed"] and infinite["finite"] is False, "bf16 non-finite input fails")
    stats = module.bf16_stats(reference)
    check(stats["bytes"] == 256 and stats["finite"] is True and len(stats["BF16_bytes_sha256"]) == 64,
          "bf16 stats tool intact")
    return {"module_sha256": sha_file(HERE / "borrowed/v5_numeric_contract.py"),
            "contrast_threshold_ULP": 1,
            "identical": {"passed": identical["passed"], "max_ULP": identical["max_ULP"]},
            "one_lsb_threshold1": {"passed": within["passed"], "max_ULP": within["max_ULP"]},
            "one_lsb_threshold0": {"passed": strict["passed"]},
            "beyond_threshold": {"passed": overflow["passed"], "exceeding": overflow["elements_exceeding_threshold"]},
            "nonfinite": {"passed": infinite["passed"], "finite": infinite["finite"]},
            "sample_sha256": stats["BF16_bytes_sha256"]}


def load_v5_module():
    spec = importlib.util.spec_from_file_location("v5_numeric_contract", HERE / "borrowed/v5_numeric_contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def m1_regression(frame_binary, build_dir):
    """K5 parametric oracle must reproduce the frozen v5 capacity6 frames."""
    v5 = load_v5_module()
    v5_evidence = KP.SPARSE / "m1_numeric_v5/evidence/runtime_v5_success"
    profile = KP.Profile(family="m1_regression", capacity=6, experts=48, k=2304, n=5120,
                         loaded_W2_shape=(48, 2304, 2560), loaded_W2_stage="original_CPU_transposed_INT8",
                         FFN_partition=2304, source_reference="frozen_v5_regression")
    records = []
    for index, (name, counts) in enumerate(v5.cases()):
        k5_frame = NC.oracle(profile, counts)
        plan_frame = NC.plan_frame(counts, profile)
        v5_frame = v5.oracle(counts)
        cpp = run_frame_reference(frame_binary, profile, counts, build_dir, "m1reg_" + name)
        same_oracle = bool(np.array_equal(k5_frame, v5_frame))
        same_plan = bool(np.array_equal(plan_frame, v5_frame))
        same_cpp = bool(cpp["frame"] is not None and np.array_equal(v5_frame, cpp["frame"]))
        check(same_oracle, "m1 regression " + name + " K5 oracle equals frozen v5 oracle")
        check(same_plan, "m1 regression " + name + " plan_frame equals frozen v5 oracle")
        check(same_cpp, "m1 regression " + name + " C++ frame equals frozen v5 oracle")
        result_path = v5_evidence / (name + "_result.json")
        device_record = {}
        if result_path.exists():
            device = json.loads(result_path.read_text())
            device_frame = np.asarray(device.get("planner_frame_words", []), dtype=np.uint32)
            device_sha = sha_bytes(device_frame.astype("<u4").tobytes())
            device_match = (len(device_frame) == 32 and
                            device_sha == sha_bytes(v5_frame.astype("<u4").tobytes()) and
                            device.get("planner_frame_byte_exact") is True)
            check(device_match, "m1 regression " + name + " matches read-back device frame sha")
            device_record = {"v5_result_sha256": sha_file(result_path), "device_frame_sha256": device_sha,
                             "device_frame_words": len(device_frame), "match": device_match}
        else:
            check(False, "m1 regression " + name + " frozen v5 result present")
        records.append({"case": name, "rows": int(v5_frame[0]), "slots": int(v5_frame[1]),
                        "frame_sha256": sha_bytes(v5_frame.astype("<u4").tobytes()),
                        "oracle_byte_exact": same_oracle, "plan_frame_byte_exact": same_plan,
                        "cpp_byte_exact": same_cpp, "v5_device_frame": device_record})
    return {"v5_contract_sha256": sha_file(HERE / "borrowed/v5_numeric_contract.py"),
            "frame_bytes": profile.frame_bytes, "cases": records}


def compile_input_checks(profiles):
    import copy
    templates = {"native": json.loads((HERE / "borrowed/compile_input_native.json").read_text()),
                 "candidate": json.loads((HERE / "borrowed/compile_input_candidate.json").read_text())}
    positives, negatives = [], []
    for kind, template in templates.items():
        for family, profile in profiles.items():
            value = KP.compile_input(template, profile)
            op = value["op_list"][0]
            inputs = op["inputs"]
            ok = (inputs[0][0]["shape"] == [profile.capacity, profile.k] and
                  inputs[1][0]["shape"] == [profile.experts, profile.n // 64, profile.k // 16, 16, 64] and
                  inputs[4:7] == [None, None, None] and inputs[7]["shape"] == [profile.experts] and
                  inputs[8]["shape"] == [profile.capacity] and
                  op["outputs"][0][0]["shape"] == [profile.capacity, profile.n] and
                  op["bin_filename"] == template["op_list"][0]["bin_filename"])
            check(ok, kind + ":" + family + " compile_input shapes")
            positives.append({"template": kind, "profile": family,
                              "x_shape": inputs[0][0]["shape"], "weight_shape": inputs[1][0]["shape"],
                              "group_list_shape": inputs[7]["shape"], "token_scales_shape": inputs[8]["shape"],
                              "y_shape": op["outputs"][0][0]["shape"]})
            for name, mutate in [
                ("x_dtype", lambda v: v["op_list"][0]["inputs"][0][0].update(dtype="int16")),
                ("weight_format", lambda v: v["op_list"][0]["inputs"][1][0].update(format="ND")),
                ("y_dtype", lambda v: v["op_list"][0]["outputs"][0][0].update(dtype="float16")),
                ("null_payload", lambda v: v["op_list"][0]["inputs"].__setitem__(4, {"name": "offset"})),
                ("attrs", lambda v: v["op_list"][0]["attrs"][1].update(value=28)),
            ]:
                broken = copy.deepcopy(template)
                mutate(broken)
                rejected = False
                try:
                    KP.compile_input(broken, profile)
                except ValueError:
                    rejected = True
                check(rejected, kind + ":" + family + " compile_input rejects " + name)
                negatives.append({"template": kind, "profile": family, "case": name,
                                  "rejected": rejected})
    return {"positives": positives, "negatives": negatives}


def factory_checks(profiles, build_dir):
    source_records, bridge_records = {}, {}
    prepare_text = (HERE / "borrowed/prepare_source.py").read_text()
    frozen_sha = sha_file(HERE / "borrowed/prepare_source.py")
    check(frozen_sha == SF.FROZEN_FACTORY_SHA, "frozen prepare_source.py sha matches source_factory guard")
    needles = list(dict.fromkeys(list(SF.replacement_map(profiles["target"])) +
                                 list(SF.replacement_map(profiles["draft"]))))
    diff_records = {}
    for tag in ("v2", "v3"):
        diff_path = KP.SPARSE / "evidence" / tag / "native.diff"
        if not diff_path.exists():
            check(False, "native.diff " + tag + " present")
            continue
        plus_lines = [line for line in diff_path.read_text().splitlines() if line.startswith("+")]
        hits = {needle: sum(line.count(needle) for line in plus_lines) for needle in needles}
        check(all(count >= 1 for count in hits.values()),
              "native.diff " + tag + " candidate side contains all replacement needles")
        diff_records[tag] = {"path": str(diff_path), "sha256": sha_file(diff_path),
                             "candidate_plus_lines": len(plus_lines), "needle_hits": hits}
    archive = KP.SPARSE / "evidence/compact_evidence.tgz"
    with tarfile.open(archive) as handle:
        member = handle.extractfile("v2/native.diff").read()
    check(sha_bytes(member) == diff_records.get("v2", {}).get("sha256"),
          "compact_evidence.tgz v2/native.diff matches disk candidate copy")
    diff_records["archive_member_v2_sha256"] = sha_bytes(member)
    for family, profile in profiles.items():
        text = SF.planner_source(profile)
        frame_words, experts, capacity, slots = profile.frame_words, profile.experts, profile.capacity, profile.slots
        assertions = {
            "counts_int64_E": ("input.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(counts),"
                               + str(experts)) in text,
            "frame_u32_words": ("output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(frame),"
                                + str(frame_words)) in text,
            "frame_zeroed": ("for(uint32_t i=0;i<" + str(frame_words) + ";++i) output.SetValue(i,0);") in text,
            "header_experts": ("output.SetValue(3," + str(experts) + ");") in text,
            "slot_presets": ("for(uint32_t i=0;i<" + str(slots)
                             + ";++i) output.SetValue(4+4*i,0xffffffffU);") in text,
            "expert_loop": ("for(uint32_t expert=0;expert<" + str(experts) + ";++expert)") in text,
            "invalid_guard": ("count<0 || count>" + str(capacity) + " || rows+count>" + str(capacity)) in text,
            "error_flag_sets_and_returns": "output.SetValue(2,1);return;" in text,
            "single_block": "GetBlockIdx()!=0" in text,
            "slot_write": "output.SetValue(pos,expert);output.SetValue(pos+1,rows);"
                          "output.SetValue(pos+2,static_cast<uint32_t>(count));" in text,
            "totals_written_last": "output.SetValue(0,rows);output.SetValue(1,slots);" in text,
        }
        for name, ok in assertions.items():
            check(ok, "planner_source " + family + " " + name)
        replacements = SF.replacement_map(profile)
        replace_ok = {needle: (prepare_text.count(needle) == 1) for needle in replacements}
        for needle, ok in replace_ok.items():
            check(ok, "replacement needle present once in frozen factory: " + needle)
        check(replacements["tiling->m == 6"] == "tiling->m == " + str(capacity),
              "replacement map capacity " + family)
        check(replacements["reinterpret_cast<__gm__ uint32_t*>(planner), 32)"].endswith(str(frame_words) + ")"),
              "replacement map frame_words " + family)
        source_records[family] = {"planner_source_sha256": sha_bytes(text.encode()),
                                  "frame_words": frame_words, "assertions": assertions,
                                  "replacement_needles_present": replace_ok,
                                  "change_header": {"executed": False,
                                                    "reason": "CANN 9.1.0 native grouped_matmul source not present on this CPU host"}}

        source_a, geometry, workspace, receipt = BF.generate(
            profile, TEST_NATIVE_WORKSPACE_BYTES, TEST_ROLE_ROOT,
            native_workspace_origin="cpu_test_input_not_compiler_metadata")
        source_b, _, _, _ = BF.generate(
            profile, TEST_NATIVE_WORKSPACE_BYTES, TEST_ROLE_ROOT,
            native_workspace_origin="cpu_test_input_not_compiler_metadata")
        guards = {
            "declared_physical_guard": "CAPACITY_EXPECTED_PHYSICAL" in source_a and "return -910814" in source_a,
            "expected_physical_used": "rtGetDeviceIndexByPhyId(expected_physical, &logical)" in source_a,
            "role_root_guard": "std::string(root) != EXPECTED_ROLE_ROOT" in source_a,
            "adapter_validate_used": "capacity_validate(&b, &PROFILE)" in source_a,
            "adapter_pack_used": "capacity_pack_lists(&b, &PROFILE, words.data())" in source_a,
            "frame_bytes_offsets": "p->workspace_base + SYSTEM + FRAME_BYTES" in source_a
                                   and "aclrtMemcpy(frame, FRAME_BYTES," in source_a,
            "profile_geometry_buffers": "uint64_t(PROFILE.capacity) * PROFILE.k" in source_a
                                        and "uint64_t(PROFILE.experts) * PROFILE.k * PROFILE.n / 2" in source_a,
            "old_role_root_removed": "gmm_lane_sparse_msd_build_20261011_v3_repro" not in source_a,
            "old_frame_size_removed": "aclrtMemcpy(frame, 128," not in source_a,
            "old_geometry_removed": "6ull * 2304" not in source_a and "48ull * 2304 * 5120 / 2" not in source_a,
            "borrowed_stream_preserved": "aclrtGetCurrentContext" in source_a
                                         and "borrowed_stream" in source_a,
            "cleanup_preserved": "cleanup:loader_v2_close" in source_a,
        }
        for name, ok in guards.items():
            check(ok, "bridge " + family + " " + name)
        check(sha_bytes(source_a.encode()) == sha_bytes(source_b.encode()),
              "bridge " + family + " generation deterministic")
        check(receipt["original_v5_bridge_sha256"] == sha_file(HERE / "borrowed/v5_probe_bridge.cpp"),
              "bridge " + family + " borrowed source sha")
        check(receipt["actual_NPU_calls"] == 0 and receipt["actual_numeric_passed"] is False,
              "bridge " + family + " receipt keeps CPU-only flags")
        geometry_expected = ("constexpr CapacityGeometry PROFILE{" + str(profile.capacity) + "," +
                             str(profile.experts) + "," + str(profile.k) + "," + str(profile.n) + "," +
                             str(TEST_NATIVE_WORKSPACE_BYTES) + "ull};")
        check(geometry_expected in geometry and ("FRAME_BYTES=" + str(profile.frame_bytes)) in geometry,
              "bridge " + family + " geometry header constants")
        check(("M1_WORKSPACE_BYTES=" + str(TEST_NATIVE_WORKSPACE_BYTES + profile.frame_bytes)) in workspace,
              "bridge " + family + " workspace header constants")
        bridge_records[family] = {
            "generated_source_sha256": sha_bytes(source_a.encode()),
            "geometry_header_sha256": sha_bytes(geometry.encode()),
            "workspace_header_sha256": sha_bytes(workspace.encode()),
            "deterministic_regeneration": True,
            "native_workspace_input_bytes": TEST_NATIVE_WORKSPACE_BYTES,
            "native_workspace_input_origin": "cpu_test_input_not_compiler_metadata",
            "role_root_input": TEST_ROLE_ROOT,
            "guards": guards,
            "receipt_flags": {"borrowed_context_stream_and_cleanup_preserved":
                              receipt["borrowed_context_stream_and_cleanup_preserved"],
                              "actual_NPU_calls": receipt["actual_NPU_calls"]}}
    return {"source_factory": {"frozen_prepare_source_sha256": frozen_sha, "profiles": source_records},
            "bridge_factory": {"profiles": bridge_records},
            "native_diff_needle_evidence": diff_records}


def source_manifest():
    entries = {}
    for name in ["__init__.py", "capacity_adapter.h", "capacity_adapter.cpp", "verify_adapter_cpu.cpp",
                 "planner_frame_cpu_reference.cpp", "numeric_contract.py", "profile.py", "bridge_factory.py",
                 "source_factory.py", "weights_cpu.py", "verify_numeric_contract_cpu.py"]:
        entries["k5_capacity_numeric_cpu/" + name] = sha_file(HERE / name)
    for path in sorted(p for p in (HERE / "borrowed").iterdir() if p.is_file()):
        entries["k5_capacity_numeric_cpu/borrowed/" + path.name] = sha_file(path)
    for name in ["actual_GMM2_geometry_all_ranks.json", "actual_config_and_profile_source.json"]:
        entries["k5_capacity_numeric_cpu/evidence/" + name] = sha_file(HERE / "evidence" / name)
    return entries


def weights_static(profiles):
    source_path = HERE / "weights_cpu.py"
    compile(source_path.read_text(), str(source_path), "exec")
    check(WC.MODEL.name == "v41-w4a8-engram-dr-vision-qrot-mtpq", "weights_cpu model constant")
    check(str(WC.SOURCE).endswith("w4a8.py"), "weights_cpu loader source constant")
    check(WC.SOURCE_SHA == "f3a8cdd205719453606fa106d47c1e556ef40d2ee4046869a029d5f12a3b6de9",
          "weights_cpu loader sha constant")
    return {"py_compile": "ok", "source_sha256": sha_file(source_path),
            "executed": False, "torch_imported": "torch" in sys.modules,
            "checkpoint_present": WC.MODEL.exists(), "loader_source_present": WC.SOURCE.exists(),
            "reason": "requires the real checkpoint and vllm-ascend loader source; neither exists on this CPU host",
            "model_constant": str(WC.MODEL), "loader_constant": str(WC.SOURCE)}


def geometry_checks(profiles):
    expected = {"target": {"capacity": 36, "experts": 48, "k": 2304, "n": 5120},
                "draft": {"capacity": 15, "experts": 16, "k": 2304, "n": 5120}}
    actual = KP.actual_profiles()
    records = {}
    for family, want in expected.items():
        got = actual[family]
        ok = (got["capacity"] == want["capacity"] and got["experts"] == want["experts"] and
              got["observed_logical_k"] == want["k"] and got["n"] == want["n"])
        check(ok, "actual_profiles " + family + " matches frozen geometry")
        check(profiles[family].capacity == want["capacity"] and profiles[family].experts == want["experts"],
              "bound profile " + family + " matches frozen geometry")
        records[family] = {"capacity": got["capacity"], "experts": got["experts"],
                           "k": got["observed_logical_k"], "n": got["n"], "matches_expected": ok}
    geometry_copy = sha_file(HERE / "evidence/actual_GMM2_geometry_all_ranks.json")
    config_copy = sha_file(HERE / "evidence/actual_config_and_profile_source.json")
    check(geometry_copy == KP.GEOMETRY_SHA, "evidence geometry copy sha equals frozen constant")
    check(config_copy == KP.CONFIG_SOURCE_SHA, "evidence config copy sha equals frozen constant")
    sibling = KP.SPARSE / "k5_geometry_cpu"
    check(sha_file(sibling / "actual_GMM2_geometry_all_ranks.json") == geometry_copy,
          "k5_geometry_cpu geometry file byte identical")
    check(sha_file(sibling / "actual_config_and_profile_source.json") == config_copy,
          "k5_geometry_cpu config file byte identical")
    contract = json.loads((sibling / "geometry_contract.json").read_text())
    for family, key in [("target", "target"), ("draft", "draft")]:
        entry = contract[key]
        check(entry["x_capacity"] == expected[family]["capacity"] and
              entry["local_experts"] == expected[family]["experts"] and
              entry["K"] == expected[family]["k"] and entry["N"] == expected[family]["n"],
              "geometry_contract.json " + family + " cross-check")
    return {"profiles": records, "geometry_json_sha256": geometry_copy,
            "config_source_sha256": config_copy,
            "k5_geometry_cpu_byte_identical": True}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compact-out", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=None)
    args = parser.parse_args()
    assert not args.out.exists(), "refuse to overwrite existing evidence: " + str(args.out)
    assert not args.compact_out.exists(), "refuse to overwrite existing evidence: " + str(args.compact_out)
    build_dir = args.build_dir or (Path.home() / "tmp/dsv41-k5-capacity" /
                                   ("run_" + __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")))
    build_dir.mkdir(parents=True, exist_ok=True)
    cxx = shutil.which("g++") or shutil.which("clang++")
    assert cxx, "host C++ compiler required"
    gxx_version = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    environment = {"cxx": cxx, "cxx_version": gxx_version, "python": sys.version.split()[0],
                   "numpy": np.__version__, "platform": platform.platform(),
                   "build_dir": str(build_dir)}

    compile_log = []
    compile_unit(cxx, ["capacity_adapter.cpp"], build_dir / "capacity_adapter.o",
                 build_dir, "capacity_adapter_object", compile_log, object_only=True)
    adapter_binary = compile_unit(cxx, ["capacity_adapter.cpp", "verify_adapter_cpu.cpp"],
                                  build_dir / "verify_adapter_cpu", build_dir,
                                  "verify_adapter_cpu", compile_log)
    frame_binary = compile_unit(cxx, ["planner_frame_cpu_reference.cpp"],
                                build_dir / "planner_frame_cpu_reference", build_dir,
                                "planner_frame_cpu_reference", compile_log)
    adapter_result = run_adapter(adapter_binary)

    profiles = {}
    for family, loaded in [("target", (48, 2304, 2560)), ("draft", (16, 2304, 2560))]:
        profiles[family] = KP.bind(family, loaded, "original_CPU_transposed_INT8", 2304,
                                   "cpu_verification_20261012")
        check(profiles[family].capacity == (36 if family == "target" else 15),
              family + " bound capacity")

    positive_records, negative_python_records, negative_cpp_records = {}, [], []
    for family, profile in profiles.items():
        records = []
        for index, (name, counts) in enumerate(positive_cases(profile)):
            oracle_frame = NC.oracle(profile, counts)
            plan_frame = NC.plan_frame(counts, profile)
            x, token = NC.activation(profile, index)
            cpp = run_frame_reference(frame_binary, profile, counts, build_dir, name)
            records.append(frame_record(profile, name, counts, index, oracle_frame, plan_frame, cpp, x, token))
            check(str(x.dtype) == "int8" and str(token.dtype) == "float32", name + " activation dtypes")
        positive_records[family] = records
        negative_python_records += python_negatives(profile)
        negative_cpp_records += cpp_negatives(frame_binary, profile, build_dir)

    compile_input = compile_input_checks(profiles)
    bf16 = bf16_selfcheck()
    m1 = m1_regression(frame_binary, build_dir)
    factories = factory_checks(profiles, build_dir)
    weights = weights_static(profiles)
    geometry = geometry_checks(profiles)
    manifest = source_manifest()

    ascend_modules = sorted(m for m in list(sys.modules)
                            if m.split(".")[0] in {"acl", "torch", "torch_npu", "ascendc", "opc"})
    check(not ascend_modules and "torch" not in sys.modules, "no device/torch modules imported")

    main_payload = {
        "schema": "k5_capacity_numeric_cpu_verification_v1",
        "date": "20261012",
        "trace": "k5_capacity_cpu",
        "passed": not failures,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "environment": environment,
        "compile_units": compile_log,
        "adapter_verify": adapter_result,
        "profiles": {"target": {"capacity": profiles["target"].capacity, "experts": profiles["target"].experts,
                                "frame_bytes": profiles["target"].frame_bytes,
                                "frame_words": profiles["target"].frame_words,
                                "cases": positive_records["target"]},
                     "draft": {"capacity": profiles["draft"].capacity, "experts": profiles["draft"].experts,
                               "frame_bytes": profiles["draft"].frame_bytes,
                               "frame_words": profiles["draft"].frame_words,
                               "cases": positive_records["draft"]}},
        "negative_python": negative_python_records,
        "negative_cpp": negative_cpp_records,
        "bf16_tool_selfcheck": bf16,
        "m1_regression": m1,
        "compile_input_checks": compile_input,
        "factory_checks": factories,
        "weights_cpu_static": weights,
        "geometry_freeze_checks": geometry,
        "source_manifest": manifest,
        "declarations": {
            "CPU_only": True, "NPU_calls": 0, "docker_used": False, "ssh_used": False,
            "network_used": False, "device_kernel_compiled": False,
            "actual_device_numeric_passed": False, "performance_receipt_granted": False,
            "admission_granted": False, "formal_receipt_granted": False,
            "bf16_contrast_threshold_note": "future device outputs compare with the retained <=1 ULP tool; "
                                            "no device output exists in this lane yet"},
        "failures": failures,
    }
    args.out.write_text(json.dumps(jsonable(main_payload), indent=2, ensure_ascii=False) + "\n")
    main_sha = sha_file(args.out)
    compact = {
        "schema": "k5_capacity_cpu_final_v1",
        "date": "20261012",
        "trace": "k5_capacity_cpu",
        "passed": not failures,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "compile_units_passed": "{}/{}".format(sum(1 for unit in compile_log if unit["exit_code"] == 0),
                                               len(compile_log)),
        "adapter_checks": adapter_result.get("checks"),
        "adapter_negative_rejections": adapter_result.get("negative_rejections"),
        "numeric_positive_cases": sum(len(v) for v in positive_records.values()),
        "numeric_negative_rejections": len(negative_python_records) + len(negative_cpp_records),
        "compile_input_negatives": len(compile_input["negatives"]),
        "m1_regression_cases": len(m1["cases"]),
        "source_manifest_sha256": sha_bytes(json.dumps(manifest, sort_keys=True).encode()),
        "main_evidence": str(args.out),
        "main_evidence_sha256": main_sha,
        "CPU_only": True, "NPU_calls": 0,
        "actual_device_numeric_passed": False, "performance_receipt_granted": False,
        "admission_granted": False, "formal_receipt_granted": False,
        "remaining_gaps": [
            "change_header/native source rebuild needs the CANN 9.1.0 grouped_matmul source tree",
            "bridge workspace bytes need the new-geometry kernel compiler metadata",
            "weights_cpu payload path needs the real checkpoint and torch on a CPU host",
            "device planner/kernel numeric and performance proof still requires an authorized NPU window"],
    }
    args.compact_out.write_text(json.dumps(jsonable(compact), indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": compact["passed"], "checks_passed": checks_passed,
                      "checks_failed": checks_failed, "main_evidence_sha256": main_sha,
                      "compact": str(args.compact_out)}))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
