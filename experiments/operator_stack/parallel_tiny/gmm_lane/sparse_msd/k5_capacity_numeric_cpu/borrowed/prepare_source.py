"""Make a private native-source copy; replace AIC scheduling only, hash all math."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


SDK = Path("/usr/local/Ascend/cann-9.1.0")
NATIVE = SDK / "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/grouped_matmul"
FRONTEND = SDK / "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/dynamic/grouped_matmul.py"


def function_span(source, name):
    match = re.search(r"__aicore__ inline void (GMMA8W4MSDCompute(?:New)?<mmType>::" + re.escape(name) + r")\s*\(", source)
    assert match, name
    begin = source.index("{", match.end())
    depth, quote, comment, i = 0, None, None, begin
    while i < len(source):
        char, nxt = source[i], source[i:i+2]
        if comment == "line":
            if char == "\n": comment = None
        elif comment == "block":
            if nxt == "*/": comment = None; i += 1
        elif quote:
            if char == "\\": i += 1
            elif char == quote: quote = None
        elif nxt == "//": comment = "line"; i += 1
        elif nxt == "/*": comment = "block"; i += 1
        elif char in "\"'": quote = char
        elif char == "{": depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0: return begin, i + 1
        i += 1
    raise AssertionError("Unclosed native function " + name)


def sparse_block(flavor):
    api_only = "        mm.SetOrgShape(A8W4preM, tiling->n, tiling->k);\n" if flavor == "api" else ""
    vector = "            if ASCEND_IS_AIV { VectorCompute(groupIdx, mnConfig); }\n" if flavor == "api" else ""
    return r'''
    // The AIV retains the complete native dense loop and numerical body.
    // Active expert order, tile/core assignment and handshakes stay identical.
    if ASCEND_IS_AIC {
        // This MSD class reads counts unconditionally. Some native CPU tiling
        // blobs retain nominal groupListType=0 even for a count-mode API call;
        // the new entry's input contract (not that unused field) requires counts.
        const bool geometry = tiling->groupNum == 48 &&
            tiling->groupType == 0 && tiling->m == 6 && tiling->k == 2304 && tiling->n == 5120 &&
            tiling->quantGroupNum == 1 && tiling->isSingleTensor == 1;
        const bool frame = sparsePlannerGm.GetValue(2) == 0 && sparsePlannerGm.GetValue(3) == 48 &&
            sparsePlannerGm.GetValue(0) <= 6 && sparsePlannerGm.GetValue(1) <= 6;
        if (geometry && frame) {
            uint32_t preCount = 0;
            for (uint32_t slot = 0; slot < sparsePlannerGm.GetValue(1); ++slot) {
                const uint32_t pos = 4 + 4 * slot;
                const uint32_t groupIdx = sparsePlannerGm.GetValue(pos);
                const uint32_t rowBegin = sparsePlannerGm.GetValue(pos + 1);
                const uint32_t rowCount = sparsePlannerGm.GetValue(pos + 2);
                mnConfig.offsetM = rowBegin * 2;
                mnConfig.m = rowCount * 2;
                mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
''' + api_only + r'''
                uint32_t curCount = preCount + mnConfig.blockDimN * mnConfig.blockDimM;
                uint32_t curBlock = coreIdx >= preCount ? coreIdx : coreIdx + tiling->coreNum;
                while (curBlock < curCount) {
                    mnConfig.mIdx = (curBlock - preCount) / mnConfig.blockDimN;
                    mnConfig.nIdx = (curBlock - preCount) % mnConfig.blockDimN;
                    MMCompute(groupIdx, mnConfig);
''' + vector + r'''
                    curBlock += tiling->coreNum;
                }
                preCount = curCount % tiling->coreNum;
            }
            return;
        }
    }
'''


def change_header(original, flavor):
    # All native mathematical function definitions remain byte-for-byte intact.
    first = "    __aicore__ inline void Process();"
    assert original.count(first) == 1
    changed = original.replace(first, first + "\n    __aicore__ inline void SetSparsePlanner(GM_ADDR planner) {\n        sparsePlannerGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(planner), 32);\n    }")
    needle = "    GlobalTensor<int64_t> groupTokensGm;"
    assert changed.count(needle) == 1
    changed = changed.replace(needle, needle + "\n    GlobalTensor<uint32_t> sparsePlannerGm;")
    start, end = function_span(changed, "Process")
    body = changed[start:end]
    loop = "    for (uint32_t groupIdx = 0, preCount = 0; groupIdx < tiling->groupNum; ++groupIdx) {"
    assert body.count(loop) == 1
    body = body.replace(loop, sparse_block(flavor) + loop)
    changed = changed[:start] + body + changed[end:]
    names = re.findall(r"GMMA8W4MSDCompute(?:New)?<mmType>::([A-Za-z0-9_]+)\s*\(", original)
    math = {}
    for name in dict.fromkeys(names):
        a, b = function_span(original, name)
        c, d = function_span(changed, name)
        if name == "Process": continue
        assert original[a:b] == changed[c:d], "Native function changed: " + name
        math[name] = hashlib.sha256(original[a:b].encode()).hexdigest()
    return changed, math


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert "gmm_lane_sparse_msd" in str(root)
    target = root / "private_impl/op_kernel"
    assert not target.exists()
    shutil.copytree(NATIVE, target)
    receipts, diffs = {}, []
    for flavor, filename in [("api", "grouped_matmul_antiquant_a8w4_msd.h"),
                             ("vector", "grouped_matmul_antiquant_a8w4_msd_new.h")]:
        path = target / filename
        original = path.read_text()
        changed, math = change_header(original, flavor)
        path.write_text(changed)
        receipts[flavor] = {"native_file": str(NATIVE/filename),
                           "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                           "candidate_sha256": hashlib.sha256(changed.encode()).hexdigest(),
                           "native_function_body_sha256_unchanged": math,
                           "scope": "Only AIC Process loop uses six slots; original AIV dense loop and math unchanged"}
        diffs += list(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                                          fromfile="native/"+filename, tofile="candidate/"+filename))
    cpp = target / "grouped_matmul.cpp"
    original = cpp.read_text()
    changed = original
    needle = "    GM_ADDR user1 = GetUserWorkspace(workspace);"
    assert changed.count(needle) == 1
    changed = changed.replace(needle, needle + "\n    GM_ADDR sparsePlanner = user1;\n    user1 += 128; // extra private workspace prefix; native workspace/tiling is unchanged")
    macro_start = changed.index("#define GMM_CV_SPLIT_IMP_A8W4_MSD(")
    macro_end = changed.index("#define GMM_CV_SPLIT_IMP_A8W4(", macro_start)
    macro = changed[macro_start:macro_end]
    assert macro.count("op.Process();") == 2
    macro = macro.replace("            op.Process();", "            op.SetSparsePlanner(sparsePlanner);                                                    \\\n            op.Process();")
    changed = changed[:macro_start] + macro + changed[macro_end:]
    assert changed.count("__global__ __aicore__ void grouped_matmul(") == 1
    changed = changed.replace("__global__ __aicore__ void grouped_matmul(", "__global__ __aicore__ void gmm2_sparse_msd(")
    cpp.write_text(changed)
    diffs += list(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                                      fromfile="native/grouped_matmul.cpp", tofile="candidate/grouped_matmul.cpp"))
    frontend = FRONTEND.read_text()
    frontend = frontend.replace('origin_func_name = "grouped_matmul"', 'origin_func_name = "gmm2_sparse_msd"')
    # This is a process-local compiler frontend. It keeps original op type,
    # descriptors and host tiling; it does not install a global SDK operator.
    (root / "private_impl/grouped_matmul.py").write_text(frontend)
    # The installed compiler frontend declares all seven tensor lists mandatory,
    # although the native op accepts null offset/antiquant payloads. A private
    # CPU frontend preserves those nulls and the labels normally attached by
    # the mandatory-list decorator; no OpInfo descriptors or attrs are changed.
    nullable, removed = re.subn(r'^@para_check\.check_op_params\(.*\)\n', '', frontend, flags=re.M)
    assert removed == 1
    labels = '''    names = ['x_in__', 'weight_in__', 'bias_in__', 'scale_in__', 'offset_in__', 'antiquant_scale_in__', 'antiquant_offset_in__', 'group_list_in__', 'per_token_scale_in__']
    for position, argument in enumerate(__inputs__):
        if argument is not None:
            argument.setdefault('param_name', names[position])
    for argument in __outputs__:
        if argument is not None:
            argument.setdefault('param_name', 'y_out_')
'''
    needle = '    return __inputs__, __outputs__, __attrs__\n'
    assert nullable.count(needle) == 1
    nullable = nullable.replace(needle, labels + needle)
    (root / "private_impl/grouped_matmul_cpu_frontend.py").write_text(nullable)
    pre_files = ["grouped_matmul_antiquant_a8w4_msd_pre.h", "grouped_matmul_antiquant_a8w4_msd_pre_nz.h"]
    receipts["pre_unchanged"] = {name: hashlib.sha256((target/name).read_bytes()).hexdigest() for name in pre_files}
    receipts["entry"] = {"original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                         "candidate_sha256": hashlib.sha256(changed.encode()).hexdigest(),
                         "native_parameter_order_unchanged": True, "native_host_op_type": "GroupedMatmul",
                         "totalExpertNum": 48, "activeSlotCount": "device <=6",
                         "extra_workspace_prefix_bytes": 128, "calls_old_WeightNzV2_API": False,
                         "host_tiling_K_N_attributes_changed": False,
                         "host_runtime_adapter_ready": False}
    receipts["private_compiler_frontend"] = {
        "native_sha256": hashlib.sha256(FRONTEND.read_bytes()).hexdigest(),
        "private_nullable_frontend_sha256": hashlib.sha256(nullable.encode()).hexdigest(),
        "nullable_payloads_preserved": True, "formal_param_name_labels_restored": True,
        "kernel_OpInfo_tiling_attributes_changed": False,
        "SDK_global_files_modified": False}
    (root / "source_contract.json").write_text(json.dumps(receipts, indent=2)+"\n")
    (root / "native.diff").write_text("".join(diffs))
    print(json.dumps({"private_source": str(target), "native_math_functions_unchanged":
                      sum(len(r["native_function_body_sha256_unchanged"]) for r in receipts.values() if "native_function_body_sha256_unchanged" in r)}))


if __name__ == "__main__":
    main()
