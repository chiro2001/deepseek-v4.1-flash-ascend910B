"""CPU-size the exact native KFC message structs and bound native clear stores."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def definition(source, name):
    match = re.search(r"\bstruct " + name + r"\s*\{", source)
    assert match, name
    begin, depth = match.start(), 1
    cursor = match.end()
    while depth:
        if source[cursor] == "{": depth += 1
        elif source[cursor] == "}": depth -= 1
        cursor += 1
    assert source[cursor] == ";"
    return source[begin:cursor + 1]


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert "gmm_lane_sparse_msd" in str(root)
    sdk = Path("/usr/local/Ascend/cann-9.1.0")
    native = sdk / "aarch64-linux/asc/impl/basic_api/dav_c220/kfc/kfc_comm.h"
    original = native.read_text(errors="replace")
    preprocessed = next((root / "kernel_meta_Gmm2SparseMsdDown_candidate_v6/kernel_meta").glob("*_mix_aic.i"))
    expanded = preprocessed.read_text()
    assert "constexpr int32_t MIX_NUM = 2;" in expanded
    assert "BIDIRECTION_NUM * 64 * AlignTo32(sizeof(KfcMsg)) * MIX_NUM" in expanded
    structs = [definition(original, name) for name in ["TilingInfo", "MatmulOrgShape", "MatmulConfigParams",
        "MatmulUserDefInfo", "Conv3DBpInputConfigParams", "Conv3DBpFilterConfigParams", "Conv3DForwardConfigParams",
        "KfcMsg", "MsgUBAvailable", "MsgMatmulCnt"]]
    directory = root / "workspace_contract"
    directory.mkdir(exist_ok=False)
    source = directory / "native_struct_sizes.cpp"
    source.write_text("#include <cstdint>\n#include <cstdio>\nusing GM_ADDR=uint8_t*;\n" + "\n".join(structs) + r'''
int main() {
    static_assert(sizeof(void*)==8);
    const uint64_t msg=(sizeof(KfcMsg)+31)/32*32;
    const uint64_t ub=(sizeof(MsgUBAvailable)+31)/32*32;
    const uint64_t cnt=(sizeof(MsgMatmulCnt)+31)/32*32;
    const uint64_t stride=2*64*msg*2;
    const uint64_t ub_offset=50*2*64*msg + 50*8*cnt;
    const uint64_t msg_end=24*stride;
    const uint64_t ub_end=ub_offset+24*2*ub;
    std::printf("{\"KfcMsg_bytes\":%llu,\"MsgUBAvailable_bytes\":%llu,\"MsgMatmulCnt_bytes\":%llu,"
        "\"msg_clear_end_exclusive\":%llu,\"ub_clear_end_exclusive\":%llu}\n",
        (unsigned long long)sizeof(KfcMsg),(unsigned long long)sizeof(MsgUBAvailable),
        (unsigned long long)sizeof(MsgMatmulCnt),(unsigned long long)msg_end,(unsigned long long)ub_end);
}
''')
    compiler = shutil.which("g++") or shutil.which("c++")
    assert compiler
    command = [compiler, "-std=c++17", "-O2", str(source), "-o", str(directory / "verify_cpu")]
    compiled = subprocess.run(command, capture_output=True)
    (directory / "compile.log").write_bytes(compiled.stdout + compiled.stderr)
    assert compiled.returncode == 0
    executed = subprocess.run([str(directory / "verify_cpu")], capture_output=True)
    (directory / "run.log").write_bytes(executed.stdout + executed.stderr)
    assert executed.returncode == 0
    result = json.loads(executed.stdout)
    assert result["KfcMsg_bytes"] == 128 and result["MsgUBAvailable_bytes"] == result["MsgMatmulCnt_bytes"] == 64
    boundary = 16 * 1024 * 1024
    assert max(result["msg_clear_end_exclusive"], result["ub_clear_end_exclusive"]) < boundary
    receipt = {"passed": True, "CPU_only": True, "NPU_launches": 0,
               "native_structs_copied_without_changes": True,
               "native_kfc_source": str(native), "native_kfc_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
               "real_AIC_preprocessed_file": str(preprocessed), "AIC_blocks": 24, "MIX_NUM": 2,
               "MAX_MSG_COUNT_expanded": 64, "system_reservation_bytes": boundary,
               "planner_frame_begin": boundary, "planner_frame_bytes": 128,
               "native_scratch_begin": boundary + 128, "compiler_command": command,
               "compile_exit": compiled.returncode, "CPU_execution_exit": executed.returncode, **result}
    (directory / "workspace_contract.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
