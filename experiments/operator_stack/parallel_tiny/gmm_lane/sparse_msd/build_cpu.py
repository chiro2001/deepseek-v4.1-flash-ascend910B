"""Reproducible private CPU-only planner, native/candidate OPC and host build.

OPC may load libruntime and enumerate an empty visible-device set. This is not
a device initialization requirement: visible guards remain empty throughout.
The host adapter links against SDK libraries; only its mock CPU test executes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from prepare_source import NATIVE, SDK


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, name, command, env, execute=False):
    log = root / "logs" / (name + ".log")
    assert not log.exists(), "Refuse to overwrite prior build evidence"
    with log.open("wb") as output:
        result = subprocess.run(command, cwd=root, env=env,
                                stdout=output, stderr=subprocess.STDOUT)
    receipt = {"command": command, "exit_code": result.returncode,
               "CPU_only": True, "NPU_launches": 0,
               "ASCEND_RT_VISIBLE_DEVICES": env["ASCEND_RT_VISIBLE_DEVICES"],
               "ASCEND_VISIBLE_DEVICES": env["ASCEND_VISIBLE_DEVICES"],
               "mock_CPU_test_executed": execute, "log": str(log)}
    (root / (name + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"stage": name, "exit_code": result.returncode}), flush=True)
    if result.returncode:
        raise RuntimeError(name + " failed; see " + str(log))
    return receipt


def planner(root, env):
    output = root / "planner_cpu_rebuild.o"
    assert not output.exists()
    source = HERE / "planner.cpp"
    command = ["/usr/local/Ascend/ascend-toolkit/latest/bin/ccec", "-x", "cce", "-std=c++17",
               "--cce-aicore-only", "--cce-aicore-arch=dav-c220-vec", "-O2", "-c", str(source), "-o", str(output)]
    for rel in ["compiler/tikcpp/tikcfw", "compiler/tikcpp/tikcfw/interface", "compiler/tikcpp/tikcfw/impl",
                "compiler/tikcpp/tikcfw/lib", "aarch64-linux/asc/include", "aarch64-linux/ascendc/include/highlevel_api"]:
        command += ["-I", str(SDK / rel)]
    receipt = run(root, "planner_cpu_rebuild", command, env)
    receipt.update(source_sha256=sha(source), object_sha256=sha(output), object_bytes=output.stat().st_size,
                   target="dav-c220-vec", expected_NPU_ARCH=2201)
    (root / "planner_cpu_rebuild.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def tiling_bytes(header):
    match = re.search(r"const uint8_t __ascendc_arr_##tiling_data\[(\d+)\]\s*=\s*\{([^}]+)\}", header.read_text())
    assert match, "Native generated constant tiling array missing"
    data = bytes(int(x.strip()) for x in match[2].split(",") if x.strip())
    assert len(data) == int(match[1]) == 2032
    return data


def full_gmm(root, env):
    run(root, "prepare_source", [sys.executable, str(HERE / "prepare_source.py"), "--root", str(root)], env)
    native = root / "native_control"
    shutil.copytree(NATIVE, native / "op_kernel")
    frontend = (root / "private_impl/grouped_matmul_cpu_frontend.py").read_text()
    needle = 'origin_func_name = "gmm2_sparse_msd"'
    assert frontend.count(needle) == 1
    (native / "grouped_matmul.py").write_text(frontend.replace(needle, 'origin_func_name = "grouped_matmul"'))
    compiled, arrays = {}, {}
    for flavor, bin_name, frontend_path in [
        ("candidate", "Gmm2SparseMsdDown_candidate_v6", root / "private_impl/grouped_matmul_cpu_frontend.py"),
        ("native", "Gmm2NativeMsdDown_control_v6", native / "grouped_matmul.py")]:
        template = json.loads((HERE / ("evidence/v2/compile_input_" + flavor + ".json")).read_text())
        assert template["op_list"][0]["bin_filename"] == bin_name
        parameter = root / ("compile_input_" + flavor + ".json")
        parameter.write_text(json.dumps(template, indent=2) + "\n")
        command = [sys.executable, str(HERE / "opc_diagnostics.py"), str(frontend_path),
                   "--input_param=" + str(parameter), "--main_func=grouped_matmul",
                   "--output=" + str(root / ("op_build_" + flavor)),
                   "--soc_version=Ascend910_9382", "--op_mode=dynamic", "--deterministic=true",
                   "--op_debug_config=dump_cce", "--log=debug"]
        receipt = run(root, "compile_" + flavor, command, env)
        object_path = root / ("op_build_" + flavor) / (bin_name + ".o")
        header = next((root / ("kernel_meta_" + bin_name) / "kernel_meta").glob("*_tiling_data.h"))
        arrays[flavor] = tiling_bytes(header)
        (root / ("tiling_" + flavor + ".bin")).write_bytes(arrays[flavor])
        receipt.update(object_bytes=object_path.stat().st_size, object_sha256=sha(object_path),
                       tiling_bytes=len(arrays[flavor]), tiling_sha256=hashlib.sha256(arrays[flavor]).hexdigest(),
                       tiling_header=str(header))
        (root / ("compile_" + flavor + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
        compiled[flavor] = receipt
    assert arrays["native"] == arrays["candidate"], "Native/candidate tiling changed"
    equality = {"passed": True, "native_candidate_byte_exact": True, "host_K_N_tiling_preserved": True,
                "bytes": 2032, "sha256": hashlib.sha256(arrays["candidate"]).hexdigest()}
    (root / "tiling_byte_equality.json").write_text(json.dumps(equality, indent=2) + "\n")
    return compiled


def host(root, env):
    compiler = shutil.which("g++") or shutil.which("c++")
    assert compiler, "A C++ host compiler is required; this script does not install one"
    common = [compiler, "-std=c++17", "-O2", "-I", str(SDK / "include"), "-I", str(HERE)]
    shared = root / "libgmm2_sparse_msd_probe.so"
    command = common + ["-fPIC", "-shared", str(HERE / "host_adapter.cpp"), str(HERE / "host_loader.cpp"),
                        "-L", str(SDK / "lib64"), "-lascendcl", "-Wl,--no-undefined", "-o", str(shared)]
    receipt = run(root, "host_adapter_build", command, env)
    receipt.update(object_sha256=sha(shared), object_bytes=shared.stat().st_size,
                   runtime_loaded_or_executed=False, initialization_functions_present=False)
    (root / "host_adapter_build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    mock = root / "verify_host_adapter_cpu"
    run(root, "host_mock_build", common + [str(HERE / "host_adapter.cpp"), str(HERE / "verify_host_adapter.cpp"),
                                           "-o", str(mock)], env)
    run(root, "host_mock_contract", [str(mock)], env, execute=True)
    contract = json.loads((root / "logs/host_mock_contract.log").read_text())
    assert contract["passed"] and contract["real_SDK_runtime_calls"] == 0 and contract["NPU_launches"] == 0
    (root / "host_mock_contract_result.json").write_text(json.dumps(contract, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=["planner", "full-gmm", "host", "all"], default="all")
    args = parser.parse_args()
    root = args.root.resolve()
    assert "gmm_lane_sparse_msd" in str(root) and SDK.exists()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(ASCEND_RT_VISIBLE_DEVICES="", ASCEND_VISIBLE_DEVICES="",
               ASCEND_HOME_PATH=str(SDK), ASCEND_OPP_PATH=str(SDK / "opp"), ASCEND_AICPU_PATH=str(SDK))
    env["PYTHONPATH"] = ":".join([str(SDK / "python/site-packages"),
        str(SDK / "opp/built-in/op_impl/ai_core/tbe"), env.get("PYTHONPATH", "")])
    env["LD_LIBRARY_PATH"] = ":".join([str(SDK / "lib64"),
        str(SDK / "opp/built-in/op_impl/ai_core/tbe/op_tiling"), env.get("LD_LIBRARY_PATH", "")])
    result = {"CPU_compilation_only": True, "NPU_launches": 0, "SDK_global_files_modified": False,
              "target": "DAV2201/Ascend910_9382", "runtime_registration_launch_validated": False}
    if args.stage in ["planner", "all"]: result["planner"] = planner(root, env)
    if args.stage in ["full-gmm", "all"]: result["full_GMM_build"] = full_gmm(root, env)
    if args.stage in ["host", "all"]: result["host_adapter"] = host(root, env)
    (root / ("build_cpu_" + args.stage + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "stage": args.stage, "CPU_only": True, "NPU_launches": 0}), flush=True)


if __name__ == "__main__":
    main()
