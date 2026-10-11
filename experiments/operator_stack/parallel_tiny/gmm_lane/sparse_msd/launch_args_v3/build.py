"""CPU-only launch marshalling build/mock; original objects remain read-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--frozen-root", type=Path, required=True)
    p.add_argument("--loader-v2-root", type=Path, required=True)
    args = p.parse_args()
    root, original, loader = args.root.resolve(), args.frozen_root.resolve(), args.loader_v2_root.resolve()
    assert "gmm_lane_sparse_msd" in str(root) and root != original and root != loader
    output = root / "build"; output.mkdir(exist_ok=False)
    source = Path(__file__).resolve().parent
    sdk = Path("/usr/local/Ascend/cann-9.1.0")
    frozen = json.loads((original / "frozen_artifacts.json").read_text())
    checks = {k: sha(Path(v["path"])) for k, v in frozen["objects"].items()}
    assert checks == {k: v["sha256"] for k, v in frozen["objects"].items()}
    loader_sha = sha(loader / "build/libgmm2_sparse_msd_loader_v2.so")
    assert loader_sha == "b2e6c3033f372a0542d96f9aa2053f230617052aacd2b37bd484a5293b4cf884"
    env = os.environ.copy(); env.update(ASCEND_RT_VISIBLE_DEVICES="", ASCEND_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    common = ["g++", "-std=c++17", "-O2", "-I", str(sdk / "include")]
    jobs = [("shared", common + ["-shared", "-fPIC", str(source / "launch_v3.cpp"), "-L", str(sdk / "lib64"),
                               "-lascendcl", "-Wl,--no-undefined", "-o", str(output / "libgmm2_launch_args_v3.so")]),
            ("mock_build", common + [str(source / "launch_v3.cpp"), str(source / "verify_launch_v3.cpp"),
                                     "-o", str(output / "verify_launch_cpu")]),
            ("mock_run", [str(output / "verify_launch_cpu")])]
    for name, cmd in jobs:
        with (output / (name + ".log")).open("wb") as f: result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        (output / (name + ".json")).write_text(json.dumps({"command": cmd, "exit_code": result.returncode,
                                                           "CPU_only": True, "NPU_launches": 0}, indent=2) + "\n")
        print(json.dumps({"stage": name, "exit_code": result.returncode}), flush=True)
        assert result.returncode == 0
    mock = json.loads((output / "mock_run.log").read_text())
    assert mock["passed"] and mock["real_SDK_runtime_calls"] == mock["NPU_launches"] == 0
    shared = output / "libgmm2_launch_args_v3.so"
    symbols = subprocess.run(["readelf", "-Ws", str(shared)], capture_output=True)
    (output / "shared_symbols.log").write_bytes(symbols.stdout + symbols.stderr)
    imports = sorted({x.split()[-1] for x in symbols.stdout.decode().splitlines() if " UND " in x and x.split()[-1].startswith(("acl", "rt"))})
    assert imports == sorted(["aclrtKernelArgsAppend", "aclrtKernelArgsFinalize", "aclrtLaunchKernelWithConfig", "aclrtLaunchKernelWithHostArgs"])
    deps = subprocess.run(["readelf", "-d", str(output / "verify_launch_cpu")], capture_output=True)
    (output / "mock_dependencies.log").write_bytes(deps.stdout + deps.stderr)
    assert b"libascendcl" not in deps.stdout and b"libruntime" not in deps.stdout
    assert checks == {k: sha(Path(v["path"])) for k, v in frozen["objects"].items()}
    assert loader_sha == sha(loader / "build/libgmm2_sparse_msd_loader_v2.so")
    for name, expected in frozen["source_sha256"].items(): assert sha(original / "source" / name) == expected
    receipt = {"passed": True, "CPU_only": True, "NPU_launches": 0, "runtime_launch_validated": False,
               "mock_contract": mock, "shared_bytes": shared.stat().st_size, "shared_sha256": sha(shared),
               "SDK_imports": imports, "frozen_v3_objects_SHA_unchanged": checks, "loader_v2_SHA_unchanged": loader_sha,
               "source_sha256": {q.name: sha(q) for q in source.iterdir() if q.is_file()},
               "parameter_bytes_unchanged": {"planner": 16, "GMM": 72}, "mathematics_tiling_objects_changed": False}
    (output / "CPU_contract.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": True, "mock": mock, "shared_sha256": receipt["shared_sha256"]}))


if __name__ == "__main__": main()
