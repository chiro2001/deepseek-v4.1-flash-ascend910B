"""CPU compile/link/mock v2 loader without touching frozen v3 math or objects."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    args = parser.parse_args()
    root, frozen = args.root.resolve(), args.frozen_root.resolve()
    assert "gmm_lane_sparse_msd" in str(root) and root != frozen
    root.mkdir(exist_ok=True)
    output = root / "build"
    output.mkdir(exist_ok=False)
    source = Path(__file__).resolve().parent
    sdk = Path("/usr/local/Ascend/cann-9.1.0")
    before = json.loads((frozen / "frozen_artifacts.json").read_text())
    objects = {k: sha(Path(v["path"])) for k, v in before["objects"].items()}
    assert all(objects[k] == v["sha256"] for k, v in before["objects"].items())
    assert sha(source.parent / "host_adapter.h") == before["source_sha256"]["host_adapter.h"]
    env = os.environ.copy()
    env.update(ASCEND_RT_VISIBLE_DEVICES="", ASCEND_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    commands = [
        ("shared", ["g++", "-std=c++17", "-O2", "-fPIC", "-shared", str(source / "loader_v2.cpp"),
                    "-I", str(sdk / "include"), "-L", str(sdk / "lib64"), "-lascendcl", "-Wl,--no-undefined",
                    "-o", str(output / "libgmm2_sparse_msd_loader_v2.so")]),
        ("mock_build", ["g++", "-std=c++17", "-O2", str(source / "loader_v2.cpp"), str(source / "verify_loader_v2.cpp"),
                        "-I", str(sdk / "include"), "-o", str(output / "verify_loader_cpu")]),
        ("mock_run", [str(output / "verify_loader_cpu"), str(frozen)]),
    ]
    receipts = {}
    for name, command in commands:
        with (output / (name + ".log")).open("wb") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipts[name] = {"command": command, "exit_code": result.returncode, "CPU_only": True, "NPU_launches": 0}
        (output / (name + ".json")).write_text(json.dumps(receipts[name], indent=2) + "\n")
        print(json.dumps({"stage": name, "exit_code": result.returncode}), flush=True)
        assert result.returncode == 0, name + " failed"
    contract = json.loads((output / "mock_run.log").read_text())
    assert contract["passed"] and contract["real_SDK_runtime_calls"] == contract["NPU_launches"] == 0
    shared = output / "libgmm2_sparse_msd_loader_v2.so"
    symbols = subprocess.run(["readelf", "-Ws", str(shared)], capture_output=True)
    (output / "shared_symbols.log").write_bytes(symbols.stdout + symbols.stderr)
    sdk_symbols = sorted({x.split()[-1] for x in symbols.stdout.decode().splitlines() if " UND " in x and x.split()[-1].startswith(("acl", "rt"))})
    assert sdk_symbols == sorted(["aclrtBinaryLoadFromData", "aclrtBinaryGetFunction", "aclrtGetFunctionAddr", "aclrtBinaryUnLoad"])
    dependencies = subprocess.run(["readelf", "-d", str(output / "verify_loader_cpu")], capture_output=True)
    (output / "mock_dependencies.log").write_bytes(dependencies.stdout + dependencies.stderr)
    assert b"libascendcl" not in dependencies.stdout and b"libruntime" not in dependencies.stdout
    assert objects == {k: sha(Path(v["path"])) for k, v in before["objects"].items()}
    for filename, expected in before["source_sha256"].items(): assert sha(frozen / "source" / filename) == expected
    receipt = {"passed": True, "CPU_only": True, "NPU_launches": 0, "runtime_registration_validated": False,
               "shared_bytes": shared.stat().st_size, "shared_sha256": sha(shared), "SDK_undefined_symbols": sdk_symbols,
               "mock_contract": contract, "frozen_v3_objects_SHA_unchanged": objects,
               "frozen_v3_source_SHA_unchanged": before["source_sha256"],
               "source_sha256": {p.name: sha(p) for p in source.iterdir() if p.is_file()},
               "borrowed_frozen_header_sha256": sha(source.parent / "host_adapter.h"),
               "remaining_runtime_boundaries": ["Data load accepts explicit vector/mixed MAGIC", "raw planner registration",
                                                "base kernelName mixed AIC/AIV pairing", "planner frame GM visibility"]}
    (output / "CPU_contract.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": True, "mock": contract, "shared_sha256": receipt["shared_sha256"]}))


if __name__ == "__main__":
    main()
