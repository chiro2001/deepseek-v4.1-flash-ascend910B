"""Execute one physical6-only HostArgs R0 gate, with frozen hashes.

Requires a fresh external npu-smi/health/container check saved as preflight.json.
This is not a general launcher. Any runtime failure is terminal; no retries.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path("/work/gmm_lane_sparse_msd_build_20261011_v3_repro")
V2 = Path("/work/gmm_lane_sparse_msd_runtime_loader_v2_cpu_20261011")
V3 = Path("/work/gmm_lane_sparse_msd_launch_args_v3_cpu_20261011")
GATE = Path("/work/gmm_lane_sparse_msd_launch_args_v3_npu_p6_20261011_v1")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == os.environ.get("ASCEND_VISIBLE_DEVICES") == "6"
    preflight = json.loads((GATE / "preflight.json").read_text())
    assert preflight["physical"] == 6 and preflight["physical6_processes"] == []
    assert preflight["health_allowed"] and preflight["container_has_dev6_permission"]
    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp = datetime.datetime.fromisoformat(preflight["UTC"])
    assert 0 <= (now - timestamp).total_seconds() <= 30
    assert not (GATE / "run_receipt.json").exists(), "No runtime retries allowed"
    frozen = json.loads((GATE / "pre_runtime_sha.json").read_text())
    objects = frozen["frozen_v3"]["objects"]
    assert all(digest(Path(v["path"])) == v["sha256"] for v in objects.values())
    assert all(digest(ROOT / "source" / p) == sha for p, sha in frozen["frozen_v3"]["source_sha256"].items())
    assert digest(V2 / "build/libgmm2_sparse_msd_loader_v2.so") == frozen["loader_v2"]["shared_sha256"]
    assert all(digest(V2 / "source/runtime_loader_v2" / p) == sha for p, sha in frozen["loader_v2"]["source_sha256"].items())
    assert digest(V3 / "build/libgmm2_launch_args_v3.so") == frozen["launch_v3"]["shared_sha256"]
    assert all(digest(V3 / "source" / q) == sha for q, sha in frozen["launch_v3"]["source_sha256"].items())
    compiled = json.loads((GATE / "compile_receipt.json").read_text())
    assert compiled["exit_code"] == 0
    executable = GATE / "runtime_hostargs_r0"
    assert digest(executable) == compiled["executable_sha256"]
    assert digest(GATE / "runtime_hostargs_r0.cpp") == compiled["source_sha256"]
    command = [str(executable), str(ROOT)]
    timeout = False
    with (GATE / "runtime.log").open("wb") as log:
        try:
            process = subprocess.run(command, cwd=GATE, env=os.environ.copy(),
                                     stdout=log, stderr=subprocess.STDOUT, timeout=60)
            code = process.returncode
        except subprocess.TimeoutExpired:
            timeout = True
            code = -999
    events = []
    for line in (GATE / "runtime.log").read_text(errors="replace").splitlines():
        if line.startswith("{"):
            try: events.append(json.loads(line))
            except json.JSONDecodeError: pass
    terminal = next((e for e in reversed(events) if e.get("terminal")), None)
    receipt = {"command": command, "exit_code": code, "timeout": timeout, "events": events,
               "terminal": terminal, "only_physical6_authorized": True,
               "ASCEND_RT_VISIBLE_DEVICES": "6", "ASCEND_VISIBLE_DEVICES": "6",
               "GMM_math_launches": 0, "weight_bytes": 0, "CPU_mutation_objects_launched": 0,
               "runtime_log_bytes": (GATE / "runtime.log").stat().st_size,
               "runtime_log_sha256": digest(GATE / "runtime.log"),
               "objects_sha256_after": {p: digest(Path(v["path"])) for p, v in objects.items()},
               "loader_v2_sha256_after": digest(V2 / "build/libgmm2_sparse_msd_loader_v2.so"),
               "launch_v3_sha256_after": digest(V3 / "build/libgmm2_launch_args_v3.so")}
    assert receipt["objects_sha256_after"] == {p: v["sha256"] for p, v in objects.items()}
    assert receipt["loader_v2_sha256_after"] == frozen["loader_v2"]["shared_sha256"]
    assert receipt["launch_v3_sha256_after"] == frozen["launch_v3"]["shared_sha256"]
    (GATE / "run_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ["events", "objects_sha256_after"]}))
    raise SystemExit(code if 0 <= code <= 255 else 1)


if __name__ == "__main__":
    main()
