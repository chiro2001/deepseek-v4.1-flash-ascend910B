"""Execute one physical8-only registration/planner gate, with frozen hashes.

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
GATE = ROOT / "runtime_registration_p8_20261011"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == os.environ.get("ASCEND_VISIBLE_DEVICES") == "8"
    preflight = json.loads((GATE / "preflight.json").read_text())
    assert preflight["physical"] == 8 and preflight["physical8_processes"] == []
    assert preflight["health_allowed"] and preflight["container_maps_physical8"]
    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp = datetime.datetime.fromisoformat(preflight["UTC"])
    assert 0 <= (now - timestamp).total_seconds() <= 30
    assert not (GATE / "run_receipt.json").exists(), "No runtime retries allowed"
    frozen = json.loads((GATE / "pre_runtime_sha.json").read_text())
    assert all(digest(ROOT / p) == sha for p, sha in frozen["object_sha256"].items())
    assert all(digest(ROOT / "source" / p) == sha for p, sha in frozen["source_sha256"].items())
    compiled = json.loads((GATE / "compile_receipt.json").read_text())
    assert compiled["exit_code"] == 0
    executable = GATE / "runtime_registration_probe"
    assert digest(executable) == compiled["executable_sha256"]
    assert digest(GATE / "runtime_registration_probe.cpp") == compiled["source_sha256"]
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
               "terminal": terminal, "only_physical8_authorized": True,
               "ASCEND_RT_VISIBLE_DEVICES": "8", "ASCEND_VISIBLE_DEVICES": "8",
               "GMM_math_launches": 0, "weight_bytes": 0, "CPU_mutation_objects_launched": 0,
               "runtime_log_bytes": (GATE / "runtime.log").stat().st_size,
               "runtime_log_sha256": digest(GATE / "runtime.log"),
               "objects_sha256_after": {p: digest(ROOT / p) for p in frozen["object_sha256"]}}
    assert receipt["objects_sha256_after"] == frozen["object_sha256"]
    (GATE / "run_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ["events", "objects_sha256_after"]}))
    raise SystemExit(code if 0 <= code <= 255 else 1)


if __name__ == "__main__":
    main()
