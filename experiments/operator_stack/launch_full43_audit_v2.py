"""Fresh TP8 launch preserving the verified native runtime configuration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from launch_healthy_probe import resources


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def docker_python(container, code, *args):
    return subprocess.check_output(["docker", "exec", container, "python3", "-c", code, *args], text=True)


NATIVE_RUNTIME_BASIS_SHA = "f645edab59417b14f02aa8035cf88f98233ed5d9a5476218bf921868c8b1038f"

def native_runtime_flags(root, startup_env):
    basis_path = root / "results/formal_dspark_native_api_v2/launched.json"
    require(sha(basis_path) == NATIVE_RUNTIME_BASIS_SHA, "Validated native runtime basis SHA changed")
    basis = json.loads(basis_path.read_text())
    require(basis["dspark"] is True and basis["audit"] is False and basis["profiler"] is False
            and basis["chips"] == list(range(8, 16)), "Runtime basis is not the actual native TP8 job")
    flags = dict(basis["env"])
    for key, value in startup_env.items():
        require(key not in flags or flags[key] == value, "Frozen environment conflicts with native runtime: " + key)
        flags[key] = value
    return flags, {"path": str(basis_path), "sha256": NATIVE_RUNTIME_BASIS_SHA,
                   "scope": "same Engram/HCCL/native runtime flags as completed actual HTTP baseline"}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for key in ("root", "source-dir", "container", "job", "model", "snapshot-sha256"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--mode", choices=("diagnostic", "audit"), default="diagnostic")
    parser.add_argument("--generations", type=int, default=32)
    parser.add_argument("--max-output-tokens", type=int, default=192)
    args = parser.parse_args()
    require(args.job.replace("_", "").isalnum() and args.generations == 32
            and args.max_output_tokens == 192, "Need a new valid job and the agreed 32/192 audit geometry")
    root, source = Path(args.root).resolve(), Path(args.source_dir)
    require(source.is_absolute() and source.is_relative_to("/work") and ".." not in source.parts, "Source not under private /work")
    host_source = root / source.relative_to("/work")
    manifest_path = host_source / "full43_snapshot.json"
    require(sha(manifest_path) == args.snapshot_sha256, "Snapshot manifest SHA changed")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["scope"] == "full43 real-weight consumer audit snapshot; no performance or formal admission"
            and manifest["mode"] == args.mode, "Snapshot mode/scope changed")
    for rel, expected in manifest["files"].items():
        require(not Path(rel).is_absolute() and ".." not in Path(rel).parts
                and sha(host_source / rel) == expected, "Frozen snapshot source changed: " + rel)
    if args.mode == "audit": require(not manifest["strict_uncovered_points"], "Audit has uncovered consumers")
    config = docker_python(args.container, "import hashlib,sys; from pathlib import Path; print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())", args.model + "/config.json").strip()
    require(config == manifest["checkpoint_config_sha256"], "Checkpoint config changed")
    container = json.loads(subprocess.check_output(["docker", "inspect", args.container], text=True))[0]
    env = dict(item.split("=", 1) for item in container["Config"]["Env"])
    visible = "8,9,10,11,12,13,14,15"; chips = list(range(8, 16))
    require(container["State"]["Running"] and env["ASCEND_RT_VISIBLE_DEVICES"] == env["STACK_PHYSICAL_CHIPS"] == visible,
            "Container is not the selected physical TP8")
    require(any(m["Destination"] == "/work" and Path(m["Source"]).resolve() == root for m in container["Mounts"]), "Private /work mount differs")
    raw = subprocess.check_output(["npu-smi", "info"], text=True); parsed = resources(raw); health = {}
    for chip in chips:
        require(parsed[chip]["health"] in ("OK", "Alarm") and not parsed[chip]["processes"], "Selected chip busy or unhealthy")
        detail = subprocess.check_output(["npu-smi", "info", "-t", "health", "-i", str(chip // 2), "-c", str(chip % 2)], text=True)
        codes = [line.split(":", 1)[1].strip() for line in detail.splitlines() if "Error Code" in line]
        require(codes == (["80C98001"] if parsed[chip]["health"] == "Alarm" else ["NA"]), "Unapproved chip health code")
        health[chip] = detail
    flags, runtime_basis = native_runtime_flags(root, manifest["startup_env"])
    require(flags.get("HCCL_DETERMINISTIC") == "strict"
            and all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in flags), "Invalid frozen environment")
    enforced = {"STACK_SRC_ROOT": str(source), "STACK_PHYSICAL_CHIPS": visible,
                "ASCEND_RT_VISIBLE_DEVICES": visible, "ASCEND_VISIBLE_DEVICES": visible,
                "HCCL_DETERMINISTIC": "strict", "LOCAL_WORLD_SIZE": "8",
                "STACK_REAL_WEIGHTS": "1", "STACK_REAL_AUDIT": "1", "STACK_TINY_PROFILE": "0"}
    for key, value in enforced.items():
        if key != "STACK_SRC_ROOT": require(key not in flags or flags[key] == value, "Frozen environment contradicts launch: " + key)
    flags.update(enforced)
    job = "/work/results/" + args.job
    out = root / "results" / args.job
    require(not out.exists(), "Do not reuse a job directory")
    script = root / (args.job + ".sh"); require(not script.exists(), "Do not overwrite a job script")
    command = ["bash", str(source / "stack/run_stack.sh"), manifest["entry_file"],
               "--model", args.model, "--physical-chips", visible, "--output", job + "/gate",
               "--generations", str(args.generations), "--max-output-tokens", str(args.max_output_tokens),
               "--mode", args.mode, "--execute"]
    body = "#!/usr/bin/env bash\nset -uo pipefail\numask 022\n"
    body += "\n".join("export " + key + "=" + shlex.quote(value) for key, value in flags.items()) + "\n"
    body += "echo $$ > " + job + "/runner.pid\ndate -u +%FT%TZ > " + job + "/start.utc\n"
    body += shlex.join(command) + " > " + job + "/run.log 2>&1\n"
    body += "driver_status=$?\nprintf '%s\\n' \"$driver_status\" > " + job + "/run.exit\nexit \"$driver_status\"\n"
    fresh = resources(subprocess.check_output(["npu-smi", "info"], text=True))
    require(all(not fresh[c]["processes"] and fresh[c]["health"] == parsed[c]["health"] for c in chips), "Fresh resource state changed")
    docker_python(args.container, "from pathlib import Path; import os,sys; p=Path(sys.argv[1]); p.mkdir(exist_ok=False); os.chown(p,int(sys.argv[2]),int(sys.argv[3]))", job, str(os.getuid()), str(os.getgid()))
    script.write_text(body)
    receipt = {"scope": manifest["scope"], "timestamp_unix": time.time(), "chips": chips,
               "container": args.container, "container_id": container["Id"], "source_dir": str(source),
               "snapshot_manifest_sha256": sha(manifest_path), "full43_freeze_sha256": manifest["full43_freeze_sha256"],
               "source_hashes": manifest["files"], "dependency_freezes": manifest["dependency_freezes"],
               "native_runtime_environment_basis": runtime_basis,
               "sdk_pair_proof_sha256": manifest["sdk_pair_proof_sha256"], "resources": parsed,
               "health_details": health, "argv": command, "env": flags, "mode": args.mode,
               "strict_uncovered_points": manifest["strict_uncovered_points"], "audit": True, "profiler": False,
               "other_tenants_signalled": False, "device_reset": False, "full_consumer_audit_passed": False,
               "performance_receipt_granted": False, "formal_receipt_granted": False, "script_sha256": hashlib.sha256(body.encode()).hexdigest()}
    (out / "launched.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (out / "resource_prelaunch.txt").write_text(raw)
    subprocess.run(["docker", "exec", "-d", args.container, "bash", "/work/" + script.name], check=True)
    print(json.dumps({"job": args.job, "mode": args.mode, "strict_uncovered_points": manifest["strict_uncovered_points"],
                      "full_consumer_audit_passed": False, "formal_receipt_granted": False}), flush=True)


if __name__ == "__main__": main()
