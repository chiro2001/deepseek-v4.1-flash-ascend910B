"""Host launcher for the authorized single chip4 fallback; creates one runner."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--container-root", required=True)
    parser.add_argument("--container", default="dsv41-tiny-prof-20261009-c4")
    parser.add_argument("--job", default="ids_audit_perf_v1")
    parser.add_argument("--supervise", action="store_true")
    args = parser.parse_args()
    assert args.container == "dsv41-tiny-prof-20261009-c4"
    assert "ids_lane_private" in str(args.root) and "ids_lane_private" in args.container_root
    assert re.fullmatch(r"[a-z0-9_]+", args.job)
    job = args.root / "results" / args.job
    if args.supervise:
        receipt = json.loads((job / "launch.json").read_text())
        with (job / "runner.log").open("wb") as log:
            result = subprocess.run(receipt["command"], stdout=log, stderr=subprocess.STDOUT)
        (job / "run.exit").write_text(str(result.returncode) + "\n")
        return
    job.mkdir(parents=True, exist_ok=False)
    smi = subprocess.check_output(["npu-smi", "info"], text=True)
    (job / "npu_before.txt").write_text(smi)
    lines = smi.splitlines()
    physical = next(i for i, line in enumerate(lines) if re.search(r"\|\s+0\s+4\s+\|\s+0000:", line))
    assert "| OK" in lines[physical - 1], "chip4 health is not OK"
    assert not any(re.search(r"\|\s+2\s+0\s+\d+\s+\|", line) for line in lines), "chip4 occupied"
    inspect = json.loads(subprocess.check_output(["docker", "inspect", args.container], text=True))[0]
    assert inspect["HostConfig"]["Privileged"], "Fallback device mapping requires separate review"
    assert inspect["State"]["Running"]
    (job / "container_inspect.json").write_text(json.dumps(inspect, indent=2) + "\n")
    visible = subprocess.check_output(["docker", "exec", args.container, "sh", "-c", "test -c /dev/davinci4 && ls -l /dev/davinci4"], text=True)
    (job / "device_mapping.txt").write_text(visible)
    cr = args.container_root
    script = "\n".join([
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh",
        "source /usr/local/Ascend/nnal/atb/set_env.sh",
        f"export PYTHONPATH={shlex.quote(cr + '/ids_lane:' + cr + '/source')}\"${{PYTHONPATH:+:$PYTHONPATH}}\"",
        f"export TRITON_CACHE_DIR={shlex.quote(cr + '/results/' + args.job + '/triton_cache')}",
        f"export ASCEND_CACHE_PATH={shlex.quote(cr + '/results/' + args.job + '/ascend_cache')}",
        "export ASCEND_RT_VISIBLE_DEVICES=4 ASCEND_VISIBLE_DEVICES=4 IDS_PHYSICAL_CHIP=4",
        "export HCCL_DETERMINISTIC=strict LCCL_DETERMINISTIC=1 OMP_NUM_THREADS=1",
        "export V41_DUMMY_WO_A_FIX=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1",
        f"exec python3 {shlex.quote(cr + '/ids_lane/bench_ids_model.py')} --output {shlex.quote(cr + '/results/' + args.job)} --pairs 12 --profile",
    ])
    command = ["docker", "exec", args.container, "bash", "-lc", script]
    receipt = {"command": command, "fixed_physical_chip": 4, "runtime_visible_index": 4,
               "container": args.container, "container_id": inspect["Id"], "source_root": cr,
               "started_unix": time.time(), "pending_new_container_may_only_sleep": "dsv41-tiny-ids-20261011-c4-v1",
               "same_instance_accuracy_before_performance": True}
    (job / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--root", str(args.root),
                             "--container-root", cr, "--container", args.container, "--job", args.job,
                             "--supervise"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    (job / "supervisor.pid").write_text(str(proc.pid) + "\n")
    print(json.dumps({"supervisor_pid": proc.pid, "job": str(job), "physical_chip": 4}), flush=True)


if __name__ == "__main__":
    main()
