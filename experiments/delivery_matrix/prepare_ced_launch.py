"""CPU-only role recipe generator; never starts a service or edits a snapshot."""
import argparse
import json
from pathlib import Path
import shlex


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--role", choices=["prefill", "decode"], required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--kv-port", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=1048576)
    parser.add_argument("--kv-bytes", type=int, default=15728022528)
    parser.add_argument("--decode-multistream", choices=["best", "off"], default="best")
    parser.add_argument("--compressor-overlay")
    parser.add_argument("--shared-engram-overlay")
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text())
    env = dict(candidate["startup_env"])
    physical = "0,1,2,3,4,5,6,7" if args.role == "prefill" else "8,9,10,11,12,13,14,15"
    env.update(STACK_SRC_ROOT=args.source, DELIVERY_TMPDIR=str(args.job_root / "tmp"),
               DELIVERY_CED_OVERLAY=args.overlay, DELIVERY_FORM="a3-ced-pd", V41_CED_ROLE=args.role,
               DELIVERY_BASE_WORKER=candidate["model_kwargs"]["worker_cls"], VLLM_ADMISSION_GATE="1",
               V41_ENGRAM_DEVICE_INDEX="0", V41_ENGRAM_ROUTE_PROBE="", V41_CED_ALLOW_DSPARK="1",
               STACK_PHYSICAL_CHIPS=physical, STACK_EXPECTED_CHIPS=physical,
               ASCEND_RT_VISIBLE_DEVICES=physical, ASCEND_VISIBLE_DEVICES=physical)
    for key in ("V41_DCP", "DELIVERY_DCP_OVERLAY", "DELIVERY_DCP_DRAFT"):
        env.pop(key, None)
    env.pop("DELIVERY_CED_COMPRESSOR", None)
    env.pop("DELIVERY_CED_COMPRESSOR_MANIFEST_SHA256", None)
    shared_keys = ("DELIVERY_SHARED_ENGRAM", "DELIVERY_SHARED_ENGRAM_MANIFEST_SHA256", "V41_ENGRAM_SHARED_HOST_MMAP")
    for key in shared_keys:
        env.pop(key, None)
    if args.shared_engram_overlay:
        # The component byte gate runs in bootstrap; the generator can embed
        # a remote path without opening/importing remote SDK modules.
        try:
            from .shared_engram_contract import MANIFEST_SHA
        except ImportError:
            from shared_engram_contract import MANIFEST_SHA
        env.update(DELIVERY_SHARED_ENGRAM=args.shared_engram_overlay,
                   DELIVERY_SHARED_ENGRAM_MANIFEST_SHA256=MANIFEST_SHA,
                   V41_ENGRAM_SHARED_HOST_MMAP="1")
    if args.compressor_overlay:
        if args.role != "decode":
            raise ValueError("The compressor candidate must not be installed on P")
        env["DELIVERY_CED_COMPRESSOR"] = args.compressor_overlay
    if args.role == "prefill":
        env.update(DRAFT_GRAPH="0", DSPARK_GRAPH_CAPTURE_METADATA="0", DSPARK_DRAFT_USE_CUDAGRAPH="0")
    else:
        env.update(V41_CED_GRAPH_PROMPT_TAIL_EAGER="1", V41_CED_SWA_CLIP="1")
    command = ["bash", args.package + "/run_ced_service.sh", "--source", args.source,
               "--overlay", args.overlay, "--role", args.role, "--output", str(args.job_root / "service"),
               "--served-model", args.served_model, "--port", str(args.port), "--kv-port", str(args.kv_port),
               "--max-model-len", str(args.max_model_len), "--kv-bytes", str(args.kv_bytes),
               "--decode-multistream", args.decode_multistream, "--execute"]
    if args.compressor_overlay:
        command.extend(["--compressor-overlay", args.compressor_overlay])
    if args.shared_engram_overlay:
        command.extend(["--shared-engram-overlay", args.shared_engram_overlay])
    body = "#!/usr/bin/env bash\nset -uo pipefail\numask 022\n"
    body += "unset " + " ".join(shared_keys) + "\n"
    body += "\n".join("export " + key + "=" + shlex.quote(value) for key, value in sorted(env.items())) + "\n"
    body += "mkdir -p " + shlex.quote(str(args.job_root)) + "\n"
    body += "echo $$ > " + shlex.quote(str(args.job_root / "runner.pid")) + "\n"
    body += shlex.join(command) + " > " + shlex.quote(str(args.job_root / "run.log")) + " 2>&1\n"
    body += 'ced_status=$?\nprintf "%s\\n" "$ced_status" > ' + shlex.quote(str(args.job_root / "run.exit")) + '\nexit "$ced_status"\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        file.write(body)
    print(json.dumps({"role": args.role, "environment": env, "argv": command,
                      "service_launch_requested": False}, indent=2))


if __name__ == "__main__":
    main()
