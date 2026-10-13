"""CPU-only launch artifact generator; does not launch a service."""
import argparse
import json
from pathlib import Path
import shlex

if __package__:
    from .service import physical_chips
else:
    from service import physical_chips


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--job-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--form", choices=["a2-tp8-a3sim", "a3-tp8-dcp8"], required=True)
    parser.add_argument("--device-index", type=int, choices=[0, 1], default=1)
    parser.add_argument("--physical-chips", type=physical_chips)
    parser.add_argument("--physical-worker")
    parser.add_argument("--dcp-overlay")
    parser.add_argument("--dcp-draft")
    parser.add_argument("--dcp-replicate-indexer", type=int, choices=[0, 1], default=1)
    parser.add_argument("--dcp-det-reduce", type=int, choices=[0, 1])
    parser.add_argument("--max-model-len", type=int, default=196608)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text())
    if args.dcp_det_reduce is not None and args.form != "a3-tp8-dcp8":
        raise ValueError("Software DCP reduction is not an A2-sim option")
    env = dict(candidate["startup_env"])
    clear = ("V41_DCP", "DELIVERY_DCP_OVERLAY", "DELIVERY_DCP_DRAFT", "V41_DCP_DET_REDUCE",
             "V41_DCP_REPLICATE_INDEXER", "V41_CED_ROLE", "DELIVERY_CED_OVERLAY",
             "DELIVERY_CED_COMPRESSOR", "DELIVERY_CED_COMPRESSOR_MANIFEST_SHA256",
             "DELIVERY_CED_MANIFEST_SHA256", "DELIVERY_CED_OPP_FINGERPRINT",
             "DELIVERY_PHYSICAL_WORKER", "DELIVERY_PHYSICAL_WORKER_MANIFEST_SHA256",
             "DELIVERY_SHARED_ENGRAM", "DELIVERY_SHARED_ENGRAM_MANIFEST_SHA256", "V41_ENGRAM_SHARED_HOST_MMAP")
    for key in clear:
        env.pop(key, None)
    env.update(STACK_SRC_ROOT=args.source, DELIVERY_TMPDIR=str(args.job_root / "tmp"),
               V41_ENGRAM_DEVICE_INDEX=str(args.device_index),
               V41_ENGRAM_ROUTE_PROBE="",
               DELIVERY_FORM=args.form, DELIVERY_BASE_WORKER=candidate["model_kwargs"]["worker_cls"])
    if args.physical_chips is not None:
        env.update(STACK_PHYSICAL_CHIPS=args.physical_chips, STACK_EXPECTED_CHIPS=args.physical_chips,
                   ASCEND_RT_VISIBLE_DEVICES=args.physical_chips, ASCEND_VISIBLE_DEVICES=args.physical_chips)
    if args.physical_worker is not None:
        if args.physical_chips != "0,1,2,3,4,5,6,7":
            raise ValueError("The physical worker candidate is explicitly limited to 0..7")
        env["DELIVERY_PHYSICAL_WORKER"] = args.physical_worker
    elif (args.physical_chips is not None and candidate["startup_env"].get("STACK_PHYSICAL_CHIPS")
          not in (None, args.physical_chips)):
        raise ValueError("Moving a frozen physical guard requires --physical-worker")
    if args.form == "a3-tp8-dcp8":
        if not args.dcp_overlay or not args.dcp_draft:
            raise ValueError("DCP8 requires the pinned DCP runtime overlay")
        env.update(DELIVERY_DCP_OVERLAY=args.dcp_overlay, V41_DCP="1",
                   DELIVERY_DCP_DRAFT=args.dcp_draft,
                   V41_DCP_REPLICATE_INDEXER=str(args.dcp_replicate_indexer))
        if args.dcp_det_reduce is not None:
            env["V41_DCP_DET_REDUCE"] = str(args.dcp_det_reduce)
    command = ["bash", args.package + "/run_service.sh", "--source", args.source,
               "--output", str(args.job_root / "service"), "--port", str(args.port),
               "--served-model", args.served_model, "--form", args.form,
               "--max-model-len", str(args.max_model_len),
               "--device-index", str(args.device_index), "--execute"]
    if args.dcp_det_reduce is not None:
        command.extend(["--dcp-det-reduce", str(args.dcp_det_reduce)])
    if args.physical_chips is not None:
        command.extend(["--physical-chips", args.physical_chips])
    if args.physical_worker is not None:
        command.extend(["--physical-worker", args.physical_worker])
    body = "#!/usr/bin/env bash\nset -uo pipefail\numask 022\n"
    # Container-level CED env must be removed BEFORE Python sitecustomize,
    # not merely overwritten after native model modules have been imported.
    body += "unset " + " ".join(clear) + "\n"
    body += "\n".join("export " + key + "=" + shlex.quote(value) for key, value in sorted(env.items())) + "\n"
    body += "mkdir -p " + shlex.quote(str(args.job_root)) + "\n"
    body += "echo $$ > " + shlex.quote(str(args.job_root / "runner.pid")) + "\n"
    body += shlex.join(command) + " > " + shlex.quote(str(args.job_root / "run.log")) + " 2>&1\n"
    body += 'delivery_status=$?\nprintf "%s\\n" "$delivery_status" > ' + shlex.quote(str(args.job_root / "run.exit")) + '\nexit "$delivery_status"\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        file.write(body)
    print(json.dumps({"argv": command, "environment": env, "service_launch_requested": False}, indent=2))


if __name__ == "__main__":
    main()
