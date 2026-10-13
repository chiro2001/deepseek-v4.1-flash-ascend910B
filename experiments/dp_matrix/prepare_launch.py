"""CPU-only shell generator. No service/process/device is started here."""
import argparse
import json
from pathlib import Path
import shlex

from .launch_plan import make_plan


CLEAR = (
    "V41_CED_ROLE", "DELIVERY_CED_OVERLAY", "DELIVERY_CED_COMPRESSOR",
    "DELIVERY_CED_COMPRESSOR_MANIFEST_SHA256", "DELIVERY_CED_MANIFEST_SHA256",
    "DELIVERY_CED_OPP_FINGERPRINT", "DELIVERY_SHARED_ENGRAM", "DELIVERY_SHARED_ENGRAM_MANIFEST_SHA256",
    "V41_ENGRAM_SHARED_HOST_MMAP", "V41_DCP", "DELIVERY_DCP_OVERLAY", "DELIVERY_DCP_DRAFT",
    "V41_DCP_DET_REDUCE", "V41_DCP_REPLICATE_INDEXER", "V41_DCP_ALLOW_CAPACITY_PROBE", "DP_MATRIX_DCP2_OVERLAY_MANIFEST_SHA256",
    "DP_MATRIX_DCP2_DRAFT_MANIFEST_SHA256", "DP_MATRIX_DCP2_CONTRACT_SHA256",
    "DP_MATRIX_DCP2_BOOTSTRAP", "DP_MATRIX_DCP2_BOOTSTRAP_SHA256", "DELIVERY_PHYSICAL_WORKER",
    "DELIVERY_PHYSICAL_WORKER_MANIFEST_SHA256", "STACK_TP8_ARM", "KV_CACHE_MEMORY_BYTES",
    "DP_MATRIX_NUMA_INTERLEAVE_NODES",
)


def shell(candidate, *, layout, source, package, generic_package, runtime,
          native_guards, job_root, served_model, port, max_model_len=196608,
          gpu_memory_utilization=0.90, loading_threads=None, host_numa_nodes=None,
          dcp_overlay=None, dcp_draft=None,
          dcp2_overlay_manifest_sha256=None, dcp_draft_manifest_sha256=None):
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("An explicit unprivileged port is required")
    job = Path(job_root)
    plan = make_plan(candidate, layout=layout, source=source, output=job / "service",
                     served_model=served_model, max_model_len=max_model_len,
                     gpu_memory_utilization=gpu_memory_utilization,
                     loading_threads=loading_threads,
                     host_numa_nodes=host_numa_nodes,
                     dcp_overlay=dcp_overlay, dcp_draft=dcp_draft,
                     dcp2_overlay_manifest_sha256=dcp2_overlay_manifest_sha256,
                     dcp_draft_manifest_sha256=dcp_draft_manifest_sha256)
    env = dict(plan["environment"])
    env.update(STACK_SRC_ROOT=str(source), DP_MATRIX_GENERIC_PACKAGE=str(generic_package),
               DP_MATRIX_TMPDIR=str(job / "tmp"), DP_MATRIX_RUNTIME=str(runtime))
    for key in CLEAR:
        env.pop(key, None)
    if host_numa_nodes is not None:
        env["DP_MATRIX_NUMA_INTERLEAVE_NODES"] = plan["environment"]["DP_MATRIX_NUMA_INTERLEAVE_NODES"]
    if layout == "dp4-tp2-dcp2":
        for key in ("V41_DCP", "V41_DCP_DET_REDUCE", "V41_DCP_REPLICATE_INDEXER",
                    "DELIVERY_DCP_OVERLAY", "DELIVERY_DCP_DRAFT",
                    "DP_MATRIX_DCP2_OVERLAY_MANIFEST_SHA256", "DP_MATRIX_DCP2_DRAFT_MANIFEST_SHA256",
                    "DP_MATRIX_DCP2_CONTRACT_SHA256"):
            env[key] = plan["environment"][key]
        env["DP_MATRIX_DCP2_BOOTSTRAP"] = str(Path(package) / "dcp2_bootstrap")
    command = ["bash", str(Path(package) / "run_service.sh"), "--source", str(source),
               "--output", str(job / "service"), "--runtime", str(runtime),
               "--native-source-guards", str(native_guards), "--layout", layout,
               "--served-model", served_model, "--port", str(port),
               "--max-model-len", str(max_model_len),
               "--gpu-memory-utilization", str(gpu_memory_utilization), "--execute"]
    if loading_threads is not None:
        command.extend(["--loading-threads", str(loading_threads)])
    if host_numa_nodes is not None:
        command.extend(["--host-numa-nodes", host_numa_nodes])
    if layout == "dp4-tp2-dcp2":
        command.extend(["--dcp2-overlay", str(dcp_overlay), "--dcp-draft", str(dcp_draft)])
    body = "#!/usr/bin/env bash\nset -uo pipefail\numask 022\n"
    body += "unset " + " ".join(CLEAR) + "\n"
    body += "\n".join("export " + key + "=" + shlex.quote(value)
                      for key, value in sorted(env.items())) + "\n"
    body += "mkdir -p " + shlex.quote(str(job)) + "\n"
    body += "echo $$ > " + shlex.quote(str(job / "runner.pid")) + "\n"
    body += shlex.join(command) + " > " + shlex.quote(str(job / "run.log")) + " 2>&1\n"
    body += 'dp_matrix_status=$?\nprintf "%s\\n" "$dp_matrix_status" > ' + shlex.quote(str(job / "run.exit"))
    body += '\nexit "$dp_matrix_status"\n'
    return body, plan


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("candidate", "source", "package", "generic-package", "runtime", "native-guards",
                 "job-root", "served-model", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--max-model-len", type=int, default=196608)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--loading-threads", type=int)
    parser.add_argument("--host-numa-nodes")
    parser.add_argument("--dcp2-overlay")
    parser.add_argument("--dcp-draft")
    parser.add_argument("--dcp2-overlay-manifest-sha256")
    parser.add_argument("--dcp-draft-manifest-sha256")
    args = parser.parse_args()
    candidate = json.loads(Path(args.candidate).read_text())
    body, plan = shell(candidate, layout=args.layout, source=args.source, package=args.package,
                       generic_package=args.generic_package, runtime=args.runtime,
                       native_guards=args.native_guards, job_root=args.job_root,
                       served_model=args.served_model, port=args.port,
                       max_model_len=args.max_model_len,
                       gpu_memory_utilization=args.gpu_memory_utilization,
                       loading_threads=args.loading_threads,
                       host_numa_nodes=args.host_numa_nodes,
                       dcp_overlay=args.dcp2_overlay, dcp_draft=args.dcp_draft,
                       dcp2_overlay_manifest_sha256=args.dcp2_overlay_manifest_sha256,
                       dcp_draft_manifest_sha256=args.dcp_draft_manifest_sha256)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        file.write(body)
    print(json.dumps({"launch_script": str(output), "plan": plan,
                      "service_started": False, "NPU_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
