"""Freeze a CPU-reviewed DP snapshot without touching a running service."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from .build_runtime import build
from .prepare_launch import shell
from .topology import DEFAULT_CHIP_LIST


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("output", "native-guards", "candidate", "remote-package", "remote-job",
                 "source", "generic-package", "served-model"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=196608)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--loading-threads", type=int)
    parser.add_argument("--host-numa-nodes")
    parser.add_argument("--physical-chips", default=DEFAULT_CHIP_LIST)
    parser.add_argument("--dcp2-base", type=Path)
    parser.add_argument("--dcp-draft-base", type=Path)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    stage = output / "stage"
    source = Path(__file__).resolve().parent
    package = stage / "dp_matrix"
    package.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".sh", ".md"):
            relative = path.relative_to(source)
            if "__pycache__" in relative.parts:
                continue
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    build(stage / "runtime")
    dcp_options = {}
    if args.layout == "dp4-tp2-dcp2":
        if args.dcp2_base is None or args.dcp_draft_base is None:
            parser.error("DCP2 freeze requires exact base and draft input directories")
        from .build_dcp2 import build as build_dcp2, verify as verify_dcp2, verify_draft
        build_dcp2(args.dcp2_base, stage / "dcp2_runtime")
        verify_dcp2(stage / "dcp2_runtime")
        draft_manifest = verify_draft(args.dcp_draft_base)
        draft_output = stage / "dcp_draft"
        draft_output.mkdir()
        for name in [*draft_manifest["files"], "manifest.json"]:
            target = draft_output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.dcp_draft_base / name, target)
        verify_draft(draft_output)
        dcp_options = {
            "dcp_overlay": str(Path(args.remote_package) / "dcp2_runtime"),
            "dcp_draft": str(Path(args.remote_package) / "dcp_draft"),
            "dcp2_overlay_manifest_sha256": hashlib.sha256((stage / "dcp2_runtime/manifest.json").read_bytes()).hexdigest(),
            "dcp_draft_manifest_sha256": hashlib.sha256((draft_output / "manifest.json").read_bytes()).hexdigest(),
        }
    elif args.dcp2_base is not None or args.dcp_draft_base is not None:
        parser.error("DCP source inputs are admitted only for the DCP2 layout")
    shutil.copy2(args.native_guards, stage / "native_source_guards.json")
    records = {}
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            records[path.relative_to(stage).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (stage / "package_files.sha256").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in records.items()))
    archive = output / "dp_matrix.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(stage.iterdir()):
            tar.add(path, arcname=path.name)
    if archive.stat().st_size >= 1_000_000:
        raise ValueError("Package exceeds authorized SSH single-file transfer bound")
    candidate = json.loads(Path(args.candidate).read_text())
    launch, plan = shell(candidate, layout=args.layout, source=args.source,
                         package=str(Path(args.remote_package) / "dp_matrix"),
                         generic_package=args.generic_package,
                         runtime=str(Path(args.remote_package) / "runtime"),
                         native_guards=str(Path(args.remote_package) / "native_source_guards.json"),
                         job_root=args.remote_job, served_model=args.served_model,
                         port=args.port, max_model_len=args.max_model_len,
                         gpu_memory_utilization=args.gpu_memory_utilization,
                         loading_threads=args.loading_threads,
                         host_numa_nodes=args.host_numa_nodes,
                         physical_chips=args.physical_chips, **dcp_options)
    launch_path = output / "launch.sh"
    launch_path.write_text(launch)
    (output / "prepared_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    receipt = {"service_started": False, "files": {
        path.name: {"bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (archive, launch_path)}, "package_source_files": records}
    (output / "freeze_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
