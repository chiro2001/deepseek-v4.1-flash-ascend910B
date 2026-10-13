"""Create a source-pinned full43 audit snapshot without importing any runtime.

Run on the remote host. The bundle contains only exact frozen text files; the
base snapshot, operator libraries and weights stay on that host.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


V10 = "stack/parallel_tiny/dspark_formal_audit/full43_consumers_v10/"
DEPENDENCIES = {
    "stack/parallel_tiny/dspark_formal_audit/formal_consumer_sdk_bridge_v9_cpu/":
        ("freeze.json", "files", "1bbdc613969add1f47d9e678ac9ade9d35ad9e187a9985e8c21b67ff881623e5", ("SHA256SUMS",)),
    "stack/parallel_tiny/dspark_formal_audit/native_initialized_worker_discovery_v8/":
        ("freeze.json", "files", "73f865b8ce39a631db991cac1d6f8b39b35e9b7ab1e5cfdc1d89e7cadae33432", ("SHA256SUMS",)),
    "stack/parallel_tiny/dspark_prepack_audit_cpu/":
        ("frozen_files.json", "text_files", "bf2becb2f9e9e87e1d72ff1bbb23c88a5414f1d914e5ce8dec7b035182a902bd", ("delivery.json",)),
}
CONFIG_SHA = "40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424"
DRIVER = "parallel_tiny/dspark_formal_audit/full43_consumers_v10/formal_audit_driver.py"


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_bundle(path):
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        require(members and len({m.name for m in members}) == len(members), "Empty/duplicate archive members")
        require(all(m.isfile() and m.size < 1_000_000 and not Path(m.name).is_absolute()
                    and ".." not in Path(m.name).parts and m.name.startswith("stack/") for m in members),
                "Bundle must contain only bounded, regular, safe stack files")
        return {m.name: archive.extractfile(m).read() for m in members}


def frozen_members(payload, prefix, manifest_name, field, expected_sha, extras):
    name = prefix + manifest_name
    require(name in payload and hashlib.sha256(payload[name]).hexdigest() == expected_sha,
            "Freeze SHA differs: " + name)
    manifest = json.loads(payload[name])
    files = manifest[field]
    require(isinstance(files, dict) and files, "Freeze has no precise file SHA list")
    for rel, spec in files.items():
        require(not Path(rel).is_absolute() and ".." not in Path(rel).parts, "Unsafe frozen path")
        expected = spec["sha256"] if isinstance(spec, dict) else spec
        require(prefix + rel in payload and hashlib.sha256(payload[prefix + rel]).hexdigest() == expected,
                "Frozen payload differs: " + prefix + rel)
    names = {prefix + rel for rel in files} | {name} | {prefix + rel for rel in extras}
    require(names <= set(payload), "Missing frozen package metadata")
    return manifest, names


def prepare(args):
    package = args.package_name
    require(package in ("full43_consumers_v10", "full43_consumers_v11", "full43_consumers_v12", "full43_consumers_v21"),
            "Unknown frozen full43 package")
    prefix = "stack/parallel_tiny/dspark_formal_audit/" + package + "/"
    driver = "parallel_tiny/dspark_formal_audit/" + package + "/formal_audit_driver.py"
    for name in (args.base_name, args.source_name, args.bundle_name):
        require(name and name.replace("_", "").replace("-", "").replace(".", "").isalnum(), "Unsafe root child name")
    root = Path(args.root).resolve()
    base, source, bundle = root / args.base_name, root / args.source_name, root / args.bundle_name
    require(base.is_dir() and not source.exists(), "Base missing or source snapshot already exists")
    old = json.loads((base / "stack/parallel_tiny/dspark_formal_prep/freeze.json").read_text())
    for rel, expected in old["files"].items():
        require(sha(base / "stack" / rel) == expected, "Base frozen source changed: " + rel)
    payload = read_bundle(bundle)
    candidate, expected_names = frozen_members(payload, prefix, "freeze.json", "files", args.full43_freeze_sha256, ("SHA256SUMS",))
    dependencies = {}
    for prefix, specification in DEPENDENCIES.items():
        _, names = frozen_members(payload, prefix, *specification)
        expected_names |= names
        dependencies[prefix] = {"manifest": specification[0], "sha256": specification[2]}
    require(set(payload) == expected_names, "Unexpected/missing bundle files")
    require("stack/" + driver in payload, "No runnable full43 driver in freeze")
    require("strict_uncovered_points" in candidate and isinstance(candidate["strict_uncovered_points"], list),
            "Freeze must list explicit uncovered points")
    if args.mode == "audit":
        require(not candidate["strict_uncovered_points"], "Audit mode cannot omit actual consumer coverage")
    require(candidate.get("full_consumer_audit_passed", False) is False
            and candidate.get("formal_model_validated", False) is False, "CPU freeze grants unsupported device admission")
    proof_name = "stack/parallel_tiny/dspark_formal_audit/formal_consumer_sdk_bridge_v9_cpu/evidence/actual_pair_protocol.json"
    proof = json.loads(payload[proof_name])
    require(proof["protocol_passed"] is True and proof["full_consumer_audit_passed"] is False,
            "SDK pair protocol missing or promoted to full consumers")
    startup_env = candidate.get("startup_env")
    require(isinstance(startup_env, dict) and startup_env.get("HCCL_DETERMINISTIC") == "strict"
            and all(isinstance(k, str) and isinstance(v, str) for k, v in startup_env.items()),
            "Need the frozen actual startup environment with HCCL strict")
    source.mkdir()
    for name in ("stack", "operator", "up950"):
        shutil.copytree(base / name, source / name, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (source / "runtime/tmp").mkdir(parents=True)
    for rel, raw in payload.items():
        target = source / rel; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
    paths = {source / "stack" / rel for rel in old["files"]} | {source / rel for rel in payload}
    paths |= {source / "stack" / rel for rel in ("run_stack.sh", "sitecustomize.py", "formal_model_contract.py")}
    manifest = {"scope": "full43 real-weight consumer audit snapshot; no performance or formal admission",
                "base": str(base), "mode": args.mode, "entry_file": driver, "full43_package": package,
                "full43_freeze_sha256": args.full43_freeze_sha256, "dependency_freezes": dependencies,
                "sdk_pair_proof_sha256": hashlib.sha256(payload[proof_name]).hexdigest(),
                "checkpoint_config_sha256": CONFIG_SHA, "startup_env": startup_env,
                "implemented_points": candidate.get("implemented_points", []),
                "strict_uncovered_points": candidate["strict_uncovered_points"],
                "bundle_sha256": sha(bundle), "files": {str(p.relative_to(source)): sha(p) for p in sorted(paths)},
                "NPU_calls": 0, "full_consumer_audit_passed": False, "formal_receipt_granted": False,
                "performance_receipt_granted": False}
    path = source / "full43_snapshot.json"; path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"source": str(source), "files": len(paths), "manifest_sha256": sha(path),
            "mode": args.mode, "strict_uncovered_points": manifest["strict_uncovered_points"], "NPU_calls": 0}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for key in ("root", "base-name", "source-name", "bundle-name", "full43-freeze-sha256"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--package-name", choices=("full43_consumers_v10", "full43_consumers_v11", "full43_consumers_v12", "full43_consumers_v21"),
                        default="full43_consumers_v21")
    parser.add_argument("--mode", choices=("diagnostic", "audit"), default="diagnostic")
    print(json.dumps(prepare(parser.parse_args())))


if __name__ == "__main__": main()
