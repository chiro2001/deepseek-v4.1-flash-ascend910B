"""Derive one strict 0..7 guard from the exact async best worker, no bypass."""
import argparse
import hashlib
import json
from pathlib import Path

BASE_SHA = "8dc2192a04f44e92081af8b85ae15df2cfcfbd837da38434612553dac2564d3d"
BASE_CLASS = "parallel_tiny.dspark_formal_prep.worker.FormalDSparkWorker"
BASE_RELATIVE = "stack/parallel_tiny/dspark_formal_prep/worker.py"
CHIPS = "0,1,2,3,4,5,6,7"
OLD = "        assert os.environ['STACK_PHYSICAL_CHIPS']=='8,9,10,11,12,13,14,15'\n"
NEW = "        assert os.environ['STACK_PHYSICAL_CHIPS']=='0,1,2,3,4,5,6,7'\n"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive(source):
    if hashlib.sha256(source.encode()).hexdigest() != BASE_SHA or source.count(OLD) != 1:
        raise ValueError("Exact pinned async full40 worker and its physical guard required")
    return source.replace(OLD, NEW)


def build(base, output):
    base, root = Path(base).resolve(strict=True), Path(output).resolve()
    source = derive(base.read_text())
    root.mkdir(parents=True, exist_ok=False)
    target = root / "worker.py"
    target.write_text(source)
    manifest = {"schema": "delivery_physical_worker_v1", "base_class": BASE_CLASS,
                "base_relative": BASE_RELATIVE, "source_sha256": BASE_SHA, "source": str(base),
                "physical_chips": CHIPS, "files": {"worker.py": {"sha256": sha(target),
                                                                 "bytes": target.stat().st_size}},
                "only_one_physical_literal_changed": True,
                "model_SDK_async_BI_graph_counter_checks_preserved": True,
                "actual_quality_passed": False, "service_launch_requested": False}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output": str(root), "manifest_sha256": sha(root / "manifest.json"),
            "actual_quality_passed": False, "service_launch_requested": False}


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != "delivery_physical_worker_v1" or manifest.get("base_class") != BASE_CLASS
            or manifest.get("base_relative") != BASE_RELATIVE or manifest.get("source_sha256") != BASE_SHA
            or manifest.get("physical_chips") != CHIPS or set(manifest.get("files", {})) != {"worker.py"}
            or manifest.get("only_one_physical_literal_changed") is not True
            or manifest.get("model_SDK_async_BI_graph_counter_checks_preserved") is not True):
        raise ValueError("Unreviewed physical worker derivation")
    target = (root / "worker.py").resolve(strict=True)
    if not target.is_relative_to(root) or sha(target) != manifest["files"]["worker.py"]["sha256"]:
        raise ValueError("Physical worker source/hash changed")
    source = target.read_text()
    if source.count(NEW) != 1 or hashlib.sha256(source.replace(NEW, OLD).encode()).hexdigest() != BASE_SHA:
        raise ValueError("Only the one strict physical guard may change")
    return manifest


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.output), indent=2))


if __name__ == "__main__":
    main()
