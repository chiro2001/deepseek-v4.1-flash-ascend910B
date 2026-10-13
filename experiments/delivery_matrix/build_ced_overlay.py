"""Build a new role-scoped CED overlay from pinned native sources.

Original vendor files, frozen best-source snapshots and existing worktrees are
never modified. Admission/replay/prompt-tail patches are applied to copies.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, repo, out = args.base_root.resolve(strict=True), args.repo.resolve(strict=True), args.output.resolve()
    scheduler = "vllm/v1/core/sched/scheduler.py"
    runner = "vllm_ascend/worker/model_runner_v1.py"
    sources = {scheduler: root / "vllm" / scheduler,
               runner: root / "vllm-ascend" / runner,
               "vllm_ascend/attention/dsa_v41.py": repo / "experimental/ced/dsa_v41.py",
               "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py":
                   repo / "experimental/ced/mooncake_hybrid_connector.py"}
    expected = {scheduler: "c67bda28", runner: "67035d97"}
    for relative, prefix in expected.items():
        if not sha(sources[relative]).startswith(prefix):
            raise ValueError("Actual native CED base changed: " + relative)
    out.mkdir(parents=True, exist_ok=False)
    provenance = {}
    for relative, source in sources.items():
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        provenance[relative] = {"source": str(source), "source_sha256": sha(source)}
    patches = [repo / "patches/admission_gate.patch",
               repo / "experimental/ced/core_scheduler_replay.patch",
               repo / "experimental/ced/core_model_runner_prompt_tail.patch"]
    for index, patch_file in enumerate(patches):
        command = ["patch", "--batch", "--forward", "--fuzz=0", "--no-backup-if-mismatch",
                   "-p1", "--directory", str(out), "--input", str(patch_file)]
        subprocess.run(command[:1] + ["--dry-run"] + command[1:], check=True, capture_output=True, text=True)
        subprocess.run(command, check=True, capture_output=True, text=True)
        if index == 0 and not sha(out / scheduler).startswith("533eed49"):
            raise ValueError("The CED admission-gated scheduler is not the pinned replay base")
    files, modules = {}, {}
    for relative in sources:
        path = out / relative
        ast.parse(path.read_text(), filename=str(path))
        files[relative] = {**provenance[relative], "sha256": sha(path), "bytes": path.stat().st_size}
        modules[relative[:-3].replace("/", ".")] = relative
    manifest = {"schema": "delivery_ced_overlay_v1", "files": files, "modules": modules,
                "patches": [{"path": str(p), "sha256": sha(p)} for p in patches],
                "best_model_and_engram_unchanged": True, "DCP": 1,
                "role_scope": "P encoder+layer20 source / D bounded128 replay",
                "actual_quality_passed": False, "actual_performance_observed": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(out), "manifest_sha256": sha(out / "manifest.json"),
                      "files": len(files), "all_patches_exact_context": True}, indent=2))


if __name__ == "__main__":
    main()
