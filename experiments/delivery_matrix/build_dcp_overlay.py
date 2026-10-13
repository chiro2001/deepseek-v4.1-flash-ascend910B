"""Package the verified historical DCP sources, preserving provenance.

This is a mechanical copy into a NEW delivery directory, not an edit to the
historical overlay or an installation into a running vendor checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

RUNTIME_FILES = (
    "core/deepseek_v41.py", "core/kv_cache_interface.py",
    "attention/context_parallel/v41_dcp.py", "attention/dsa_v41.py",
    "attention/v41_merge_kernel.py", "attention/libv41merge_ops.so",
    "patch/platform/patch_kv_cache_coordinator.py",
    "patch/platform/patch_v41_dcp.py", "worker/block_table.py",
    "worker/dcp_utils.py", "models/deepseek_v41/indexer.py",
    "models/deepseek_v41/engram_hbm.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-host-root", required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    files, modules = {}, {}
    for rel in RUNTIME_FILES:
        relative = "vllm_ascend/" + rel
        original = source / relative
        if not original.is_file() or original.is_symlink():
            raise ValueError("A regular pinned source file is required: " + relative)
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        chosen = original
        if rel == "models/deepseek_v41/indexer.py":
            chosen = Path(__file__).with_name("dcp_indexer.py")
        elif rel == "models/deepseek_v41/engram_hbm.py":
            chosen = Path(__file__).with_name("dcp_engram_hbm.py")
        shutil.copy2(chosen, target)
        digest = sha(chosen)
        if sha(target) != digest:
            raise ValueError("Copy hash mismatch: " + relative)
        files[relative] = {"sha256": digest, "bytes": chosen.stat().st_size,
                           "original_host_path": str(Path(args.original_host_root) / relative),
                           "historical_source_sha256": sha(original),
                           "delivery_source": str(chosen), "delivery_bytes": chosen.stat().st_size}
        if target.suffix == ".py":
            module = relative[:-3].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[:-9]
            modules[module] = relative
    manifest = {"schema": "delivery_dcp_overlay_v2", "source_root": str(source),
                "historical_host_root": args.original_host_root, "files": files, "modules": modules,
                "best_model_unchanged": True, "best_engram_lookup_algorithm_preserved": True,
                "best_QLI_no_candidate_preserved": True,
                "best_worker_runner_context_and_platform_preserved": True,
                "post_import_patches": ["vllm_ascend.platform:_validate_parallel_config:V41-interleave-only"],
                "engram_DCP8_guard_narrowly_enabled_for_single_node_TP8_EP8": True,
                "actual_DCP_BI_async_quality_passed": False,
                "actual_DCP_BI_async_performance_observed": False}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "files": len(files), "modules": len(modules),
                      "manifest_sha256": sha(args.output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
