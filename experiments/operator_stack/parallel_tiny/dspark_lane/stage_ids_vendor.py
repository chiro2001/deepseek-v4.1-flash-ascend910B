"""Clone the successful IDS runtime and restore exact pre-audit native sources."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path("/home/l00886679/projects/dsv41-tiny-prof-20261009")
IDS = ROOT / "ids_lane_private_v3"
REPLACEMENTS = {
    "model": ("vllm_ascend/models/deepseek_v41/model.py", [
        ("        import ids_patches\n        ids_patches.record_model_ids(input_ids, moe_input_ids)\n", ""),
    ]),
    "router": ("vllm_ascend/ops/fused_moe/router/fused_topk_router.py", [
        ("                import ids_patches\n                ids_patches.record_router_before(self, input_ids)\n", ""),
        ("                ids_patches.record_router_after_cast(input_ids)\n", ""),
        ("            if input_ids is not None:\n                ids_patches.record_router_consumer(input_ids)\n", ""),
        ("            if input_ids is not None:\n                ids_patches.record_router_result(topk_weights, topk_ids)\n", ""),
    ]),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    dest = args.destination.absolute()
    assert dest.name == "vendor" and dest.parent.parent == ROOT
    assert dest.parent.name.startswith("dspark_lane_20261011_c4_")
    assert not dest.exists()
    assert (IDS / "results/ids_audit_perf_v3/run.exit").read_text().strip() == "0"
    controls = json.loads((IDS / "instrumentation_receipt.json").read_text())
    source = IDS / "source"
    factory = source / "vllm_ascend/patch/platform/patch_fused_moe.py"
    factory_text = factory.read_text()
    tree = ast.parse(factory_text)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.name == "_ascend_FusedMoE"]
    assert len(functions) == 1
    params = {a.arg for a in functions[0].args.args + functions[0].args.kwonlyargs}
    assert {"bias_vl", "image_sentinel_lo"} <= params
    assert "bias_vl=bias_vl" in factory_text and "image_sentinel_lo=image_sentinel_lo" in factory_text
    shutil.copytree(source, dest, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", ".git", "*.pyc"))
    restored = {}
    for role, (rel, replacements) in REPLACEMENTS.items():
        original = source / rel
        target = dest / rel
        assert not original.is_symlink() and not target.is_symlink(), "Never rewrite a shared symlink"
        assert sha(original) == controls[role]["instrumented_sha256"]
        text = target.read_text()
        for before, after in replacements:
            assert text.count(before) == 1, (rel, before)
            text = text.replace(before, after)
        assert "ids_patches" not in text
        ast.parse(text, filename=str(target))
        target.write_text(text)
        assert sha(target) == controls[role]["original_sha256"]
        restored[role] = {"relative_path": rel, "instrumented_sha256": sha(original),
                          "restored_native_sha256": sha(target), "exact_pre_audit_source": True}
    files = []
    digest = hashlib.sha256()
    for path in sorted(dest.rglob("*.py")):
        relative = str(path.relative_to(dest))
        h = sha(path)
        files.append({"path": relative, "sha256": h, "bytes": path.stat().st_size})
        digest.update(relative.encode() + b"\0" + h.encode() + b"\n")
    receipt = {"successful_ids_source": str(source), "ids_job_exit": 0,
               "destination": str(dest), "restored_sources": restored,
               "factory_path": str(factory), "factory_sha256": sha(factory),
               "factory_preserves_bias_vl": True, "factory_preserves_image_sentinel_lo": True,
               "source_py_tree_sha256": digest.hexdigest(), "source_py_files": len(files),
               "sources": files, "no_global_factory_mutation": True,
               "no_dummy_weight_change": True, "cpu_only_no_npu_init": True}
    (dest.parent / "vendor_source_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "sources"}), flush=True)


if __name__ == "__main__":
    main()
