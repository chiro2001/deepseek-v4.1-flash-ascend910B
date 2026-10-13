"""Derive only four masked residual projection addresses from actual SDK."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

BASE_SHA = "0d4d78ebb701c52364308f294dca59c8f94a0cdbf877011e60b31c91b8aa2b39"
MODULE = "vllm_ascend.ops.triton.compressor.compressor_triton"
RELATIVE = MODULE.replace(".", "/") + ".py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive(source):
    if hashlib.sha256(source.encode()).hexdigest() != BASE_SHA:
        raise ValueError("The exact actual compressor SDK source is required")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_pooled_blocked")
    rows = source.splitlines(keepends=True)
    body = "".join(rows[function.lineno - 1:function.end_lineno])
    anchor = "(seg_row_base + seg_off)[:, None] * HEAD_DIM"
    if body.count(anchor) != 4:
        raise ValueError("Expected exactly four masked residual projection loads")
    body = body.replace(anchor, "(seg_row_base + tl.maximum(seg_off, 0))[:, None] * HEAD_DIM")
    result = "".join(rows[:function.lineno - 1]) + body + "".join(rows[function.end_lineno:])
    ast.parse(result)
    return result


def build(base, output):
    base, root = Path(base).resolve(strict=True), Path(output).resolve()
    derived = derive(base.read_text())
    root.mkdir(parents=True, exist_ok=False)
    target = root / RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(derived)
    manifest = {"schema": "delivery_ced_compressor_patch_v1", "role": "decode",
                "source_sha256": BASE_SHA, "source": str(base),
                "files": {RELATIVE: {"sha256": sha(target), "bytes": target.stat().st_size}},
                "modules": {MODULE: RELATIVE}, "masked_projection_load_addresses_clamped": 4,
                "original_masks_ring_selection_math_unchanged": True,
                "actual_failure_cause_proven": False, "actual_quality_passed": False,
                "service_launch_requested": False}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output": str(root), "manifest_sha256": sha(root / "manifest.json"),
            "actual_failure_cause_proven": False, "service_launch_requested": False}


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != "delivery_ced_compressor_patch_v1"
            or manifest.get("role") != "decode" or manifest.get("source_sha256") != BASE_SHA
            or manifest.get("modules") != {MODULE: RELATIVE}
            or set(manifest.get("files", {})) != {RELATIVE}
            or manifest.get("masked_projection_load_addresses_clamped") != 4
            or manifest.get("original_masks_ring_selection_math_unchanged") is not True):
        raise ValueError("Unreviewed CED compressor component")
    target = (root / RELATIVE).resolve(strict=True)
    if not target.is_relative_to(root) or sha(target) != manifest["files"][RELATIVE]["sha256"]:
        raise ValueError("CED compressor payload/source changed")
    # Reversing the only permitted address change must reproduce exact SDK.
    source = target.read_text()
    anchor = "(seg_row_base + tl.maximum(seg_off, 0))[:, None] * HEAD_DIM"
    if (source.count(anchor) != 4 or hashlib.sha256(source.replace(
            anchor, "(seg_row_base + seg_off)[:, None] * HEAD_DIM").encode()).hexdigest() != BASE_SHA):
        raise ValueError("The compressor component changes more than four addresses")
    return manifest


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.output), indent=2))


if __name__ == "__main__":
    main()
