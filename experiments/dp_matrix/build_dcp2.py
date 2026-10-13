"""Derive a new DCP2 runtime without modifying frozen DCP8 sources."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

from .dcp2_contract import derive_engram

if __package__.startswith("experiments."):
    from experiments.delivery_matrix.build_dcp_stability import verify_candidate
else:
    from delivery_matrix.build_dcp_stability import verify_candidate


ENGRAM_SHA = "16c7cb674cf28390e42c1032350ced6aeeeca84ba4ea572d18d46d3c7de1541d"
BASE_MANIFEST_SHA = "b0c1d36e44707c872707477677a5ef60a4d3d53f86c5df0d1092e47e41706a89"
DRAFT_MANIFEST_SHA = "abb1a60cd24f8040a9f87a3a69471a0e38e2a166aed6d073f43c6229316d5074"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(base, output):
    base = Path(base).resolve(strict=True)
    output = Path(output).resolve()
    verify_candidate(base)
    manifest = json.loads((base / "manifest.json").read_text())
    base_manifest_sha = sha(base / "manifest.json")
    if (manifest.get("schema") != "delivery_dcp_overlay_v2"
            or manifest.get("best_model_unchanged") is not True
            or manifest.get("best_engram_lookup_algorithm_preserved") is not True
            or manifest.get("stable_reduction_candidate", {}).get("all_five_gates_use_bounded_helper") is not True):
        raise ValueError("The frozen bounded-reduction DCP8 source candidate is required")
    for relative, record in manifest["files"].items():
        source = (base / relative).resolve(strict=True)
        if not source.is_relative_to(base) or sha(source) != record["sha256"]:
            raise ValueError("Frozen DCP base changed: " + relative)
    relative = "vllm_ascend/models/deepseek_v41/engram_hbm.py"
    if sha(base / relative) != ENGRAM_SHA:
        raise ValueError("Unreviewed Engram source base")
    if base_manifest_sha != BASE_MANIFEST_SHA:
        raise ValueError("The exact reviewed DCP v6 manifest is required")
    updated = derive_engram((base / relative).read_text())
    ast.parse(updated)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(base / "manifest.json", output / "dcp2_base_manifest.json")
    shutil.copy2(base / relative, output / "dcp2_base_engram.py")
    for name in manifest["files"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / name, target)
    target = output / relative
    target.write_text(updated)
    manifest["files"][relative].update(sha256=sha(target), bytes=target.stat().st_size,
        derivation_base_sha256=ENGRAM_SHA, delivery_source="dp_matrix.dcp2_contract.derive_engram")
    manifest["engram_DCP8_guard_narrowly_enabled_for_single_node_TP8_EP8"] = False
    manifest["native_dp_geometry"] = {
        "layout": "dp4-tp2-dcp2", "dp": 4, "tp": 2, "dcp": 2, "ep": 8,
        "base_manifest_sha256": base_manifest_sha,
        "dcp2_contract_sha256": sha(Path(__file__).with_name("dcp2_contract.py")),
        "engram_config_and_actual_groups_required": True,
        "engram_lookup_owner_metadata_and_broadcast_unchanged": True,
        "quality_inherited": False, "actual_quality_passed": False,
        "actual_performance_observed": False, "actual_capacity_observed": False,
        "service_started": False,
    }
    # No source or historical receipt can grant this new geometry a PASS.
    manifest["actual_DCP_BI_async_quality_passed"] = False
    manifest["actual_DCP_BI_async_performance_observed"] = False
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output": str(output), "manifest_sha256": sha(output / "manifest.json"),
            "service_started": False, "quality_inherited": False}


def verify(root):
    """Verify the DCP2 derivation and its runtime contract before importing vendor code."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError("DCP2 overlay root must not be a symlink")
    root = root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    base_manifest_path = root / "dcp2_base_manifest.json"
    base_engram_path = root / "dcp2_base_engram.py"
    if (base_manifest_path.is_symlink() or base_engram_path.is_symlink()
            or sha(base_manifest_path) != BASE_MANIFEST_SHA or sha(base_engram_path) != ENGRAM_SHA):
        raise ValueError("DCP2 derivation originals differ from the exact reviewed base")
    base_manifest = json.loads(base_manifest_path.read_text())
    geometry = manifest.get("native_dp_geometry") or {}
    if (manifest.get("schema") != "delivery_dcp_overlay_v2"
            or geometry.get("layout") != "dp4-tp2-dcp2"
            or (geometry.get("dp"), geometry.get("tp"), geometry.get("dcp"), geometry.get("ep")) != (4, 2, 2, 8)
            or geometry.get("base_manifest_sha256") != BASE_MANIFEST_SHA
            or geometry.get("dcp2_contract_sha256") != sha(Path(__file__).with_name("dcp2_contract.py"))
            or geometry.get("quality_inherited") is not False
            or geometry.get("actual_quality_passed") is not False
            or geometry.get("actual_performance_observed") is not False
            or geometry.get("actual_capacity_observed") is not False
            or geometry.get("service_started") is not False
            or manifest.get("actual_DCP_BI_async_quality_passed") is not False
            or manifest.get("actual_DCP_BI_async_performance_observed") is not False):
        raise ValueError("DCP2 overlay manifest does not bind the reviewed native geometry/contract")
    if (manifest.get("best_model_unchanged") is not True
            or manifest.get("best_engram_lookup_algorithm_preserved") is not True
            or manifest.get("stable_reduction_candidate", {}).get("all_five_gates_use_bounded_helper") is not True):
        raise ValueError("DCP2 overlay lost the reviewed bounded-reduction/Engram base")
    if (set(manifest.get("files", {})) != set(base_manifest["files"])
            or manifest.get("modules") != base_manifest["modules"]):
        raise ValueError("DCP2 overlay changed the frozen module/payload set")
    for relative, record in manifest.get("files", {}).items():
        raw_path = root / relative
        if raw_path.is_symlink():
            raise ValueError("DCP2 overlay payload is a symlink: " + relative)
        path = raw_path.resolve(strict=True)
        if (not path.is_relative_to(root) or not path.is_file()
                or sha(path) != record.get("sha256")
                or path.stat().st_size != record.get("bytes")):
            raise ValueError("DCP2 overlay payload SHA/size mismatch: " + relative)
        if (relative != "vllm_ascend/models/deepseek_v41/engram_hbm.py"
                and sha(path) != base_manifest["files"][relative]["sha256"]):
            raise ValueError("DCP2 changed a frozen non-Engram payload: " + relative)
    engram = "vllm_ascend/models/deepseek_v41/engram_hbm.py"
    row = manifest["files"].get(engram, {})
    if (row.get("derivation_base_sha256") != ENGRAM_SHA
            or row.get("delivery_source") != "dp_matrix.dcp2_contract.derive_engram"):
        raise ValueError("DCP2 Engram file is not the narrow reviewed derivation")
    if (root / engram).read_bytes() != derive_engram(base_engram_path.read_text()).encode():
        raise ValueError("DCP2 Engram payload differs from the exact bounded derivation")
    return manifest


def verify_draft(root):
    """Verify the original best DCP draft graph patch and every payload byte."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError("DCP2 draft root must not be a symlink")
    root = root.resolve(strict=True)
    manifest_raw = (root / "manifest.json").read_bytes()
    if sha(root / "manifest.json") != DRAFT_MANIFEST_SHA:
        raise ValueError("DCP2 requires the exact frozen best draft manifest")
    manifest = json.loads(manifest_raw)
    if (manifest.get("schema") != "delivery_dcp_best_draft_patch_v1"
            or manifest.get("parallel_token_layout_preserved") is not True
            or manifest.get("best_graph_BI_async_patches_preserved") is not True
            or manifest.get("actual_quality_passed") is not False
            or manifest.get("actual_performance_observed") is not False):
        raise ValueError("DCP2 draft patch is not the pinned best BI/async graph candidate")
    for relative, record in manifest.get("files", {}).items():
        raw_path = root / relative
        if raw_path.is_symlink():
            raise ValueError("DCP2 draft payload is a symlink: " + relative)
        path = raw_path.resolve(strict=True)
        if (not path.is_relative_to(root)
                or sha(path) != record.get("sha256")
                or path.stat().st_size != record.get("bytes")):
            raise ValueError("DCP2 draft payload SHA/size mismatch: " + relative)
    return manifest


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.output), indent=2))


if __name__ == "__main__":
    main()
