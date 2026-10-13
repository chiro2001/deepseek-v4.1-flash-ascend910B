"""Derive a NEW pinned DCP candidate; never change a frozen runtime in place."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil

BASE_DSA_SHA = "8f2f11fc2c097579040957b3eb42c2f2d97ac9213199282cca1a7f2ec45a7d76"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(source):
    if hashlib.sha256(source.encode()).hexdigest() != BASE_DSA_SHA:
        raise ValueError("The frozen v3 DCP attention source is required")
    pattern = r'_perf_flags\(\)\.get\("det_reduce"\) == "1"(?: or _DCP_DET_REDUCE)?'
    updated, count = re.subn(pattern, "_v41_det_reduce_enabled()", source)
    if count != 5:
        raise ValueError("All five actual reduction gates must be explicitly covered")
    # Merely replacing their predicates leaves the kernel and folded pack
    # branches using a full, unbounded gather. Route those bodies through the
    # same helper too; retain the kernel's resident pack pointer for replay.
    rows = updated.splitlines(keepends=True)
    inline = []
    for branch in ast.walk(ast.parse(updated)):
        if (not isinstance(branch, ast.If) or not isinstance(branch.test, ast.Call)
                or not isinstance(branch.test.func, ast.Name)
                or branch.test.func.id != "_v41_det_reduce_enabled"):
            continue
        gathers = [node for statement in branch.body for node in ast.walk(statement)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "all_gather_into_tensor"]
        if not gathers:
            continue
        if len(gathers) != 1 or not isinstance(gathers[0].args[1], ast.Name):
            raise ValueError("Unexpected inline ordered gather body")
        payload = gathers[0].args[1].id
        if payload == "_pk":
            replacement = "_pk.copy_(_v41_ordered_allreduce(_pk, group))"
        elif payload == "_pack":
            replacement = "_pack = _v41_ordered_allreduce(_pack, group)"
        else:
            raise ValueError("Unreviewed inline gather payload: " + payload)
        indent = " " * branch.body[0].col_offset
        inline.append((branch.body[0].lineno - 1, branch.body[-1].end_lineno,
                       indent + replacement + "\n"))
    if len(inline) != 2:
        raise ValueError("Both full kernel/pack gathers must be removed")
    for start, stop, replacement in sorted(inline, reverse=True):
        rows[start:stop] = [replacement]
    updated = "".join(rows)
    node = next(node for node in ast.parse(updated).body
                if isinstance(node, ast.FunctionDef) and node.name == "_v41_ordered_allreduce")
    rows = updated.splitlines(keepends=True)
    replacement = '''def _v41_det_reduce_enabled():
    # One switch for pack, sepw denominator, no-pack, legacy and kernel paths.
    return _DCP_DET_REDUCE or _perf_flags().get("det_reduce") == "1"


def _v41_ordered_allreduce(t, group):
    from delivery_matrix.dcp_reduction import ordered_allreduce
    return ordered_allreduce(t, group)
'''
    rows[node.lineno - 1:node.end_lineno] = [replacement]
    result = "".join(rows)
    ast.parse(result)
    verify_bounded_branches(result)
    return result


def verify_bounded_branches(source):
    merge = next(node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "_v41_dcp_merge_attention")
    branches = [node for node in ast.walk(merge) if isinstance(node, ast.If)
                and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name)
                and node.test.func.id == "_v41_det_reduce_enabled"]
    if len(branches) != 5:
        raise ValueError("All five reduction branches must share the switch")
    for branch in branches:
        calls = [node for statement in branch.body for node in ast.walk(statement)
                 if isinstance(node, ast.Call)]
        if (not any(isinstance(node.func, ast.Name) and node.func.id == "_v41_ordered_allreduce"
                    for node in calls)
                or any(isinstance(node.func, ast.Attribute)
                       and node.func.attr in ("all_gather_into_tensor", "all_reduce") for node in calls)):
            raise ValueError("An ordered reduction branch still bypasses the bounded helper")


def build(base, output):
    base, output = Path(base).resolve(strict=True), Path(output).resolve()
    manifest = json.loads((base / "manifest.json").read_text())
    if manifest.get("schema") != "delivery_dcp_overlay_v2":
        raise ValueError("A pinned minimal v3 DCP runtime is required")
    for relative, record in manifest["files"].items():
        source = (base / relative).resolve(strict=True)
        if not source.is_relative_to(base) or sha(source) != record["sha256"]:
            raise ValueError("Frozen DCP base payload hash mismatch: " + relative)
    dsa = "vllm_ascend/attention/dsa_v41.py"
    derived = derive((base / dsa).read_text())
    output.mkdir(parents=True, exist_ok=False)
    for relative in manifest["files"]:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / relative, target)
    (output / dsa).write_text(derived)
    manifest["files"][dsa].update(sha256=sha(output / dsa), bytes=(output / dsa).stat().st_size,
                                  derivation_base_sha256=BASE_DSA_SHA)
    helper = "delivery_matrix/dcp_reduction.py"
    target = output / helper
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("dcp_reduction.py"), target)
    manifest["files"][helper] = {"sha256": sha(target), "bytes": target.stat().st_size,
                                 "delivery_source": str(Path(__file__).with_name("dcp_reduction.py"))}
    manifest["modules"]["delivery_matrix.dcp_reduction"] = helper
    manifest["stable_reduction_candidate"] = {
        "base_manifest_sha256": sha(base / "manifest.json"), "source_attention_sha256": BASE_DSA_SHA,
        "all_five_reduction_gates_covered": True, "environment_switch": "V41_DCP_DET_REDUCE=1",
        "all_five_gates_use_bounded_helper": True,
        "fixed_kernel_pack_address_preserved": True,
        "max_gather_staging_bytes": 128 * 1024 * 1024, "max_chunk_rows": 64,
        "gather_input_alignment_bytes": 512,
        "GSM_failure_cause_established": False, "actual_stability_or_quality_passed": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output": str(output), "manifest_sha256": sha(output / "manifest.json"),
            "actual_quality_passed": False, "service_launch_requested": False}


def verify_candidate(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    contract = manifest.get("stable_reduction_candidate") or {}
    if (contract.get("source_attention_sha256") != BASE_DSA_SHA
            or contract.get("all_five_reduction_gates_covered") is not True
            or contract.get("all_five_gates_use_bounded_helper") is not True
            or contract.get("fixed_kernel_pack_address_preserved") is not True
            or contract.get("max_gather_staging_bytes") != 128 * 1024 * 1024
            or contract.get("max_chunk_rows") != 64 or contract.get("gather_input_alignment_bytes") != 512):
        raise ValueError("Software ordered reduction requires the all-branch bounded candidate, not v3")
    for relative in ("vllm_ascend/attention/dsa_v41.py", "delivery_matrix/dcp_reduction.py"):
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or sha(path) != manifest["files"][relative]["sha256"]:
            raise ValueError("Bounded DCP candidate payload changed")
    if sha(root / "delivery_matrix/dcp_reduction.py") != sha(Path(__file__).with_name("dcp_reduction.py")):
        raise ValueError("The reduction cap/helper is not the reviewed source")
    verify_bounded_branches((root / "vllm_ascend/attention/dsa_v41.py").read_text())
    return contract


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.output), indent=2))


if __name__ == "__main__":
    main()
