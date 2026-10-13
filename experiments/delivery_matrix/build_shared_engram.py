"""Derive an opt-in host mmap component from the exact best Engram source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

BEST_ENGRAM_SHA = "b422e9899204b7a55af05832073bdea41e24cdc14900ea739c1dcd82822c77ae"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(source):
    if hashlib.sha256(source.encode()).hexdigest() != BEST_ENGRAM_SHA:
        raise ValueError("Shared Engram must derive from the actual frozen best source")
    marker = "        shard_rows = 1 if self.device_index else self.end - self.start\n"
    if source.count(marker) != 1:
        raise ValueError("Unknown best Engram allocation boundary")
    source = source.replace(marker, '''        self.shared_host_mmap = os.environ.get("V41_ENGRAM_SHARED_HOST_MMAP", "0") == "1"
        if self.shared_host_mmap and (
            storage_format != "int8" or os.environ.get("V41_ENGRAM_HOST_RESIDENT") != "1"
            or _ENGRAM_DEVICE_INDEX
        ):
            raise ValueError("Shared mmap requires host-resident INT8 Engram and device index disabled")
        shard_rows = 1 if self.device_index or self.shared_host_mmap else self.end - self.start
''')
    insertions = {
        "set_rows": '''        if self.shared_host_mmap:
            raise RuntimeError("Read-only shared Engram must bind checkpoint views, never copy/quantize rows")
''',
        "load_checkpoint": '''        if self.shared_host_mmap:
            if hasattr(self, "_shared_host_mappings"):
                raise RuntimeError("Shared Engram cannot replace a live checkpoint mapping")
            from .engram_shared_host import map_checkpoint_rows
            mappings = map_checkpoint_rows(model_path, key, self.rows, self.width, self.start, self.end)
            codes, scales = (mapping.torch_view() for mapping in mappings)
            self.weight = nn.Parameter(codes, requires_grad=False)
            self.weight_scale = scales
            self._shared_host_mappings = mappings  # Keep mmap lifetime >= tensor/model lifetime.
            self._shared_host_receipt = [mapping.receipt for mapping in mappings]
            return
''',
    }
    rows = source.splitlines(keepends=True)
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "NodeShardedEngram")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    for name in sorted(insertions, key=lambda name: methods[name].lineno, reverse=True):
        doc = methods[name].body[0]
        if not isinstance(doc, ast.Expr) or not isinstance(doc.value, ast.Constant) or not isinstance(doc.value.value, str):
            raise ValueError("The reviewed Engram method docstring boundary changed")
        rows[doc.end_lineno:doc.end_lineno] = [insertions[name]]
    result = "".join(rows)
    ast.parse(result)
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.base.resolve(strict=True)
    derived = derive(source.read_text())
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    relative = "vllm_ascend/models/deepseek_v41/engram_hbm.py"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(derived)
    helper = "vllm_ascend/models/deepseek_v41/engram_shared_host.py"
    shutil.copy2(Path(__file__).with_name("shared_engram.py"), root / helper)
    files = {name: {"sha256": digest(root / name), "bytes": (root / name).stat().st_size}
             for name in (relative, helper)}
    manifest = {"schema": "delivery_shared_engram_component_v1", "source": str(source),
                "source_sha256": BEST_ENGRAM_SHA, "files": files,
                "modules": {name[:-3].replace("/", "."): name for name in files},
                "opt_in": "V41_ENGRAM_SHARED_HOST_MMAP=1", "requires_device_index": 0,
                "requires_host_resident": 1, "requires_prequantized": "INT8 weight + FP32 group32 scale",
                "routing_and_lookup_math_unchanged": True, "private_table_allocation_skipped": True,
                "actual_TP8_PD_physical_sharing_proven": False, "actual_quality_passed": False,
                "service_launch_requested": False}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(root), "manifest_sha256": digest(root / "manifest.json"),
                      "actual_quality_passed": False, "service_launch_requested": False}, indent=2))


if __name__ == "__main__":
    main()
