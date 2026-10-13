"""Pinned component and live CPU-table backing checks; never NPU probes.

These contracts prove selected source, actual tensor/mmap aliasing and P/D
file/range identity. Physical PSS and workload correctness are separate gates.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

MODULE = "vllm_ascend.models.deepseek_v41.engram_hbm"
HELPER = "vllm_ascend.models.deepseek_v41.engram_shared_host"
MANIFEST_SHA = "37ec7b7a9bb9a2812cabc35535784dc7a7b5c643270d2dd4f59e444a06e04e75"
BASE_SHA = "b422e9899204b7a55af05832073bdea41e24cdc14900ea739c1dcd82822c77ae"
SOURCE_SHA = "895222b92ef67e5d79044e3bec8a334d8f37eca937359aab8c0f34ff416d75b6"
FILES = {MODULE: "vllm_ascend/models/deepseek_v41/engram_hbm.py",
         HELPER: "vllm_ascend/models/deepseek_v41/engram_shared_host.py"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_component(root):
    root = Path(root).resolve(strict=True)
    if sha(root / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Shared Engram requires the pinned immutable component manifest")
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != "delivery_shared_engram_component_v1"
            or manifest.get("source_sha256") != BASE_SHA
            or manifest.get("modules") != FILES
            or set(manifest.get("files", {})) != set(FILES.values())
            or manifest.get("requires_device_index") != 0
            or manifest.get("requires_host_resident") != 1
            or manifest.get("routing_and_lookup_math_unchanged") is not True
            or manifest.get("private_table_allocation_skipped") is not True):
        raise ValueError("Unsupported shared Engram component contract")
    for relative, receipt in manifest["files"].items():
        path = (root / relative).resolve(strict=True)
        if (not path.is_relative_to(root) or not path.is_file()
                or path.stat().st_size != receipt["bytes"] or sha(path) != receipt["sha256"]):
            raise ValueError("Shared Engram payload changed: " + relative)
    return manifest


def _loaded_component(root, manifest):
    loaded = {}
    for name, relative in FILES.items():
        module = sys.modules.get(name)
        if module is None or not getattr(module, "__file__", None):
            raise ValueError("Selected shared Engram module was not actually loaded: " + name)
        path = Path(module.__file__).resolve(strict=True)
        if path != (root / relative).resolve(strict=True) or sha(path) != manifest["files"][relative]["sha256"]:
            raise ValueError("Loaded Engram source is not the selected immutable component")
        loaded[name] = module
    return loaded


def describe_tables(model, component_root, expected_manifest_sha):
    root = Path(component_root).resolve(strict=True)
    manifest = verify_component(root)
    if expected_manifest_sha != MANIFEST_SHA:
        raise ValueError("Worker selection differs from the shared component fingerprint")
    loaded = _loaded_component(root, manifest)
    cls, mapping_cls = loaded[MODULE].NodeShardedEngram, loaded[HELPER].SharedTensorRows
    tables = []
    for name, table in model.named_modules():
        if not isinstance(table, cls):
            continue
        if (getattr(table, "shared_host_mmap", None) is not True or table.device_index
                or table.weight.device.type != "cpu" or table.weight_scale.device.type != "cpu"
                or table.weight.requires_grad
                or tuple(table.weight.shape) != (table.end - table.start, table.width)
                or tuple(table.weight_scale.shape) != (table.end - table.start, table.width // 32)):
            raise ValueError("An actual Engram table still has private/device-index storage")
        mappings = getattr(table, "_shared_host_mappings", ())
        saved = getattr(table, "_shared_host_receipt", ())
        if len(mappings) != 2 or list(saved) != [item.receipt for item in mappings]:
            raise ValueError("Missing actual weight/scale mappings or stale loader receipts")
        receipts = []
        for index, (mapping, tensor) in enumerate(zip(mappings, (table.weight, table.weight_scale))):
            receipt = mapping.receipt
            if (not isinstance(mapping, mapping_cls) or mapping.mapping is None
                    or receipt.get("MAP_SHARED") is not True
                    or receipt.get("PROT_READ_only") is not True
                    or receipt.get("whole_table_private_copy") is not False
                    or receipt.get("ACL_host_registered") is not False
                    or receipt.get("dtype") != ("I8" if index == 0 else "F32")
                    or receipt.get("row_start") != table.start or receipt.get("row_stop") != table.end
                    or receipt.get("table_shape") != [table.rows, table.width if index == 0 else table.width // 32]):
                raise ValueError("Actual Engram mapping has the wrong access mode or geometry")
            info = os.stat(receipt["path"])
            if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
                    receipt["device"], receipt["inode"], receipt["file_bytes"], receipt["mtime_ns"]):
                raise ValueError("Shared Engram backing checkpoint changed after mapping")
            # A same-shaped clone would pass shape/dtype checks but lose physical
            # sharing. Reconstruct only a zero-copy tensor view, no row reads.
            view = mapping.torch_view()
            if tensor.data_ptr() != view.data_ptr() or tensor.dtype != view.dtype:
                raise ValueError("Engram tensor does not alias its actual file mapping")
            del view
            receipts.append(dict(receipt))
        if receipts[1]["key"] != receipts[0]["key"].removesuffix(".weight") + ".scale":
            raise ValueError("Weight/scale checkpoint keys do not belong to one table")
        tables.append({"module_name": name, "mappings": receipts,
                       "tensor_aliases_file_mapping": True, "private_table_copy": False})
    if not tables:
        raise ValueError("Shared Engram selected but no actual NodeShardedEngram tables found")
    keys = [table["mappings"][0]["key"] for table in tables]
    if len(set(keys)) != len(keys):
        raise ValueError("Ambiguous duplicate Engram table checkpoint keys")
    return {"manifest_sha256": MANIFEST_SHA, "source_sha256": manifest["files"][FILES[MODULE]]["sha256"],
            "pid": os.getpid(), "tables": sorted(tables, key=lambda item: item["mappings"][0]["key"]),
            "actual_tensor_mmap_aliasing_passed": True,
            "actual_TP8_PD_physical_sharing_proven": False, "actual_PSS_capacity_or_quality_passed": False}


def describe_selected(model):
    root = os.environ.get("DELIVERY_SHARED_ENGRAM")
    if not root:
        return None
    return describe_tables(model, root, os.environ.get("DELIVERY_SHARED_ENGRAM_MANIFEST_SHA256"))


def verify_pair_shared_tables(prefill_identity, decode_identity):
    """Validate all eight per-side actual mapping receipts, not measured PSS."""
    sides = []
    for identity, role in ((prefill_identity, "prefill"), (decode_identity, "decode")):
        if identity.get("role") != role:
            raise ValueError("Shared Engram comparison requires the selected P and D roles")
        descriptions = identity.get("rank_descriptions", [])
        if (len(descriptions) != 8 or any(type(row.get("rank")) is not int for row in descriptions)
                or {row["rank"] for row in descriptions} != set(range(8))):
            raise ValueError("Shared Engram requires exactly eight unique ranks per side")
        ranks = {}
        for row in descriptions:
            shared = row.get("shared_engram")
            if (not shared or shared.get("manifest_sha256") != MANIFEST_SHA
                    or shared.get("source_sha256") != SOURCE_SHA
                    or shared.get("actual_tensor_mmap_aliasing_passed") is not True):
                raise ValueError("One P/D rank did not load the selected shared mappings")
            mappings = {}
            for table in shared.get("tables", []):
                if table.get("tensor_aliases_file_mapping") is not True or table.get("private_table_copy") is not False:
                    raise ValueError("A rank still holds a private table copy")
                for item in table["mappings"]:
                    key = item["key"]
                    if (item.get("MAP_SHARED") is not True or item.get("PROT_READ_only") is not True
                            or item.get("whole_table_private_copy") is not False
                            or item.get("ACL_host_registered") is not False):
                        raise ValueError("Serialized mapping access mode is not read-only shared")
                    if key in mappings:
                        raise ValueError("Duplicate mapping key in a worker receipt")
                    mappings[key] = item
            if not mappings:
                raise ValueError("Empty actual shared Engram table set")
            ranks[row["rank"]] = mappings
        sides.append(ranks)
    p, d = sides
    identity_fields = ("key", "device", "inode", "file_bytes", "mtime_ns", "header_sha256", "dtype",
                       "table_shape", "row_start", "row_stop", "tensor_begin", "view_begin", "view_bytes",
                       "map_begin", "map_bytes", "MAP_SHARED", "PROT_READ_only", "whole_table_private_copy")
    for rank in range(8):
        if set(p[rank]) != set(d[rank]) or set(p[rank]) != set(p[0]):
            raise ValueError("P/D or ranks do not use the same Engram checkpoint table set")
        for key in p[rank]:
            if any(p[rank][key].get(field) != d[rank][key].get(field) for field in identity_fields):
                raise ValueError("P/D same-rank mappings do not share the same file and byte range")
    for key in p[0]:
        common_fields = ("device", "inode", "file_bytes", "mtime_ns", "header_sha256",
                         "dtype", "table_shape", "tensor_begin")
        if any(any(p[rank][key].get(field) != p[0][key].get(field) for field in common_fields)
               for rank in range(8)):
            raise ValueError("Ranks mapped different files for the same Engram table")
        ranges = sorted((p[rank][key]["row_start"], p[rank][key]["row_stop"]) for rank in range(8))
        stop = 0
        for start, end in ranges:
            if start != stop or end <= start:
                raise ValueError("TP8 Engram shard ranges overlap or have gaps")
            stop = end
        if stop != p[0][key]["table_shape"][0]:
            raise ValueError("TP8 mappings do not cover the complete Engram table")
    return {"PD_shared_file_and_range_contract_passed": True, "ranks_per_side": 8,
            "mapped_tensors_per_rank": len(p[0]), "actual_PSS_capacity_or_quality_passed": False}
