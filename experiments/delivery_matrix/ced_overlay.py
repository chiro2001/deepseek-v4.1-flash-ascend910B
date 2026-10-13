"""CED-only imports. A CED-PD deployment is distinct from the DCP8 form."""
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys


class CEDFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root, manifest, role):
        self.root, self.manifest, self.role = root, manifest, role

    def find_spec(self, fullname, path=None, target=None):
        relative = self.manifest["modules"].get(fullname)
        if relative is None:
            return None
        # P needs producer contracts and the admission gate, but no D replay
        # attention or D-specific prompt-tail graph behavior.
        if self.role == "prefill" and fullname in (
            "vllm_ascend.attention.dsa_v41", "vllm_ascend.worker.model_runner_v1"
        ):
            return None
        source = self.root / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != self.manifest["files"][relative]["sha256"]:
            raise ImportError("Pinned CED source changed: " + relative)
        return importlib.util.spec_from_file_location(fullname, source)


def install(root, role):
    if role not in ("prefill", "decode"):
        raise ValueError("An explicit CED P or D role is required")
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != "delivery_ced_overlay_v1" or manifest.get("DCP") != 1
            or manifest.get("best_model_and_engram_unchanged") is not True):
        raise ValueError("A separate pinned zero-DCP CED overlay is required")
    for relative, row in manifest["files"].items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("CED source/file hash mismatch: " + relative)
    if any(name in sys.modules for name in manifest["modules"]):
        raise RuntimeError("CED overlay must be installed before vendor module imports")
    sys.meta_path.insert(0, CEDFinder(root, manifest, role))
    return manifest
