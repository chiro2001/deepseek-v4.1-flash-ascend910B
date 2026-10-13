"""Import historical DCP modules before applying the frozen best-stack patches."""
import hashlib
import functools
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys


def patch_platform(module):
    original = module._validate_parallel_config

    @functools.wraps(original)
    def validate(config):
        result = original(config)
        parallel = config.parallel_config
        if parallel.decode_context_parallel_size > 1:
            from vllm_ascend.patch.platform.patch_v41_dcp import v41_dcp_active
            if v41_dcp_active():
                import math
                from vllm_ascend.core.deepseek_v41 import STATE_RING_ROWS
                interleave = math.gcd(int(config.cache_config.block_size), int(STATE_RING_ROWS))
                if interleave <= 0 or interleave % 2:
                    raise ValueError("V41 DCP interleave must preserve complete compressed token groups")
                parallel.cp_kv_cache_interleave_size = interleave
        return result

    module._validate_parallel_config = validate


class PlatformLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        method = getattr(self.original, "create_module", None)
        return method(spec) if method else None

    def exec_module(self, module):
        self.original.exec_module(module)
        patch_platform(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class DCPFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root, manifest):
        self.root, self.manifest = root, manifest

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm_ascend.platform":
            # Respect other source redirects/editable finders. Only omit this
            # finder during the nested lookup, then restore its exact position.
            index = sys.meta_path.index(self)
            sys.meta_path.pop(index)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(index, self)
            if spec is None or spec.loader is None:
                raise ImportError("Actual vendor platform module is unavailable")
            spec.loader = PlatformLoader(spec.loader)
            return spec
        relative = self.manifest["modules"].get(fullname)
        if relative is None:
            return None
        source = self.root / relative
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != self.manifest["files"][relative]["sha256"]:
            raise ImportError("Pinned DCP runtime source changed: " + relative)
        package_paths = [str(source.parent)] if source.name == "__init__.py" else None
        # Existing vendor-only modules remain available in the original package.
        if package_paths is not None:
            from importlib.machinery import PathFinder
            existing = PathFinder.find_spec(fullname, path)
            if existing and existing.submodule_search_locations:
                package_paths.extend(existing.submodule_search_locations)
        return importlib.util.spec_from_file_location(fullname, source, submodule_search_locations=package_paths)


def install(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != "delivery_dcp_overlay_v2"
            or manifest.get("best_model_unchanged") is not True
            or manifest.get("best_engram_lookup_algorithm_preserved") is not True):
        raise ValueError("A provenance-pinned minimal DCP overlay is required")
    for relative, row in manifest["files"].items():
        source = (root / relative).resolve(strict=True)
        if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("DCP overlay file/SHA mismatch: " + relative)
    if "vllm_ascend.platform" in sys.modules or any(name in sys.modules for name in manifest["modules"]):
        raise RuntimeError("DCP overlay must be installed before vendor module imports")
    sys.meta_path.insert(0, DCPFinder(root, manifest))
    return manifest


def install_best_draft(root):
    """Retarget the frozen graph finder to a narrowly patched best payload."""
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != "delivery_dcp_best_draft_patch_v1"
            or manifest.get("parallel_token_layout_preserved") is not True
            or manifest.get("best_graph_BI_async_patches_preserved") is not True):
        raise ValueError("A pinned best-source DSpark/DCP graph patch is required")
    if any(name in sys.modules for name in manifest["modules"]):
        raise RuntimeError("DSpark/DCP patches must precede proposer imports")
    import draft_graph_overlay
    for name, relative in manifest["modules"].items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][relative]["sha256"]:
            raise ValueError("DSpark/DCP graph patch file/hash mismatch")
        if name not in draft_graph_overlay.OVERLAY:
            raise ValueError("Only the original frozen draft-graph modules may be retargeted")
        draft_graph_overlay.OVERLAY[name] = path
    return manifest
