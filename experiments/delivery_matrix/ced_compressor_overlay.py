"""D-only import redirect; verify actual SDK before selecting derived source."""
import importlib.abc
from importlib.machinery import PathFinder
import importlib.util
from pathlib import Path
import sys

from .build_ced_compressor import BASE_SHA, MODULE, RELATIVE, sha, verify


class CompressorFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        if fullname != MODULE:
            return None
        actual = PathFinder.find_spec(fullname, path)
        if actual is None or not actual.origin or sha(actual.origin) != BASE_SHA:
            raise ImportError("Actual compressor SDK source differs from the pinned derivation base")
        verify(self.root)
        return importlib.util.spec_from_file_location(fullname, self.root / RELATIVE)


def install(root, role):
    if role != "decode":
        raise ValueError("Compressor residual candidate is D-only; do not alter P")
    root = Path(root).resolve(strict=True)
    manifest = verify(root)
    if MODULE in sys.modules or any(isinstance(finder, CompressorFinder) for finder in sys.meta_path):
        raise RuntimeError("Compressor candidate must be selected once, before SDK imports")
    sys.meta_path.insert(0, CompressorFinder(root))
    return manifest
