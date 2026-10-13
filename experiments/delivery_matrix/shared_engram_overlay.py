"""CED-only opt-in mmap loader; retain the real editable SDK import chain."""
import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys

from .shared_engram_contract import BASE_SHA, FILES, MANIFEST_SHA, MODULE, sha, verify_component


class SharedEngramFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.resolving = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in FILES or self.resolving:
            return None
        if fullname == MODULE:
            self.resolving = True
            try:
                native = None
                # PEP660 editable and other supported SDK finders may supply
                # the original. Never bypass them by guessing site-packages.
                for finder in list(sys.meta_path):
                    if finder is self:
                        continue
                    find = getattr(finder, "find_spec", None)
                    if find is None:
                        continue
                    native = find(fullname, path, target)
                    if native is not None:
                        break
                if native is None or not native.origin or sha(native.origin) != BASE_SHA:
                    raise ImportError("Actual Engram source differs from the exact best derivation base")
            finally:
                self.resolving = False
        verify_component(self.root)
        return importlib.util.spec_from_file_location(fullname, self.root / FILES[fullname])


def install(root, role):
    if (role not in ("prefill", "decode") or os.environ.get("V41_CED_ROLE") != role
            or os.environ.get("V41_ENGRAM_DEVICE_INDEX") != "0"
            or os.environ.get("V41_ENGRAM_HOST_RESIDENT") != "1"
            or os.environ.get("V41_ENGRAM_SHARED_HOST_MMAP") != "1"):
        raise ValueError("Shared Engram is a host-INT8 CED P/D option with device index disabled")
    if (any(name in sys.modules for name in FILES)
            or any(isinstance(item, SharedEngramFinder) for item in sys.meta_path)):
        raise RuntimeError("Shared Engram must be selected once, before its SDK import")
    manifest = verify_component(root)
    if os.environ.get("DELIVERY_SHARED_ENGRAM_MANIFEST_SHA256") != MANIFEST_SHA:
        raise ValueError("Bootstrap manifest selection is not the pinned mmap component")
    sys.meta_path.insert(0, SharedEngramFinder(root))
    return manifest
