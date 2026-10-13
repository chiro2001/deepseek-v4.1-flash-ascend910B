"""Load a separate derived class; never patch the original worker/module/env."""
import importlib
import importlib.util
from pathlib import Path
import os
import sys

from .build_physical_worker import BASE_CLASS, BASE_RELATIVE, BASE_SHA, CHIPS, sha, verify


def load_base(root, source_root, class_name):
    root, source_root = Path(root).resolve(strict=True), Path(source_root).resolve(strict=True)
    manifest = verify(root)
    if class_name != BASE_CLASS:
        raise ValueError("Only the exact original best worker may use this derivation")
    original = (source_root / BASE_RELATIVE).resolve(strict=True)
    if not original.is_relative_to(source_root) or sha(original) != BASE_SHA:
        raise ValueError("Actual immutable best worker differs from the derivation base")
    for key in ("STACK_PHYSICAL_CHIPS", "STACK_EXPECTED_CHIPS", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
        if os.environ.get(key) != CHIPS:
            raise ValueError("Requested physical chips and every runtime visibility guard must agree")
    # A sibling module keeps .sdk_gate pointing at the ORIGINAL package. Its
    # full40, async, graph and counter code executes unchanged; no env spoofing.
    parent = BASE_CLASS.rsplit(".", 2)[0]
    importlib.import_module(parent)
    name = parent + "._delivery_physical_0_7"
    if name in sys.modules:
        loaded = sys.modules[name]
        if Path(loaded.__file__).resolve() != root / "worker.py" or sha(loaded.__file__) != manifest["files"]["worker.py"]["sha256"]:
            raise RuntimeError("A different physical worker is already loaded")
    else:
        spec = importlib.util.spec_from_file_location(name, root / "worker.py")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        try:
            spec.loader.exec_module(loaded)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return loaded.FormalDSparkWorker, {"manifest_sha256": sha(root / "manifest.json"),
        "source_sha256": BASE_SHA, "derived_source_sha256": sha(loaded.__file__), "physical_chips": CHIPS,
        "original_worker_unchanged": True, "original_private_API_contract_not_reused": True,
        "actual_quality_passed": False}
