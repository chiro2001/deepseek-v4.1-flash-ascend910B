"""Fail-closed bootstrap for the native DP4/TP2/DCP2 candidate."""
import hashlib
import os
from pathlib import Path
import runpy
import sys


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


try:
    if os.environ.get("DP_MATRIX_LAYOUT") != "dp4-tp2-dcp2":
        raise RuntimeError("DCP2 bootstrap requires the exact native DP layout")
    if (os.environ.get("V41_DCP") != "1"
            or os.environ.get("V41_DCP_REPLICATE_INDEXER") != "1"
            or os.environ.get("V41_ENGRAM_REUSE_EP_GROUP") != "1"
            or os.environ.get("V41_ENGRAM_DEVICE_INDEX") != "0"):
        raise RuntimeError("DCP2 and Engram settings differ from the frozen plan")
    overlay = Path(os.environ["DELIVERY_DCP_OVERLAY"]).resolve(strict=True)
    draft = Path(os.environ["DELIVERY_DCP_DRAFT"]).resolve(strict=True)
    if _digest(overlay / "manifest.json") != os.environ["DP_MATRIX_DCP2_OVERLAY_MANIFEST_SHA256"]:
        raise RuntimeError("DCP2 overlay manifest SHA differs from the frozen service plan")
    if _digest(draft / "manifest.json") != os.environ["DP_MATRIX_DCP2_DRAFT_MANIFEST_SHA256"]:
        raise RuntimeError("DCP2 draft manifest SHA differs from the frozen service plan")

    from dp_matrix.build_dcp2 import verify, verify_draft
    from dp_matrix.dcp2_contract import DCP2_CONTRACT_SHA256

    if DCP2_CONTRACT_SHA256 != os.environ["DP_MATRIX_DCP2_CONTRACT_SHA256"]:
        raise RuntimeError("DCP2 Engram admission source SHA differs from the frozen service plan")
    verify(overlay)
    verify_draft(draft)

    # This must run before the vendor platform and DSpark proposer modules load.
    from delivery_matrix.dcp_overlay import install, install_best_draft

    install(overlay)
    install_best_draft(draft)
    source_sitecustomize = Path(os.environ["STACK_SRC_ROOT"]) / "stack/sitecustomize.py"
    runpy.run_path(str(source_sitecustomize), run_name="_dp_matrix_dcp2_frozen_sitecustomize")
except BaseException:
    import traceback

    traceback.print_exc()
    # Python swallows sitecustomize exceptions; never continue into model load.
    os._exit(2)
