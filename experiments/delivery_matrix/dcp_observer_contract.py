"""Admit exactly the derived best proposer to the unchanged graph observer.

This changes the expected source identity for a NEW DCP candidate, not the
observer's execution checks or any existing frozen snapshot/receipt.
"""
import hashlib
import json
from pathlib import Path

BEST_SOURCES = {
    "llm_base_proposer.py": "36d7c8870538e10e310863f9b628b90b946b534861563542cf7fba4cd88d7c39",
    "dspark_proposer.py": "56943080f99da2f2dee516aa3e76605c5150408df9a26691cc6529f327ae29b4",
}
STOCK_BASE_SHA256 = "d5003b772a1982f032f59a65d9ad4970d598311898c487a50791ee2337bdaae4"


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != "delivery_dcp_best_draft_patch_v1"
            or manifest.get("best_graph_BI_async_patches_preserved") is not True
            or manifest.get("parallel_token_layout_preserved") is not True):
        raise ValueError("An explicit derived best-source DCP candidate is required")
    for name, original_sha in BEST_SOURCES.items():
        record = manifest["files"].get(name, {})
        source = (root / name).resolve(strict=True)
        if record.get("source_sha256") != original_sha or not source.is_relative_to(root):
            raise ValueError("DCP proposer is not derived from the verified best snapshot")
        if hashlib.sha256(source.read_bytes()).hexdigest() != record.get("sha256"):
            raise ValueError("DCP proposer payload differs from its derivation receipt")
    return {"original_best_sources": dict(BEST_SOURCES),
            "derived_sources": {name: manifest["files"][name]["sha256"] for name in BEST_SOURCES},
            "derivation_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
            "original_stock_claim": False, "graph_execution_checks_unchanged": True}


def install(root, observer):
    receipt = verify(root)
    original = observer.STOCK_PROPOSER_SHA256
    if (original != STOCK_BASE_SHA256
            or getattr(observer, "OVERLAY_PROPOSER_SHA256", None) != BEST_SOURCES["llm_base_proposer.py"]):
        raise ValueError("The frozen graph observer is not the original best-source observer")
    observer.STOCK_PROPOSER_SHA256 = receipt["derived_sources"]["llm_base_proposer.py"]
    return receipt
