"""Read-only CED startup gate for the actual A3 batch-invariant OPP.

The image tag alone is insufficient: the current best container installed this
vendor into its writable layer. These four hashes bind the version and ABI;
the full-tree copy manifest remains a separate deployment receipt.
"""
import hashlib
import json
from pathlib import Path


A3_BI_FILES = {
    "version.info": "cc3e7001ed18e748e400d78bbcd3a8c51c6d292bd7f252a3ed9c4268dc1a5371",
    "op_api/lib/libcust_opapi.so": "ff82d965a8a08bfa2862567022a04a1fd04df56d142f4393f1fb527592111069",
    "op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so": "f8308be503301621117cd421e411240a077dc4fcf04972321e403e7efc818010",
    "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so":
        "68221652dffb7fa13351616ecdf4e3445425697452fed29ae147f0cc335b02c1",
}


def inspect_opp(root, expected=None):
    expected = A3_BI_FILES if expected is None else expected
    if not expected:
        raise ValueError("A nonempty pinned A3 OPP contract is required")
    root = Path(root).resolve(strict=True)
    files = {}
    for relative, wanted in expected.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("BI OPP key must be a file within the selected vendor: " + relative)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != wanted:
            raise ValueError("Actual A3 BI OPP file SHA mismatch: " + relative)
        files[relative] = {"sha256": actual, "bytes": path.stat().st_size}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"schema": "delivery_ced_opp_contract_v1", "actual_root": str(root), "files": files,
            "artifact_fingerprint": hashlib.sha256(encoded).hexdigest(),
            "scope": "actual A3 BI version.info and three ABI libraries; not a full-tree checksum",
            "full_tree_verified_by_this_gate": False}
