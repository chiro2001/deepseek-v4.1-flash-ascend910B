"""Assemble the frozen v12 overlay bundle accepted by prepare_full43_audit_snapshot_v5.

Members are exactly: the v12 package (freeze.json + SHA256SUMS + every frozen
file), the v9 SDK-provenance bridge, the v8 discovery pack and the prepack text
package. No runtime artifact, weight or __pycache__ enters the bundle.
"""
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

STACK = Path(__file__).resolve().parent
PKG = "parallel_tiny/dspark_formal_audit/full43_consumers_v14/"
BRIDGE = "parallel_tiny/dspark_formal_audit/formal_consumer_sdk_bridge_v9_cpu/"
V8 = "parallel_tiny/dspark_formal_audit/native_initialized_worker_discovery_v8/"
PREPACK = "parallel_tiny/dspark_prepack_audit_cpu/"


def die(message):
    raise SystemExit(message)


def collect():
    out = {}

    def add(prefix, rel, path, expected=None):
        raw = path.read_bytes()
        if len(raw) >= 1_000_000:
            die("member reaches the 1MB SSH bound: " + str(path))
        sha = hashlib.sha256(raw).hexdigest()
        if expected is not None and sha != expected:
            die("member SHA mismatch: " + str(path))
        out["stack/" + prefix + rel] = raw

    package = STACK / PKG
    frozen = json.loads((package / "freeze.json").read_text())
    for rel, sha in frozen["files"].items():
        add(PKG, rel, package / rel, sha)
    add(PKG, "freeze.json", package / "freeze.json")
    add(PKG, "SHA256SUMS", package / "SHA256SUMS")
    for prefix, manifest_name in ((BRIDGE, "freeze.json"), (V8, "freeze.json")):
        directory = STACK / prefix
        manifest = json.loads((directory / manifest_name).read_text())
        for rel, spec in manifest["files"].items():
            expected = spec["sha256"] if isinstance(spec, dict) else spec
            add(prefix, rel, directory / rel, expected)
        add(prefix, manifest_name, directory / manifest_name)
        add(prefix, "SHA256SUMS", directory / "SHA256SUMS")
    directory = STACK / PREPACK
    manifest = json.loads((directory / "frozen_files.json").read_text())
    for rel, spec in manifest["text_files"].items():
        expected = spec["sha256"] if isinstance(spec, dict) else spec
        add(PREPACK, rel, directory / rel, expected)
    add(PREPACK, "frozen_files.json", directory / "frozen_files.json")
    add(PREPACK, "delivery.json", directory / "delivery.json")
    return out


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        STACK / "parallel_tiny/dspark_formal_audit/full43_consumers_v14/evidence/full43_v14_overlay.tgz")
    if target.exists():
        die("Do not overwrite an existing bundle: " + str(target))
    data = collect()
    with tarfile.open(target, "w:gz") as archive:
        for name in sorted(data):
            info = tarfile.TarInfo(name)
            info.size = len(data[name])
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data[name]))
    size = target.stat().st_size
    if size >= 1_000_000:
        die("bundle reaches the 1MB SSH bound: %d" % size)
    print(json.dumps({"bundle": str(target), "members": len(data), "bytes": size,
                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
