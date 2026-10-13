"""Bounded derivation of the actual best async worker and counter.

All model, SDK, graph observer, request/round and fence logic is retained.
Only the old TP8 admission and TP8 terminal-group constants are generalized.
The terminal join remains an actual CPU Gloo join of the request's TP group;
it is not a global EP barrier and is not proof of idle DP completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BASE_HASHES = {
    "worker.py": "8dc2192a04f44e92081af8b85ae15df2cfcfbd837da38434612553dac2564d3d",
    "formal_dspark_counters.py": "4877900b492640e292d9daa9755d8699767deeae7eec17a8c2edcf85659b8e8c",
}
FROZEN = Path(__file__).resolve().parent / "frozen_sources"
WORKER_EDIT = (
    "        assert os.getenv('STACK_TP8_ARM')=='tp8base'\n",
    "        from dp_matrix.runtime_contract import validate_worker_config\n"
    "        validate_worker_config(self.vllm_config, os.environ)\n",
    1,
)
WORKER_PHYSICAL_EDIT = (
    "        assert os.environ['STACK_PHYSICAL_CHIPS']=='8,9,10,11,12,13,14,15'\n",
    "        # Exact physical placement is checked by validate_worker_config below.\n",
    1,
)
COUNTER_EDITS = (
    ("    if len(receipts) != 8 or {r['rank'] for r in receipts} != set(range(8)):\n",
     "    from dp_matrix.runtime_contract import expected_tp_size\n"
     "    tp_size = expected_tp_size()\n"
     "    if len(receipts) != tp_size or {r['rank'] for r in receipts} != set(range(tp_size)):\n", 1),
    ("Expected all eight TP ranks", "Expected all actual active-DP TP ranks", 1),
    ("join.get('world_size') != 8", "join.get('world_size') != tp_size", 2),
    ("CPU Gloo TP8 join", "CPU Gloo active-DP TP join (not a global EP barrier)", 1),
    ("    if (tp_group.world_size != 8 or tp_group.rank_in_group != counter.rank or\n",
     "    from dp_matrix.runtime_contract import expected_tp_size\n"
     "    tp_size = expected_tp_size()\n"
     "    if (tp_group.world_size != tp_size or tp_group.rank_in_group != counter.rank or\n", 1),
    ("actual eight-rank CPU Gloo TP group", "actual active-DP CPU Gloo TP group", 1),
    ("'backend': 'gloo', 'world_size': 8,", "'backend': 'gloo', 'world_size': tp_size,", 1),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def derive(name, data):
    if name not in BASE_HASHES or digest(data) != BASE_HASHES[name]:
        raise ValueError("Only the exact actual best async source can be derived")
    text = data.decode()
    edits = (WORKER_EDIT, WORKER_PHYSICAL_EDIT) if name == "worker.py" else COUNTER_EDITS
    for old, new, count in edits:
        if text.count(old) != count:
            raise ValueError("Unexpected derivation anchor count: " + old)
        text = text.replace(old, new)
    return text.encode()


def build(output):
    root = Path(output).resolve()
    payloads = {name: derive(name, (FROZEN / name).read_bytes()) for name in BASE_HASHES}
    root.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        (root / name).write_bytes(data)
    manifest = {
        "schema": "dp_best_async_runtime_derivation_v1",
        "base_sources": dict(BASE_HASHES),
        "files": {name: {"sha256": digest(data), "bytes": len(data)}
                  for name, data in payloads.items()},
        "terminal_join_scope": "actual active-DP TP group, not global EP",
        "model_SDK_graph_round_fence_logic_preserved": True,
        "actual_quality_passed": False, "service_started": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != "dp_best_async_runtime_derivation_v1"
            or manifest.get("base_sources") != BASE_HASHES
            or set(manifest.get("files", {})) != set(BASE_HASHES)
            or manifest.get("model_SDK_graph_round_fence_logic_preserved") is not True
            or manifest.get("terminal_join_scope") != "actual active-DP TP group, not global EP"):
        raise ValueError("Unreviewed DP runtime derivation")
    for name in BASE_HASHES:
        source = (root / name).resolve(strict=True)
        if not source.is_relative_to(root):
            raise ValueError("Runtime source escapes its pinned component")
        data = source.read_bytes()
        record = manifest["files"][name]
        if (record.get("sha256") != digest(data) or record.get("bytes") != len(data)
                or data != derive(name, (FROZEN / name).read_bytes())):
            raise ValueError("DP source differs from the bounded derivation: " + name)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output), indent=2))
