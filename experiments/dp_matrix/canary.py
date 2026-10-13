"""One separately archived measured Dihuo canary per native DP replica."""
import argparse
import json
from pathlib import Path
import uuid

from .benchmark import http_json, load_frozen_dihuo_inputs, metrics_after, stream
from .measurement import request_metrics
from .topology import Topology, validate_identity


def run(args):
    topology = Topology.named(args.layout)
    inputs = load_frozen_dihuo_inputs(args.manifest, args.manifest_sha256, 2048, 256)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    identity = http_json(args.base_url, "/dp/identity")
    validate_identity(identity, topology, served_model=args.model,
                      sid=args.sid, fingerprint=args.fingerprint)
    (output / "identity_before.json").write_text(json.dumps(identity, indent=2) + "\n")
    results = []
    for rank in range(topology.dp):
        tag = "dp-canary-" + uuid.uuid4().hex
        arm = http_json(args.base_url, "/dp/arm", payload={
            "dp_rank": rank, "tag": tag, "max_tokens": 256})
        row = stream(args.base_url, args.model, inputs["samples"][rank]["messages"],
                     dp_rank=rank, tag=tag, max_tokens=256, seed=rank,
                     ignore_eos=True, timeout=args.timeout)
        receipt = {"excluded_from_formal_benchmark": True, "dp_rank": rank,
                   "arm": arm, "stream": row, "metrics": None,
                   "reduction": None, "error": row.get("error")}
        try:
            if row.get("error"):
                raise ValueError(row["error"])
            actual = metrics_after(args.base_url, rank, tag, args.timeout)
            receipt["metrics"] = actual
            receipt["reduction"] = request_metrics(
                arm, actual, row, topology, dp_rank=rank, tag=tag,
                sid=args.sid, fingerprint=args.fingerprint,
                prompt_tokens=2048, max_tokens=256, ignore_eos=True)
            if not row["clean"]["clean"]:
                raise ValueError("Canary output cleanliness failed")
        except Exception as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
        (output / f"dp{rank}.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        results.append(receipt)
        if receipt["error"] is not None:
            raise ValueError("Canary failed; preserve raw receipt and do not retry")
    after = http_json(args.base_url, "/dp/identity")
    validate_identity(after, topology, served_model=args.model,
                      sid=args.sid, fingerprint=args.fingerprint)
    (output / "identity_after.json").write_text(json.dumps(after, indent=2) + "\n")
    result = {"layout": args.layout, "service_instance_id": args.sid,
              "runtime_fingerprint": args.fingerprint, "canary_requests": len(results),
              "all_canaries_verified": True, "formal_quality_passed": False}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("base-url", "model", "layout", "sid", "fingerprint",
                 "manifest", "manifest-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
