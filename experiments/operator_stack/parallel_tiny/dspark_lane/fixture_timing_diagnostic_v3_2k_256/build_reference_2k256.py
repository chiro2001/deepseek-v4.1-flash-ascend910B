"""Extract the 2K/256 reference token ids from the v3 geometry gate terminal evidence.

The reference is only ever copied from the local results/requests.json of the terminal
dspark_lane_20261011_c4_graph_geometry2k256_v3 job; this tool never fabricates token ids.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

BASE = Path(__file__).resolve().parent
REFERENCE_JOB = "dspark_lane_20261011_c4_graph_geometry2k256_v3"
REFERENCE_NAME = "reference_token_ids_2k256.json"
OUTPUT_TOKENS = 256
VOCAB_SIZE = 129280
REQUIRED_TAGS = ("native-off-{seed}-A", "native-off-{seed}-B", "candidate-{seed}-off",
                 "candidate-{seed}-on", "native-on-{seed}-A", "native-on-{seed}-B")
MINIMAL_TAGS = ("native-off-{seed}-A", "candidate-{seed}-off", "candidate-{seed}-on")


def extract_reference(records, require_all_tags=True):
    """Pure extraction: all records of a seed must agree bitwise; returns seed -> tags/token_ids."""
    assert isinstance(records, list) and len(records) >= 18
    by_seed = {}
    for row in records:
        assert isinstance(row, dict), row
        assert isinstance(row.get("tag"), str) and row["tag"]
        assert row.get("arm") in ("off", "on"), row.get("tag")
        assert isinstance(row.get("seed"), int) and row["seed"] in (0, 1, 2), row.get("tag")
        tokens = row.get("token_ids")
        assert isinstance(tokens, list) and len(tokens) == OUTPUT_TOKENS, row["tag"]
        assert all(isinstance(token, int) and 0 <= token < VOCAB_SIZE for token in tokens), row["tag"]
        by_seed.setdefault(row["seed"], []).append(row)
    assert set(by_seed) == {0, 1, 2}, sorted(by_seed)
    seeds = {}
    for seed in (0, 1, 2):
        rows = by_seed[seed]
        tags = [row["tag"] for row in rows]
        assert len(set(tags)) == len(tags), (seed, tags)
        required = [tag.format(seed=seed) for tag in (REQUIRED_TAGS if require_all_tags else MINIMAL_TAGS)]
        missing = [tag for tag in required if tag not in tags]
        assert not missing, (seed, missing, sorted(tags))
        reference = rows[0]["token_ids"]
        assert all(row["token_ids"] == reference for row in rows), ("cross-record token mismatch", seed)
        seeds[str(seed)] = {"tags": sorted(tags), "token_ids": list(reference)}
    assert len({tuple(seeds[str(seed)]["token_ids"]) for seed in (0, 1, 2)}) == 3, "Seed references must not collide"
    return seeds


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--job", default=REFERENCE_JOB)
    parser.add_argument("--source", required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--allow-tag-subset", action="store_true")
    args = parser.parse_args()
    assert args.job == REFERENCE_JOB, args.job
    source = Path(args.source).resolve()
    assert source.is_file(), source
    assert REFERENCE_JOB in str(source), "Source path must contain the v3 job name for provenance"
    raw = source.read_bytes()
    records = json.loads(raw)
    seeds = extract_reference(records, require_all_tags=not args.allow_tag_subset)
    payload = {"pending": False, "source_job": REFERENCE_JOB, "source_requests_path": str(source),
               "source_requests_sha256": hashlib.sha256(raw).hexdigest(), "source_records": len(records),
               "extractor": "build_reference_2k256.py", "extracted_unix": time.time(),
               "tag_subset_allowed": bool(args.allow_tag_subset), "fabricated_reference_bytes": False,
               "seeds": seeds}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    target = BASE / REFERENCE_NAME
    if args.write:
        assert not target.exists() or target.read_text() == rendered, "Never overwrite a changed reference"
        target.write_text(rendered)
    print(json.dumps({"job": args.job, "source": str(source), "source_requests_sha256": payload["source_requests_sha256"],
                      "records": len(records), "target": str(target), "written": bool(args.write),
                      "tag_subset_allowed": bool(args.allow_tag_subset),
                      "seed_tags": {seed: seeds[seed]["tags"] for seed in seeds},
                      "reference_sha256": hashlib.sha256(rendered.encode()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
