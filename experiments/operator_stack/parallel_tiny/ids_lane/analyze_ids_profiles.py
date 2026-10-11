"""Verify actual chip4 decode windows and classify integer Cast kernels."""
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path


def analyze(root, arm):
    files = list((root / "prof" / arm).rglob("kernel_details.csv"))
    assert len(files) == 1, (arm, files)
    path = files[0]
    with path.open() as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    assert rows and {r["Device_id"] for r in rows} == {"4"}
    hc = [r for r in rows if r["Type"].lower() == "hcpre"]
    assert len(hc) == 800, (arm, "expected ten decode steps x 80 native HcPre", len(hc))
    hash_router = [r for r in rows if "gating" in r["Type"].lower() and "hash" in r["Type"].lower()]
    assert len(hash_router) == 400, (arm, "expected ten decode steps x forty actual hash router consumers", len(hash_router))
    casts = [r for r in rows if r["Type"].lower() == "cast"]
    groups = defaultdict(lambda: {"count": 0, "duration_us": 0.})
    for row in casts:
        key = (row["Name"], row["Input Data Types"], row["Output Data Types"],
               row["Input Shapes"], row["Output Shapes"])
        groups[key]["count"] += 1
        groups[key]["duration_us"] += float(row["Duration(us)"])
    all_int64 = [r for r in casts if r["Input Data Types"] in ("INT32", "DT_INT32")
                 and r["Output Data Types"] in ("INT64", "DT_INT64")]
    scalar_int64 = [r for r in all_int64 if r["Input Shapes"].strip('" ') == "1"
                    and r["Output Shapes"].strip('" ') == "1"]
    output = path.parent
    trace = output / "trace_view.json"
    with trace.open() as handle:
        trace_data = json.load(handle)
    events = trace_data if isinstance(trace_data, list) else trace_data.get("traceEvents")
    assert isinstance(events, list) and events and all(isinstance(r, dict) for r in events)
    assert (output / "op_statistic.csv").exists() and (output / "api_statistic.csv").exists()
    return {"arm": arm, "passed": True, "physical_chip": 4, "actual_decode_steps": 10,
            "hc_pre_calls": len(hc), "hash_router_calls": len(hash_router), "kernel_rows": len(rows),
            "kernel_csv": str(path), "kernel_csv_bytes": path.stat().st_size,
            "kernel_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "fields": fields,
            "cast_calls": len(casts), "int32_to_int64_calls": len(all_int64),
            "scalar_int32_to_int64_calls": len(scalar_int64),
            "scalar_int32_to_int64_duration_us_per_step": sum(float(r["Duration(us)"]) for r in scalar_int64) / 10,
            "casts": [{"name": k[0], "input_dtype": k[1], "output_dtype": k[2],
                       "input_shape": k[3], "output_shape": k[4], **v,
                       "calls_per_step": v["count"] / 10, "duration_us_per_step": v["duration_us"] / 10}
                      for k, v in sorted(groups.items())],
            "profile_timing_is_not_performance_quote": True, "trace_json_valid": True}


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    off, on = [analyze(args.root, arm) for arm in ("off_perf", "on_perf")]
    removed = off["scalar_int32_to_int64_calls"] - on["scalar_int32_to_int64_calls"]
    # Other integer conversions remain in the native forward. This delta is
    # cross-checked with capture witnesses for forty material router casts vs
    # one model-level hoist, never inferred from the environment variable.
    assert removed == 390, ("expected 39 real Cast removals per step", removed)
    assert off["cast_calls"] - on["cast_calls"] == 390
    witnesses = json.loads((args.root / "consumer_witnesses.json").read_text())
    by_arm = {r["arm"]: r for r in witnesses}
    assert by_arm["off_repeat"]["router_material_casts"] == 40
    assert by_arm["on"]["router_material_casts"] == 0
    assert by_arm["on"]["model_hoist_material_casts"] == 1
    records = json.loads((args.root / "requests.json").read_text())
    direct_checks = []
    pairs = [("native_AA", f"native-{i}-A", "off", f"native-{i}-B", "off") for i in range(3)]
    pairs += [("postcapture_AA", f"capture-AA-{i}-A", "off", f"capture-AA-{i}-B", "off_repeat") for i in range(3)]
    pairs += [("off_on", f"candidate-{i}", "off_repeat", f"candidate-{i}", "on") for i in range(6)]
    for kind, tag_a, arm_a, tag_b, arm_b in pairs:
        a = next(r for r in records if r["tag"] == tag_a and r["arm"] == arm_a)
        b = next(r for r in records if r["tag"] == tag_b and r["arm"] == arm_b)
        # np.save headers include shape/dtype; compare full original route files
        # byte for byte on the remote host, not only their recorded SHA256.
        pa, pb = [args.root / Path(r["route_file"]).name for r in (a, b)]
        assert pa.read_bytes() == pb.read_bytes(), (kind, tag_a, "actual route files")
        assert a["token_ids"] == b["token_ids"]
        delta = 0.
        for left, right in zip(a["logprobs"], b["logprobs"]):
            assert left.keys() == right.keys()
            delta = max(delta, max(abs(left[k] - right[k]) for k in left))
        assert delta < 1e-3
        direct_checks.append({"kind": kind, "tag": tag_a, "routes_byte_exact": True,
                              "tokens_equal": True, "top5_keys_equal": True, "max_logprob_delta": delta})
    result = {"passed": True, "scope": "TP1 tiny FULL NPU decode graph; original integer router consumer",
              "off": off, "on": on, "real_cast_removed_per_step": removed / 10,
              "capture_material_id_casts_off": 40, "capture_material_id_casts_on": 1,
              "capture_all_40_consumers_equal": all(r["all_40_consumers_equal"] for r in witnesses),
              "direct_stored_route_comparisons": direct_checks,
              "formal_tp8_cast_profile_claim": False, "formal_17ms_claim": False}
    (args.root / "profile_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "off_casts": off["cast_calls"], "on_casts": on["cast_calls"],
                      "real_cast_removed_per_step": removed / 10, "scope": result["scope"]}), flush=True)


if __name__ == "__main__":
    main()
