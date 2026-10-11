"""Stdlib CPU parse of the frozen eager-draft baseline; raw trace stays remote."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics
import sys


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def union(intervals):
    total = 0.0
    begin = end = None
    for left, right in sorted(intervals):
        if begin is None:
            begin, end = left, right
        elif left <= end:
            end = max(end, right)
        else:
            total += end - begin
            begin, end = left, right
    return total + (0.0 if begin is None else end - begin)


def normalize(row, base, time_key, duration_key):
    start = float(Decimal(row[time_key].strip()) - base)
    duration = float(row[duration_key])
    return row | {"start_us": start, "duration_us": duration, "end_us": start + duration}


def summarize(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    assert "torch" not in sys.modules, "Offline parser must not import/initialize NPU"
    folder = args.profile
    files = {name: folder / name for name in (
        "kernel_details.csv", "task_time.csv", "api_statistic.csv", "step_trace_time.csv")}
    originals = list(csv.DictReader(files["kernel_details.csv"].open()))
    assert {r["Device_id"] for r in originals} == {"4"}
    base = min(Decimal(r["Start Time(us)"].strip()) for r in originals)
    kernels = [normalize(r, base, "Start Time(us)", "Duration(us)") for r in originals]
    task_originals = list(csv.DictReader(files["task_time.csv"].open()))
    assert {r["Device_id"] for r in task_originals} == {"4"}
    tasks = [normalize(r, base, "task_start(us)", "task_time(us)") for r in task_originals]
    grouped = defaultdict(list)
    for row in kernels:
        grouped[int(row["Step Id"])].append(row)
    assert len(grouped) == 10 and sorted(grouped) == list(range(5, 15))
    steps, event_examples = [], []
    for step, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r["start_us"])
        hc = [r for r in rows if r["Type"] == "HcPre"]
        target = [r for r in hc if r["Input Shapes"].strip('"').startswith("6,4,5120;")]
        draft = [r for r in hc if r["Input Shapes"].strip('"').startswith("5,4,5120;")]
        assert len(target) == 80 and len(draft) == 6 and len(hc) == 86
        draft_attention = [r for r in rows if r["Type"] == "SparseAttnSharedkv"
                           and r["OP State"] == "dynamic" and r["Stream ID"] == "47"]
        assert len(draft_attention) == 3, "Every actual draft layer must execute its cache reader"
        for index, attention in enumerate(draft_attention):
            assert draft[2 * index]["start_us"] <= attention["start_us"] < draft[2 * index + 1]["start_us"]
        assert {r["OP State"] for r in target} == {"static"}
        assert {r["OP State"] for r in draft} == {"dynamic"}
        assert {r["Stream ID"] for r in target} == {"13"}
        assert {r["Stream ID"] for r in draft} == {"47"}
        models = {r["Model ID"] for r in target}
        assert len(models) == 1
        model = next(iter(models))
        graph = [r for r in rows if r["Model ID"] == model]
        assert all(r["OP State"] == "static" for r in graph)
        main_proj = [r for r in rows if r["Type"] == "MatMulV2"
                     and r["Input Shapes"].strip('"') == "6,15360;5120,15360"]
        assert len(main_proj) == 1
        projection = main_proj[0]
        start, end = min(r["start_us"] for r in rows), max(r["end_us"] for r in rows)
        graph_start, graph_end = min(r["start_us"] for r in graph), max(r["end_us"] for r in graph)
        draft_start = min(r["start_us"] for r in draft)
        draft_scatter = [r for r in rows if r["Type"] == "ScatterNdUpdateSk"
                         and r["OP State"] == "dynamic" and r["Stream ID"] == "47"]
        context_scatter = [r for r in draft_scatter if projection["start_us"] <= r["start_us"] < draft_start]
        query_scatter = [r for r in draft_scatter if r["start_us"] >= draft_start]
        assert len(context_scatter) == 3 and len(query_scatter) == 3
        assert start <= graph_start < graph_end <= projection["start_us"] < draft_start <= end
        windows = {"pre_target": (start, graph_start), "target_graph": (graph_start, graph_end),
                   "target_output_verify_draft_prepare": (graph_end, projection["start_us"]),
                   "draft_context_projection_cache": (projection["start_us"], draft_start),
                   "draft_body_logits": (draft_start, end)}
        phases = {}
        for name, (left, right) in windows.items():
            entries = [r for r in rows if left <= r["start_us"] < right]
            interval_tasks = [r for r in tasks if r["start_us"] < right and r["end_us"] > left]
            events = [r for r in interval_tasks if r["kernel_type"].startswith("EVENT_")]
            waits = [r for r in events if r["kernel_type"] == "EVENT_WAIT"]
            phases[name] = {"begin_relative_step_us": left - start, "end_relative_step_us": right - start,
                           "span_us": right - left, "kernel_count": len(entries),
                           "kernel_duration_sum_us": sum(r["duration_us"] for r in entries),
                           "kernel_interval_union_us": union([(max(left, r["start_us"]), min(right, r["end_us"]))
                                                              for r in entries]),
                           "streams": dict(Counter(r["Stream ID"] for r in entries)),
                           "event_type_counts": dict(Counter(r["kernel_type"] for r in events)),
                           "wait_interval_union_us": union([(max(left, r["start_us"]), min(right, r["end_us"]))
                                                            for r in waits]),
                           "wait_duration_sum_us": sum(r["duration_us"] for r in waits)}
        steps.append({"step_id": step, "kernel_count": len(rows), "physical_device": 4,
                      "target_hc_pre": 80, "draft_hc_pre": 6, "target_model_id": model,
                      "actual_draft_attention_cache_readers": 3,
                      "actual_draft_context_cache_writes": 3, "actual_draft_query_cache_writes": 3,
                      "draft_attention_input_shapes": [r["Input Shapes"] for r in draft_attention],
                      "context_cache_scatter_input_shapes": [r["Input Shapes"] for r in context_scatter],
                      "query_cache_scatter_input_shapes": [r["Input Shapes"] for r in query_scatter],
                      "kernel_step_span_us": end - start, "phases": phases,
                      "main_projection_us": projection["duration_us"],
                      "draft_layer_hc_relative_step_us": [r["start_us"] - start for r in draft]})
        if step == 9:
            selected = [r for r in tasks if r["kernel_type"].startswith("EVENT_")
                        and graph_start - 30 <= r["start_us"] <= graph_start + 50]
            selected += [r for r in tasks if r["kernel_type"].startswith("EVENT_")
                         and graph_end - 30 <= r["start_us"] <= projection["start_us"] + 50]
            event_examples = [{"type": r["kernel_type"], "stream": r["stream_id"],
                               "task_id": r["task_id"], "relative_step_us": r["start_us"] - start,
                               "duration_us": r["duration_us"]} for r in selected[:100]]
    types = defaultdict(lambda: {"count": 0, "duration_sum_us": 0.0})
    for row in kernels:
        stats = types[(row["Type"], row["OP State"], row["Stream ID"])]
        stats["count"] += 1
        stats["duration_sum_us"] += row["duration_us"]
    apis = list(csv.DictReader(files["api_statistic.csv"].open()))
    replay = [r for r in apis if r["Level"] == "acl" and r["API Name"] == "aclmdlRIExecuteAsync"]
    assert len(replay) == 1 and int(replay[0]["Count"]) == 10
    launch = [r for r in apis if r["Level"] == "acl" and r["API Name"] == "aclrtLaunchKernelWithHostArgs"]
    assert len(launch) == 1
    metrics = {phase: {metric: summarize([s["phases"][phase][metric] for s in steps])
                       for metric in ("span_us", "kernel_count", "kernel_duration_sum_us", "kernel_interval_union_us",
                                      "wait_interval_union_us", "wait_duration_sum_us")}
               for phase in steps[0]["phases"]}
    receipt = {"profile": str(folder), "offline_cpu_no_torch_import": True,
               "parser_sha256": args.source_sha256, "target_full_graph": True, "draft_enforce_eager": True,
               "draft_graph": False, "private_legacy_helper_scope": True, "formal_tp8_claim": False,
               "actual_device_steps": 10, "physical_device": 4,
               "actual_target_layers_per_step": 40, "actual_draft_layers_per_step": 3,
               "target_hc_pre_per_step": 80, "draft_hc_pre_per_step": 6,
               "actual_draft_attention_cache_readers_per_step": 3,
               "actual_draft_context_cache_writes_per_step": 3, "actual_draft_query_cache_writes_per_step": 3,
               "cache_value_precision_audit_passed": False,
               "actual_replay_api_calls_per_step": 1, "actual_replay_scope": "target only",
               "launch_api_calls_per_step": int(launch[0]["Count"]) / 10,
               "launch_api_duration_sum_us_per_step": float(launch[0]["Time(us)"]) / 10,
               "api_time_is_not_recoverable_e2e_budget": True,
               "profile_intervals_not_unobserved_performance": True,
               "task_time_has_event_id": "event_id" in task_originals[0],
               "event_rows_do_not_identify_python_event_objects": True,
               "timestamp_math": "Decimal absolute timestamps; relative float intervals",
               "steps": steps, "phase_statistics": metrics, "event_example_step9": event_examples,
               "operator_by_type_state_stream": [{"type": key[0], "state": key[1], "stream": key[2],
                                                   "count_per_step": val["count"] / 10,
                                                   "duration_sum_us_per_step": val["duration_sum_us"] / 10}
                                                  for key, val in sorted(types.items(), key=lambda x: -x[1]["duration_sum_us"])],
               "source_sha256": {name: {"sha256": sha(path), "bytes": path.stat().st_size}
                                 for name, path in files.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in
                      ("steps", "operator_by_type_state_stream", "event_example_step9", "phase_statistics")}, indent=2))


if __name__ == "__main__":
    main()
