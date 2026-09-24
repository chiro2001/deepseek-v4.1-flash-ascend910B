#!/usr/bin/env python3
"""Summarize eager/graph CANN traces by the recorded HTTP request windows.

Raw traces are downloaded from private COS archives and extracted under
prof_{eager,graph}/extracted/prof. The request IDs are labels for the time
windows; CANN trace events themselves do not carry vLLM request IDs.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
PROFILES = {
    "eager": ROOT / "prof_eager",
    "graph": ROOT / "prof_graph",
}
RELEVANT_API = {
    "aclrtRecordEvent": "EVENT_RECORD",
    "aclrtStreamWaitEvent": "EVENT_WAIT",
    "aclrtMemcpyAsync": "MEMCPY_ASYNC",
}
PATTERNS = {
    "smla": re.compile(r"sparseflashmla|smla", re.I),
    "indexer": re.compile(r"indexer", re.I),
    "copy": re.compile(r"memcpy|copy", re.I),
}


def find_one(root: Path, name: str) -> Path:
    paths = list(root.glob(f"extracted/prof/**/{name}"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {name} below {root}, got {len(paths)}")
    return paths[0]


def window_us(item: dict) -> tuple[float, float]:
    start = dt.datetime.fromisoformat(item["start_iso"]).timestamp() * 1_000_000
    return start, start + float(item["wall_s"]) * 1_000_000


def load_events(trace: Path) -> list[dict]:
    if trace.suffix == ".gz":
        with gzip.open(trace, "rt", encoding="utf-8") as f:
            return json.load(f)
    with trace.open(encoding="utf-8") as f:
        return json.load(f)


def event_us(e: dict) -> float | None:
    try:
        return float(e["ts"])
    except (KeyError, TypeError, ValueError):
        return None


def in_window(e: dict, lo: float, hi: float) -> bool:
    t = event_us(e)
    return t is not None and lo <= t < hi and e.get("ph") == "X"


def api_id(e: dict) -> str | None:
    value = e.get("args", {}).get("connection_id")
    return None if value is None else str(value)


def summarize(profile: str, base: Path) -> dict:
    windows = json.loads((base / "profile_windows.json").read_text())
    trace_path = find_one(base, "trace_view.json")
    kernel_path = find_one(base, "kernel_details.csv")
    events = load_events(trace_path)

    parsed_kernels = []
    with kernel_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                row["_ts"] = float(row["Start Time(us)"].strip())
                row["_dur"] = float(row["Duration(us)"].strip())
            except (KeyError, ValueError):
                continue
            parsed_kernels.append(row)

    result = {"trace_events_total": len(events), "windows": []}
    for w in windows:
        lo, hi = window_us(w)
        selected = [e for e in events if in_window(e, lo, hi)]
        name_counts = Counter(e.get("name", "") for e in selected)
        device_connection_reuse = {}
        for name in ("EVENT_RECORD", "EVENT_WAIT", "MEMCPY_ASYNC", "EVENT_RESET"):
            ids = Counter(
                api_id(e) for e in selected
                if e.get("name") == name and api_id(e) is not None
            )
            device_connection_reuse[name] = {
                "events": sum(ids.values()),
                "unique_connection_ids": len(ids),
                "ids_used_more_than_once": sum(1 for count in ids.values() if count > 1),
                "top_reused_ids": [
                    {"connection_id": cid, "events": count}
                    for cid, count in ids.most_common(5) if count > 1
                ],
            }
        kernels = [r for r in parsed_kernels if lo <= r["_ts"] < hi]
        stream_counts = Counter(r.get("Stream ID", "?") for r in kernels)
        kernel_patterns = {}
        for label, rx in PATTERNS.items():
            matched = [r for r in kernels if rx.search(r.get("Name", "")) or rx.search(r.get("Type", ""))]
            kernel_patterns[label] = {
                "count": len(matched),
                "by_stream": dict(sorted(Counter(r.get("Stream ID", "?") for r in matched).items())),
                "names": dict(Counter(r.get("Name", "") for r in matched).most_common(12)),
            }

        pair_summary = {}
        for api, task in RELEVANT_API.items():
            selected_apis = [
                e for e in selected if e.get("name") == f"AscendCL@{api}"
            ]
            device_events = [
                e for e in selected
                if e.get("args", {}).get("Task Type") == task or e.get("name") == task
            ]
            api_ids = Counter(api_id(e) for e in selected_apis if api_id(e) is not None)
            device_ids = Counter(api_id(e) for e in device_events if api_id(e) is not None)
            common_ids = set(api_ids) & set(device_ids)
            # Only summarize a lag for IDs with exactly one API and exactly one
            # device task in this same request window. Graph replay may reuse
            # one connection_id for many device executions.
            one_to_one_ids = {
                cid for cid in common_ids if api_ids[cid] == 1 and device_ids[cid] == 1
            }
            by_id_api = {api_id(e): e for e in selected_apis if api_id(e) in one_to_one_ids}
            by_id_dev = {api_id(e): e for e in device_events if api_id(e) in one_to_one_ids}
            lags = [event_us(by_id_dev[cid]) - event_us(by_id_api[cid]) for cid in one_to_one_ids]
            pair_summary[api] = {
                "api_calls_in_window": len(selected_apis),
                "device_task_events_in_window": len(device_events),
                "unique_api_connection_ids": len(api_ids),
                "unique_device_connection_ids": len(device_ids),
                "connection_ids_common_within_window": len(common_ids),
                "one_to_one_connection_ids_within_window": len(one_to_one_ids),
                "device_minus_api_us_median_for_one_to_one_ids": median(lags) if lags else None,
                "device_minus_api_us_min": min(lags) if lags else None,
                "device_minus_api_us_max": max(lags) if lags else None,
            }

        event_times = [(event_us(e), e) for e in selected if event_us(e) is not None]
        event_times.sort(key=lambda p: p[0])
        smla_rows = [
            r for r in kernels
            if "SparseFlashMla" in r.get("Name", "")
            and "Metadata" not in r.get("Name", "")
        ]
        last_smla_row = max(smla_rows, key=lambda r: r["_ts"], default=None)
        last_smla = last_smla_row["_ts"] if last_smla_row else None
        tail_names = Counter(
            e.get("name", "") for t, e in event_times
            if last_smla is not None and t >= last_smla and e.get("cat") not in ("cpu_op",)
        )
        result["windows"].append({
            "label": w["label"],
            "request_id": w["request_id"],
            "start_iso": w["start_iso"],
            "wall_s": w["wall_s"],
            "start_us": lo,
            "end_us": hi,
            "trace_events_in_window": len(selected),
            "trace_event_names_top30": dict(name_counts.most_common(30)),
            "trace_selected_counts": {
                name: sum(1 for e in selected if e.get("name") == name)
                for name in (
                    "AscendCL@aclrtRecordEvent", "AscendCL@aclrtStreamWaitEvent",
                    "AscendCL@aclrtMemcpyAsync", "EVENT_RECORD", "EVENT_WAIT",
                    "MEMCPY_ASYNC", "Event::synchronize",
                )
            },
            "device_connection_id_reuse": device_connection_reuse,
            "kernel_rows_in_window": len(kernels),
            "kernel_rows_by_stream": dict(sorted(stream_counts.items())),
            "kernel_patterns": kernel_patterns,
            "smla_attention_kernel_count": sum(
                1 for r in smla_rows
            ),
            "smla_metadata_kernel_count": sum(
                1 for r in kernels if "SparseFlashMlaMetadata" in r.get("Name", "")
            ),
            "host_api_to_device_task_by_connection_id": pair_summary,
            "last_smla_start_us": last_smla,
            "last_smla_stream": last_smla_row.get("Stream ID") if last_smla_row else None,
            "last_smla_task_id": last_smla_row.get("Task ID") if last_smla_row else None,
            "http_end_minus_last_smla_end_us": (
                hi - (last_smla + last_smla_row["_dur"])
                if last_smla_row else None
            ),
            "trace_event_names_at_or_after_last_smla": dict(tail_names.most_common(30)),
        })
    return result


def main() -> None:
    output = {
        "scope": "HTTP wall-time windows; CANN events have no vLLM request ID",
        "matching_limits": [
            "aclrt API and device tasks are paired by connection_id",
            "trace does not expose the event handle linking a producer EVENT_RECORD to a consumer EVENT_WAIT",
        ],
        "profiles": {name: summarize(name, root) for name, root in PROFILES.items()},
    }
    out_path = ROOT / "profile_window_summary.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(out_path)
    for profile, data in output["profiles"].items():
        print(f"[{profile}]")
        for w in data["windows"]:
            print(
                w["label"], w["request_id"],
                "trace", w["trace_events_in_window"],
                "SMLA", w["kernel_patterns"]["smla"]["count"],
                "streams", w["kernel_patterns"]["smla"]["by_stream"],
                "waits", w["trace_selected_counts"]["AscendCL@aclrtStreamWaitEvent"],
                "records", w["trace_selected_counts"]["AscendCL@aclrtRecordEvent"],
                "copies", w["trace_selected_counts"]["AscendCL@aclrtMemcpyAsync"],
            )


if __name__ == "__main__":
    main()
