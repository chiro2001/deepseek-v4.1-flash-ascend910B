#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p36_capacity15.py -- 15x1Mi serial admission + checkpoint(3) + hold + metrics.

Protocol
--------
1. build a 1,024,000-token prompt via /tokenize, submit streaming /v1/completions
   (temperature=0, ignore_eos, max_tokens=24576 => sequence == 1,048,576 == max-model-len);
2. admissions are serial: the next request is submitted only after the previous one
   produced its first content token (prefill done);
3. after the CHECKPOINT-th request is accepted, hold briefly and evaluate
   (running==checkpoint, waiting==0, kv_usage<100%, preemptions==0, health ok);
   on failure stop and report;
4. after target N accepted, hold --hold-seconds with periodic /metrics sampling;
5. abort client connections, wait drain, parse serve log for startup accounting.

Per-request chunk arrival times give the decode step-latency distribution while all
N requests are online (SSE chunk == one decode step incl. speculative tokens).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

MI = 1048576
FILLER = "甲乙丙丁"
_METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE.+-]+)\s*$")
_LE_RE = re.compile(r'le="([^"]+)"')


def _open(base, path, payload=None, timeout=30.0):
    url = base.rstrip("/") + path
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def http_json(base, path, payload, timeout=60.0):
    with _open(base, path, payload, timeout) as resp:
        return json.loads(resp.read())


def health_ok(base, timeout=5.0):
    try:
        with _open(base, "/health", None, timeout) as resp:
            return getattr(resp, "status", 200) == 200
    except Exception:
        return False


def build_ids(base, model, target, timeout):
    toks = http_json(base, "/tokenize", {"model": model, "prompt": FILLER * 40000}, timeout).get("tokens") or []
    if not toks:
        raise RuntimeError("/tokenize returned no tokens")
    return (toks * (target // len(toks) + 1))[:target]


def sample_metrics(base, timeout=10.0):
    out = {
        "running": 0.0, "waiting": 0.0, "kv_max": 0.0, "kv_by_engine": {},
        "running_by_engine": {}, "preemptions": 0.0, "gen_tokens": 0.0,
        "accepted_tokens": 0.0, "draft_tokens": 0.0,
        "tpot_hist": {}, "tpot_count": 0.0, "tpot_sum": 0.0,
    }
    try:
        with _open(base, "/metrics", None, timeout) as resp:
            text = resp.read().decode("utf-8", "ignore")
    except Exception as exc:  # noqa: BLE001
        out["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:120])
        return out
    for raw in text.splitlines():
        m = _METRIC_RE.match(raw.strip())
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        eng = ""
        em = re.search(r'engine="([^"]*)"', labels)
        if em:
            eng = em.group(1)
        if name == "vllm:num_requests_running":
            out["running"] += val
            out["running_by_engine"][eng] = val
        elif name == "vllm:num_requests_waiting":
            out["waiting"] += val
        elif name == "vllm:kv_cache_usage_perc":
            out["kv_max"] = max(out["kv_max"], val)
            out["kv_by_engine"][eng] = val
        elif name == "vllm:num_preemptions_total":
            out["preemptions"] += val
        elif name == "vllm:generation_tokens_total":
            out["gen_tokens"] += val
        elif name in ("vllm:spec_decode_num_accepted_tokens_total",):
            out["accepted_tokens"] += val
        elif name in ("vllm:spec_decode_num_draft_tokens_total",):
            out["draft_tokens"] += val
        elif name == "vllm:time_per_output_token_seconds_bucket":
            lm = _LE_RE.search(labels)
            if lm:
                out["tpot_hist"][lm.group(1)] = out["tpot_hist"].get(lm.group(1), 0.0) + val
        elif name == "vllm:time_per_output_token_seconds_count":
            out["tpot_count"] += val
        elif name == "vllm:time_per_output_token_seconds_sum":
            out["tpot_sum"] += val
    return out


def hist_percentiles(hist, qs=(50, 95)):
    """Approximate percentiles from Prometheus bucket counts (upper-bound interp)."""
    if not hist:
        return {}
    items = []
    for k, v in hist.items():
        try:
            items.append((float(k), float(v)))
        except ValueError:
            continue
    items.sort()
    total = items[-1][1] if items else 0.0
    res = {}
    for q in qs:
        target = total * q / 100.0
        prev_le, prev_c = 0.0, 0.0
        for le, c in items:
            if c >= target:
                span = c - prev_c
                frac = (target - prev_c) / span if span > 0 else 0.0
                res["p%d" % q] = prev_le + (le - prev_le) * frac
                break
            prev_le, prev_c = le, c
    res["count"] = total
    return res


def submit_stream(base, model, prompt, max_tokens, timeout, rec, q, stop_event, chunk_cap=8000):
    body = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True, "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    rec["t_submit"] = time.time()
    resp = None
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        rec["http_status"] = 200
        for raw in resp:
            now = time.time()
            if stop_event.is_set():
                break
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if obj.get("usage"):
                rec["usage"] = obj["usage"]
            for ch in obj.get("choices") or []:
                txt = ch.get("text") or (ch.get("delta") or {}).get("content") or ""
                if txt:
                    rec["chunks"] += 1
                    if len(rec["chunk_times"]) < chunk_cap:
                        rec["chunk_times"].append(now)
                    if rec.get("t_first") is None:
                        rec["t_first"] = now
                        q.put(("first", rec["idx"]))
    except urllib.error.HTTPError as exc:
        body_txt = ""
        try:
            body_txt = exc.read()[:400].decode("utf-8", "ignore")
        except Exception:
            pass
        rec["error"] = "HTTP %s: %s" % (exc.code, body_txt)
        q.put(("error", rec["idx"]))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        q.put(("error", rec["idx"]))
    finally:
        rec["t_end"] = time.time()
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass


def parse_serve_log(path):
    out = {
        "available_kv_gib": None, "gpu_kv_cache_tokens": None, "max_concurrency": None,
        "int8_capacity_lines": [], "state_slots_lines": [], "admission_gate_lines": [],
        "dequant_lines": [], "capturing1_count": 0, "capturing0_count": 0,
        "host_tier_markers": 0, "err00100": 0, "ai_core_markers": 0,
        "error_markers": 0, "preempt_lines": 0, "n_lines": 0,
    }
    if not path or not os.path.exists(path):
        return out
    for line in open(path, errors="ignore"):
        out["n_lines"] += 1
        m = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", line)
        if m:
            out["available_kv_gib"] = float(m.group(1))
        m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", line)
        if m:
            out["gpu_kv_cache_tokens"] = int(m.group(1).replace(",", ""))
        m = re.search(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x", line)
        if m:
            out["max_concurrency"] = {"tokens_per_request": int(m.group(1).replace(",", "")), "x": float(m.group(2))}
        if "int8 long-KV capacity" in line:
            out["int8_capacity_lines"].append(line.strip())
        if "state slots" in line:
            out["state_slots_lines"].append(line.strip())
        if "[admission_gate]" in line:
            out["admission_gate_lines"].append(line.strip())
        if "int8 window: first dequant" in line:
            out["dequant_lines"].append(line.strip())
        if re.search(r"int8 window: first dequant.*capturing=1", line):
            out["capturing1_count"] += 1
        if re.search(r"int8 window: first dequant.*capturing=0", line):
            out["capturing0_count"] += 1
        if re.search(r"host[_-]?tier|V41KVOffload|offload_pool|OffloadPool", line, re.I):
            out["host_tier_markers"] += 1
        if "ERR00100" in line:
            out["err00100"] += 1
        if re.search(r"AI core|aicore error|AIV fault|vector core", line, re.I):
            out["ai_core_markers"] += 1
        if re.search(r"ERROR.*HCCL|HCCL.*ERROR|RuntimeError|Traceback \(most recent", line):
            out["error_markers"] += 1
        if "preempt" in line.lower():
            out["preempt_lines"] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="deepseek-v41")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--checkpoint", type=int, default=3)
    ap.add_argument("--prompt-tokens", type=int, default=1024000)
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--full-context", type=int, default=MI)
    ap.add_argument("--first-timeout", type=float, default=2700.0)
    ap.add_argument("--http-timeout", type=float, default=21600.0)
    ap.add_argument("--hold-seconds", type=float, default=120.0)
    ap.add_argument("--checkpoint-hold", type=float, default=30.0)
    ap.add_argument("--sample-interval", type=float, default=2.0)
    ap.add_argument("--settle-seconds", type=float, default=2.0)
    ap.add_argument("--queue-grace", type=float, default=12.0)
    ap.add_argument("--serve-log", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="p36")
    args = ap.parse_args()

    if args.prompt_tokens + args.max_tokens > args.full_context:
        raise SystemExit("prompt+max_tokens > full_context")
    if not health_ok(args.base_url):
        raise SystemExit("/health not OK at %s" % args.base_url)

    print("[p36cap] tokenizing %d tokens ..." % args.prompt_tokens, flush=True)
    prompt_ids = build_ids(args.base_url, args.model, args.prompt_tokens, args.http_timeout)
    print("[p36cap] tokenize ok: %d ids (head=%s)" % (len(prompt_ids), prompt_ids[:4]), flush=True)

    m0 = sample_metrics(args.base_url)
    gen0, preempt0 = m0.get("gen_tokens", 0.0), m0.get("preemptions", 0.0)
    peaks = {"running": 0.0, "waiting": 0.0, "kv_max": 0.0}
    samples = []
    health_fail = 0
    accepted = []
    records = []
    stop_event = threading.Event()
    threads = []
    full_reason = None
    waiting_since = None
    checkpoint_eval = None
    t_start = time.time()

    def bump(m):
        for k in peaks:
            peaks[k] = max(peaks[k], float(m.get(k, 0.0)))
        if len(samples) < 20000:
            samples.append({"t": round(time.time() - t_start, 2), "running": m.get("running", 0.0),
                            "waiting": m.get("waiting", 0.0), "kv_max": m.get("kv_max", 0.0),
                            "kv_eng": m.get("kv_by_engine", {}), "run_eng": m.get("running_by_engine", {}),
                            "preempt": m.get("preemptions", 0.0), "gen": m.get("gen_tokens", 0.0)})

    def sample_and_bump():
        nonlocal health_fail
        m = sample_metrics(args.base_url)
        bump(m)
        if not health_ok(args.base_url):
            health_fail += 1
        return m

    for i in range(1, args.n + 1):
        if stop_event.is_set():
            break
        rec = {"idx": i, "prompt_tokens_target": args.prompt_tokens, "max_tokens": args.max_tokens,
               "t_submit": None, "t_first": None, "t_end": None, "http_status": None,
               "chunks": 0, "chunk_times": [], "usage": None, "error": None}
        q = queue.Queue()
        th = threading.Thread(target=submit_stream,
                              args=(args.base_url, args.model, prompt_ids, args.max_tokens,
                                    args.http_timeout, rec, q, stop_event), daemon=True)
        th.start()
        threads.append(th)
        accepted_now = False
        deadline = time.time() + args.first_timeout
        while True:
            if time.time() > deadline:
                full_reason = "first_timeout_after_%.0fs" % args.first_timeout
                stop_event.set()
                break
            try:
                kind, _ = q.get(timeout=args.sample_interval)
            except queue.Empty:
                m = sample_and_bump()
                if rec.get("t_first") is None and m.get("waiting", 0.0) >= 1:
                    if waiting_since is None:
                        waiting_since = time.time()
                    elif time.time() - waiting_since >= args.queue_grace:
                        full_reason = "kv_cache_full_or_queue: waiting>=1 for %.0fs (req %d)" % (args.queue_grace, i)
                        stop_event.set()
                        break
                else:
                    waiting_since = None
                continue
            if kind == "first":
                accepted_now = True
            else:
                full_reason = "request_error req%d: %s" % (i, rec.get("error") or "unknown")
            break
        records.append(rec)
        if accepted_now:
            accepted.append(rec)
            m = sample_and_bump()
            print("[p36cap] admitted %d/%d wall=%.1fs running=%.0f waiting=%.0f kv_max=%.3f run_eng=%s"
                  % (i, args.n, rec["t_first"] - rec["t_submit"], m.get("running", 0), m.get("waiting", 0),
                     m.get("kv_max", 0.0), m.get("running_by_engine", {})), flush=True)
            if i == args.checkpoint:
                t_ck = time.time()
                while time.time() - t_ck < args.checkpoint_hold:
                    sample_and_bump()
                    time.sleep(args.sample_interval)
                mck = sample_and_bump()
                ok = (mck.get("running", 0) >= args.checkpoint and mck.get("waiting", 0) == 0
                      and mck.get("kv_max", 0.0) < 1.0
                      and (mck.get("preemptions", 0.0) - preempt0) == 0 and health_fail == 0)
                checkpoint_eval = {
                    "at_req": i, "running": mck.get("running", 0), "waiting": mck.get("waiting", 0),
                    "kv_max": mck.get("kv_max", 0.0), "kv_by_engine": mck.get("kv_by_engine", {}),
                    "preemptions_delta": mck.get("preemptions", 0.0) - preempt0,
                    "health_fail": health_fail, "pass": bool(ok),
                }
                print("[p36cap] checkpoint(3) eval: %s" % json.dumps(checkpoint_eval, ensure_ascii=False), flush=True)
                if not ok:
                    full_reason = "checkpoint3_failed"
                    stop_event.set()
                    break
            if args.settle_seconds > 0:
                time.sleep(args.settle_seconds)
        else:
            stop_event.set()
            break

    all_accepted = len(accepted) >= args.n
    m_hold0 = sample_metrics(args.base_url)
    bump(m_hold0)
    hold_start = time.time()
    hold_samples = 0
    while time.time() - hold_start < args.hold_seconds and all_accepted:
        sample_and_bump()
        hold_samples += 1
        time.sleep(args.sample_interval)
    hold_end = time.time()
    m_end = sample_metrics(args.base_url)
    bump(m_end)

    # ---- aggregate throughput / step latency over the hold window ----
    gen_end, preempt_end = m_end.get("gen_tokens", 0.0), m_end.get("preemptions", 0.0)
    hold_s = max(0.1, hold_end - hold_start)
    agg_tok_s = ((gen_end - m_hold0.get("gen_tokens", gen0)) / hold_s) if all_accepted else 0.0
    step_deltas = []
    per_req = []
    for rec in accepted:
        cts = [t for t in rec["chunk_times"] if hold_start <= t <= hold_end]
        n_ch = len(cts)
        toks = len([t for t in rec["chunk_times"] if hold_start <= t <= hold_end])
        per_req.append({"idx": rec["idx"], "chunks_in_hold": n_ch,
                        "tok_s_in_hold": round(n_ch / hold_s, 3)})
        for a, b in zip(cts, cts[1:]):
            dt = (b - a) * 1000.0
            if 0 < dt < 60000:
                step_deltas.append(dt)
    tpot = hist_percentiles(m_end.get("tpot_hist", {}), (50, 95))

    stop_event.set()
    for th in threads:
        th.join(timeout=15.0)
    drain_start = time.time()
    drained = False
    while time.time() - drain_start < 180.0:
        m = sample_metrics(args.base_url)
        bump(m)
        if m.get("running", 0.0) == 0 and m.get("waiting", 0.0) == 0:
            drained = True
            break
        time.sleep(2.0)

    zinfo = parse_serve_log(args.serve_log)
    summary = {
        "tag": args.tag, "n_target": args.n, "checkpoint": args.checkpoint,
        "accepted": len(accepted), "requested": len(records),
        "all_target_resident": bool(all_accepted and full_reason is None),
        "full_reason": full_reason,
        "checkpoint3": checkpoint_eval,
        "kv_usage_peak_pct": round(peaks["kv_max"] * 100.0, 2),
        "kv_usage_end_pct": round(m_end.get("kv_max", 0.0) * 100.0, 2),
        "kv_end_by_engine_pct": {k: round(v * 100.0, 2) for k, v in m_end.get("kv_by_engine", {}).items()},
        "running_peak": peaks["running"], "waiting_peak": peaks["waiting"],
        "health_fail_samples": health_fail,
        "preemptions_delta": preempt_end - preempt0,
        "preemption_log_lines": zinfo["preempt_lines"],
        "drained": drained,
        "hold_seconds_actual": round(hold_s, 1),
        "hold_samples": hold_samples,
        "agg_gen_tokens_per_s": round(agg_tok_s, 3),
        "per_request_hold_chunks": per_req,
        "step_latency_ms": {
            "n": len(step_deltas),
            "p50": round(statistics.median(step_deltas), 2) if step_deltas else None,
            "p95": round(_pct(step_deltas, 95), 2) if step_deltas else None,
            "min": round(min(step_deltas), 2) if step_deltas else None,
            "max": round(max(step_deltas), 2) if step_deltas else None,
        },
        "server_time_per_output_token_s": tpot,
        "admission_wall_s": {
            "per_req": [round(r["t_first"] - r["t_submit"], 1) for r in accepted],
            "total_s": round(sum(r["t_first"] - r["t_submit"] for r in accepted), 1),
        },
        "elapsed_s": round(time.time() - t_start, 1),
        "z_startup": zinfo,
        "errors": {"err00100": zinfo["err00100"], "ai_core_markers": zinfo["ai_core_markers"],
                   "error_markers": zinfo["error_markers"], "host_tier_markers": zinfo["host_tier_markers"],
                   "http_errors": [r["error"] for r in records if r.get("error")]},
    }
    summary["pass"] = bool(
        summary["all_target_resident"] and summary["kv_usage_peak_pct"] < 100.0
        and summary["preemptions_delta"] == 0 and health_fail == 0
        and zinfo["err00100"] == 0 and zinfo["ai_core_markers"] == 0
        and not summary["errors"]["http_errors"]
    )
    out = {"plan": {"base_url": args.base_url, "model": args.model, "n": args.n,
                    "prompt_tokens": args.prompt_tokens, "max_tokens": args.max_tokens,
                    "hold_seconds": args.hold_seconds, "checkpoint": args.checkpoint,
                    "protocol": "serial long-prefill admission; checkpoint(3); hold all; abort"},
           "summary": summary, "metrics_samples": samples, "per_request": records}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print("== p36cap %s: accepted=%d/%d reason=%s kv_peak=%.2f%% waiting_peak=%.0f preempt=%.0f "
          "hold=%.0fs agg=%.2f tok/s step_p50=%s step_p95=%s pass=%s =="
          % (args.tag, len(accepted), args.n, full_reason, summary["kv_usage_peak_pct"], peaks["waiting"],
             summary["preemptions_delta"], hold_s, agg_tok_s, summary["step_latency_ms"]["p50"],
             summary["step_latency_ms"]["p95"], summary["pass"]), flush=True)
    sys.exit(0 if summary["pass"] else 2)


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


if __name__ == "__main__":
    main()
