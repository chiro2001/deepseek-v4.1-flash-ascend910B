#!/usr/bin/env python3
"""Single-stream long-context decode curve for the DSV4.1-Flash server (stdlib only).

For each requested context length this script:
  * verifies the server is idle (concurrency=1 / exclusive protocol),
  * optionally warms up the decode graph,
  * streams one greedy request with ignore_eos=true,
  * derives ms/step, tok/step (spec acceptance length) and tok/s from the
    server-side Prometheus counters (vllm:spec_decode_num_drafts_total,
    vllm:spec_decode_num_accepted_tokens_total, vllm:iteration_tokens_total_count,
    vllm:generation_tokens_total), not from client SSE chunk timing,
  * appends one JSON object per context point to --out (JSONL, crash-safe).

Typical use (P15 Phase 0, TP8 server already running on :8001):

  python3 scripts/stream_ctx_curve.py \
      --base-url http://127.0.0.1:8001 \
      --tokens 32768,131072,262144,524288 \
      --max-tokens 192 \
      --out logs/perf/stream_ctx_curve_tp8u094.jsonl

Then inspect / fit the model:

  python3 scripts/stream_ctx_curve.py --fit --in logs/perf/stream_ctx_curve_tp8u094.jsonl

Notes / protocol:
  * One request in flight at any time; a background sampler asserts that
    vllm:num_requests_running never exceeds --max-running (default 1).
  * The first token may take minutes at 512Ki (1M prefill is ~315 s under the
    admission gate); --request-timeout default is 2 h.
  * The decode window used for ms/step starts at the first streamed token, so
    prefill/TTFT is reported separately and never contaminates ms/step.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_FILLER = "甲乙丙丁戊己庚辛壬癸" * 4


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def _http(base: str, path: str, payload: dict | None = None, timeout: float = 30.0) -> tuple[int, str]:
    url = base.rstrip("/") + path
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:  # keep the body for diagnostics
        return exc.code, exc.read().decode("utf-8", "replace")


def _get_text(base: str, path: str, timeout: float = 10.0) -> str | None:
    try:
        with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Prometheus parsing
# --------------------------------------------------------------------------- #
_METRIC_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[0-9eE+\-.]+)\s*$")


def parse_metrics(text: str) -> dict[str, float]:
    """Sum every sample of each metric name (labels are intentionally summed).

    Counter names in the vLLM multiprocess exporter appear as ``foo_total``;
    ``foo`` (if it exists) is folded into ``foo_total`` so callers can use one
    canonical key. Histograms are exposed as ``foo_count`` / ``foo_sum`` which
    this parser keeps verbatim.
    """
    out: dict[str, float] = {}
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        m = _METRIC_LINE.match(raw)
        if not m:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        out[m.group("name")] = out.get(m.group("name"), 0.0) + value
    for name in list(out):
        if not name.endswith("_total") and f"{name}_total" in out:
            out[name + "_total"] = out.get(name + "_total", 0.0) + out[name]
    return out


def parse_labelled(text: str, metric: str, label: str = "position") -> dict[str, float]:
    """Extract one metric's samples keyed by a label value (e.g. per-position)."""
    out: dict[str, float] = {}
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or not raw.startswith(metric):
            continue
        m = _METRIC_LINE.match(raw)
        if not m:
            continue
        labels = m.group("labels") or ""
        mm = re.search(rf'{label}="([^"]*)"', labels)
        key = mm.group(1) if mm else "_"
        try:
            out[key] = out.get(key, 0.0) + float(m.group("value"))
        except ValueError:
            continue
    return out


def mval(m: dict[str, float], *names: str) -> float | None:
    for n in names:
        if n in m:
            return m[n]
    return None


def counter(m: dict[str, float], *names: str) -> float:
    v = mval(m, *names)
    return float(v) if v is not None else 0.0


# --------------------------------------------------------------------------- #
# Idle / exclusive check
# --------------------------------------------------------------------------- #
class MetricsSampler:
    """Background /metrics sampler used for both idle checks and step timing."""

    def __init__(self, base: str, path: str, interval: float) -> None:
        self.base = base
        self.path = path
        self.interval = max(0.05, interval)
        self.samples: list[tuple[float, dict[str, float]]] = []
        self.raw: list[tuple[float, str]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.max_running = 0.0
        self.max_waiting = 0.0
        self.max_kv = 0.0
        self.errors = 0

    def sample_once(self, timeout: float = 10.0) -> dict[str, float] | None:
        text = _get_text(self.base, self.path, timeout)
        if text is None:
            return None
        m = parse_metrics(text)
        running = counter(m, "vllm:num_requests_running", "vllm:num_requests_running_total")
        waiting = counter(m, "vllm:num_requests_waiting", "vllm:num_requests_waiting_total")
        kv = counter(m, "vllm:kv_cache_usage_perc")
        self.max_running = max(self.max_running, running)
        self.max_waiting = max(self.max_waiting, waiting)
        self.max_kv = max(self.max_kv, kv)
        now = time.perf_counter()
        self.samples.append((now, m))
        self.raw.append((now, text))
        return m

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            if self.sample_once() is None:
                self.errors += 1

    def start(self) -> None:
        self.sample_once()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.sample_once()

    # ---- lookups -------------------------------------------------------- #
    def at_or_before(self, t: float) -> tuple[float, dict[str, float], str] | None:
        best = None
        for (ts, m), (_, raw) in zip(self.samples, self.raw):
            if ts <= t:
                best = (ts, m, raw)
            else:
                break
        return best

    def last(self) -> tuple[float, dict[str, float], str] | None:
        if not self.samples:
            return None
        return (self.samples[-1][0], self.samples[-1][1], self.raw[-1][1])


def wait_idle(base: str, metrics_path: str, timeout: float, max_running: float) -> tuple[bool, str]:
    """Block until the server has no running/waiting requests."""
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        text = _get_text(base, metrics_path, 10.0)
        if text is None:
            last = "metrics endpoint unreachable"
            time.sleep(2.0)
            continue
        m = parse_metrics(text)
        running = counter(m, "vllm:num_requests_running", "vllm:num_requests_running_total")
        waiting = counter(m, "vllm:num_requests_waiting", "vllm:num_requests_waiting_total")
        if running <= 0 and waiting <= 0:
            return True, f"idle (running={running:.0f} waiting={waiting:.0f})"
        last = f"busy (running={running:.0f} waiting={waiting:.0f})"
        time.sleep(2.0)
    return False, last or "timeout"


# --------------------------------------------------------------------------- #
# Request helpers
# --------------------------------------------------------------------------- #
def _tokenize_file(base: str, path: str, model: str, timeout: float = 900.0) -> list[int]:
    text = open(path, encoding="utf-8", errors="ignore").read()
    code, body = _http(base, "/tokenize", {"model": model, "prompt": text}, timeout)
    if code != 200:
        raise RuntimeError("tokenize %s failed HTTP %s: %s" % (path, code, body[:200]))
    return json.loads(body)["tokens"]


def build_ids(base: str, tokenize_path: str, target: int, filler: str, timeout: float, model: str = "") -> list[int]:
    """Return exactly `target` token ids by tiling a tokenized filler block."""
    code, body = _http(base, tokenize_path, {"model": model, "prompt": filler}, timeout)
    if code != 200:
        # some servers require the served model name; retry with a probe model
        raise RuntimeError(f"tokenize failed: HTTP {code}: {body[:300]}")
    obj = json.loads(body)
    base_ids = obj.get("tokens") or []
    if not base_ids:
        raise RuntimeError("tokenizer returned no tokens")
    reps = target // len(base_ids) + 1
    return (base_ids * reps)[:target]


def stream_completion(base: str, model: str, ids: list[int], max_tokens: int, timeout: float) -> dict[str, Any]:
    """Stream one greedy completion; return timing + usage."""
    payload = {
        "model": model,
        "prompt": ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    t_first: float | None = None
    token_times: list[float] = []
    usage: dict[str, Any] | None = None
    err: str | None = None
    n_chunks = 0
    text_chars = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
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
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    txt = ch.get("text")
                    if txt is None:
                        txt = (ch.get("delta") or {}).get("content") or ""
                    if txt:
                        now = time.perf_counter()
                        if t_first is None:
                            t_first = now
                        token_times.append(now)
                        n_chunks += 1
                        text_chars += len(txt)
    except Exception as exc:  # noqa: BLE001 - report and let the caller record it
        err = f"{type(exc).__name__}: {str(exc)[:300]}"
    t_end = time.perf_counter()
    return {
        "t0": t0,
        "t_first": t_first,
        "t_end": t_end,
        "token_times": token_times,
        "usage": usage,
        "error": err,
        "n_chunks": n_chunks,
        "text_chars": text_chars,
    }


def warmup(base: str, model: str, warm_tokens: int, out_tokens: int, timeout: float) -> dict[str, Any]:
    ids = build_ids(base, "/tokenize", warm_tokens, DEFAULT_FILLER, timeout, model)
    rec = stream_completion(base, model, ids, out_tokens, timeout)
    rec.pop("token_times", None)
    rec["t0"] = round(rec["t0"], 4)
    rec["t_first"] = rec.get("t_first")
    rec["t_end"] = round(rec["t_end"], 4)
    return rec


# --------------------------------------------------------------------------- #
# Per-point measurement
# --------------------------------------------------------------------------- #
def measure_point(
    base: str,
    model: str,
    target: int,
    max_tokens: int,
    sample_interval: float,
    idle_timeout: float,
    request_timeout: float,
    metrics_path: str,
    warmup_out: int,
    max_running: float,
    filler: str,
    prefix_file: str | None = None,
    suffix_file: str | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {"context_tokens_target": target, "ts": time.strftime("%F %T")}
    ok, why = wait_idle(base, metrics_path, idle_timeout, max_running)
    rec["pre_idle"] = {"ok": ok, "detail": why}
    if not ok:
        rec["error"] = "server not idle before point: " + why
        return rec

    # warmup (small prompt, same decode graph shape family; keeps the measured
    # interval free of first-touch compilation/capture effects)
    if warmup_out > 0:
        try:
            rec["warmup"] = warmup(base, model, 16, warmup_out, 600.0)
        except Exception as exc:  # noqa: BLE001
            rec["warmup"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    rec["prompt_tokens_target"] = target
    try:
        if prefix_file:
            pids = _tokenize_file(base, prefix_file, model, 900.0)
            sids = _tokenize_file(base, suffix_file, model, 900.0) if suffix_file else []
            cut = max(1, target - len(sids))
            ids = pids[:cut] + sids
        else:
            ids = build_ids(base, "/tokenize", target, filler, 900.0, model)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"tokenize failed: {type(exc).__name__}: {str(exc)[:200]}"
        return rec

    sampler = MetricsSampler(base, metrics_path, sample_interval)
    sampler.start()
    run = stream_completion(base, model, ids, max_tokens, request_timeout)
    sampler.stop()

    ttft = (run["t_first"] - run["t0"]) if run["t_first"] else None
    decode_s = (run["t_end"] - run["t_first"]) if run["t_first"] else None
    usage = run["usage"] or {}
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")

    rec.update(
        {
            "prompt_tokens_actual": prompt_tokens,
            "completion_tokens_actual": completion_tokens,
            "ttft_s": round(ttft, 3) if ttft else None,
            "total_s": round(run["t_end"] - run["t0"], 3),
            "decode_s": round(decode_s, 3) if decode_s else None,
            "n_sse_chunks": run["n_chunks"],
            "error": run["error"],
        }
    )
    # ---- client-side fallback (chunk intervals; only indicative under spec) --
    tt = run["token_times"]
    if len(tt) > 2:
        iv = [tt[i + 1] - tt[i] for i in range(len(tt) - 1)]
        rec["client_ms_per_chunk"] = {
            "p50": round(statistics.median(iv) * 1000, 3),
            "p95": round(sorted(iv)[min(len(iv) - 1, int(len(iv) * 0.95))] * 1000, 3),
            "n": len(iv),
        }

    # ---- server-side counters over the decode window ---------------------- #
    t_first = run["t_first"]
    t_end = run["t_end"]
    a = sampler.at_or_before(t_first) if t_first else None
    b = sampler.last()
    metrics_ok = False
    if a and b and completion_tokens:
        dt = b[0] - a[0]
        ma, mb = a[1], b[1]
        d_drafts = counter(mb, "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_drafts") - counter(
            ma, "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_drafts"
        )
        d_acc = counter(mb, "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens") - counter(
            ma, "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens"
        )
        d_gen = counter(mb, "vllm:generation_tokens_total", "vllm:generation_tokens") - counter(
            ma, "vllm:generation_tokens_total", "vllm:generation_tokens"
        )
        d_iter = counter(mb, "vllm:iteration_tokens_total_count") - counter(ma, "vllm:iteration_tokens_total_count")
        steps = d_drafts if d_drafts > 0 else d_iter
        steps_fallback = d_gen if 0 < d_gen < 1e9 else 0
        if steps <= 0 and steps_fallback > 0:
            steps = steps_fallback
        if steps > 0 and dt > 0:
            metrics_ok = True
            rec["decode_window_s"] = round(dt, 3)
            rec["steps_est"] = steps
            rec["steps_source"] = "drafts" if d_drafts > 0 else ("iteration_hist" if d_iter > 0 else "generation")
            rec["ms_per_step"] = round(dt / steps * 1000, 3)
            rec["tok_per_step"] = round((d_gen if d_gen > 0 else completion_tokens) / steps, 3)
            rec["accept_length"] = round(1 + d_acc / d_drafts, 3) if d_drafts > 0 else None
            rec["decode_tok_s"] = round((d_gen if d_gen > 0 else completion_tokens) / dt, 3)
            rec["gen_tokens_delta"] = d_gen
            rec["accepted_tokens_delta"] = d_acc
            rec["draft_tokens_delta"] = counter(mb, "vllm:spec_decode_num_draft_tokens_total") - counter(
                ma, "vllm:spec_decode_num_draft_tokens_total"
            )
            # per-draft-position accepted counters (label position=0..S-1)
            pos_a = parse_labelled(a[2] if len(a) > 2 else "", "vllm:spec_decode_num_accepted_tokens_per_pos")
            pos_b = parse_labelled(b[2] if len(b) > 2 else "", "vllm:spec_decode_num_accepted_tokens_per_pos")
            if pos_b:
                pos_delta = {k: round(pos_b[k] - pos_a.get(k, 0.0), 1) for k in sorted(pos_b, key=lambda z: int(z) if z.isdigit() else 0)}
                denom = d_drafts if d_drafts > 0 else steps
                rec["accepted_per_pos"] = {k: round(v / denom, 4) for k, v in pos_delta.items()} if denom else None
        rec["e2e_tok_s"] = round(completion_tokens / (run["t_end"] - run["t0"]), 3) if completion_tokens else None
        rec["prefill_tok_s"] = round(prompt_tokens / ttft, 1) if prompt_tokens and ttft and ttft > 0 else None
    rec["metrics_ok"] = metrics_ok
    rec["exclusive_check"] = {
        "max_running": sampler.max_running,
        "max_waiting": sampler.max_waiting,
        "max_kv_cache_usage_perc": round(sampler.max_kv, 3),
        "ok": sampler.max_running <= max_running and sampler.max_waiting <= 0,
        "metrics_errors": sampler.errors,
    }
    if not metrics_ok:
        rec.setdefault("warnings", []).append(
            "metrics-derived ms/step unavailable; use client_ms_per_chunk only as a rough fallback"
        )
    if prompt_tokens is not None and abs(prompt_tokens - target) > 8:
        rec.setdefault("warnings", []).append(f"prompt_tokens_actual={prompt_tokens} != target={target}")
    if completion_tokens is None:
        rec.setdefault("warnings", []).append("usage.completion_tokens missing")
    return rec


# --------------------------------------------------------------------------- #
# Fit helper
# --------------------------------------------------------------------------- #
def fit_file(path: str, tokens_key: str = "prompt_tokens_actual") -> int:
    pts = []
    try:
        fh = open(path, "r", encoding="utf-8")
    except OSError as exc:
        print(f"[fit] cannot open {path}: {exc}")
        return 1
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            x = obj.get(tokens_key)
            y = obj.get("ms_per_step")
            if x and y:
                pts.append((float(x), float(y), obj))
    if len(pts) < 2:
        print(f"[fit] need >=2 valid points in {path}, got {len(pts)}")
        return 1
    n = len(pts)
    sx = sum(x for x, _, _ in pts)
    sy = sum(y for _, y, _ in pts)
    sxx = sum(x * x for x, _, _ in pts)
    sxy = sum(x * y for x, y, _ in pts)
    den = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / den  # ms per context token
    intercept = (sy - slope * sx) / n
    print(f"[fit] ms/step = {intercept:.3f} + {slope * 1000:.6f} * L(1k tokens)   (n={n})")
    print(f"[fit] slope per 1M context tokens = {slope * 1e6:.2f} ms")
    print(f"{'L':>9} {'measured':>9} {'fit':>9} {'resid':>8} {'tok/step':>9} {'tok/s':>8}")
    for x, y, obj in sorted(pts):
        print(
            f"{x:9.0f} {y:9.3f} {intercept + slope * x:9.3f} {y - (intercept + slope * x):8.3f}"
            f" {str(obj.get('tok_per_step')):>9} {str(obj.get('decode_tok_s')):>8}"
        )
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001", help="OpenAI server base url")
    ap.add_argument("--model", default="deepseek-v41", help="served model name")
    ap.add_argument("--tokens", default="32768,131072,262144,524288",
                    help="comma separated target context lengths (prompt tokens)")
    ap.add_argument("--max-tokens", type=int, default=192, help="generated tokens per point")
    ap.add_argument("--warmup-tokens", type=int, default=1, help="warmup requests per point (0 disables)")
    ap.add_argument("--warmup-output-tokens", type=int, default=24)
    ap.add_argument("--filler", default=DEFAULT_FILLER, help="text tiled to build the prompt")
    ap.add_argument("--filler-file", default=None, help="read filler text from this file")
    ap.add_argument("--prefix-file", default=None, help="use the first tokens of this file as the prompt prefix")
    ap.add_argument("--suffix-file", default=None, help="append the tokens of this file after the prefix")
    ap.add_argument("--corpus-label", default="", help="label written into each jsonl record")
    ap.add_argument("--tokenize-path", default="/tokenize")
    ap.add_argument("--metrics-path", default="/metrics")
    ap.add_argument("--sample-interval", type=float, default=0.25, help="metrics sampling period (s)")
    ap.add_argument("--idle-timeout", type=float, default=300.0, help="wait this long for an idle server")
    ap.add_argument("--request-timeout", type=float, default=7200.0)
    ap.add_argument("--max-running", type=float, default=1.0, help="exclusive protocol: fail point if exceeded")
    ap.add_argument("--out", default="logs/perf/stream_ctx_curve.jsonl")
    ap.add_argument("--fit", action="store_true", help="only fit an existing jsonl")
    ap.add_argument("--in", dest="in_file", default=None, help="jsonl to fit with --fit")
    args = ap.parse_args()

    if getattr(args, "filler_file", None):
        args.filler = open(args.filler_file, encoding="utf-8").read()

    if args.fit:
        return fit_file(args.in_file or args.out)

    targets = [int(x) for x in args.tokens.split(",") if x.strip()]
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"_meta": {
            "base_url": args.base_url, "model": args.model, "tokens": targets,
            "max_tokens": args.max_tokens, "ts": time.strftime("%F %T"),
            "note": "one JSON object per context point follows this _meta line",
        }}, ensure_ascii=False) + "\n")
        fh.flush()

    rcs = []
    for target in targets:
        print(f"[curve] === context {target} ===", flush=True)
        rec = measure_point(
            base=args.base_url,
            model=args.model,
            target=target,
            max_tokens=args.max_tokens,
            sample_interval=args.sample_interval,
            idle_timeout=args.idle_timeout,
            request_timeout=args.request_timeout,
            metrics_path=args.metrics_path,
            warmup_out=args.warmup_output_tokens if args.warmup_tokens else 0,
            max_running=args.max_running,
            filler=args.filler,
            prefix_file=args.prefix_file,
            suffix_file=args.suffix_file,
        )
        rec["corpus_label"] = args.corpus_label
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
        rcs.append(rec)
        print(
            "[curve] ctx=%s prompt=%s steps=%s ms/step=%s tok/step=%s tok/s=%s ttft=%ss maxrun=%s err=%s"
            % (
                target,
                rec.get("prompt_tokens_actual"),
                rec.get("steps_est"),
                rec.get("ms_per_step"),
                rec.get("tok_per_step"),
                rec.get("decode_tok_s"),
                rec.get("ttft_s"),
                (rec.get("exclusive_check") or {}).get("max_running"),
                rec.get("error"),
            ),
            flush=True,
        )
        if rec.get("error"):
            print(f"[curve] stop: first error at ctx={target}", flush=True)
            break
    print(f"[curve] results appended to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
