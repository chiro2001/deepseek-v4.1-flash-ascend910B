#!/usr/bin/env python3
"""DCP8 性能 A/B 与逐层阶段计时驱动（**在 a3-21 上运行**）。

为什么需要它：主线的消融（`/tmp/v41_perf_flags`）之前是「一次性跑一个臂」，
测得 43.19 / 42.09 / 42.49，彼此只差 ~1 ms，**低于方案间漂移**。
本脚本把「臂」放进同一个会话内**交替**执行，再取中位数，消除漂移。

用法（在 a3-21）：
  # 模式一：交替 A/B，输出 (ms/step, A, tok/s) 三元组中位数
  python3 ~/tmp/dcp_ab.py ab --reps 3 \
      --arm base: --arm nopack:no_pack=1 --arm skip2nd:skip_2nd=1

  # 模式二：逐层阶段计时（抓 [V41-TIME] 行）
  python3 ~/tmp/dcp_ab.py timing --steps 24

口径（必须写进结论）：流式请求，相邻 token 间隔中位数 = ms/step；
首个 `--drop-first` 个间隔丢弃（warmup）；A=1（未开推测解码）⇒ tok/s=1000/ms。
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

PROMPT = "请用中文写一段关于分布式推理系统的详细说明，不少于三百字。"
TIME_RE = re.compile(r"\[V41-TIME\] (\S+) total=([\d.]+)ms (.*)")


def sh(cmd, timeout=60):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def set_flags(container, flags):
    """把开关写进容器内的 /tmp/v41_perf_flags（空 = 清空）。"""
    body = (flags or "").replace("\\n", "\n")
    cmd = "docker exec %s bash -lc %s" % (container, json.dumps("printf %s " + json.dumps(body) + " > /tmp/v41_perf_flags"))
    r = sh(cmd)
    if r.returncode != 0:
        print("[warn] set_flags rc=%d %s" % (r.returncode, r.stderr.strip()[:200]), flush=True)


def run_stream(base_url, model, max_tokens, timeout=1800, drop_first=8):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    stamps = []
    n_chunks = 0
    t0 = time.time()
    ttft = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                d = json.loads(body)
            except Exception:
                continue
            ch = (d.get("choices") or [{}])[0]
            delta = (ch.get("delta") or {}).get("content")
            if delta:
                now = time.time()
                n_chunks += 1
                if ttft is None:
                    ttft = now - t0
                stamps.append(now)
    end = time.time()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])][drop_first:]
    return {
        "n_chunks": n_chunks,
        "ttft_s": round(ttft, 3) if ttft else None,
        "elapsed_s": round(end - t0, 3),
        "ms_per_step_median": round(statistics.median(gaps) * 1000, 2) if gaps else None,
        "ms_per_step_mean": round(statistics.mean(gaps) * 1000, 2) if gaps else None,
        "tok_s_api": round(n_chunks / (end - t0), 2) if end > t0 else None,
    }


def mode_ab(a):
    arms = []
    for spec in a.arm:
        label, _, flags = spec.partition(":")
        arms.append((label, flags))
    if not arms:
        print("[ab] 至少给一个 --arm", file=sys.stderr)
        return 2
    rows = []
    # 先整体热身一次（不进统计）
    set_flags(a.container, arms[0][1])
    time.sleep(1.0)
    w = run_stream(a.base_url, a.model, 24, drop_first=0)
    print("[ab] warmup: %s" % json.dumps(w, ensure_ascii=False), flush=True)
    for rep in range(a.reps):
        for label, flags in arms:
            set_flags(a.container, flags)
            time.sleep(a.settle)
            r = run_stream(a.base_url, a.model, a.max_tokens, drop_first=a.drop_first)
            r.update({"label": label, "flags": flags, "rep": rep})
            rows.append(r)
            print("[ab] rep%d %-10s %s" % (rep, label, json.dumps(r, ensure_ascii=False)), flush=True)
    set_flags(a.container, "")
    print("\n=== 汇总（每臂 %d 次，中位数）===" % a.reps, flush=True)
    print("%-12s %-24s %10s %10s %10s" % ("arm", "flags", "ms/step", "tok/s", "A"))
    summary = {}
    for label, flags in arms:
        ms = [x["ms_per_step_median"] for x in rows if x["label"] == label and x["ms_per_step_median"]]
        if not ms:
            continue
        med = statistics.median(ms)
        summary[label] = {"ms_per_step": med, "tok_s": round(1000 / med, 2), "reps_ms": ms, "flags": flags}
        print("%-12s %-24s %10.2f %10.2f %10.1f" % (label, flags or "(base)", med, 1000 / med, 1.0))
    if summary:
        base = summary.get("base") or summary[arms[0][0]]
        print("\n相对 base 的差值：", flush=True)
        for label, s in summary.items():
            print("  %-12s Δ=%+6.2f ms/step（%.3f×）" % (label, s["ms_per_step"] - base["ms_per_step"], s["ms_per_step"] / base["ms_per_step"]), flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "summary": summary}, f, ensure_ascii=False, indent=2)
        print("[ab] 原始数据 -> %s" % a.out, flush=True)
    return 0


def _log_lines(log):
    try:
        with open(log, "rb") as f:
            return f.read().decode("utf-8", "ignore").splitlines()
    except OSError:
        return []


def mode_timing(a):
    n0 = len(_log_lines(a.log))
    set_flags(a.container, "timing=1")
    time.sleep(a.settle)
    r = run_stream(a.base_url, a.model, a.steps, drop_first=0)
    print("[timing] 请求完成：%s" % json.dumps(r, ensure_ascii=False), flush=True)
    time.sleep(2.0)
    set_flags(a.container, "")
    lines = _log_lines(a.log)[n0:]
    # 逐 rank 逐层聚合
    per = {}   # (rank, layer) -> {stage: ms}
    for ln in lines:
        if "[V41-TIME]" not in ln:
            continue
        m = re.search(r"\((Worker_\S+) pid=\d+\)\s+.*?\[V41-TIME\] (\S+) total=([\d.]+)ms (.*)$", ln)
        if not m:
            m2 = TIME_RE.search(ln)
            if not m2:
                continue
            rank, tag, total, parts = "?", m2.group(1), float(m2.group(2)), m2.group(3)
        else:
            rank, tag, total, parts = m.group(1), m.group(2), float(m.group(3)), m.group(4)
        st = {}
        for kv in parts.split():
            k, _, v = kv.partition("=")
            if v.endswith("ms(0%)") or "(" in v:
                v = v.split("(")[0]
            if v.endswith("ms"):
                v = v[:-2]
            try:
                st[k] = float(v)
            except ValueError:
                pass
        per.setdefault((rank, tag), {}).update({"total": total, **st})
    if not per:
        print("[timing] 没抓到 [V41-TIME] 行 —— 检查 /tmp/v41_perf_flags 是否真的写进容器，以及日志路径 %s" % a.log, file=sys.stderr)
        return 3
    # 阶段聚合：对 (rank, layer) 求中位数
    stages = {}
    for (_rank, _layer), st in per.items():
        for k, v in st.items():
            stages.setdefault(k, []).append(v)
    print("\n=== 阶段中位数（跨 %d 个 (rank, layer) 样本）===" % len(per), flush=True)
    tot = stages.get("total")
    tot_med = statistics.median(tot) if tot else 1.0
    order = sorted(stages, key=lambda k: -statistics.median(stages[k]))
    for k in order:
        vals = stages[k]
        med = statistics.median(vals)
        print("  %-12s 中位 %8.3f ms   占比 %5.1f%%   样本 %d" % (k, med, 100 * med / tot_med if tot_med else 0, len(vals)), flush=True)
    print("\n  单层 total 中位数 = %.3f ms ⇒ 40 层 ≈ %.2f ms/step（仅 attention 部分）" % (tot_med, tot_med * 40), flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"per_rank_layer": {"%s|%s" % k: v for k, v in per.items()}, "stage_medians": {k: statistics.median(v) for k, v in stages.items()}, "request": r}, f, ensure_ascii=False, indent=2)
        print("[timing] 原始数据 -> %s" % a.out, flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ab", "timing"])
    ap.add_argument("--base-url", default="http://127.0.0.1:19210")
    ap.add_argument("--model", default="deepseek-v41")
    ap.add_argument("--container", default="dsv41-dcpcap")
    ap.add_argument("--log", default=os.path.expanduser("~/cedpd-repo/results/dcpcap_0929_190852/serve.log"))
    ap.add_argument("--arm", action="append", default=[], help="LABEL:FLAGS，可重复")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--steps", type=int, default=24, help="timing 模式下的 max_tokens")
    ap.add_argument("--drop-first", type=int, default=8)
    ap.add_argument("--settle", type=float, default=1.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    return mode_ab(a) if a.mode == "ab" else mode_timing(a)


if __name__ == "__main__":
    raise SystemExit(main())
