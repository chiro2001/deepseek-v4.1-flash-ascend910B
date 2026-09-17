#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""interleave_ab.py —— 同会话交错 A/B：用 **pos0** 作为「该请求是否被污染」的敏感判据。

### 为什么需要它
`A`（整段 decode 的平均接受长度）方差过大（163 发里 1.18–3.45），要几百发才能分辨两臂。
但 `accepted_per_pos[0]`（第一个 draft token 的接受率，同一请求内平均几十步）观测上**近双峰**：

* **干净**请求 `pos0 ≈ 0.83–0.95`
* **被污染**请求 `pos0 ≈ 0.17–0.63`

⇒ `pos0 ≥ 0.8` 就是"该请求干净吗"的判据（clean-rate），比 A 灵敏得多。
本脚本在**同一会话内交错**切换运行时开关（容器内 `/tmp` 文件轮询，0.25 s 生效），
消除"会话抽签"这个最大的混淆因素。**v4 的 4 个负结果就是用这个工具判的**
（`MOE_ZERO` / `MOE_NONFINITE` / `LOCAL_OWNER` / `HCCL_DET`）。

### 用法（宿主机；服务已就绪，且镜像里带对应 patch）
```bash
# Engram local-owner 慢路径 vs 快路径
python3 tools/interleave_ab.py --container dsv41-a2 --knob owner --values fast,on --n 12
# MoE 无效行全量清零（负结果臂）
python3 tools/interleave_ab.py --container dsv41-a2 --knob zero  --values 0,1  --n 12
# MoE 只清零非有限元素（需要 MOE_NF 挂载；否则写文件无效）
python3 tools/interleave_ab.py --container dsv41-a2 --knob nf    --values 0,1  --n 24
```
输出末行 `clean 2x2 = [[...],[...]] fisher_p = ...`（**p 用双侧口径**，与
`tools/fisher_recheck.py` 一致；相同表必为 1.0）。

### 判据（**必须同时报两条线**）
* `clean(pos0>=0.8)` 的计数比 —— 这是"数值是否干净"；
* `A_med` 与 `uniq2` —— A 高可能是复读吸引子（A 与质量**反相关**），所以 A **不能**单独下结论。
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

_METRIC_LINE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)')
_POS_METRIC = "vllm:spec_decode_num_accepted_tokens_per_pos"
_DRAFT_NAMES = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_drafts")
_ACCEPT_NAMES = ("vllm:spec_decode_num_accepted_tokens_total",
                 "vllm:spec_decode_num_accepted_tokens")

# 容器内热切换文件（与 scripts/serve_a2.sh / A3-node1 完全一致）
KNOB_PATH = {
    "owner": "/tmp/v41_engram_localowner",
    "zero": "/tmp/v41_moe_zero_file",
    "nf": "/tmp/v41_moe_nf",
}


def http(base, path, payload, timeout=900.0):
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parse_metrics(text):
    """把 /metrics 里每个指标的所有样本**求和**（vLLM 多进程 exporter 会分进程暴露）。"""
    out = {}
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        m = _METRIC_LINE.match(raw)
        if not m:
            continue
        try:
            out[m.group("name")] = out.get(m.group("name"), 0.0) + float(m.group("value"))
        except ValueError:
            continue
    for name in list(out):
        if not name.endswith("_total") and f"{name}_total" in out:
            out[name + "_total"] = out.get(name + "_total", 0.0) + out[name]
    return out


def counter(m, *names):
    for n in names:
        if n in m:
            return float(m[n])
    return 0.0


def per_pos(text):
    out = {}
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw.startswith(_POS_METRIC):
            continue
        m = _METRIC_LINE.match(raw)
        if not m:
            continue
        mm = re.search(r'position="([^"]*)"', m.group("labels") or "")
        if not mm:
            continue
        try:
            out[mm.group(1)] = out.get(mm.group(1), 0.0) + float(m.group("value"))
        except ValueError:
            continue
    return out


def uniq2(t):
    g = [t[j:j + 2] for j in range(max(len(t) - 1, 0))]
    return (len(set(g)) / len(g)) if g else 1.0


def fetch(base):
    with urllib.request.urlopen(base + "/metrics", timeout=20) as r:
        return r.read().decode()


def tokenize_file(base, path, model, timeout=900.0):
    text = open(path, encoding="utf-8", errors="ignore").read()
    return http(base, "/tokenize", {"model": model, "prompt": text}, timeout)["tokens"]


def build_prompt(base, prefix, suffix, target, model):
    pids = tokenize_file(base, prefix, model, 900.0)
    sids = tokenize_file(base, suffix, model, 900.0)
    return pids[:max(1, target - len(sids))] + sids


def set_knob(docker, container, knob, value):
    path = KNOB_PATH[knob]
    subprocess.run(docker + ["exec", container, "bash", "-c", f"printf '%s' {value} > {path}"],
                   check=True, capture_output=True, text=True)
    got = subprocess.run(docker + ["exec", container, "bash", "-c", f"cat {path}"],
                         capture_output=True, text=True).stdout.strip()
    return got


def one(base, ids, max_tokens, model="deepseek-v41"):
    raw0 = fetch(base); m0 = parse_metrics(raw0); pp0 = per_pos(raw0)
    d0 = counter(m0, *_DRAFT_NAMES); a0 = counter(m0, *_ACCEPT_NAMES)
    body = json.dumps({"model": model, "prompt": ids, "max_tokens": max_tokens,
                       "temperature": 0.0, "top_p": 1.0, "seed": 1234, "ignore_eos": True,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(base + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); tf = None; pieces = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            s = line.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            d = s[5:].strip()
            if d == "[DONE]":
                break
            try:
                o = json.loads(d)
            except Exception:
                continue
            for ch in o.get("choices") or []:
                tx = ch.get("text") or ""
                if tx:
                    if tf is None:
                        tf = time.time()
                    pieces.append(tx)
    tend = time.time()
    raw1 = fetch(base); m1 = parse_metrics(raw1); pp1 = per_pos(raw1)
    d1 = counter(m1, *_DRAFT_NAMES); a1 = counter(m1, *_ACCEPT_NAMES)
    steps = d1 - d0
    A = 1 + (a1 - a0) / steps if steps else float("nan")
    pos = [((pp1.get(str(k), 0.0) - pp0.get(str(k), 0.0)) / steps) if steps else 0.0
           for k in range(5)]
    ms = (tend - (tf or t0)) * 1000 / steps if steps else float("nan")
    return {"A": A, "pos": pos, "steps": steps, "ms": ms,
            "uniq2": uniq2("".join(pieces)), "text": "".join(pieces)[:80]}


def _fisher():
    """复用包内带自检的实现（`tools/fisher_recheck.py`），**不要**自己再写一遍。"""
    sys.path.insert(0, HERE)
    from fisher_recheck import fisher  # type: ignore

    def two_sided(a, b):
        _, _, p2, _ = fisher(a[0], a[1], b[0], b[1])
        return p2
    return two_sided


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--container", default="dsv41-a2")
    ap.add_argument("--knob", required=True, choices=["owner", "zero", "nf"])
    ap.add_argument("--values", required=True, help="逗号分隔，如 fast,on 或 0,1")
    ap.add_argument("--n", type=int, default=12, help="每臂轮数（交错 n 轮 × 每轮遍历所有臂）")
    ap.add_argument("--tokens", type=int, default=131072)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--settle", type=float, default=1.2, help="切换后等待生效秒数（文件轮询 0.25 s）")
    ap.add_argument("--prefix", default=os.path.join(PKG, "data", "hongloumeng.txt"))
    ap.add_argument("--suffix", default=os.path.join(PKG, "data", "hlm", "suffix_quote.txt"))
    ap.add_argument("--label", default="iab")
    ap.add_argument("--docker", default="", help="docker 命令（默认 docker；失败自动试 sudo -n docker）")
    a = ap.parse_args()

    docker_cmd = a.docker.split() if a.docker else ["docker"]
    try:
        subprocess.run(docker_cmd + ["ps", "--format", "{{.Names}}"],
                       check=True, capture_output=True, text=True)
    except Exception:
        docker_cmd = ["sudo", "-n", "docker"]

    vals = a.values.split(",")
    ids = build_prompt(a.url, a.prefix, a.suffix, a.tokens, "deepseek-v41")
    print(f"[iab] label={a.label} knob={a.knob} values={vals} n={a.n} tokens={len(ids)}", flush=True)

    res = {v: [] for v in vals}
    for i in range(a.n):
        for v in vals:
            got = set_knob(docker_cmd, a.container, a.knob, v)
            time.sleep(a.settle)
            r = one(a.url, ids, a.max_tokens)
            r["knob_echo"] = got
            res[v].append(r)
            print(f"[iab] {a.label} round{i+1} {a.knob}={v}(echo={got}) A={r['A']:.3f} "
                  f"pos0={r['pos'][0]:.3f} steps={r['steps']:.0f} ms={r['ms']:.1f} "
                  f"uniq2={r['uniq2']:.2f} pos={[round(x, 3) for x in r['pos']]}", flush=True)
            print(f"[iab]    text={r['text']!r}", flush=True)

    print(f"\n[iab] ===== summary {a.knob} =====", flush=True)
    for v in vals:
        rs = res[v]
        clean = sum(1 for r in rs if r["pos"][0] >= 0.8)
        steep = sum(1 for r in rs if r["A"] >= 3.0 and r["pos"][0] >= 0.8)
        As = sorted(r["A"] for r in rs); mss = sorted(r["ms"] for r in rs)
        print(f"[iab] {a.knob}={v}: n={len(rs)} clean(pos0>=0.8)={clean}/{len(rs)} "
              f"steepA={steep}/{len(rs)} A_med={As[len(As)//2]:.3f} A_min={min(As):.3f} "
              f"A_max={max(As):.3f} ms_med={mss[len(mss)//2]:.1f}", flush=True)
    if len(vals) == 2:
        v0, v1 = vals
        t = [[sum(1 for r in res[v0] if r["pos"][0] >= 0.8),
              sum(1 for r in res[v0] if r["pos"][0] < 0.8)],
             [sum(1 for r in res[v1] if r["pos"][0] >= 0.8),
              sum(1 for r in res[v1] if r["pos"][0] < 0.8)]]
        p2 = _fisher()(t[0], t[1])
        print(f"[iab] clean 2x2 = {t}  fisher_p(two-sided) = {p2:.4f}", flush=True)
        print("[iab] ⚠️ 相同表必为 1.0；若见 0.0000 说明 Fisher 实现有 bug（见 tools/fisher_recheck.py）",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
