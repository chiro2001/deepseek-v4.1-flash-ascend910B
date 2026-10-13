#!/usr/bin/env python3
"""**接受长度是否随并发下降**的对照实验（同一批 prompt，只改并发）。

动机：`bench_concurrency` 的 N=1 与 N=8 用的是**同一批 prompt**，
但实测接受长度 A 从 2.87 掉到 2.52（−12%）。若这是可修的（例如 draft 在
多请求下拿到错的 KV / 掩码），修好就能在 N=8 直接换 ~10% 吞吐。

本脚本把同一批 prompt 先**串行**打一遍、再**并发**打一遍，
用 `/metrics` 的 `spec_decode_num_accepted_tokens_per_pos_total` 差分算每位置接受率。
两种模式的总 forward 次数接近，因此可直接比较。

用法: accept_vs_concurrency.py <port> [n_reqs] [max_tokens] [prompt_tokens] [suffix_dir]

`suffix_dir` 给出后，prompt 会**轮换该目录下的所有后缀文件**（与
`bench_concurrency.py` 的做法一致 —— 默认 `data/hlm_local/`，含
continue / extract / qa / quote 四种任务，它们的接受率差别很大）。
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 19210
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
MAX_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 192
PROMPT_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
SUFFIX_DIR = sys.argv[5] if len(sys.argv) > 5 else ""

BASE = f"http://127.0.0.1:{PORT}"
CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"
MARK = "正文"


def get_metrics() -> dict:
    """返回 {'<name>|<label_kv>': value}，同时给出不带标签的裸名（若唯一）。"""
    txt = urllib.request.urlopen(f"{BASE}/metrics", timeout=30).read().decode()
    out: dict[str, float] = {}
    for line in txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        key, _, val = line.rpartition(" ")
        try:
            v = float(val)
        except ValueError:
            continue
        if "{" in key and key.endswith("}"):
            name, _, labels = key.partition("{")
            out[f"{name}|{labels[:-1]}"] = v
        else:
            out[key] = v
    return out


def _pick(m: dict, name: str, must: tuple[str, ...] = ()) -> float:
    """取某个 metric：优先精确匹配标签，其次取唯一一条裸名/带标签记录。"""
    if name in m:
        return m[name]
    hits = [(k, v) for k, v in m.items() if k.startswith(name + "|") and all(s in k for s in must)]
    if len(hits) == 1:
        return hits[0][1]
    return sum(v for _, v in hits)  # 多条时求和（单 engine 场景下通常只有一条）


def snap(m: dict) -> dict:
    d = {
        "accepted": _pick(m, "vllm:spec_decode_num_accepted_tokens_total"),
        "drafted": _pick(m, "vllm:spec_decode_num_draft_tokens_total"),
        "drafts": _pick(m, "vllm:spec_decode_num_drafts_total"),
    }
    for i in range(16):
        d[f"pos{i}"] = _pick(
            m, "vllm:spec_decode_num_accepted_tokens_per_pos_total", (f'position="{i}"',)
        )
    return d


def load_suffixes() -> list[str]:
    if not SUFFIX_DIR:
        return ["\n\n请继续写下去。"]
    import os

    out = []
    for name in sorted(os.listdir(SUFFIX_DIR)):
        if name.endswith(".txt"):
            with open(os.path.join(SUFFIX_DIR, name), encoding="utf-8") as fh:
                out.append(fh.read())
    return out or ["\n\n请继续写下去。"]


def make_prompts(n: int) -> list[str]:
    text = open(CORPUS, encoding="utf-8").read()
    if MARK in text:
        text = text.split(MARK, 1)[1]
    sufs = load_suffixes()
    per = PROMPT_TOK * 2  # 中文字符≈token 量级，够长即可（服务端不做精确校准）
    step = max(1, (len(text) - per) // max(1, n))
    return [
        text[i * step : i * step + per] + sufs[i % len(sufs)]
        for i in range(n)
    ]


def fire(prompt: str) -> tuple[int, float]:
    payload = {
        "model": "deepseek-v41",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOK,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return d.get("usage", {}).get("prompt_tokens", -1), d.get("usage", {}).get("completion_tokens", -1)


def phase(name: str, prompts: list[str], concurrent: bool) -> dict:
    m0 = snap(get_metrics())
    if concurrent:
        with cf.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
            list(ex.map(fire, prompts))
    else:
        for p in prompts:
            fire(p)
    m1 = snap(get_metrics())
    out = {k: m1[k] - m0[k] for k in m0}
    acc, drf, nst = out["accepted"], out["drafted"], out["drafts"]
    # A = 1 + 每步平均接受数（vLLM 的 "Mean acceptance length" 口径）
    a_len = 1.0 + acc / nst if nst else 0.0
    print(f"--- {name}: drafts={nst:.0f} accepted={acc:.0f} drafted={drf:.0f} "
          f"接受率={acc/drf*100 if drf else 0:.1f}%  **A={a_len:.3f}**")
    pos = [out[f"pos{i}"] / drf * 5 if drf else 0 for i in range(5)]
    print(f"    每位置接受率（×5 归一）: " + " ".join(f"{v:.3f}" for v in pos))
    return out


def main() -> int:
    prompts = make_prompts(N)
    print(f"prompt {len(prompts)} 条 × ~{PROMPT_TOK} tok，max_tokens={MAX_TOK}，"
          f"后缀 {'轮换 ' + SUFFIX_DIR if SUFFIX_DIR else '单一'}")
    seq = phase("串行（N=1）", prompts, concurrent=False)
    con = phase(f"并发（N={N}）", prompts, concurrent=True)
    if seq["drafts"] and con["drafts"]:
        r1 = seq["accepted"] / seq["drafted"]
        r2 = con["accepted"] / con["drafted"]
        a1 = 1.0 + seq["accepted"] / seq["drafts"]
        a2 = 1.0 + con["accepted"] / con["drafts"]
        print()
        print(f"★ 接受率：串行 {r1*100:.1f}% vs 并发 {r2*100:.1f}%  → 相对 {r2/r1-1:+.1%}")
        print(f"★ A（平均接受长度）：串行 {a1:.3f} vs 并发 {a2:.3f}  → 相对 {a2/a1-1:+.1%}")
        print("  若并发显著更低 ⇒ 存在批量相关的机制（draft 预测/KV/掩码或批内数值差异）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
