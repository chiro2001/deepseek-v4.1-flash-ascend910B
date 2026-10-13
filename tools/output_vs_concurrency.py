#!/usr/bin/env python3
"""同一批 prompt 在 N=1 与 N=k 下的**生成文本是否相同**（决定性判定）。

为什么关键：`bench_concurrency` 是 `temperature=0` + `ignore_eos`，即贪心解码。
同一个 prompt 在 N=1 与 N=8 下，如果生成文本**相同**，那么接受长度 A 的任何下降
都只能来自「draft 本身在多请求下变差」⇒ 是可修的性能 bug；
如果文本**不同**，说明批内数值差异已经改变了输出 ⇒ 是数值/复现性问题，
「A 下降」只是不同文本的副产品，不能靠调 draft 修。

用法: output_vs_concurrency.py <port> <concurrency> [prompt_tokens] [max_tokens] [suffix_dir]
输出：逐条 prompt 的报告（N=1 vs N=k 的前缀是否一致、分叉位置、各自 SHA）。
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 19210
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 8
PROMPT_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
MAX_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 128
SUFFIX_DIR = sys.argv[5] if len(sys.argv) > 5 else ""

BASE = f"http://127.0.0.1:{PORT}"
CORPUS = "/home/l00886679/cedpd-repo/data/hongloumeng.txt"
MARK = "正文"


def load_suffixes() -> list[str]:
    if not SUFFIX_DIR:
        return ["\n\n请继续写下去。"]
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
    per = PROMPT_TOK * 2
    step = max(1, (len(text) - per) // max(1, n))
    return [text[i * step : i * step + per] + sufs[i % len(sufs)] for i in range(n)]


def fire(prompt: str) -> str:
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
    return (d.get("choices", [{}])[0].get("message", {}).get("content") or "")


def main() -> int:
    prompts = make_prompts(max(CONC, 4))
    print(f"{len(prompts)} 条 prompt；比较 N=1（串行）与 N={CONC}（并发）的生成文本")
    serial = [fire(p) for p in prompts]
    if CONC > 1:
        with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
            conc = list(ex.map(fire, prompts))
    else:
        conc = serial
    same = 0
    for i, (a, b) in enumerate(zip(serial, conc)):
        if a == b:
            same += 1
            print(f"  [{i}] 完全相同（{len(a)} 字符）")
            continue
        n = min(len(a), len(b))
        k = next((j for j in range(n) if a[j] != b[j]), n)
        print(
            f"  [{i}] **不同**：首个差异在第 {k} 个字符"
            f"（{k/max(1,n)*100:.1f}%）"
            f" len {len(a)} vs {len(b)}  sha {hashlib.sha256(a.encode()).hexdigest()[:8]}"
            f" / {hashlib.sha256(b.encode()).hexdigest()[:8]}"
        )
    print()
    print(f"★ 完全相同的 prompt：{same}/{len(prompts)}")
    if same == len(prompts):
        print("  ⇒ 输出与并发**无关** ⇒ 接受长度若仍下降，必是 draft 在多请求下变差（可修 bug）")
    else:
        print("  ⇒ 输出与并发**有关** ⇒ 批内数值差异已改变结果；A 的变化是文本变化的副产品")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
