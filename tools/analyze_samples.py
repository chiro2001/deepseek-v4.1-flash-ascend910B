#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复算 EXPECTED_PERF.md 里的数字（直接读 p42_t4_quote_*.jsonl）。

用法：
    python3 tools/analyze_samples.py logs_meta/samples
    python3 tools/analyze_samples.py "logs_meta/samples/p42_t4_quote_131072_faB_128k_r*.jsonl"
"""

import json, glob, statistics, sys


def rows(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        for ln in open(f):
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            if '_meta' in d:
                continue
            out.append((f, d))
    return out


for pat in sys.argv[1:]:
    rs = rows(pat)
    ms = [d['ms_per_step'] for _, d in rs]
    A = [d['accept_length'] for _, d in rs]
    tok = [d['decode_tok_s'] for _, d in rs]
    print(f"== {pat}  n={len(rs)}")
    for f, d in rs:
        print(f"   {f.split('/')[-1]:52s} ms={d['ms_per_step']:7.3f} A={d['accept_length']:6.3f} "
              f"tps={d['decode_tok_s']:7.2f} ttft={d.get('ttft_s')} prefill_tps={d.get('prefill_tok_s')}")
    if ms:
        print(f"   MED ms={statistics.median(ms):.3f} A={statistics.median(A):.3f} "
              f"tps={statistics.median(tok):.2f} | minA={min(A)} maxA={max(A)} ratio={max(A)/min(A):.2f} "
              f"| msmin={min(ms):.3f} msmax={max(ms):.3f} spread={(max(ms)-min(ms))/statistics.median(ms)*100:.1f}%")
