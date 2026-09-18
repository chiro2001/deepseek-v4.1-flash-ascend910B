#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""长上下文检索探针 —— 目前**最灵敏**的长上下文正确性判据。

## 为什么需要它

仓库既有的长上下文测试都是"能不能发出工具调用"。实测发现那**太粗**：
在 97K token 上它可以连续 10/10 通过，而同一时刻"检索文档中间的一个唯一事实"
已经 0/10。也就是说**工具调用测试会漏掉已经严重退化的服务**。

本探针把探针事实（needle）埋在长文档正中间，只问一个答案唯一的问题，
用字符串精确匹配判分 —— 无歧义、可自动统计、对退化敏感。

## 用法

    python3 longctx_retrieval.py --base http://127.0.0.1:8020 \\
        --reps 10 --out results/needle.json

    # 指定剂量点（token 数）
    python3 longctx_retrieval.py --tokens 10000,20000,40000,60000,80000

## 关键设计（都是踩过的坑）

1. **每个样本换一个 nonce**，让 prompt 内容不同。
   实测同一长度下"内容"决定了成败（同一 nonce 重复结果稳定，换 nonce 结果翻转），
   所以只测一个内容会得到毫无代表性的数字。
2. **固定长度的文本用"唯一 n-gram"构造**（编号 + 大数乘积），
   避免自然语料的高频重复干扰。
3. **needle 放在正中间**（50%），不放开头/结尾 —— 短程注意力会掩盖问题。
4. 本文件**不含真实控制符**（DSML 等一律用 \\u 转义构造）。

## 基线（<test-host>，2026-09-18，每档 10 个 nonce）

| prompt_tokens | 通过率 |
|---:|---:|
| 10,396 | 10/10 |
| 20,318 | 8/10 |
| 40,163 | 7/10 |
| 60,011 | 5/10 |
| 79,856 | 3/10 |

对照：**官方 `api.deepseek.com` 在同 4 个长度上 12/12 全部通过**（含 252K token）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
import uuid

# 用转义构造 needle，避免本文件被分词器当成控制符来源
MARK = "\u3010\u91cd\u8981\u6807\u8bb0\u3011"
NEEDLE = f"\n{MARK}\u672c\u9879\u76ee\u7684\u5185\u90e8\u4ee3\u53f7\u662f ZQ-7749\u3002\n"
ANSWER = "ZQ-7749"

# 实测：本构造的文本约 0.4434 token/字符
TOK_PER_CHAR = 0.4434


def tool_read():
    return {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }


def uniq_text(nchar: int, nonce: str) -> str:
    """构造 n-gram 几乎不重复的长文本。"""
    parts = [f"\u4f1a\u8bdd{nonce}\u5f00\u59cb\u3002"]
    tot = len(parts[0])
    i = 0
    while tot < nchar:
        p = (f"\u8bb0\u5f55\u7f16\u53f7{i:08d}\u7684\u5185\u5bb9\u662f\uff1a"
             f"{i * 7919:08d}-{i * 104729:010d}-{i * 15485863:012d}\u3002")
        parts.append(p)
        tot += len(p)
        i += 1
    return "".join(parts)[:nchar]


def build(nchar: int, nonce: str):
    doc = uniq_text(nchar, nonce)
    cut = len(doc) // 2
    body = doc[:cut] + NEEDLE + doc[cut:]
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "\u8bf7\u53ea\u56de\u7b54\u4e00\u4e2a\u95ee\u9898\uff0c\u4e0d\u8981\u89e3\u91ca\u3002"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "read",
                             "arguments": json.dumps({"path": "README.md"})},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": body},
        {"role": "user",
         "content": "\u6839\u636e\u4e0a\u9762\u8bfb\u53d6\u5230\u7684\u6587\u6863\uff0c"
                    "\u672c\u9879\u76ee\u7684\u5185\u90e8\u4ee3\u53f7\u662f\u4ec0\u4e48\uff1f"
                    "\u53ea\u56de\u7b54\u4ee3\u53f7\u672c\u8eab\u3002"},
    ]


def probe(base: str, target_tok: int, reps: int, max_tokens: int,
          temperature: float | None, timeout: int) -> dict:
    nchar = int(target_tok / TOK_PER_CHAR) + 200
    hits, pat, tokens, errs, outs = 0, [], None, 0, []
    t0 = time.time()
    for _ in range(reps):
        body = {
            "model": os.environ.get("GATE_MODEL", "deepseek-v41"),
            "messages": build(nchar, uuid.uuid4().hex[:6]),
            "tools": [tool_read()],
            "max_tokens": max_tokens,
            "reasoning_effort": "high",
        }
        if temperature is not None:
            body["temperature"] = temperature
        req = urllib.request.Request(
            base.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                x = json.loads(r.read().decode())
        except Exception:
            errs += 1
            pat.append("E")
            continue
        tokens = x["usage"]["prompt_tokens"]
        m = x["choices"][0]["message"]
        txt = (m.get("content") or "") + (m.get("reasoning_content") or "")
        ok = ANSWER in txt or ANSWER.replace("-", "") in txt
        hits += ok
        pat.append("Y" if ok else "N")
        outs.append(x["usage"]["completion_tokens"])
    valid = reps - errs
    lo, hi = (0.0, 0.0)
    if valid:
        z, p = 1.96, hits / valid
        d = 1 + z * z / valid
        c = (p + z * z / (2 * valid)) / d
        h = z * math.sqrt(p * (1 - p) / valid + z * z / (4 * valid * valid)) / d
        lo, hi = max(0.0, c - h), min(1.0, c + h)
    return {
        "target_tokens": target_tok,
        "actual_prompt_tokens": tokens,
        "reps": reps,
        "hit": hits,
        "valid": valid,
        "errors": errs,
        "hit_rate": round(hits / valid, 4) if valid else None,
        "hit_rate_ci95": [round(lo, 4), round(hi, 4)],
        "pattern": "".join(pat),
        "out_tokens_median": sorted(outs)[len(outs) // 2] if outs else None,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8020")
    ap.add_argument("--tokens", default="10000,20000,40000,60000,80000",
                    help="剂量点（prompt token 数），逗号分隔")
    ap.add_argument("--reps", type=int, default=10, help="每个剂量点的样本数")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pts = [int(x) for x in a.tokens.split(",") if x.strip()]
    print(f"[needle] base={a.base} reps={a.reps} points={pts}")
    results = []
    for t in pts:
        r = probe(a.base, t, a.reps, a.max_tokens, a.temperature, a.timeout)
        results.append(r)
        print(f"  {t:>7} tok (实际 {r['actual_prompt_tokens']}): "
              f"{r['hit']}/{r['valid']}  {r['pattern']}  "
              f"ci95={r['hit_rate_ci95']}", flush=True)

    out = {
        "base": a.base, "reps": a.reps, "max_tokens": a.max_tokens,
        "temperature": a.temperature,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "points": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[needle] 写入 {a.out}")

    # 供报告直接引用的表
    print("\n| prompt_tokens | 命中 | 命中率 | ci95 |")
    print("|---:|---:|---:|---|")
    for r in results:
        print(f"| {r['actual_prompt_tokens']} | {r['hit']}/{r['valid']} | "
              f"{r['hit_rate']} | {r['hit_rate_ci95']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
