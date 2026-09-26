#!/usr/bin/env python3
"""二分 D 侧"块号上界 B"：找出"多大块号开始必然失败"。

已知规律（见 evidence/ced_swa_clip_ab_clip1_20260924/README.md 第 Q 节）：
  请求失败 ⟺ 它的块列表里出现 id ≥ B 的块，B ∈ (28204, 29560]（C=29600 时实测）。

思路（不重启服务，全部跑在现成 D 上）：
  1. 发一个 22-token 短请求，读它刚落盘的 `[CED-BLOCK-DUMP]` 得到当前分配游标 c；
  2. 用**一次定制长度的请求**把游标推到 28200 附近（长度→R 的映射可精确构造）；
  3. 之后只用 22-token 请求：每次游标只前进 ~22 块、耗时 ~0.6 s，
     记录 (max_id, verdict)，直到结果翻转 ⇒ 把 B 夹到 ±22 块。

每一步都落盘请求体与结果，并打印 (max_id, verdict)。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import urllib.request

CORPUS_DEFAULT = "/tmp/hongloumeng.txt"
SHORT_PROMPT = "校验码是 ZQ7K-3341。请只回复这个校验码。"
SHORT_EXPECT = "ZQ7K-3341"
NEEDLE = "【运维备忘 D】构建机临时令牌是 RB9N-6014，用完即废。"
QUESTION = "运维备忘 D 里的构建机临时令牌是什么？只给令牌本身。"
SYSTEM = ("你是一个严谨的中文助手。回答要直接、简短；"
          '被要求"只给密码/口令/访问码/令牌"时就只输出它本身，不要解释。')
BLOCK = 128
EXTRA_GROUPS = 21


def post_json(url, payload, timeout):
    body = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def chat(url, user_text, timeout, max_tokens=64):
    payload = {"model": None, "messages": [{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": user_text}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
    return payload


def send(url, model, user_text, timeout, expect):
    payload = chat(url, user_text, timeout)
    payload["model"] = model
    data = post_json(url, payload, timeout)
    ch = (data.get("choices") or [{}])[0]
    content = (ch.get("message") or {}).get("content")
    usage = data.get("usage") or {}
    ok = isinstance(content, str) and expect in content
    return ok, content, usage


def newest_dump(dump_dir, seen):
    files = [f for f in glob.glob(os.path.join(dump_dir, "*_g0.txt")) if f not in seen]
    if not files:
        return None, None
    f = max(files, key=os.path.getmtime)
    ids = [int(x) for x in open(f) if x.strip()]
    return f, ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--tokenize", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--corpus", default=CORPUS_DEFAULT)
    ap.add_argument("--target-cursor", type=int, default=28200)
    ap.add_argument("--walk-limit", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    corpus = open(args.corpus, encoding="utf-8", errors="replace").read()
    seen: set[str] = set(glob.glob(os.path.join(args.dump_dir, "*_g0.txt")))
    log = []

    def record(tag, ok, content, usage, ids):
        entry = {"tag": tag, "passed": ok, "content": content,
                 "completion": usage.get("completion_tokens"),
                 "prompt_tokens": usage.get("prompt_tokens"),
                 "max_id": max(ids) if ids else None,
                 "min_id": min(ids) if ids else None, "n": len(ids) if ids else None}
        log.append(entry)
        json.dump(log, open(os.path.join(args.out, "log.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  [{tag}] {'PASS' if ok else 'FAIL'} prompt={entry['prompt_tokens']} "
              f"n={entry['n']} max_id={entry['max_id']} content={str(content)[:16]!r}", flush=True)
        return entry

    # ---------- 1) 测当前游标 ----------
    print("=== 1) 22-token 探针，测当前分配游标 ===", flush=True)
    ok, content, usage = send(args.url, args.model, SHORT_PROMPT, 300, SHORT_EXPECT)
    time.sleep(3)
    f, ids = newest_dump(args.dump_dir, seen)
    if f:
        seen.add(f)
    record("probe", ok, content, usage, ids)
    if not ids:
        print("没读到 block dump —— 无法继续（检查 V41_CED_BLOCK_DUMP_DIR）", flush=True)
        return 2
    cursor = max(ids)

    # ---------- 2) 用定制长度把游标推到 target ----------
    delta = (args.target_cursor - cursor) % 29600
    print(f"\n=== 2) 把游标从 {cursor} 推到 ~{args.target_cursor}（需前进 {delta} 块）===", flush=True)
    if delta < 100:
        print("  已经足够接近，跳过跳跃步", flush=True)
    else:
        # R = ceil((N-1)/128) + 21 ⇒ g0 = delta - 21
        g0 = max(1, delta - EXTRA_GROUPS)
        est_tokens = g0 * BLOCK
        # 校准：迭代调整字数使 /tokenize 落在目标
        chars = est_tokens
        body = ""
        for _ in range(6):
            body = (corpus * (chars // len(corpus) + 1))[:chars]
            cut = int(len(body) * 0.8)
            text = body[:cut] + "\n" + NEEDLE + "\n" + body[cut:] + "\n\n" + QUESTION
            got = post_json(args.tokenize, {"model": args.model, "prompt": text}, 300)
            n_tok = int(got.get("count") or 0)
            if abs(n_tok - (est_tokens + 60)) <= 200:
                break
            chars = max(1, round(chars * (est_tokens + 60) / max(1, n_tok)))
        ok, content, usage = send(args.url, args.model, text, 3600, "RB9N-6014")
        time.sleep(3)
        f2, ids2 = newest_dump(args.dump_dir, seen)
        if f2:
            seen.add(f2)
        record("jump", ok, content, usage, ids2)

    # ---------- 3) 22-token 逐步走，找翻转点 ----------
    print(f"\n=== 3) 用 22-token 请求逐步越过上界（最多 {args.walk_limit} 次）===", flush=True)
    prev = None
    for i in range(1, args.walk_limit + 1):
        ok, content, usage = send(args.url, args.model, SHORT_PROMPT, 300, SHORT_EXPECT)
        time.sleep(0.5)
        f3, ids3 = newest_dump(args.dump_dir, seen)
        if f3:
            seen.add(f3)
        e = record(f"walk{i}", ok, content, usage, ids3)
        if prev is not None and prev["passed"] != e["passed"]:
            a, b = prev["max_id"], e["max_id"]
            print(f"\n★ 翻转：max_id {a}（{'PASS' if prev['passed'] else 'FAIL'}）→ "
                  f"{b}（{'PASS' if e['passed'] else 'FAIL'}） ⇒ B ∈ ({min(a,b)}, {max(a,b)}]",
                  flush=True)
            break
        prev = e
    else:
        print("\n在步数上限内没有观察到翻转（B 可能高于当前 max_id）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
