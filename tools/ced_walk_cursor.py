#!/usr/bin/env python3
"""用**短请求**逐步推动 D 的分配游标，把"块号上界 B"夹出来。

已知规律：请求失败 ⟺ 它的块列表出现 id ≥ B 的块（B ∈ (28204, 29560]，C=29600）。
短请求（约 60 prompt tokens）每次只推进 ~22 块、耗时 ~0.6 s，
因此可以密集采样 (max_id, verdict) 对，找到结果翻转点。

★★ 修过的坑：初版用"最新的 dump 文件"判断本次请求的块列表，而上一轮遗留的
dump 更早落盘但更晚被 glob 到 ⇒ 读到**别人的**块列表（于是算出一个 3.6M token 的
非法跳跃）。现在改为：发请求前记录已存在的 dump 集合，发完后**轮询等待出现新文件**，
只在出现新文件时才读，并且带超时。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import urllib.request

PROMPT = "校验码是 ZQ7K-3341。请只回复这个校验码。"
EXPECT = "ZQ7K-3341"
SYSTEM = ("你是一个严谨的中文助手。回答要直接、简短；"
          '被要求"只给密码/口令/访问码/令牌"时就只输出它本身，不要解释。')


def send(url, model, timeout=300.0):
    payload = {"model": model,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": PROMPT}],
               "max_tokens": 64, "temperature": 0.0, "stream": False}
    body = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    ch = (data.get("choices") or [{}])[0]
    content = (ch.get("message") or {}).get("content")
    return (isinstance(content, str) and EXPECT in content), content, data.get("usage") or {}


def wait_new_dump(dump_dir, before, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        now = set(glob.glob(os.path.join(dump_dir, "*_g0.txt")))
        fresh = now - before
        if fresh:
            f = max(fresh, key=os.path.getmtime)
            ids = [int(x) for x in open(f) if x.strip()]
            return f, ids, now
        time.sleep(0.3)
    return None, None, before


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--steps", type=int, default=1400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "walk_log.json")
    log = []
    before = set(glob.glob(os.path.join(args.dump_dir, "*_g0.txt")))
    prev = None
    for i in range(1, args.steps + 1):
        t0 = time.time()
        try:
            ok, content, usage = send(args.url, args.model)
        except Exception as err:  # noqa: BLE001
            print(f"step {i}: 请求失败 {err!r}", flush=True)
            time.sleep(2)
            continue
        f, ids, before = wait_new_dump(args.dump_dir, before)
        entry = {"step": i, "passed": ok, "content": content,
                 "prompt": usage.get("prompt_tokens"),
                 "max_id": max(ids) if ids else None,
                 "n": len(ids) if ids else None,
                 "wall_s": round(time.time() - t0, 2),
                 "dump": os.path.basename(f) if f else None}
        log.append(entry)
        if i % 20 == 0 or prev is None or prev["passed"] != ok:
            print(f"step {i:>4} max_id={entry['max_id']} n={entry['n']} "
                  f"{'PASS' if ok else 'FAIL'} wall={entry['wall_s']}s "
                  f"content={str(content)[:14]!r}", flush=True)
        json.dump(log, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        if prev is not None and prev["passed"] != ok and entry["max_id"] and prev["max_id"]:
            a, b = prev["max_id"], entry["max_id"]
            print(f"\n★ 翻转：max_id {a}（{'PASS' if prev['passed'] else 'FAIL'}）→ "
                  f"{b}（{'PASS' if ok else 'FAIL'}）", flush=True)
            break
        prev = entry
    else:
        print("\n达到步数上限未观察到翻转", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
