#!/usr/bin/env python3
"""并发 2 路、各带**不同针**的正确性探针（动态 K 的 K=0 路径专用）。

为什么必须单独做这一条：
  * `tools/ced_pd_acceptance.py` 的四针用例是**顺序**发的 ⇒ 在动态 K 的表
    `1,1,7;2,32,0` 下全部走 **K=7（query_len=8）**；它**覆盖不到 K=0 那条路径**；
  * 并发 ≥2 才会切到 K=0，如果两路用**同一根针**，即使两路都把对方的上下文算错、
    或都只答对了"某一路"的内容，判据也看不出来。
  ⇒ 两路必须**各带不同针、并且互相不能答出对方的针**，这才能证明 K=0 的步
    没有把两路的 KV/元数据串在一起（本仓记过的 `dsa_v1` gather 行数错位正是这类失效）。

判据（全部满足才算 PASS）：
  1. 两路 HTTP 200 且都自然停止（有 finish_reason）；
  2. 第 i 路的回答**包含**自己的针；**不包含**另一路的针；
  3. 两路的回答**不相同**（若相同说明串了）；
  4. 重复 R 轮全部通过（默认 3 轮，取"全过"而不是"过一半"）。

只依赖标准库。复用 `tools/ced_pd_acceptance.py` 的针文本与嵌入/校准函数，保证与
既有四针证据**同口径**。

用法：
    python3 probe_concurrent_needles.py --base-url http://127.0.0.1:19210 \
        --model deepseek-v41 --corpus data/hongloumeng.txt \
        --context-tokens 131072 --repeat 3

退出码：0 = 全过；1 = 有失败；2 = 用法/连接错误。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))

try:
    from ced_pd_acceptance import (  # type: ignore
        KEY_TEXT,
        NEEDLE_Q,
        count_tokens,
        embed_needles,
        judge,
        load_corpus,
    )
except Exception as exc:  # pragma: no cover
    print(f"[usage][FAIL] 无法从 tools/ced_pd_acceptance.py 导入助手：{exc!r}")
    raise SystemExit(2)


def ask(url: str, model: str, prompt: str, max_tokens: int, timeout: float):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    choice = data["choices"][0]
    return {
        "content": (choice.get("message", {}).get("content") or ""),
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": data.get("usage", {}).get("completion_tokens"),
        "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
    }


def build_prompt(corpus: str, tokenize_url: str, model: str, target: int, key: str, offset: int) -> str:
    """把语料裁到约 target token，并插入指定针。用 /tokenize 校准。"""
    # 先按字符粗裁，再二分收敛——与 ced_pd_acceptance 同思路，但这里只需"够长且两路不同"。
    lo, hi = 1, len(corpus)
    best = corpus[: lo]
    for _ in range(24):
        mid = (lo + hi) // 2
        candidate = embed_needles(corpus[offset : offset + mid], [key])
        try:
            n = count_tokens(tokenize_url, model, candidate)
        except Exception as exc:
            raise SystemExit(
                f"[pre][FAIL] 校准过程中 /tokenize 失败（key={key}, chars={mid}）：{exc!r}"
            )
        if n < target:
            lo = mid + 1
            best = candidate
        else:
            hi = mid
        if hi - lo <= 64:
            break
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="实例 URL（单实例时同时用于 /tokenize）")
    ap.add_argument("--tokenize-url", default="", help="留空则用 --base-url")
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default="data/hongloumeng.txt")
    ap.add_argument("--context-tokens", type=int, default=131072)
    ap.add_argument("--needles", default="A,D", help="两路各用哪根针（默认 A 与 D）")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tokenize_url = args.tokenize_url or args.base_url
    keys = [k.strip() for k in args.needles.split(",") if k.strip()]
    if len(keys) != 2:
        print("[usage][FAIL] --needles 必须正好两根，例如 A,D")
        return 2
    for k in keys:
        if k not in KEY_TEXT:
            print(f"[usage][FAIL] 未知针 {k!r}（合法：{sorted(KEY_TEXT)}）")
            return 2

    if not os.path.isfile(args.corpus):
        print(f"[usage][FAIL] 语料不存在：{args.corpus}")
        return 2
    corpus = load_corpus(args.corpus)
    # 两路用**不同起点**，避免两路的上下文完全一样（否则"答对"也可能是串了）
    offset_b = max(0, (len(corpus) // 2) - args.context_tokens)

    print(f"[cfg] base={args.base_url} model={args.model} ctx≈{args.context_tokens} "
          f"needles={keys} repeat={args.repeat}", flush=True)

    # 预检：连不上要**干净退出 2**，不要在 build_prompt 里抛裸 traceback
    # （本仓纪律：脚本要能把"服务不可达"与"判据失败"分开）。
    try:
        probe = count_tokens(tokenize_url, args.model, "自检")
        print(f"[pre] /tokenize OK（自检串计 {probe} token）", flush=True)
    except Exception as exc:
        print(f"[pre][FAIL] /tokenize 不可达或异常：{exc!r}\n"
              f"  ⇒ 这是**连接/用法问题**（VPN？端口？），不是正确性判据失败。exit=2")
        return 2

    reports = []
    all_pass = True
    for rep in range(args.repeat):
        prompts = [
            build_prompt(corpus, tokenize_url, args.model, args.context_tokens, keys[0], 0),
            build_prompt(corpus, tokenize_url, args.model, args.context_tokens, keys[1], offset_b),
        ]
        # ★ 题面用 cz_pd_acceptance 的 NEEDLE_Q（官方口径）——**绝不能把针文本写进问题**，
        #   否则"照抄题面"就能骗过子串判据（judge() 为此专门拒绝含"运维备忘/请只回复"的答案）。
        questions = [NEEDLE_Q[k][0] for k in keys]
        expected = [NEEDLE_Q[k][1] for k in keys]
        full = [p + "\n\n" + q for p, q in zip(prompts, questions)]

        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(ask, args.base_url, args.model, f, args.max_tokens, args.timeout) for f in full]
            try:
                outs = [f.result() for f in futs]
            except Exception as exc:
                print(f"[rep{rep}][FAIL] 请求异常：{exc!r}", flush=True)
                all_pass = False
                reports.append({"rep": rep, "error": repr(exc)})
                continue

        checks = []
        for i in range(2):
            other = 1 - i
            txt = outs[i]["content"]
            # 官方判据：出现自己的码 **且** 不复述题面/针文本
            ok_own = judge(txt, expected[i])
            # 反向判据：**不能**出现对方的码（串了 KV/元数据最典型的症状）
            leak = expected[other] in txt
            checks.append(
                {
                    "i": i,
                    "finish": outs[i]["finish_reason"],
                    "ctok": outs[i]["completion_tokens"],
                    "ok_own": ok_own,
                    "leak_other": leak,
                    "answer": txt[:160],
                }
            )
        same = outs[0]["content"] == outs[1]["content"]
        ok = (
            all(c["finish"] for c in checks)
            and all(c["ok_own"] for c in checks)
            and not any(c["leak_other"] for c in checks)
            and not same
        )
        all_pass = all_pass and ok
        print(f"[rep{rep}] {'PASS' if ok else 'FAIL'} "
              f"own={[c['ok_own'] for c in checks]} leak={[c['leak_other'] for c in checks]} "
              f"same={same}", flush=True)
        for c in checks:
            print(f"    #{c['i']} finish={c['finish']} ctok={c['ctok']} ans={c['answer']!r}", flush=True)
        reports.append({"rep": rep, "ok": ok, "checks": checks, "same": same})

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\nVERDICT: {verdict}  （并发 2 路各带不同针，{args.repeat} 轮全过才 PASS）")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"verdict": verdict, "needles": keys, "reports": reports}, fh,
                      ensure_ascii=False, indent=2)
        print(f"[out] {args.out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
