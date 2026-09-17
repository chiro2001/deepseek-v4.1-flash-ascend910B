#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""multibatch_gate.py -- 多 batch / 多轮对话的正确性验证。

**为什么需要它**：v3 及以前的所有"并发"测试都是 **decode 并发**——
历史 GSM8K / C-Eval 用 `--conc 4 --serialize-prefill 1`，而该开关的语义是
"hold a global lock until first token (avoid concurrent prefills)"，
即 **prefill 被故意串行化**。⇒ 「多轮对话」与「prefill+decode 真同时在跑」**从未测过**。
本脚本补上这块空白，全部**不走 lmeval**，只用标准 OpenAI 接口 + 精确判据：

  [A] 多轮对话（顺序，单流）
      一个 N 轮的对话：第 1 轮埋入 3 个"针"（只有该轮说过的随机串），
      第 k 轮问其中一个。判据：能逐字复述 => 长历史 + 前缀复用没坏。
      同时记录每轮的 uniq2（复读判据）与耗时。

  [B] 并发 batch（同一批 item，conc=1 vs conc=N，**逐 item 比对**）
      同一组算术题（自带答案，可精确判分），先 conc=1 跑一遍，再 conc=N 跑一遍。
      判据：两组**逐 item 正确性一致**（不一致的 item 数 / 总 item 数）。
      这比"比较准确率"灵敏得多 —— 能发现"只有个别请求在某些并发排列下坏掉"。

  [C] 交错污染：并发批里混入一条**长上下文**请求，看短请求是否受影响。
      （模拟生产：一个 128K 请求 + 几个短请求同时在跑 ⇒ 真正的 prefill+decode 混合）

用法（在**宿主机**上跑；服务已就绪）:
  python3 tests/multibatch/multibatch_gate.py \
      --base http://127.0.0.1:8100 --model deepseek-v41 \
      --out results/mbg --rounds 8 --conc 8

只跑某几块：`--skip A,C`；不跑长请求（省时间）：`--skip A,B,C`… 或 `--long-ctx 0`。
"""
import argparse
import json
import os
import random
import re
import string
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_CORPUS = os.path.join(PKG, "data", "hongloumeng.txt")
DEFAULT_OUT = os.path.join(PKG, "results", "mbg")


def post(base, path, payload, timeout=900):
    req = urllib.request.Request(base.rstrip("/") + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def chat(base, model, messages, max_tokens=256, temperature=0.0, seed=1234):
    t0 = time.time()
    out = post(base, "/v1/chat/completions", {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "seed": seed,
    })
    dt = time.time() - t0
    txt = (out["choices"][0].get("message") or {}).get("content") or ""
    return txt, dt, out.get("usage") or {}


def uniq2(t):
    """复读判据：len(set(bigrams)) / len(bigrams)。1.0=完全不重复，<0.5 基本复读。"""
    if len(t) < 4:
        return 1.0
    g = [t[i:i + 2] for i in range(len(t) - 1)]
    return len(set(g)) / max(len(g), 1)


def rand_token(n=8):
    return "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))


# ------------------------------------------------------------------ [A] 多轮
def test_multiturn(base, model, rounds, out_dir, max_tokens=128):
    print(f"\n[mbg] ===== [A] 多轮对话  rounds={rounds} =====", flush=True)
    needles = {f"k{i}": rand_token(10) for i in range(3)}
    # 第 1 轮：把三个针塞进用户消息
    first = ("请记住下面三条凭据，后面我会考你：\n"
             + "\n".join(f"- {k} = {v}" for k, v in needles.items())
             + "\n\n只回复 OK。")
    messages = [{"role": "user", "content": first}]
    txt, dt, usage = chat(base, model, messages, max_tokens=16)
    messages.append({"role": "assistant", "content": txt})
    rec = [{"turn": 1, "role": "setup", "needle": None, "answer": txt,
            "ok": True, "uniq2": uniq2(txt), "dt": dt, "usage": usage}]
    print(f"[mbg] A turn1 setup dt={dt:.1f}s reply={txt[:40]!r}", flush=True)

    # 后续轮：每轮考一个针（上下文随之增长，模拟真实多轮对话）
    for i in range(2, rounds + 1):
        key = f"k{(i - 2) % 3}"
        ask = (f"第 {i} 问：请**逐字**输出 {key} 的值，只输出那个 10 位字符串，不要任何其他字符。")
        messages.append({"role": "user", "content": ask})
        txt, dt, usage = chat(base, model, messages, max_tokens=max_tokens)
        messages.append({"role": "assistant", "content": txt})
        want = needles[key]
        ok = want in txt
        u = uniq2(txt)
        rec.append({"turn": i, "probe": key, "want": want, "answer": txt,
                    "ok": ok, "uniq2": u, "dt": dt, "usage": usage})
        flag = "OK " if ok else "FAIL"
        print(f"[mbg] A turn{i:2d} probe={key} {flag} uniq2={u:.2f} dt={dt:5.1f}s "
              f"prompt_tok={usage.get('prompt_tokens')} ans={txt[:46]!r}", flush=True)

    hit = sum(1 for r in rec[1:] if r["ok"])
    n = len(rec) - 1
    # 把长历史整体再问一次（考验前缀复用/缓存的一致性：同一 prompt 前缀 + 新后缀）
    recalled = {}
    for k, v in needles.items():
        messages.append({"role": "user", "content": f"再确认一次 {k} 的值（只输出字符串）。"})
        txt, dt, usage = chat(base, model, messages, max_tokens=max_tokens)
        messages.append({"role": "assistant", "content": txt})
        recalled[k] = (v in txt)
        print(f"[mbg] A recall {k}: {'OK' if recalled[k] else 'FAIL'} ans={txt[:46]!r}", flush=True)

    res = {"test": "multiturn", "rounds": rounds, "needle_hits": hit, "needle_total": n,
           "recall": recalled, "final_prompt_tokens": usage.get("prompt_tokens"), "records": rec}
    with open(os.path.join(out_dir, "multiturn.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[mbg] A 结果: 轮内召回 {hit}/{n}, 末次全长召回 {sum(recalled.values())}/3, "
          f"最终 prompt_tokens={usage.get('prompt_tokens')}", flush=True)
    return res


# ------------------------------------------------------------- [B] 并发 batch
ITEMS = [
    # (问题, 期望答案)  —— 纯算术，可精确判分，且答案唯一
    ("计算 4729 * 13 的值，只输出数字。", "61477"),
    ("计算 8834 + 2917 的值，只输出数字。", "11751"),
    ("计算 6012 - 4783 的值，只输出数字。", "1229"),
    ("计算 9408 / 16 的值，只输出数字。", "588"),
    ("计算 37 * 41 + 19 的值，只输出数字。", "1536"),
    ("计算 1234 + 5678 + 9012 的值，只输出数字。", "15924"),
    ("计算 512 * 64 的值，只输出数字。", "32768"),
    ("计算 10000 - 8642 的值，只输出数字。", "1358"),
    ("计算 729 + 846 + 915 的值，只输出数字。", "2490"),
    ("计算 88 * 125 的值，只输出数字。", "11000"),
    ("计算 2048 / 8 的值，只输出数字。", "256"),
    ("计算 3333 + 4444 的值，只输出数字。", "7777"),
    ("计算 121 * 121 的值，只输出数字。", "14641"),
    ("计算 9999 - 1111 的值，只输出数字。", "8888"),
    ("计算 15 * 15 * 4 的值，只输出数字。", "900"),
    ("计算 777 * 3 的值，只输出数字。", "2331"),
]


def extract_num(t):
    m = re.findall(r"-?\d+", (t or "").replace(",", ""))
    return m[-1] if m else None


def run_items(base, model, items, conc, max_tokens=64, tag=""):
    res = [None] * len(items)

    def worker(idx_chunk):
        for i in idx_chunk:
            q, want = items[i]
            try:
                txt, dt, usage = chat(base, model, [{"role": "user", "content": q}],
                                      max_tokens=max_tokens)
                got = extract_num(txt)
                res[i] = {"i": i, "q": q, "want": want, "got": got, "ok": got == want,
                          "uniq2": uniq2(txt), "dt": dt, "raw": txt[:80],
                          "ptok": (usage or {}).get("prompt_tokens")}
            except Exception as exc:
                res[i] = {"i": i, "q": q, "want": want, "got": None, "ok": False,
                          "err": f"{type(exc).__name__}: {str(exc)[:120]}"}

    chunks = [list(range(j, len(items), conc)) for j in range(conc)]
    threads = [threading.Thread(target=worker, args=(c,)) for c in chunks]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    ok = sum(1 for r in res if r and r.get("ok"))
    print(f"[mbg] B conc={conc:2d} {tag}: {ok}/{len(items)} 正确  wall={wall:.1f}s", flush=True)
    return res, wall


def test_concurrency(base, model, conc, out_dir, items=None, max_tokens=64):
    items = items or ITEMS
    print(f"\n[mbg] ===== [B] 并发 batch  conc=1 vs conc={conc} =====", flush=True)
    seq, w1 = run_items(base, model, items, 1, max_tokens=max_tokens, tag="(串行基线)")
    con, wn = run_items(base, model, items, conc, max_tokens=max_tokens, tag="(并发)")
    diff = []
    for a, b in zip(seq, con):
        if a.get("ok") != b.get("ok"):
            diff.append({"i": a["i"], "q": a["q"], "serial_ok": a.get("ok"),
                         "conc_ok": b.get("ok"), "serial_raw": a.get("raw"),
                         "conc_raw": b.get("raw")})
    res = {"test": "concurrency", "conc": conc, "n": len(items),
           "serial_ok": sum(1 for r in seq if r.get("ok")),
           "conc_ok": sum(1 for r in con if r.get("ok")),
           "mismatch": diff, "serial_wall": w1, "conc_wall": wn,
           "serial": seq, "concurrent": con}
    with open(os.path.join(out_dir, f"concurrency_c{conc}.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[mbg] B 结果: 串行 {res['serial_ok']}/{len(items)}  并发 {res['conc_ok']}/{len(items)}  "
          f"逐项不一致 {len(diff)} 项", flush=True)
    for d in diff:
        print(f"[mbg]   MISMATCH i={d['i']} serial={'OK' if d['serial_ok'] else 'FAIL'} "
              f"conc={'OK' if d['conc_ok'] else 'FAIL'} | {d['q'][:34]}", flush=True)
    return res


# --------------------------------------------------- [C] 长请求与短请求交错
def test_mixed(base, model, out_dir, corpus=DEFAULT_CORPUS, long_ctx=131072, n_short=6,
               max_tokens=64):
    if long_ctx <= 0:
        print("\n[mbg] ===== [C] 跳过（--long-ctx<=0）=====", flush=True)
        return None
    print(f"\n[mbg] ===== [C] 长请求({long_ctx}) + {n_short} 条短请求 同时跑 =====", flush=True)
    long_prompt = None
    if os.path.exists(corpus):
        # 用 /tokenize 裁到目标长度（与 p42 同口径：token-id 数组直提）
        txt = open(corpus, encoding="utf-8", errors="ignore").read()
        try:
            tok = post(base, "/tokenize", {"model": model, "prompt": txt}, timeout=300)["tokens"]
            long_prompt = tok[:long_ctx]
        except Exception as exc:
            print(f"[mbg] C tokenize 失败: {exc}", flush=True)
    else:
        print(f"[mbg] C 语料不存在: {corpus}", flush=True)
    if not long_prompt:
        print("[mbg] C 跳过（无法构造长 prompt）", flush=True)
        return None

    baseline, _ = run_items(base, model, ITEMS[:n_short], 1, max_tokens=max_tokens,
                            tag="(短请求单独)")
    res_slot = {}

    def long_worker():
        try:
            r = post(base, "/v1/completions", {
                "model": model, "prompt": long_prompt, "max_tokens": max_tokens,
                "temperature": 0.0, "seed": 1234, "ignore_eos": True,
            }, timeout=1800)
            res_slot["long"] = ((r["choices"][0].get("text") or "")[:60],
                                (r.get("usage") or {}).get("prompt_tokens"))
        except Exception as exc:
            res_slot["long"] = (f"ERR {type(exc).__name__}: {str(exc)[:80]}", None)

    tl = threading.Thread(target=long_worker)
    tl.start()
    time.sleep(2)  # 让长 prefill 先进入（此时它会被 chunked prefill 切成多个 chunk）
    mixed, _ = run_items(base, model, ITEMS[:n_short], 3, max_tokens=max_tokens,
                         tag="(与长请求并发)")
    tl.join()

    diff = [a["i"] for a, b in zip(baseline, mixed) if a.get("ok") != b.get("ok")]
    res = {"test": "mixed_long_short", "long_ctx": long_ctx, "n_short": n_short,
           "baseline_ok": sum(1 for r in baseline if r.get("ok")),
           "mixed_ok": sum(1 for r in mixed if r.get("ok")),
           "mismatch": diff, "long_result": res_slot.get("long"),
           "baseline": baseline, "mixed": mixed}
    with open(os.path.join(out_dir, "mixed_long_short.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[mbg] C 结果: 基线 {res['baseline_ok']}/{n_short}  与长请求并发 {res['mixed_ok']}/{n_short}  "
          f"不一致 {len(diff)} 项", flush=True)
    print(f"[mbg] C 长请求返回: {res_slot.get('long')}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="deepseek-v41")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS, help="[C] 长 prompt 语料（txt）")
    ap.add_argument("--long-ctx", type=int, default=131072, help="[C] 长请求 token 数；<=0 跳过 C")
    ap.add_argument("--n-short", type=int, default=6, help="[C] 与长请求并发的短请求数")
    ap.add_argument("--max-tokens", type=int, default=64, help="[B]/[C] 生成上限（[A] 用 128）")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    skip = set(x for x in a.skip.split(",") if x)
    print(f"[mbg] base={a.base} model={a.model} out={a.out} rounds={a.rounds} conc={a.conc} "
          f"corpus={a.corpus}", flush=True)
    summary = {}
    if "A" not in skip:
        summary["multiturn"] = test_multiturn(a.base, a.model, a.rounds, a.out)
    if "B" not in skip:
        summary["concurrency"] = test_concurrency(a.base, a.model, a.conc, a.out,
                                                  max_tokens=a.max_tokens)
    if "C" not in skip:
        summary["mixed"] = test_mixed(a.base, a.model, a.out, corpus=a.corpus,
                                      long_ctx=a.long_ctx, n_short=a.n_short,
                                      max_tokens=a.max_tokens)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---------- 判据（PASS/FAIL 机器可读）----------
    verdict = {}
    mt = summary.get("multiturn")
    if mt:
        verdict["A_needle"] = "PASS" if mt["needle_hits"] == mt["needle_total"] else "FAIL"
        verdict["A_recall"] = "PASS" if all(mt["recall"].values()) else "FAIL"
    cc = summary.get("concurrency")
    if cc:
        verdict["B_itemwise"] = "PASS" if not cc["mismatch"] else "FAIL"
    mx = summary.get("mixed")
    if mx:
        verdict["C_short_vs_long"] = "PASS" if not mx["mismatch"] else "FAIL"
    summary["verdict"] = verdict
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[mbg] ===== 汇总 =====", flush=True)
    if mt:
        print(f"[mbg] 多轮: 轮内 {mt['needle_hits']}/{mt['needle_total']}  全长 {sum(mt['recall'].values())}/3  "
              f"最终 prompt={mt['final_prompt_tokens']} tokens", flush=True)
    if cc:
        print(f"[mbg] 并发(conc={cc['conc']}): 串行 {cc['serial_ok']}/{cc['n']}  并发 {cc['conc_ok']}/{cc['n']}  "
              f"不一致 {len(cc['mismatch'])}", flush=True)
    if mx:
        print(f"[mbg] 长短交错: 基线 {mx['baseline_ok']}/{mx['n_short']}  混合 {mx['mixed_ok']}/{mx['n_short']}  "
              f"不一致 {len(mx['mismatch'])}", flush=True)
    print(f"[mbg] 判据: {verdict}", flush=True)
    return 0 if verdict and all(v == "PASS" for v in verdict.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
