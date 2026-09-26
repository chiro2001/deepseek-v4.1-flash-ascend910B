#!/usr/bin/env python3
"""在**并发 N** 下发起一批流式请求，供 profiler 采集 decode 段。

为什么不用 `ced_pd_bench.py`：它是单流的。而"并发 4 下单流效率只有 48%"
这个现象必须在**真的 4 并发**下才能复现和研究。

口径固定（保证 A/B 两次采集可比）：
  * 4 条 prompt，**2048 token 精确校准**，语料切片互不重叠、问题轮换
  * 输出 `--max-tokens`（默认 256），`ignore_eos` 强制跑满
  * 4 条**同时**发出（threading），各自记 token 时间线

输出 JSON 含每条请求的 ttft / decode 时长 / token 数，以及**汇总的
decode 窗口**（首 token 到末 token），用于与 profiler 的时间窗对齐。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

# 脚本可能被放在包根（a3-21 的 ~/tmp/.../dspark/）或 experiments/dspark/ 下，
# 两种布局都要能找到 tools/。向上找到含 tools/bench_concurrency.py 的那一层。
_here = os.path.dirname(os.path.abspath(__file__))
_cands = [os.path.join(_here, "tools"),
          os.path.join(os.path.dirname(os.path.dirname(_here)), "tools"),
          os.path.join(os.path.dirname(_here), "tools")]
for _c in _cands:
    if os.path.isfile(os.path.join(_c, "bench_concurrency.py")):
        sys.path.insert(0, _c)
        break
else:
    raise SystemExit(f"找不到 tools/bench_concurrency.py；试过 {_cands}")
from bench_concurrency import (  # noqa: E402
    DEFAULT_CORPUS,
    load_corpus,
    load_suffixes,
    prepare_prompts,
    wait_idle,
)


class Res:
    def __init__(self, idx):
        self.idx = idx
        self.t0 = 0.0
        self.t_first = 0.0
        self.t_last = 0.0
        self.n_chunk = 0          # SSE 内容块数（**不是** token 数）
        self.n_out = 0            # 真实 token 数，来自 usage.completion_tokens
        self.finish = None
        self.ok = False
        self.err = ""


def one_request(base, model, idx, prompt, max_tokens, timeout, res):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    res.t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                now = time.monotonic()
                piece = ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content")
                if piece:
                    res.n_chunk += 1
                    if res.t_first == 0.0:
                        res.t_first = now
                    res.t_last = now
                usage = chunk.get("usage")
                if usage:
                    # ★ 必须用 usage 里的真实 token 数：开了推测解码后
                    # vLLM 会把**一个 step 接受的多个 token 合进一个 SSE 块**，
                    # 于是"块数"与"token 数"差 A 倍（A≈2.2 时块数只有 token 数的 45%）。
                    # 用块数算吞吐会让 DSpark 看起来比实际慢一倍以上 —— 已踩。
                    res.n_out = int(usage.get("completion_tokens") or res.n_out)
                fr = ((chunk.get("choices") or [{}])[0]).get("finish_reason")
                if fr:
                    res.finish = fr
        if res.n_out == 0:
            # 没拿到 usage 时退回块数，但**记录警告**（口径会偏乐观，不可当对照）
            res.n_out = res.n_chunk
            res.err = "no-usage-fallback"
        res.ok = res.n_out > 0
        if not res.ok:
            res.err = "no content"
    except Exception as exc:  # noqa: BLE001
        res.err = f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18992")
    ap.add_argument("--tokenize-url", default="http://127.0.0.1:18990")
    ap.add_argument("--model", default="deepseek-v41-ced-pd")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suffix-dir", default="data/hlm_local")
    ap.add_argument("--corpus-mark", default="========正文========")
    ap.add_argument("--warmup-rounds", type=int, default=0,
                    help="正式采集**之前**先跑的、不计入的热身轮数。"
                         "必须 >0：第一轮要付编译/首访/cache 冷启动，"
                         "不热身会把暖机成本算进被比较的臂里。")
    ap.add_argument("--warmup-tokens", type=int, default=64)
    ap.add_argument("--profile-url", default="",
                    help="给了就在**热身之后**、正式采集之前 POST /start_profile，"
                         "采集结束再 POST /stop_profile。放在这里而不是 shell 里，"
                         "是为了保证顺序严格是 热身→start→采集→stop。")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    tok_base = (args.tokenize_url or base).rstrip("/")

    # prepare_prompts 依赖 bench_concurrency 的**模块级** _CORPUS / _SUFFIXES，
    # 它们平时由该脚本的 main() 设置。直接 import 调用会拿到空列表
    # ⇒ `_SUFFIXES[rid % 0]` 触发 ZeroDivisionError（已踩）。
    import bench_concurrency as bc
    bc._CORPUS = load_corpus(args.corpus, skip_preamble=True,
                             mark=args.corpus_mark)
    bc._SUFFIXES, _suffix_names = load_suffixes(args.suffix_dir)
    if not bc._SUFFIXES:
        raise SystemExit(f"问题后缀为空：--suffix-dir={args.suffix_dir}")
    print(f"[conc-prof] 语料 {len(bc._CORPUS)} 字符；后缀 {len(bc._SUFFIXES)} 个"
          f"（{', '.join(_suffix_names)}）", flush=True)

    prompts = prepare_prompts(tok_base, args.model,
                              args.concurrency, args.prompt_tokens)
    wait_idle(base)

    # ---- 热身（不计入）----
    # 第一轮请求会付：ACLGraph 首次 replay、HCCL 首次建链、KV 首次分配、
    # Python/JIT 首次执行。这些成本与被比较的臂无关，但会污染短窗口采集。
    for w in range(args.warmup_rounds):
        wres = [Res(-1 - i) for i in range(len(prompts))]
        wthreads = [
            threading.Thread(target=one_request,
                             args=(base, args.model, i, p, args.warmup_tokens,
                                   args.timeout, wres[i]))
            for i, p in enumerate(prompts)
        ]
        tw0 = time.perf_counter()
        for t in wthreads:
            t.start()
        for t in wthreads:
            t.join()
        ok = sum(1 for r in wres if r.ok)
        print(f"[conc-prof] 热身 {w + 1}/{args.warmup_rounds}：ok={ok}/{len(wres)} "
              f"wall={time.perf_counter() - tw0:.2f}s", flush=True)
        wait_idle(base)

    # ---- 热身结束后才开始采集 ----
    if args.profile_url:
        purl = args.profile_url.rstrip("/")
        st = urllib.request.urlopen(
            urllib.request.Request(purl + "/start_profile", data=b"", method="POST"),
            timeout=60).status
        print(f"[conc-prof] start_profile={st}", flush=True)
        time.sleep(2)   # 让 profiler 真正开始记录

    results = [Res(i) for i in range(len(prompts))]
    threads = [
        threading.Thread(target=one_request,
                         args=(base, args.model, i, p, args.max_tokens,
                               args.timeout, results[i]))
        for i, p in enumerate(prompts)
    ]
    wall0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall1 = time.perf_counter()

    if args.profile_url:
        time.sleep(3)
        try:
            st = urllib.request.urlopen(
                urllib.request.Request(purl + "/stop_profile", data=b"", method="POST"),
                timeout=120).status
            print(f"[conc-prof] stop_profile={st}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[conc-prof] stop_profile 失败：{exc!r}", flush=True)

    good = [r for r in results if r.ok]
    if not good:
        print(f"[conc-prof] 全部失败：{results[0].err}", file=sys.stderr)
        return 1
    ttfts = [r.t_first - r.t0 for r in good]
    per_stream = [(r.n_out - 1) / (r.t_last - r.t_first)
                  for r in good if r.t_last > r.t_first and r.n_out > 1]
    win_start = min(r.t_first for r in good)
    win_end = max(r.t_last for r in good)
    out = {
        "concurrency": len(prompts),
        "ok": len(good),
        "prompt_tokens": args.prompt_tokens,
        "out_per_req": statistics.median([r.n_out for r in good]),
        "chunks_per_req": statistics.median([r.n_chunk for r in good]),
        "usage_fallback": sum(1 for r in good if r.err == "no-usage-fallback"),
        "ttft_med_s": round(statistics.median(ttfts), 3),
        "per_stream_med_tok_s": round(statistics.median(per_stream), 2) if per_stream else None,
        "decode_window_s": round(win_end - win_start, 3),
        "total_decode_tok_s": round(sum(r.n_out for r in good) / max(win_end - win_start, 1e-6), 2),
        "wall_s": round(wall1 - wall0, 3),
        "per_req": [{"idx": r.idx, "tokens": r.n_out, "chunks": r.n_chunk,
                     "finish": r.finish,
                     "ttft_s": round(r.t_first - r.t0, 3),
                     "decode_s": round(r.t_last - r.t_first, 3),
                     "tok_s": round((r.n_out - 1) / max(r.t_last - r.t_first, 1e-6), 2)}
                    for r in good],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
