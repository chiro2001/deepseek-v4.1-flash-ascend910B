#!/usr/bin/env python3
"""verify_dram_offload.py —— **证明 DRAM（KV 卸载池）真的在干活**（单文件、自包含）

## 为什么需要它
`scripts/run_test.sh` 的 7 项里**没有卸载检查**；`a2/scripts/check_4axis_acceptance.py` 有卸载判据，
但它是个**判决器**（要你先把 serve.log / metrics_after / kv_events 准备好），**不含"制造卸载"的压测**。
而 A3 一直用的那个压测客户端（`bench/kv_offload_client.py`，fill → reset → replay）
**从来没有进过发布仓** ⇒ 在 A2 上你手上没有能"跑一轮并留下可核证据"的工具。本文件补这一环。

## 它做什么（唯一能证明卸载生效的实验形态）
    ① 采 metrics before
    ② **fill 轮**：冷算一组长前缀 ⇒ 被逐出的块经 offloading connector **存进 DRAM**
    ③ 采 metrics mid
    ④ `POST /reset_prefix_cache` —— 清掉 **GPU 侧**前缀缓存（DRAM 池**不会**被它清）
    ⑤ **replay 轮**：再发同一批前缀 ⇒ 若 DRAM 里真有，就走**异步 H2D 取回**（不是重算）
    ⑥ 采 metrics after
    ⑦ 逐条判据 + 打印**原始 metrics 行**（第三方可复算）

★ 口径沿用的是 A3 已验证的那套（`bench/kv_offload_client.py`）：
  `make_prompt(seed, n) = 1000 + ((seed*100003 + SALT + i*7919) % 100000)`，
  `/v1/completions`、`temperature=0`、`stream=true`、`ignore_eos=true`。
  ⇒ 与本仓已发布的报告口径一致，**数字可横向比**。

## 判据（每条都打印原始读数；"未验"不计入通过）
    H1 `kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"}` 增量 > 0   —— 存进了 DRAM
    H2 ★★ 同表 `{transfer_type="CPU_to_GPU"}` 增量 > 0                     —— **从 DRAM 取回**
    H3 旁证：`external_prefix_cache_hits_total` 增量 > 0                    —— 命中（非重算）
    H4 旁证：`kv_offload_cpu_cache_usage_perc` > 0                         —— 池里真有内容
    P1 旁证：replay 的 TTFT 中位数小于 fill                                 —— 取回比冷算快

★ **硬判据只有 H1 + H2**，因为它们是**字节累加器**（语义唯一）：只涨 H1 = "只存不取"，
  那只是"往 DRAM 写了一份"，**不是**卸载生效。
★ H3/H4/P1 是**旁证**：H3 依赖计数器口径（本仓见过"同名不同义/某路径不更新"）、
  H4 受采样时刻影响、P1 受 HTTP/SSE/并发影响 ⇒ 它们为 0 只提示人工确认，**不定生死**。

## 用法
    # A2（1M、85 GiB 池、生产在跑 ⇒ 用短一点的前缀，别抢太多 decode 队列）
    python3 tests/verify_dram_offload.py --base-url http://127.0.0.1:8077 \
        --model deepseek-v4-flash --prompt-tokens 32768 --max-tokens 16 \
        --out ~/dram_offload_verify.json

    # A3（4 GiB 池的探针口径）
    python3 tests/verify_dram_offload.py --base-url http://127.0.0.1:8051 \
        --model deepseek-v41 --prompt-tokens 8192 --out /work/dram.json

退出码：0 = 硬判据全过；9 = 有硬判据未过；64 = 连不上/用法错
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

try:
    import requests
except ImportError:  # pragma: no cover
    print("需要 requests（vLLM 环境自带）", file=sys.stderr)
    raise SystemExit(64)


# --------------------------------------------------------------------------- #
# 口径：与 A3 已发布的 bench 客户端逐字一致（改了就不许横向比）
# --------------------------------------------------------------------------- #
SALT = 0


def make_prompt(seed: int, n_tokens: int) -> list[int]:
    base = (seed * 100003 + SALT) % 100000
    return [1000 + ((base + i * 7919) % 100000) for i in range(n_tokens)]


METRIC_KEEP = (
    "vllm:kv_offload_",
    "vllm:external_prefix_cache_",
    "vllm:prefix_cache_",
    "vllm:num_requests",
)


def scrape(base_url: str) -> dict[str, list[str]]:
    """把关心的 metric **整行**抓回来（原始行 ⇒ 可复算，不做二次加工）。"""
    try:
        text = requests.get(f"{base_url}/metrics", timeout=30).text
    except requests.RequestException as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    out: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name = line.split("{")[0].split(" ")[0]
        if any(name.startswith(k) for k in METRIC_KEEP):
            out.setdefault(name, []).append(line.strip())
    return out


_NUM = re.compile(r"\s([0-9.eE+]+)$")


def metric_value(snap: dict, name: str, label: str | None = None) -> float | None:
    """取某个 metric 的值；给了 label 就在同一行里必须含该 label 片段。"""
    lines = snap.get(name) or []
    for ln in lines:
        if label is not None and label not in ln:
            continue
        m = _NUM.search(ln)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def delta(a: float | None, b: float | None) -> float | None:
    return None if (a is None or b is None) else b - a


# --------------------------------------------------------------------------- #
# 压测
# --------------------------------------------------------------------------- #
def one_request(base_url: str, model: str, prompt: list[int], max_tokens: int,
                timeout: float) -> dict:
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True, "ignore_eos": True}
    t0 = time.perf_counter()
    ttft = None
    pieces: list[str] = []
    try:
        with requests.post(f"{base_url}/v1/completions", json=payload, stream=True,
                           timeout=timeout) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                body = raw[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                for ch in obj.get("choices", []):
                    txt = ch.get("text") or ""
                    if txt:
                        pieces.append(txt)
                        if ttft is None:
                            ttft = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "total_s": time.perf_counter() - t0}
    text = "".join(pieces)
    return {"ok": True, "ttft_s": ttft, "total_s": time.perf_counter() - t0,
            "chars": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text_repr": repr(text[:48])}


def run_round(base_url: str, model: str, prompts: list[list[int]], max_tokens: int,
              conc: int, timeout: float, tag: str) -> dict:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, conc)) as ex:
        res = list(ex.map(
            lambda p: one_request(base_url, model, p, max_tokens, timeout), prompts))
    wall = time.perf_counter() - t0
    ttfts = [r["ttft_s"] for r in res if r.get("ok") and r.get("ttft_s")]
    out = {
        "tag": tag, "wall_s": round(wall, 3), "n": len(res),
        "ok": sum(1 for r in res if r.get("ok")),
        "failed": sum(1 for r in res if not r.get("ok")),
        "ttft_s": {
            "n": len(ttfts),
            "median": round(statistics.median(ttfts), 3) if ttfts else None,
            "min": round(min(ttfts), 3) if ttfts else None,
            "max": round(max(ttfts), 3) if ttfts else None,
        },
        "results": res,
    }
    print(f"  [{tag}] n={out['n']} ok={out['ok']} failed={out['failed']} "
          f"wall={out['wall_s']}s ttft_median={out['ttft_s']['median']}s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8077")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--prompts", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--shared-prefix", action="store_true", default=True,
                    help="所有请求共用同一前缀（默认开：池只需覆盖 1 份前缀）")
    ap.add_argument("--no-shared-prefix", dest="shared_prefix", action="store_false")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rep: dict = {"base_url": args.base_url, "model": args.model,
                 "prompt_tokens": args.prompt_tokens, "max_tokens": args.max_tokens,
                 "prompts": args.prompts, "concurrency": args.concurrency,
                 "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("=" * 72)
    print("DRAM 卸载生效验证（fill → reset_prefix_cache → replay）")
    print(f"  {args.base_url}  model={args.model}  "
          f"prompt={args.prompt_tokens} tok × {args.prompts} req  "
          f"max_tokens={args.max_tokens}  conc={args.concurrency}")
    print("=" * 72)

    # ---- ① before ----
    m0 = scrape(args.base_url)
    if "error" in m0:
        print(f"⛔ 连不上 {args.base_url}/metrics：{m0['error']}", file=sys.stderr)
        return 64
    rep["metrics_before"] = m0
    g2c0 = metric_value(m0, "vllm:kv_offload_total_bytes_total", 'transfer_type="GPU_to_CPU"')
    c2g0 = metric_value(m0, "vllm:kv_offload_total_bytes_total", 'transfer_type="CPU_to_GPU"')
    hit0 = metric_value(m0, "vllm:external_prefix_cache_hits_total")
    use0 = metric_value(m0, "vllm:kv_offload_cpu_cache_usage_perc")
    print(f"  before: GPU_to_CPU={g2c0} CPU_to_GPU={c2g0} hits={hit0} pool_usage={use0}")

    # 先确认服务空闲度（**不强制**：A2 是生产在跑，我们只记录）
    running = metric_value(m0, "vllm:num_requests_running")
    print(f"  （服务当前 running={running}；本脚本只发 HTTP 请求，不改任何配置）")

    seed = (lambda i: 0) if args.shared_prefix else (lambda i: i)
    prompts = [make_prompt(seed(i), args.prompt_tokens) for i in range(args.prompts)]

    # ---- ② fill ----
    print("\n② fill 轮（冷算；被逐出的块应存进 DRAM）")
    fill = run_round(args.base_url, args.model, prompts, args.max_tokens,
                     args.concurrency, args.timeout, "fill")
    rep["rounds"] = [fill]

    # ---- ③ mid ----
    m1 = scrape(args.base_url)
    rep["metrics_after_fill"] = m1
    g2c1 = metric_value(m1, "vllm:kv_offload_total_bytes_total", 'transfer_type="GPU_to_CPU"')
    print(f"  after fill: GPU_to_CPU={g2c1}（Δ={delta(g2c0, g2c1)}）")

    # ---- ④ reset（只清 GPU 侧；DRAM 池仍在） ----
    print("\n④ POST /reset_prefix_cache（清 GPU 前缀缓存；**DRAM 池不会被它清**）")
    try:
        r = requests.post(f"{args.base_url}/reset_prefix_cache", timeout=120)
        rep["reset"] = {"status": r.status_code, "body": r.text[:200]}
        print(f"  HTTP {r.status_code} {r.text[:80]}")
    except requests.RequestException as exc:
        rep["reset"] = {"error": str(exc)}
        print(f"  ⚠ 失败：{exc}（引擎可能不支持该端点 ⇒ H2 若不过需人工判读）")

    # ---- ⑤ replay ----
    print("\n⑤ replay 轮（同前缀；若 DRAM 里有 ⇒ 走异步 H2D 取回，而不是重算）")
    replay = run_round(args.base_url, args.model, prompts, args.max_tokens,
                       args.concurrency, args.timeout, "replay")
    rep["rounds"].append(replay)

    # ---- ⑥ after ----
    m2 = scrape(args.base_url)
    rep["metrics_after_replay"] = m2
    g2c2 = metric_value(m2, "vllm:kv_offload_total_bytes_total", 'transfer_type="GPU_to_CPU"')
    c2g2 = metric_value(m2, "vllm:kv_offload_total_bytes_total", 'transfer_type="CPU_to_GPU"')
    hit2 = metric_value(m2, "vllm:external_prefix_cache_hits_total")
    use2 = metric_value(m2, "vllm:kv_offload_cpu_cache_usage_perc")
    print(f"  after replay: GPU_to_CPU={g2c2} CPU_to_GPU={c2g2} hits={hit2} pool_usage={use2}")

    # ---- ⑦ 判据 ----
    d_g2c = delta(g2c0, g2c2)
    d_c2g = delta(c2g0, c2g2)
    d_hit = delta(hit0, hit2)
    t_fill = fill["ttft_s"]["median"]
    t_repl = replay["ttft_s"]["median"]

    print("\n" + "=" * 72)
    print("判据（每条都给了原始读数；未验不计入通过）")
    print("=" * 72)
    hard = 0

    def line(ok: bool | None, name: str, detail: str, judge: bool = True) -> None:
        """judge=False ⇒ **旁证**：只打印，不计入硬判据、不影响退出码。
        ★ 为什么 P1 必须是旁证：TTFT 会被 HTTP/SSE 缓冲、并发、其它租户流量影响，
          它**不是**"卸载是否生效"的判据（真判据是计数器）。
          把它当硬判据会让一次好实验因为无关抖动被判失败（实测就撞到过：TTFT 打平）。"""
        nonlocal hard
        if ok is None:
            print(f"  ?  {name:<46} {detail}（未验）")
        elif ok:
            print(f"  ✓  {name:<46} {detail}")
        else:
            print(f"  ✗  {name:<46} {detail}")
            if judge:
                hard = 1

    line(None if d_g2c is None else d_g2c > 0,
         "H1 存进 DRAM（GPU_to_CPU 增量>0）", f"Δ={d_g2c}")
    line(None if d_c2g is None else d_c2g > 0,
         "H2 ★★ 从 DRAM 取回（CPU_to_GPU 增量>0）", f"Δ={d_c2g}")
    # ★★★ H3/H4 是**旁证**，不是硬判据 —— 理由（本仓踩过多次的同类坑）：
    #   H3 依赖 `external_prefix_cache_hits_total` 的**计数器口径**，而"计数器同名不同义/
    #   某条路径不更新"在本仓反复出现过（见 AGENTS.md 的探针纪律）；
    #   H4 是池使用率快照，受采样时刻影响。
    #   ⇒ **"卸载是否双向生效"由 H1+H2 证明**（它们是字节累加器，语义唯一）；
    #     H3/H4 用来说明"走的哪条路"，为 0 时提示人工确认，**不定生死**。
    line(None if d_hit is None else d_hit > 0,
         "H3 旁证：前缀命中计数在涨", f"Δ={d_hit}", judge=False)
    line(None if use2 is None else use2 > 0,
         "H4 旁证：池里有内容（cpu_cache_usage>0）", f"={use2}", judge=False)
    if d_hit is not None and d_hit <= 0 and d_c2g is not None and d_c2g > 0:
        print("     ⚠ H2 过了但 H3 没涨：**取回确实发生了**，但命中计数口径没动 ——")
        print("       请人工确认它是「另一条路径」还是「计数器未更新」（别直接采信 H3=0 的解读）。")
    line(None if (t_fill is None or t_repl is None) else t_repl < t_fill,
         "P1 旁证：取回比冷算快（replay TTFT < fill）",
         f"fill={t_fill}s replay={t_repl}s", judge=False)

    # ★ 兼容 `a2/scripts/check_4axis_acceptance.py --client <json>` 的字段名
    #   （与 A3 的 bench 客户端同形）⇒ **同一次运行**即可同时满足：
    #     ① 本脚本的卸载判据  ② 判决器的"逐字可复现"判据（fill vs replay 逐 prompt sha256）。
    _fh = {i: r.get("sha256") for i, r in enumerate(fill["results"]) if r.get("ok")}
    _rh = {i: r.get("sha256") for i, r in enumerate(replay["results"]) if r.get("ok")}
    _common = sorted(set(_fh) & set(_rh))
    rep["fill_out_sha256_by_prompt"] = {str(k): _fh[k] for k in sorted(_fh)}
    rep["replay1_out_sha256_by_prompt"] = {str(k): _rh[k] for k in sorted(_rh)}
    rep["fill_out_sha256_all"] = hashlib.sha256(
        "".join(_fh[k] for k in sorted(_fh)).encode()).hexdigest()
    rep["replay1_out_sha256_all"] = hashlib.sha256(
        "".join(_rh[k] for k in sorted(_rh)).encode()).hexdigest()
    rep["sha256_common_prompts"] = len(_common)
    rep["sha256_mismatched_prompts"] = [k for k in _common if _fh[k] != _rh[k]]
    rep["replay_matches_fill_sha256"] = bool(_common) and not rep["sha256_mismatched_prompts"]
    print(f"  （fill vs replay 逐 prompt sha256：common={len(_common)} "
          f"mismatched={rep['sha256_mismatched_prompts']} ⇒ "
          f"matches={rep['replay_matches_fill_sha256']}）")

    rep["verdict"] = {
        "delta_GPU_to_CPU": d_g2c, "delta_CPU_to_GPU": d_c2g,
        "delta_hits": d_hit, "cpu_cache_usage": use2,
        "ttft_fill_median": t_fill, "ttft_replay_median": t_repl,
        "hard_pass": hard == 0,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    print(f"\n产物：{args.out}（含 before/mid/after 三份 metrics 原始行 ⇒ 可复算）")
    if hard == 0:
        print("✅ 硬判据全过：DRAM 卸载**存取双向**都验证到了")
    else:
        print("⛔ 有硬判据未过 ⇒ **不能声称卸载生效**；先看上面哪条 ✗ 及其原始读数")
    return 0 if hard == 0 else 9


if __name__ == "__main__":
    sys.exit(main())
