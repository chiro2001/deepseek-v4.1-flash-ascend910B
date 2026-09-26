#!/usr/bin/env python3
"""并发吞吐扫描 —— 对每个并发级别测「单流吞吐」与「总吞吐」。

用法::

    python3 tools/bench_concurrency.py \
        --base-url http://127.0.0.1:8020 --model deepseek-v41 \
        --concurrency 1,2,4,8,16,32,64 \
        --prompt-tokens 1024 --output-tokens 256 --repeats 1

口径（重要，别把两个数混为一谈）
--------------------------------
* **单流吞吐**：每个请求自身的 decode 速率 ``(out_tokens-1) / (t_last - t_first)``，
  取所有请求的**中位数**。它回答"我自己发一条，能有多快"。
* **总吞吐**：全部请求输出 token 之和 ``/`` 整批墙钟（首 token ~ 末 token）。
  它回答"服务整体每秒能吐多少 token"。

两者都只计 **decode 阶段**（从首 token 到末 token），不含 prefill/TTFT；
TTFT 单独列出，便于判断 prefill 是否成为瓶颈。

其它纪律
--------
* 每个请求用《红楼梦》不同位置的正文切片 + 轮换的四个问题后缀，长度用服务端 /tokenize 校准到**正好 N 个 token**；
  测的是真实冷 prefill；若你的生产场景高度复用前缀，实际会更快。
* 每个并发级别跑 ``--repeats`` 次取中位数，级别之间等待服务回到空闲。
* 服务端的 ``--max-num-seqs`` 必须 >= 最大并发，否则会被调度器排队，
  数字失真（脚本会检测并警告）。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request


def _post_json(url: str, payload: dict, timeout: float):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


_SPEC_MAP = {"spec_decode_num_accepted_tokens_total": "num_accepted",
             "spec_decode_num_draft_tokens_total": "num_draft",
             "generation_tokens_total": "num_gen"}


def spec_counters(base: str) -> dict:
    """读投机解码计数器（用于算接受长度）。拿不到就返回空 dict。"""
    out = {}
    try:
        txt = urllib.request.urlopen(f"{base}/metrics", timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return out
    for line in txt.splitlines():
        if not line.startswith("vllm:"):
            continue
        name = line.split("{")[0].split(":", 1)[1]
        if name in _SPEC_MAP:
            try:
                out[_SPEC_MAP[name]] = float(line.rsplit(" ", 1)[-1])
            except ValueError:
                pass
    return out


def accept_length(m0: dict, m1: dict, n_spec: int = 7) -> tuple[float, float]:
    """返回 (接受长度, 接受率%)。接受长度 = 每步接受的 draft 数 + 1。

    ⚠️ 接受长度异常高（> 3.5）时先怀疑**复读退化**：
    模型卡在重复同一小段，重复 token 极易被草稿模型命中，
    会把接受长度虚高到 5 左右、吞吐冲到 140+ tok/s。
    详见 docs/BENCH-METHODOLOGY.md。

    ⚠️ `n_spec` 必须等于服务端的 `SP_TOKENS`。历史默认值 5 是旧口径
    （v4 时期），当前交付口径是 **7**（`serve_a2.sh` 的 `SP_TOKENS=${SP_TOKENS:-7}`）。
    对不上会让 A 静默错算：A = 1 + n_spec × acc/draft。
    """
    acc = m1.get("num_accepted", 0.0) - m0.get("num_accepted", 0.0)
    drf = m1.get("num_draft", 0.0) - m0.get("num_draft", 0.0)
    if drf <= 0:
        return 0.0, 0.0
    return acc / drf * n_spec + 1, acc / drf * 100


# 默认语料：包内自带的《红楼梦》全本（data/hongloumeng.txt）。
# 正文从 "========正文========" 之后开始，前面是出版说明，测吞吐时跳过。
DEFAULT_CORPUS = "data/hongloumeng.txt"
CORPUS_BODY_MARK = "========正文========"

# 默认问题后缀目录：**短切片专用**（每个问题只依赖切片内部信息，任务类型分散）。
# 每个请求轮换一个 ⇒ "不同的问题"。
#
# 另一套 data/hlm/suffix_*.txt 是**为全本设计**的（引第五回、三人感情走向、
# 全书词频…）。配短切片时 3/4 在上下文里无解，模型更容易掉进复读循环，
# 而复读会把接受长度虚高到 4.9+、吞吐冲到 139+ tok/s（见 docs/BENCH-METHODOLOGY.md）。
DEFAULT_SUFFIX_DIR = "data/hlm_local"

# 唯一标记放在**最前面**：vLLM prefix cache 用链式块哈希（block N 的哈希含
# block N-1 的哈希），第 0 块不同 ⇒ 后续所有块都不匹配 ⇒ 一块也不复用。
REQ_MARK_TMPL = "[req-{rid:05d}] "


def tokenize_count(base: str, model: str, text: str) -> int:
    """调用服务端 /tokenize 取精确 token 数（不要用字符数估算）。"""
    req = urllib.request.Request(
        f"{base}/tokenize",
        data=json.dumps({"model": model, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(json.loads(r.read().decode())["count"])


def load_corpus(path: str, skip_preamble: bool = True,
                mark: str | None = None) -> str:
    """读语料。若给了 mark，就从该标记**之后**开始取（用于跳过出版说明之类的前言）；
    找不到标记就整篇使用。"""
    if not os.path.exists(path):
        hint = ("\n  这份语料受版权保护，仓库不附带原文。先执行：\n"
                "      bash tools/fetch_corpus.sh\n"
                "  或改用别的语料：--corpus-file <你的文本> --suffix-dir <问题目录>"
                if "dihuo" in path else "")
        raise SystemExit(f"[bench] 语料不存在：{path}{hint}")
    text = open(path, encoding="utf-8", errors="replace").read()
    if skip_preamble:
        marker = mark or CORPUS_BODY_MARK
        k = text.find(marker)
        if k >= 0:
            text = text[k + len(marker):]
    return text


def load_suffixes(suffix_dir: str) -> tuple[list[str], list[str]]:
    """载入该目录下所有 *.txt 作为问题后缀（按文件名排序）。

    返回 (文本列表, 文件名列表)。每个请求按序轮换 ⇒ "不同的问题"。
    """
    out, names = [], []
    if not os.path.isdir(suffix_dir):
        raise SystemExit(f"[bench] 后缀目录不存在：{suffix_dir}/")
    for name in sorted(os.listdir(suffix_dir)):
        if not name.endswith(".txt") or name.lower().startswith("readme"):
            continue
        p = os.path.join(suffix_dir, name)
        out.append(open(p, encoding="utf-8", errors="replace").read())
        names.append(name)
    if not out:
        raise SystemExit(f"[bench] 找不到任何 *.txt 后缀于 {suffix_dir}/")
    return out, names


def _slice(corpus: str, offset: int, n_chars: int) -> str:
    """从 corpus 的 offset 处取 n_chars 个字符；越界则回绕（保证长度足够）。"""
    if offset + n_chars <= len(corpus):
        return corpus[offset:offset + n_chars]
    head = corpus[offset:]
    need = n_chars - len(head)
    return head + corpus[:need]


def _search_at(base: str, model: str, corpus: str, suffix: str, mark: str,
               offset: int, target_tokens: int) -> dict:
    """在固定 offset 上二分切片长度，返回最接近 target 的结果。"""
    lo, hi = 1, max(64, target_tokens * 4)     # 上界：最坏 1 token ≈ 1 字符
    best = {"chars": 0, "tokens": -1}
    while lo <= hi:
        mid = (lo + hi) // 2
        n = tokenize_count(base, model, mark + _slice(corpus, offset, mid) + suffix)
        if best["chars"] == 0 or abs(n - target_tokens) < abs(best["tokens"] - target_tokens):
            best = {"chars": mid, "tokens": n}
        if n == target_tokens:
            break
        if n < target_tokens:
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def calibrate_request(base: str, model: str, corpus: str, suffix: str,
                      rid: int, target_tokens: int,
                      offset_retries: int = 8) -> dict:
    """为**单个请求**校准出正好 target_tokens 个 token 的 prompt。

    每个请求用不同的正文起点 + 轮换的问题后缀 ⇒ 内容天然互异。

    逐字符二分通常能命中，但中文分词的字符粒度会让某些位置只能取到
    N-1 或 N+1（相邻字符跨在 token 边界上）。命中不了时**换个起点重试**——
    换个对齐方式往往就能正好卡在 target 上。仍不中则保留最小偏差并在汇总里报告。
    """
    mark = REQ_MARK_TMPL.format(rid=rid)
    base_off = _offset_for(rid)
    best = None
    for k in range(max(1, offset_retries)):
        # 37：质数微调量，避免重试时总对齐到同一个词边界
        off = (base_off + k * 37) % max(1, len(corpus) - 64)
        b = _search_at(base, model, corpus, suffix, mark, off, target_tokens)
        if best is None or abs(b["tokens"] - target_tokens) < abs(best["tokens"] - target_tokens):
            best = b
            best["offset"] = off
        if b["tokens"] == target_tokens:
            break
    return {"text": mark + _slice(corpus, best["offset"], best["chars"]) + suffix,
            "chars": best["chars"], "tokens": best["tokens"], "offset": best["offset"]}


_CORPUS = ""
_SLICE_STRIDE = 0           # 相邻请求的正文起点间隔（字符）；load 语料后自适应设定


def _set_slice_stride(n_requests: int, slice_chars: int) -> int:
    """按语料长度自适应设定切片步长，并把"重叠度"报告出来。

    理想情况：n_requests × stride ≤ 语料长度（完全互不重叠）。
    语料不够长时只能重叠 —— 这不是错误（每条 prompt 开头的唯一 id 会让
    prefix cache 的链式块哈希全部不同，**不会有任何块被复用**），
    但正文重复度升高会削弱"内容互异"的程度，所以要显式报告，别悄悄糊过去。
    """
    global _SLICE_STRIDE
    L = len(_CORPUS)
    if L <= slice_chars:
        _SLICE_STRIDE = max(1, L // 4)
        return -1
    stride = L // max(1, n_requests)
    if stride < 64:                      # 步长太小 ⇒ 切片几乎相同，没意义
        stride = max(64, slice_chars // 4)
    stride = min(stride, L - slice_chars)
    _SLICE_STRIDE = max(1, stride)
    return stride


def _offset_for(rid: int) -> int:
    """第 rid 个请求的正文起点：按 stride 铺开，取模回绕。"""
    span = max(1, len(_CORPUS) - min(len(_CORPUS) // 4, 4000))
    return (rid * max(1, _SLICE_STRIDE)) % span


def prepare_prompts(base: str, model: str, n: int, target_tokens: int) -> list[str]:
    """预先为 n 个请求各自校准出正好 target_tokens 个 token 的 prompt。"""
    # 先按目标长度估一次切片字符数（中文约 1.2 字符/token），据此定步长
    est_chars = int(target_tokens * 1.3)
    stride = _set_slice_stride(n, est_chars)
    L = len(_CORPUS)
    if stride > 0:
        covered = n * stride
        note = ("完全互不重叠" if covered <= L
                else f"需要回绕 {(covered + L - 1) // L} 圈，正文会重复")
        print(f"[bench] 切片步长={stride} 字符（语料 {L} 字符，{n} 个请求 × 步长 = {covered}）⇒ {note}",
              flush=True)
    prompts, exact, diffs = [], 0, []
    for rid in range(n):
        r = calibrate_request(base, model, _CORPUS, _SUFFIXES[rid % len(_SUFFIXES)],
                              rid, target_tokens)
        prompts.append(r["text"])
        d = r["tokens"] - target_tokens
        diffs.append(d)
        if d == 0:
            exact += 1
    hi = max(abs(d) for d in diffs) if diffs else 0
    print(f"[bench] prompt 校准：{n} 个请求，目标 {target_tokens} tok；"
          f"完全命中 {exact}/{n}，最大偏差 {hi} tok", flush=True)
    return prompts


_SUFFIXES: list[str] = []
_SUFFIX_NAMES: list[str] = []


class StreamResult:
    __slots__ = ("ok", "err", "t0", "t_first", "t_last", "n_out", "prompt_tokens")

    def __init__(self) -> None:
        self.ok = False
        self.err = ""
        self.t0 = self.t_first = self.t_last = 0.0
        self.n_out = 0
        self.prompt_tokens = 0


def one_request(base: str, model: str, rid: int, prompt: str, max_tokens: int,
                timeout: float, res: StreamResult, ignore_eos: bool = True) -> None:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # 压测必须强制生成满 max_tokens：否则模型遇到 EOS 就停，各并发级别输出
    # 长度参差不齐（实测 55~85 token），decode 窗口太短 ⇒ 中位数噪声极大。
    if ignore_eos:
        payload["ignore_eos"] = True
    t0 = time.perf_counter()
    res.t0 = t0
    try:
        with _post_json(f"{base}/v1/completions", payload, timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                now = time.perf_counter()
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage")
                if usage:
                    res.n_out = int(usage.get("completion_tokens") or res.n_out)
                    res.prompt_tokens = int(usage.get("prompt_tokens") or res.prompt_tokens)
                    continue
                for ch in obj.get("choices") or []:
                    if ch.get("text"):
                        if res.t_first == 0.0:
                            res.t_first = now
                        res.t_last = now
        res.ok = res.t_first > 0.0
        if not res.ok:
            res.err = "no content"
    except Exception as exc:  # noqa: BLE001 - 客户端要吞掉任何网络异常并记录
        res.err = f"{type(exc).__name__}: {exc}"


def run_level(base: str, model: str, conc: int, prompts: list[str],
              max_tokens: int, timeout: float, ignore_eos: bool = True,
              metrics_base: str | None = None, n_spec: int = 7) -> dict:
    """以**并发度 conc** 跑完**全部** prompts（按 conc 切批），返回该级别的统计。

    prompts 已由 prepare_prompts 预先校准为**正好 target_tokens 个 token**，
    且每条的正文切片起点与问题后缀都不同。

    ★ 每个并发档都用**同一批 prompts**（不是只取前 conc 条）：
      否则各档的内容集合不同（conc=2 只有 2 条、conc=64 有 64 条），
      投机解码接受率又和文本强相关 ⇒ 曲线不可比。
      这里 64 条按 conc 切成若干批，逐批并发、批间等空闲。
    """
    results: list[StreamResult] = []
    dec_win_total = 0.0        # Σ 各批的 decode 窗口（批间有间隔，不能算成一段）
    m_start = spec_counters(metrics_base or base)
    t_start = time.perf_counter()
    for b0 in range(0, len(prompts), conc):
        batch = prompts[b0:b0 + conc]
        rs = [StreamResult() for _ in batch]
        threads = [
            threading.Thread(target=one_request,
                             args=(base, model, b0 + i, p, max_tokens,
                                   timeout, rs[i], ignore_eos))
            for i, p in enumerate(batch)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        results.extend(rs)
        ok_rs = [r for r in rs if r.ok and r.t_last > r.t_first]
        if ok_rs:
            dec_win_total += max(r.t_last for r in ok_rs) - min(r.t_first for r in ok_rs)
        # conc=1 时批与批之间自然有序，不需要再等空闲（省掉 64 次轮询开销）
        if conc > 1 and b0 + conc < len(prompts):
            wait_idle(base)
    t_end = time.perf_counter()
    m_end = spec_counters(metrics_base or base)
    acc_len, acc_rate = accept_length(m_start, m_end, n_spec)

    good = [r for r in results if r.ok]
    if not good:
        return {"conc": conc, "ok": 0, "fail": len(prompts),
                "err": results[0].err if results else "unknown"}

    per_stream = [(r.n_out - 1) / (r.t_last - r.t_first)
                  for r in good if r.t_last > r.t_first and r.n_out > 1]
    ttfts = [r.t_first - r.t0 for r in good]
    total_out = sum(r.n_out for r in good)

    # 总吞吐只计 decode 窗口：Σ 各批（批内首 token → 批内末 token）
    dec_wall = max(dec_win_total, 1e-6)
    e2e_wall = t_end - t_start

    return {
        "conc": conc,
        "ok": len(good),
        "fail": len(prompts) - len(good),
        "prompt_tokens": statistics.median([r.prompt_tokens for r in good if r.prompt_tokens] or [0]),
        "out_per_req": statistics.median([r.n_out for r in good]),
        "per_stream_med": statistics.median(per_stream) if per_stream else 0.0,
        "per_stream_mean": statistics.mean(per_stream) if per_stream else 0.0,
        "total_decode": total_out / dec_wall,
        "total_e2e": total_out / e2e_wall,
        "ttft_med": statistics.median(ttfts),
        "decode_wall": dec_wall,
        "e2e_wall": e2e_wall,
        "accept_len": acc_len,
        "accept_rate": acc_rate,
    }



def wait_idle(base: str, timeout: float = 120.0) -> None:
    """等 /metrics 报 running==0（拿不到 metrics 就退化为 sleep）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            txt = urllib.request.urlopen(f"{base}/metrics", timeout=5).read().decode()
            running = None
            for line in txt.splitlines():
                if line.startswith("vllm:num_requests_running"):
                    running = float(line.rsplit(" ", 1)[-1])
            if running == 0.0:
                return
        except Exception:  # noqa: BLE001
            time.sleep(2.0)
            return
        time.sleep(1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="并发吞吐扫描（单流 + 总吞吐）")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--tokenize-url", default="",
                    help="/tokenize 只在 P 上有，PD 代理不提供。默认与 --base-url 相同；"
                         "走代理测吞吐时必须显式指到 P（否则 404），"
                         "例如 --tokenize-url http://127.0.0.1:18990")
    # ★ 默认从 "deepseek-v41" 改成空串：非空默认值会让"未指定"这个状态
    # **不存在**，于是在网关场景下静默假定一个模型名，而实际压的是别的东西
    # （issue #2 报告者撞的就是这类）。改成空 ⇒ 未指定时明确用 /v1/models 的
    # data[0] 并打印提示，而不是假装用户选过了。
    ap.add_argument("--model", default="",
                    help="served model name。留空则用 /v1/models 的 data[0]（会打印提示）；"
                         "走聚合网关时**建议显式指定**")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--metrics-url", default="",
                    help="读 /metrics 的地址。PD 分离时**必须**指到 D"
                         "（官方代理不透传 /metrics，否则 A 恒报 0.00）")
    ap.add_argument("--spec-tokens", type=int, default=7,
                    help="服务端 SP_TOKENS；A = 1 + spec_tokens × acc/draft。"
                         "历史默认 5 是旧口径，当前交付是 7")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--no-ignore-eos", action="store_true",
                    help="不要强制生成满 max_tokens（默认强制；关掉会让各并发输出长度不一）")
    ap.add_argument("--corpus-file", default=DEFAULT_CORPUS,
                    help=f"正文语料（默认 {DEFAULT_CORPUS}）")
    ap.add_argument("--corpus-mark", default=CORPUS_BODY_MARK,
                    help=f"从这个标记之后开始取正文（默认 {CORPUS_BODY_MARK!r}；找不到则用全篇）")
    ap.add_argument("--suffix-dir", default=DEFAULT_SUFFIX_DIR,
                    help=f"问题后缀目录（默认 {DEFAULT_SUFFIX_DIR}，四个问题轮换使用）")
    ap.add_argument("--keep-preamble", action="store_true",
                    help="不跳过语料开头的出版说明")
    a = ap.parse_args()
    ignore_eos = not a.no_ignore_eos
    global _CORPUS, _SUFFIXES

    base = a.base_url.rstrip("/")
    concs = [int(x) for x in a.concurrency.split(",") if x.strip()]

    try:
        models = _get_json(f"{base}/v1/models")
        ids = [m.get("id") for m in (models.get("data") or []) if m.get("id")]
    except Exception as exc:  # noqa: BLE001
        print(f"[bench] 服务不可达：{exc}", file=sys.stderr)
        return 3
    if not ids:
        print("[bench] /v1/models 没返回任何 id", file=sys.stderr)
        return 3

    # ★ 优先**在列表里找 `--model` 指定的那个**，找不到才回退 `data[0]`。
    #
    # 原先直接 `served = data[0]["id"]` 并用它**覆盖** `--model`。走聚合网关时
    # `data[0]` 可能是**别的模型**（issue #2 报告者那边第一项是 bge-embedding），
    # 于是压测目标被**悄悄改成 embedding 模型、结果全废**，而且没有任何报错 ——
    # 典型的"判据绑错对象"。
    if a.model:
        if a.model in ids:
            served = a.model
            if ids[0] != a.model:
                print(f"[bench] /v1/models 里 data[0]={ids[0]}，但 --model={a.model} 也在列表里 "
                      f"⇒ 用 --model（**不覆盖**）", file=sys.stderr)
        else:
            served = ids[0]
            print(f"[bench] WARNING: --model={a.model} 不在 /v1/models（{ids[:3]}…）⇒ "
                  f"回退 data[0]={served}。压测目标可能不是你想要的，请核对。", file=sys.stderr)
            a.model = served
    else:
        served = ids[0]
        if len(ids) > 1:
            print(f"[bench] 未指定 --model，用 /v1/models 的 data[0]={served}"
                  f"（列表里还有 {len(ids) - 1} 个；走网关时建议显式指定）", file=sys.stderr)
        a.model = served

    print(f"[bench] base={base} model={a.model} label={a.label or '-'}")
    print(f"[bench] prompt {a.prompt_tokens} tok（**精确校准**）, output={a.output_tokens} tok, "
          f"repeats={a.repeats}, 并发级别={concs}")

    # 载入语料与问题后缀
    _CORPUS = load_corpus(a.corpus_file, skip_preamble=not a.keep_preamble,
                          mark=a.corpus_mark)
    _SUFFIXES, _SUFFIX_NAMES = load_suffixes(a.suffix_dir)
    print(f"[bench] 语料={a.corpus_file}（正文 {len(_CORPUS)} 字符）  "
          f"问题后缀={len(_SUFFIXES)} 个（{', '.join(_SUFFIX_NAMES)}）")
    print("[bench] 每个请求：不同正文切片 + 轮换问题 ⇒ 内容互异，无 prefix cache 复用")

    # 为最大并发数预先校准出每条正好 target_tokens 个 token 的 prompt
    max_conc = max(concs)
    tok_base = (a.tokenize_url or a.base_url).rstrip('/')
    if tok_base != base:
        print(f"[bench] /tokenize 走 {tok_base}（代理不提供该端点）")
    prompts = prepare_prompts(tok_base, a.model, max_conc, a.prompt_tokens)

    # 预热（不计入结果）
    wait_idle(base)
    met = a.metrics_url.rstrip("/") or base
    if a.metrics_url:
        print(f"[bench] /metrics 走 {met}（A 与 step 口径依赖它）")
    else:
        print("[bench] ⚠️ 未指定 --metrics-url：PD 分离下代理不透传 /metrics，"
              "接受长度会恒报 0.00")
    warm = run_level(base, a.model, 1, prompts, 32, a.timeout, ignore_eos,
                     metrics_base=met, n_spec=a.spec_tokens)
    if warm.get("ok", 0) == 0:
        print(f"[bench] 预热失败：{warm.get('err')}", file=sys.stderr)
        return 4

    rows = []
    for conc in concs:
        reps = []
        for rep in range(a.repeats):
            wait_idle(base)
            r = run_level(base, a.model, conc, prompts, a.output_tokens,
                          a.timeout, ignore_eos,
                          metrics_base=met, n_spec=a.spec_tokens)
            if r.get("ok", 0) == 0:
                print(f"[bench] conc={conc:3d} 失败：{r.get('err')}")
                break
            reps.append(r)
            tag = f"rep{rep + 1}/{a.repeats}" if a.repeats > 1 else ""
            print(f"[bench] conc={conc:3d} {tag:9s} ok={r['ok']:3d}/{len(prompts)} "
                  f"prompt={r['prompt_tokens']:.0f}tok out/req={r['out_per_req']:.0f} | "
                  f"单流={r['per_stream_med']:7.1f} tok/s  总吞吐={r['total_decode']:8.1f} tok/s "
                  f"(e2e {r['total_e2e']:7.1f})  接受长度={r['accept_len']:5.2f} "
                  f"({r['accept_rate']:4.1f}%)  TTFT={r['ttft_med']:6.2f}s", flush=True)
        if reps:
            # 多次采样取**中位数**（不是取最优——取最优会让发布数字偏乐观）
            if len(reps) == 1:
                rows.append(reps[0])
            else:
                med = {}
                for k in ("conc", "ok", "fail", "prompt_tokens", "out_per_req",
                          "per_stream_med", "per_stream_mean", "total_decode",
                          "total_e2e", "ttft_med", "decode_wall", "e2e_wall",
                          "accept_len", "accept_rate"):
                    med[k] = statistics.median([r[k] for r in reps])
                med["n_reps"] = len(reps)
                rows.append(med)

    if not rows:
        return 5

    print("\n[bench] ==== 汇总（单流 = 每请求 decode 速率中位数；总吞吐 = decode 窗口聚合）====")
    print(f"{'并发':>5} {'单流 tok/s':>11} {'总吞吐 tok/s':>13} {'加速比':>8} "
          f"{'接受长度':>9} {'TTFT s':>8} {'单流效率':>9}")
    base_conc = rows[0]
    for r in rows:
        gain = r["total_decode"] / base_conc["total_decode"] if base_conc["total_decode"] else 0.0
        eff = r["per_stream_med"] / base_conc["per_stream_med"] * 100 if base_conc["per_stream_med"] else 0.0
        print(f"{r['conc']:>5} {r['per_stream_med']:>11.1f} {r['total_decode']:>13.1f} "
              f"{gain:>7.2f}x {r['accept_len']:>9.2f} {r['ttft_med']:>8.2f} {eff:>8.1f}%")
    print("\n  加速比 = 该并发总吞吐 / 并发1 总吞吐；单流效率 = 该并发单流吞吐 / 并发1 单流吞吐")

    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump({"label": a.label, "base_url": base, "model": a.model,
                       "prompt_tokens": a.prompt_tokens, "output_tokens": a.output_tokens,
                       "repeats": a.repeats, "rows": rows}, fh, ensure_ascii=False, indent=2)
        print(f"\n[bench] 明细已写入 {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
