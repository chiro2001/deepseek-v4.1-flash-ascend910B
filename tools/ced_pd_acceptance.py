#!/usr/bin/env python3
"""CED-PD 端到端验收 runner（正确性 + 流式/多轮/缓存命中 + 性能口径）。

目标（见 docs/CED-PD-ACCEPTANCE.md）要求 144K 与 1M 上下文的正确性、流式、多轮、
缓存命中，以及 prefill/TTFT/TPOT/吞吐与资源开销。本脚本把这些做成**一条命令**，
并把每条请求的原始字节、SHA-256、首 token、usage 和判据落盘，便于复核。

针与判据沿用 2026-09-24 在 A3-21 上实测的那一套（四针 A/B/C/D），因此新结果能与
既有证据直接对照：
    A 机房门禁密码     ZQ7K-3341
    B 备份账号口令     VX2M-8890
    C 监控面板访问码   HT4P-5527
    D 构建机临时令牌   RB9N-6014

只依赖标准库。token 长度用 P 的 `/tokenize` 校准（proxy 不提供该端点）。

典型用法：
    python3 tools/ced_pd_acceptance.py \
        --base-url http://127.0.0.1:18992 \
        --tokenize-url http://127.0.0.1:18990 \
        --model deepseek-v41-ced-pd \
        --corpus data/hongloumeng.txt \
        --mode all --context-tokens 144000,1000000 \
        --out results/ced_acceptance.json

退出码：0 = 所有正确性判据通过；1 = 有失败；2 = 用法/连接错误。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

SYSTEM = (
    "你是一个严谨的中文助手。回答要直接、简短；"
    '被要求"只给密码/口令/访问码/令牌"时就只输出它本身，不要解释。'
)

NEEDLES = [
    ("A", "【运维备忘 A】机房门禁密码是 ZQ7K-3341，仅限值班人员使用。"),
    ("B", "【运维备忘 B】备份账号口令是 VX2M-8890，有效期 30 天。"),
    ("C", "【运维备忘 C】监控面板访问码是 HT4P-5527，请勿外传。"),
    ("D", "【运维备忘 D】构建机临时令牌是 RB9N-6014，用完即废。"),
]
NEEDLE_Q = {
    "A": ("运维备忘 A 里的机房门禁密码是什么？只给密码本身。", "ZQ7K-3341"),
    "B": ("运维备忘 B 里的备份账号口令是什么？只给口令本身。", "VX2M-8890"),
    "C": ("运维备忘 C 里的监控面板访问码是什么？只给访问码本身。", "HT4P-5527"),
    "D": ("运维备忘 D 里的构建机临时令牌是什么？只给令牌本身。", "RB9N-6014"),
}
KEY_TEXT = {key: text for key, text in NEEDLES}
SHORT_Q = ("校验码是 ZQ7K-3341。请只回复这个校验码。", "ZQ7K-3341")


# ----------------------------------------------------------------- HTTP
def post_json(url: str, payload: dict, timeout: float):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, raw, time.monotonic() - started, body
    except urllib.error.HTTPError as error:
        return error.code, error.read(), time.monotonic() - started, body
    except Exception as error:  # noqa: BLE001 - 连接层失败也要留证据
        return -1, json.dumps({"transport_error": repr(error)}).encode(), (
            time.monotonic() - started
        ), body


def stream_chat(url: str, payload: dict, timeout: float):
    """跑一次 SSE；返回 (状态, 文本, 首 token 延迟, usage, finish_reason)。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    started = time.monotonic()
    first = None
    chunks: list[str] = []
    usage = None
    finish_reason = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    text = (choice.get("delta") or {}).get("content") or ""
                    if text:
                        if first is None:
                            first = time.monotonic()
                        chunks.append(text)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
    except Exception as error:  # noqa: BLE001
        return -1, "", None, None, repr(error), body
    return (
        status,
        "".join(chunks),
        None if first is None else first - started,
        usage,
        finish_reason,
        body,
    )


def count_tokens(tokenize_url: str, model: str, prompt: str, timeout: float = 300.0) -> int:
    status, raw, _wall, _body = post_json(
        tokenize_url.rstrip("/") + "/tokenize",
        {"model": model, "prompt": prompt},
        timeout,
    )
    if status != 200:
        raise RuntimeError(f"/tokenize 失败：status={status} body={raw[:200]!r}")
    data = json.loads(raw)
    return int(data.get("count") or len(data.get("tokens") or []))


# ----------------------------------------------------------------- 语料
def load_corpus(path: str) -> str:
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        if text.strip():
            return text
    raise SystemExit(f"语料不可用：{path!r}（用 --corpus 指定，如 data/hongloumeng.txt）")


def slice_for_tokens(
    tokenize_url: str, model: str, corpus: str, want_tokens: int, offset: int = 0
) -> tuple[str, int]:
    """按目标 token 数切语料；用 /tokenize 迭代校准。长度不足时响亮失败。"""
    if want_tokens <= 0:
        return "", 0
    chars = max(1, want_tokens)
    prompt = ""
    count = 0
    for _ in range(6):
        rotated = corpus[offset % len(corpus):] + corpus[: offset % len(corpus)]
        body = (rotated * (chars // max(1, len(rotated)) + 1))[:chars]
        prompt = body
        count = count_tokens(tokenize_url, model, prompt)
        if abs(count - want_tokens) <= max(128, want_tokens // 200):
            break
        chars = max(1, round(chars * want_tokens / max(1, count)))
    else:
        raise SystemExit(
            f"语料长度校准失败：目标 {want_tokens}，实得 {count}（检查 corpus 与 tokenizer）"
        )
    return prompt, count


def embed_needles(body: str, keys: list[str]) -> str:
    """把针按深度均匀插入 body（保持原文顺序）。必须按 key 取针。"""
    if not keys:
        return body
    marks: list[tuple[int, str]] = []
    for index, key in enumerate(keys):
        if key not in KEY_TEXT:
            raise KeyError(f"未知针 key={key!r}（合法：{sorted(KEY_TEXT)}）")
        position = int(len(body) * (index + 1) / (len(keys) + 1))
        marks.append((position, KEY_TEXT[key]))
    marks.sort()
    out: list[str] = []
    previous = 0
    for position, text in marks:
        out.append(body[previous:position])
        out.append("\n" + text + "\n")
        previous = position
    out.append(body[previous:])
    return "".join(out)


def chat_payload(model: str, user: str, max_tokens: int, stream: bool) -> dict:
    return {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }


# ----------------------------------------------------------------- 判据
def judge(answer: str, expected: str) -> bool:
    """答案里出现期望串，**且没有复述题面/针文本**，才算通过。

    只做子串匹配会被"把题面复述一遍"这种退化输出骗过（模型照抄含针的原文，
    期望串自然出现）。所以额外拒绝两类：
      1. 答案里出现针文本的特征词（`运维备忘` / `校验码是`）；
      2. 答案过长（正常只回一个码，>200 字符基本是复述或跑题）。
    """
    if expected not in answer:
        return False
    if len(answer) > 200:
        return False
    for label in ("运维备忘", "校验码是", "请只回复", "只给"):
        if label in answer:
            return False
    return True


def record(
    out_dir: str,
    tag: str,
    payload: dict,
    status: int,
    wall: float,
    answer: str,
    expected: str,
    ttft: float | None = None,
    usage: dict | None = None,
    finish_reason: str | None = None,
    body: bytes | None = None,
    note: str = "",
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, tag)
    if body is not None:
        with open(stem + ".request.json", "wb") as handle:
            handle.write(body)
        request_sha = hashlib.sha256(body).hexdigest()
    else:
        request_sha = ""
    result = {
        "tag": tag,
        "http_status": status,
        "wall_s": round(wall, 3),
        "ttft_s": None if ttft is None else round(ttft, 3),
        "answer": answer,
        "answer_repr": repr(answer)[:400],
        "expected": expected,
        "passed": judge(answer, expected),
        "answer_len": len(answer),
        "u_fffd": answer.count("\ufffd"),
        "usage": usage,
        "finish_reason": finish_reason,
        "request_sha256": request_sha,
        "note": note,
    }
    if usage and usage.get("completion_tokens"):
        decode_s = max(1e-6, wall - (ttft or 0.0))
        result["tpot_ms"] = round(1000.0 * decode_s / max(1, usage["completion_tokens"]), 2)
    with open(stem + ".result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


# ----------------------------------------------------------------- 模式
def run_short(args, results: list[dict]) -> None:
    question, expected = SHORT_Q
    payload = chat_payload(args.model, question, args.max_tokens, False)
    status, raw, wall, body = post_json(
        args.base_url.rstrip("/") + "/v1/chat/completions", payload, args.timeout
    )
    answer = ""
    usage = None
    try:
        data = json.loads(raw)
        choice = (data.get("choices") or [{}])[0]
        answer = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage")
    except Exception:  # noqa: BLE001
        pass
    results.append(record(args.out_dir, "short22", payload, status, wall, answer, expected,
                          usage=usage, body=body))


def run_needle(args, target: int, offset: int, repeat: int, results: list[dict]) -> None:
    base, base_count = slice_for_tokens(
        args.tokenize_url, args.model, args.corpus_text, target, offset
    )
    for rep in range(repeat):
        for key, (question, expected) in NEEDLE_Q.items():
            body = embed_needles(base, [key])
            prompt = body + "\n\n" + question
            prompt_tokens = count_tokens(args.tokenize_url, args.model, prompt)
            payload = chat_payload(args.model, prompt, args.max_tokens, False)
            status, raw, wall, sent = post_json(
                args.base_url.rstrip("/") + "/v1/chat/completions", payload, args.timeout
            )
            answer = ""
            usage = None
            try:
                data = json.loads(raw)
                choice = (data.get("choices") or [{}])[0]
                answer = (choice.get("message") or {}).get("content") or ""
                usage = data.get("usage")
            except Exception:  # noqa: BLE001
                pass
            tag = f"needle{target//1000}k_off{offset}_r{rep}_{key}"
            results.append(
                record(args.out_dir, tag, payload, status, wall, answer, expected,
                       usage=usage, body=sent,
                       note=f"corpus_prompt_tokens={base_count} full_prompt_tokens={prompt_tokens}")
            )
            print(
                f"    {tag}: {'PASS' if results[-1]['passed'] else 'FAIL'} "
                f"status={status} wall={wall:.1f}s prompt={prompt_tokens} "
                f"answer={answer[:40]!r}",
                flush=True,
            )


def run_stream(args, target: int, results: list[dict]) -> None:
    base, base_count = slice_for_tokens(
        args.tokenize_url, args.model, args.corpus_text, target, 0
    )
    body = embed_needles(base, ["D"])
    prompt = body + "\n\n" + NEEDLE_Q["D"][0]
    payload = chat_payload(args.model, prompt, args.max_tokens, True)
    status, answer, ttft, usage, finish_reason, sent = stream_chat(
        args.base_url.rstrip("/") + "/v1/chat/completions", payload, args.timeout
    )
    result = record(args.out_dir, f"stream{target//1000}k_D", payload, status,
                    (usage or {}).get("total_latency_s", 0.0) or 0.0, answer,
                    NEEDLE_Q["D"][1], ttft=ttft, usage=usage,
                    finish_reason=finish_reason, body=sent,
                    note=f"corpus_prompt_tokens={base_count}")
    results.append(result)
    print(
        f"    stream{target//1000}k_D: {'PASS' if result['passed'] else 'FAIL'} "
        f"status={status} ttft={ttft} answer={answer[:40]!r}",
        flush=True,
    )


def run_multiturn(args, target: int, results: list[dict]) -> None:
    """同一会话连续三轮；第一轮埋针，后两轮追问 + 复述，检查状态是否被带坏。"""
    base, base_count = slice_for_tokens(
        args.tokenize_url, args.model, args.corpus_text, max(4096, target // 8), 0
    )
    body = embed_needles(base, ["A", "B", "C", "D"])
    turns = [NEEDLE_Q["D"], NEEDLE_Q["A"], NEEDLE_Q["B"]]
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": body + "\n\n" + turns[0][0]},
    ]
    for turn, (_question, expected) in enumerate(turns, start=1):
        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        status, raw, wall, sent = post_json(
            args.base_url.rstrip("/") + "/v1/chat/completions", payload, args.timeout
        )
        answer = ""
        usage = None
        try:
            data = json.loads(raw)
            choice = (data.get("choices") or [{}])[0]
            answer = (choice.get("message") or {}).get("content") or ""
            usage = data.get("usage")
        except Exception:  # noqa: BLE001
            pass
        results.append(
            record(args.out_dir, f"multiturn{target//1000}k_t{turn}", payload, status, wall,
                   answer, expected, usage=usage, body=sent,
                   note=f"corpus_prompt_tokens={base_count}")
        )
        print(
            f"    multiturn t{turn}: {'PASS' if results[-1]['passed'] else 'FAIL'} "
            f"answer={answer[:40]!r}",
            flush=True,
        )
        messages.append({"role": "assistant", "content": answer})
        if turn < len(turns):
            # 下一轮问的是**下一个**针；把当前问题再问一遍会让判据与提问错位。
            messages.append({"role": "user", "content": turns[turn][0]})


def run_prefix(args, target: int, results: list[dict]) -> None:
    """同前缀连发两次：第二次应命中前缀缓存（usage.prompt_tokens_details.cached_tokens>0）。"""
    base, base_count = slice_for_tokens(
        args.tokenize_url, args.model, args.corpus_text, target, 0
    )
    body = embed_needles(base, ["C"])
    prompt = body + "\n\n" + NEEDLE_Q["C"][0]
    for attempt in (1, 2):
        payload = chat_payload(args.model, prompt, args.max_tokens, False)
        status, raw, wall, sent = post_json(
            args.base_url.rstrip("/") + "/v1/chat/completions", payload, args.timeout
        )
        answer = ""
        usage = None
        try:
            data = json.loads(raw)
            choice = (data.get("choices") or [{}])[0]
            answer = (choice.get("message") or {}).get("content") or ""
            usage = data.get("usage")
        except Exception:  # noqa: BLE001
            pass
        cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        results.append(
            record(args.out_dir, f"prefix{target//1000}k_a{attempt}", payload, status, wall,
                   answer, NEEDLE_Q["C"][1], usage=usage, body=sent,
                   note=f"corpus_prompt_tokens={base_count} cached_tokens={cached}")
        )
        print(
            f"    prefix a{attempt}: {'PASS' if results[-1]['passed'] else 'FAIL'} "
            f"cached_tokens={cached} wall={wall:.1f}s",
            flush=True,
        )


def collect_metrics(urls: list[str], timeout: float = 10.0) -> dict:
    """抓 P/D/proxy 的 /metrics 关键项，作为「资源开销」的粗口径证据。"""
    keys = (
        "vllm:kv_cache_usage_perc",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
    )
    snapshot: dict[str, dict[str, float]] = {}
    for url in urls:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=timeout) as resp:
                text = resp.read().decode("utf-8", "replace")
        except Exception as error:  # noqa: BLE001
            snapshot[url] = {"error": repr(error)}
            continue
        values: dict[str, float] = {}
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            for key in keys:
                if line.startswith(key):
                    try:
                        values[key] = float(line.rsplit(" ", 1)[1])
                    except (ValueError, IndexError):
                        pass
        snapshot[url] = values
    return snapshot


def parse_contexts(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def selfcheck() -> int:
    """用一个内置 mock 服务验证 runner 自身：判据、SSE 解析、证据落盘。

    mock 的行为是「在 prompt 里找到哪条针就回哪个码」，因此正确性判据应当全过；
    另外验证一条**故意答错**的请求会被判 FAIL（负控），避免判据永远为真。
    """
    import http.server
    import socketserver
    import tempfile
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a):  # 静音
            return

        def _send(self, code: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path.endswith("/tokenize"):
                prompt = payload.get("prompt", "")
                self._send(200, {"count": max(1, len(prompt) // 3)})
                return
            messages = payload.get("messages") or []
            text = "\n".join(str(m.get("content", "")) for m in messages)
            answer = ""
            # 真实模型回答的是**最近一次**提问：取所有针问题里出现位置最靠后的那个。
            best_at, best_code = -1, ""
            for _key, (question, code) in NEEDLE_Q.items():
                at = text.rfind(question)
                if at > best_at:
                    best_at, best_code = at, code
            if best_at >= 0:
                answer = best_code
            if not answer and SHORT_Q[0] in text:
                answer = SHORT_Q[1]
            if "故意答错" in text:
                answer = "WRONG"
            if "复述题面" in text:
                # 退化输出：把含针的原文整段照抄 ⇒ 旧的子串判据会假通过。
                answer = text[-400:]
            if "长答案" in text:
                # 退化输出：答案里确实含码，但拖沓到 200 字符以上。
                answer = best_code + "，" + ("废话" * 200)
            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for piece in (answer[:2], answer[2:]):
                    chunk = {"choices": [{"delta": {"content": piece}}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                    b'"usage":{"completion_tokens":7,"prompt_tokens":10}}\n\n'
                )
                self.wfile.write(b"data: [DONE]\n\n")
                return
            self._send(200, {
                "choices": [{"message": {"role": "assistant", "content": answer},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": len(answer),
                          "total_tokens": 10 + len(answer)},
            })

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                base_url=base, tokenize_url=base, model="mock", corpus="",
                corpus_text=("这是一段用于自检的中文语料。" * 5000), mode="all",
                context_tokens=[1000], max_tokens=16, timeout=30.0, repeat=1,
                offset=0, out=os.path.join(tmp, "out.json"),
                out_dir=os.path.join(tmp, "evidence"), metrics_urls="",
            )
            results: list[dict] = []
            run_short(args, results)
            run_needle(args, 1000, 0, 1, results)
            run_stream(args, 1000, results)
            run_prefix(args, 1000, results)
            run_multiturn(args, 1000, results)

            # 三条负控，都必须被判 FAIL：
            #   1. 答错；
            #   2. 复述题面（含期望串，但明显是照抄）；
            #   3. 拖沓长答案（含期望串，但远超正常长度）。
            results.append(record(args.out_dir, "negative_wrong", {}, 200, 1.0,
                                  "WRONG", "ZQ7K-3341"))
            results.append(record(
                args.out_dir, "negative_echo", {}, 200, 1.0,
                "【运维备忘 A】机房门禁密码是 ZQ7K-3341，仅限值班人员使用。", "ZQ7K-3341"))
            results.append(record(
                args.out_dir, "negative_long", {}, 200, 1.0,
                "ZQ7K-3341，" + ("废话" * 200), "ZQ7K-3341"))
            server.shutdown()

    failed = [r for r in results if not r["passed"]]
    expected_negative = [r for r in failed if r["tag"].startswith("negative_")]
    print(f"[selfcheck] 共 {len(results)} 条，失败 {len(failed)} 条 "
          f"（其中负控 {len(expected_negative)} 条）")
    for row in results:
        print(f"  {row['tag']:>22} {'PASS' if row['passed'] else 'FAIL'} "
              f"answer={row['answer'][:24]!r}")
    ok = len(failed) == 3 and len(expected_negative) == 3 and all(
        r["request_sha256"] for r in results if not r["tag"].startswith("negative_")
    )
    print("[selfcheck] " + ("通过 ✅" if ok else "不通过 ❌"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="PD proxy（或单实例）URL")
    parser.add_argument("--tokenize-url", default="", help="P 的 URL（提供 /tokenize）")
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", default="data/hongloumeng.txt")
    parser.add_argument("--mode", default="all",
                        choices=["all", "short", "needle", "stream", "multiturn", "prefix"])
    parser.add_argument("--context-tokens", default="144000,1000000")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--repeat", type=int, default=1, help="needle 模式重复次数")
    parser.add_argument("--offset", type=int, default=0, help="语料起点偏移")
    parser.add_argument("--out", default="", help="汇总 JSON 路径")
    parser.add_argument("--out-dir", default="", help="逐请求证据目录（默认 out 同名目录）")
    parser.add_argument("--metrics-urls", default="",
                        help="逗号分隔的 /metrics 端点，用于记录资源开销快照")
    parser.add_argument("--selfcheck", action="store_true",
                        help="用内置 mock 服务验证 runner 自身（不连真实服务）")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="只做长度校准并打印（不发起推理请求）")
    args = parser.parse_args()

    if args.selfcheck:
        return selfcheck()
    if not args.tokenize_url:
        args.tokenize_url = args.base_url
    args.corpus_text = load_corpus(args.corpus)

    if args.calibrate_only:
        # 投前检查：确认 /tokenize 可用、语料够长、目标长度能收敛。
        contexts = parse_contexts(args.context_tokens)
        print(f"[calibrate] tokenize={args.tokenize_url} corpus={args.corpus} "
              f"({len(args.corpus_text)} chars)")
        probe = "你好，这是一次 tokenize 自检。"
        n_probe = count_tokens(args.tokenize_url, args.model, probe)
        print(f"[calibrate] 探针 {probe!r} -> {n_probe} tokens")
        bad = 0
        for target in contexts:
            base, got = slice_for_tokens(
                args.tokenize_url, args.model, args.corpus_text, target, args.offset
            )
            keys = list(NEEDLE_Q)
            for key in keys:
                full = embed_needles(base, [key]) + "\n\n" + NEEDLE_Q[key][0]
                full_n = count_tokens(args.tokenize_url, args.model, full)
                err = abs(full_n - target) / max(1, target)
                flag = "OK " if err <= 0.005 + 128 / max(1, target) else "FAIL"
                if flag == "FAIL":
                    bad += 1
                print(f"[calibrate] 目标 {target:>8}  针 {key}  base={got:>8} "
                      f"含提问={full_n:>8}  偏差={err*100:.3f}%  {flag}")
        print(f"[calibrate] 不达标项 {bad}（需 0）")
        return 0 if bad == 0 else 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    args.out = args.out or f"results/ced_acceptance_{stamp}.json"
    args.out_dir = args.out_dir or os.path.splitext(args.out)[0] + "_evidence"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    metrics_before = collect_metrics(args.metrics_urls.split(",")) if args.metrics_urls else {}
    results: list[dict] = []
    contexts = parse_contexts(args.context_tokens)
    print(f"[ced-accept] base={args.base_url} tokenize={args.tokenize_url} "
          f"mode={args.mode} contexts={contexts} out={args.out}", flush=True)

    if args.mode in ("all", "short"):
        print("  -- 22-token 短针 --", flush=True)
        run_short(args, results)
    if args.mode in ("all", "needle"):
        for target in contexts:
            print(f"  -- needle {target} --", flush=True)
            run_needle(args, target, args.offset, args.repeat, results)
    if args.mode in ("all", "stream"):
        for target in contexts:
            print(f"  -- stream {target} --", flush=True)
            run_stream(args, target, results)
    if args.mode in ("all", "multiturn"):
        for target in contexts:
            print(f"  -- multiturn {target} --", flush=True)
            run_multiturn(args, target, results)
    if args.mode in ("all", "prefix"):
        for target in contexts:
            print(f"  -- prefix {target} --", flush=True)
            run_prefix(args, target, results)

    metrics_after = collect_metrics(args.metrics_urls.split(",")) if args.metrics_urls else {}
    failed = [r for r in results if not r["passed"]]
    summary = {
        "started_from": args.base_url,
        "model": args.model,
        "mode": args.mode,
        "context_tokens": contexts,
        "total": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print()
    print(f"{'tag':>28} {'status':>7} {'wall_s':>9} {'ttft_s':>8} {'tpot_ms':>8} {'verdict':>8}")
    for row in results:
        print(f"{row['tag']:>28} {row['http_status']:>7} {row['wall_s']:>9.1f} "
              f"{str(row['ttft_s']):>8} {str(row.get('tpot_ms')):>8} "
              f"{'PASS' if row['passed'] else 'FAIL':>8}")
    print(f"\n总计 {len(results)} 条，通过 {len(results) - len(failed)}，失败 {len(failed)}")
    print(f"汇总已写 {args.out}；逐请求证据在 {args.out_dir}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
