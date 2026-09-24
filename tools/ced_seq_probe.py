#!/usr/bin/env python3
"""串行重复探针：对同一 chat 请求连续发送 N 次，抓取 token 级证据。

用途：CED D 图上偶发失败（1M 请求每第 4 次返回 content=null）需要区分
“模型采样出 EOS”“采样出普通 token 但被丢弃”“detokenizer/parser 丢内容”。
本工具在不改变模型前向的前提下，只给请求附加只读输出字段：

    logprobs / top_logprobs / return_token_ids

每次迭代都会落盘：
    <tag>_<i>.request.json   实际发送的请求体（字节级一致）
    <tag>_<i>.response.json  原始响应体
    <tag>_<i>.summary.json   解析后的关键字段与首 token 的 logprob 明细
    <tag>_<i>.transport.txt  http 状态码 / 墙钟耗时 / 字节数

该脚本不写任何服务端状态，只能通过 HTTP 影响服务，因此必须在明确知道
目标端口（proxy/prefill/decode）的前提下运行。
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_override(raw: str):
    key, _, value = raw.partition("=")
    if not key:
        raise SystemExit(f"--set 需要 key=value，收到 {raw!r}")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def build_request(template: dict, overrides: list[str], probe_fields: bool) -> dict:
    request = json.loads(json.dumps(template))
    if probe_fields:
        request["logprobs"] = True
        request["top_logprobs"] = 20
        request["return_token_ids"] = True
        request["include_reasoning"] = True
    for raw in overrides:
        key, value = parse_override(raw)
        request[key] = value
    return request


def post(url: str, body: bytes, timeout: float):
    http_request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    started = time.time()
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            return response.status, response.read(), time.time() - started
    except urllib.error.HTTPError as error:
        return error.code, error.read(), time.time() - started
    except Exception as error:  # 连接层失败也要留下证据，不能中断序列
        return -1, json.dumps({"transport_error": repr(error)}).encode("utf-8"), time.time() - started


def describe(raw: bytes) -> dict:
    """把响应压缩成可比对的关键字段，同时保留首 token 的 logprob 明细。"""
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as error:
        return {"parse_error": str(error), "raw_head": raw[:400].decode("utf-8", "replace")}

    summary: dict = {"id": payload.get("id")}
    choices = payload.get("choices") or []
    if not choices:
        summary["choices"] = []
        return summary

    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    reasoning = message.get("reasoning")
    token_ids = choice.get("token_ids")
    logprobs = (choice.get("logprobs") or {}).get("content") or []

    first_token = None
    if logprobs:
        head = logprobs[0]
        alternatives = []
        for entry in (head.get("top_logprobs") or [])[:5]:
            alternatives.append(
                {"token": entry.get("token"), "logprob": round(entry.get("logprob", 0.0), 6)}
            )
        first_token = {
            "token": head.get("token"),
            "logprob": round(head.get("logprob", 0.0), 6),
            "top_logprobs": alternatives,
        }

    summary.update(
        {
            "content": content,
            "content_repr": repr(content),
            "content_chars": len(content) if isinstance(content, str) else None,
            "reasoning": reasoning,
            "token_ids": token_ids,
            "token_ids_len": len(token_ids) if isinstance(token_ids, list) else None,
            "finish_reason": choice.get("finish_reason"),
            "stop_reason": choice.get("stop_reason"),
            "num_logprob_entries": len(logprobs),
            "first_token": first_token,
            "usage": payload.get("usage"),
            "u_fffd": (content or "").count("\ufffd") if isinstance(content, str) else 0,
        }
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, help="请求体模板 JSON 文件")
    parser.add_argument("--url", required=True, help="目标 chat/completions 端点")
    parser.add_argument("--outdir", required=True, help="证据落盘目录")
    parser.add_argument("--tag", required=True, help="本次实验标签（文件名前缀）")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--gap", type=float, default=0.0, help="两次请求之间的间隔秒数")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖请求字段（JSON 解析）"
    )
    parser.add_argument(
        "--no-probe-fields",
        action="store_true",
        help="不自动附加 logprobs/return_token_ids（用于复现原始请求 SHA）",
    )
    args = parser.parse_args()

    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    request = build_request(template, args.set, not args.no_probe_fields)
    body = json.dumps(request, ensure_ascii=False).encode("utf-8")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(
        f"[probe] tag={args.tag} count={args.count} url={args.url} bytes={len(body)} "
        f"outdir={outdir}",
        flush=True,
    )

    for index in range(1, args.count + 1):
        stem = f"{args.tag}_{index:02d}"
        started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        (outdir / f"{stem}.request.json").write_bytes(body)
        (outdir / f"{stem}.start.txt").write_text(f"{started}\n", encoding="utf-8")

        status, raw, wall = post(args.url, body, args.timeout)
        finished = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        (outdir / f"{stem}.response.json").write_bytes(raw)
        (outdir / f"{stem}.end.txt").write_text(f"{finished}\n", encoding="utf-8")
        (outdir / f"{stem}.transport.txt").write_text(
            f"http_status={status} wall_s={wall:.3f} bytes={len(raw)}\n", encoding="utf-8"
        )
        summary = describe(raw)
        summary["http_status"] = status
        summary["wall_s"] = round(wall, 3)
        summary["start"] = started
        summary["end"] = finished
        (outdir / f"{stem}.summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[probe] {stem} status={status} wall={wall:.1f}s "
            f"completion={((summary.get('usage') or {}).get('completion_tokens'))} "
            f"token_ids_len={summary.get('token_ids_len')} "
            f"content={summary.get('content_repr')!r}",
            flush=True,
        )
        if args.gap and index < args.count:
            time.sleep(args.gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
