#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent 轨迹精度门 —— 用真实 agent 形态的请求测"是否调工具"。

**为什么需要它**：仓库里既有的精度测试全是"一问一答"（GSM8K/Vision），
而线上暴露退化的是 **agent 轨迹**（长工具输出 + 多工具定义 + 多轮）。
两者失败模式完全不同：问答测试看答案对不对，本门看**模型还发不发工具调用**。

设计要点（都是踩过的坑）：

  1. **prompt 必须逐字节固定**。用随机 nonce 会让每次请求变成不同 prompt，
     无法区分"状态漂移"与"prompt 差异"。
  2. **判据要看 `finish_reason`**。被 `max_tokens` 截断的样本（`finish=length`）
     不算失败也不算成功 —— 单独统计，否则会被误读成退化。
  3. **必须支持重复 N 次**（默认 10），报 Wilson 置信区间，而不是单发结论。
  4. **脚本自身不含真实控制符**（全角竖线等一律用 \\u 转义构造），
     否则本文件会变成"读到就中毒"的地雷。

用法：
    python3 accuracy_gate.py --base http://127.0.0.1:8020 --reps 10 \\
        --arms canary,8k,32k,128k,256k,real \\
        --corpus /path/to/hongloumeng.txt --out results/accuracy_gate.json

产出 JSON 里每个 arm 一条记录：成功数 / 有效数 / 截断数 / 失败签名分布。
把修复前后两份 JSON 对比即可满足"统计显著的前后对比"。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

FULLWIDTH_BAR = "\uFF5C"          # 全角竖线（控制符的组成部分，此处仅用于转义构造）
DSML = FULLWIDTH_BAR + "DSML" + FULLWIDTH_BAR


# --------------------------------------------------------------------------
# 请求构造
# --------------------------------------------------------------------------

def tool_read():
    return {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "file path"}},
                "required": ["path"],
            },
        },
    }


def tool_bash():
    return {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run commands in a bash shell",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "bash command"}},
                "required": ["command"],
            },
        },
    }


def synth_messages(doc: str, tools):
    """自洽的 agent 轨迹：用户要求依次读两个文件；README 已读过（工具结果=doc）。
    正确行为：再发一次 read(DESIGN.md)。"""
    return [
        {"role": "system", "content": "You are a helpful software engineer assistant."},
        {"role": "user", "content": "请依次读取 README.md 和 DESIGN.md。"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": json.dumps({"path": "README.md"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": doc},
    ]


def build_doc(corpus: str, n_chars: int) -> str:
    if not corpus:
        corpus = "占位文本。" * 100
    return (corpus * (n_chars // max(1, len(corpus)) + 1))[:n_chars]


# --------------------------------------------------------------------------
# 判据
# --------------------------------------------------------------------------

_CYRILLIC = re.compile(r"[\u0400-\u04ff]")
_ARABIC = re.compile(r"[\u0600-\u06ff]")


def judge(resp: dict, expect_tool: str | None) -> dict:
    """返回 {'verdict': 'PASS'|'FAIL'|'TRUNCATED', 'signature': str, ...}"""
    ch = resp["choices"][0]
    msg = ch["message"]
    finish = ch.get("finish_reason")
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    content = msg.get("content") or ""
    text = reasoning + content
    tool_calls = msg.get("tool_calls") or []

    # 截断：既非成功也非失败，单独归类（否则会污染失败率）
    if finish == "length" and not tool_calls:
        return {"verdict": "TRUNCATED", "signature": "finish=length",
                "out_tokens": resp["usage"]["completion_tokens"]}

    if tool_calls:
        try:
            name = tool_calls[0]["function"]["name"]
            args = tool_calls[0]["function"]["arguments"]
            json.loads(args)
        except Exception as e:
            return {"verdict": "FAIL", "signature": f"malformed_tool_call:{type(e).__name__}",
                    "out_tokens": resp["usage"]["completion_tokens"]}
        if expect_tool and name != expect_tool:
            return {"verdict": "FAIL", "signature": f"wrong_tool:{name}",
                    "out_tokens": resp["usage"]["completion_tokens"]}
        return {"verdict": "PASS", "signature": f"tool:{name}",
                "out_tokens": resp["usage"]["completion_tokens"]}

    # 无工具调用 —— 细分成可诊断的签名
    if not text.strip():
        sig = "empty_output"
    elif _CYRILLIC.search(text) or _ARABIC.search(text):
        sig = "mixed_script"
    else:
        words = re.findall(r"\w+", text)
        uniq = len(set(words)) / max(1, len(words))
        sig = "repetitive" if uniq < 0.45 else "no_tool_call"
    return {"verdict": "FAIL", "signature": sig,
            "out_tokens": resp["usage"]["completion_tokens"]}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 置信区间 —— 小样本下比 k/n 正态近似稳。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def post(base: str, path: str, body: dict, timeout: int = 1800):
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run_arm(base: str, name: str, messages, tools, reps: int, max_tokens: int,
            expect_tool: str | None, temperature: float | None,
            reasoning_effort: str | None = None) -> dict:
    tally = Counter()
    sigs = Counter()
    outs = []
    prompt_tokens = None
    t0 = time.time()
    for i in range(reps):
        body = {
            "model": os.environ.get("GATE_MODEL", "deepseek-v41"),
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        # ★ 关键维度：不发 reasoning_effort 会落到 **chat 模式**（prompt 以 </think> 结尾），
        #   发则落到 thinking 模式。两者在长上下文下的工具调用行为完全不同
        #   （实测 97K：chat 0/5，thinking 5/5）。所以必须显式记录在结果里。
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        try:
            resp = post(base, "/v1/chat/completions", body)
        except urllib.error.HTTPError as e:
            tally["ERROR"] += 1
            sigs[f"http_{e.code}"] += 1
            continue
        except Exception as e:
            tally["ERROR"] += 1
            sigs[f"exc_{type(e).__name__}"] += 1
            continue
        if prompt_tokens is None:
            prompt_tokens = resp["usage"]["prompt_tokens"]
        v = judge(resp, expect_tool)
        tally[v["verdict"]] += 1
        sigs[v["signature"]] += 1
        outs.append(v["out_tokens"])
    valid = tally["PASS"] + tally["FAIL"]
    lo, hi = wilson(tally["PASS"], valid)
    return {
        "arm": name,
        "prompt_tokens": prompt_tokens,
        "reps": reps,
        "reasoning_effort": reasoning_effort,
        "pass": tally["PASS"],
        "fail": tally["FAIL"],
        "truncated": tally["TRUNCATED"],
        "error": tally["ERROR"],
        "valid": valid,
        "pass_rate": round(tally["PASS"] / valid, 4) if valid else None,
        "pass_rate_ci95": [round(lo, 4), round(hi, 4)],
        "fail_signatures": dict(sigs.most_common()),
        "out_tokens_median": sorted(outs)[len(outs) // 2] if outs else None,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8020")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--arms", default="canary,8k,32k,128k,256k,real")
    ap.add_argument("--corpus", default=None, help="长文本语料文件（重复填充到目标长度）")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=None,
                    help="不传则用服务端默认（通常 1.0）；对齐历史数据请传 0")
    ap.add_argument("--reasoning-effort", default="high",
                    help="传给服务端的 reasoning_effort。'none' 或不传 = chat 模式；"
                         "其余（low/high/xhigh/max）= thinking 模式。生产口径为 high")
    ap.add_argument("--real", action="append", default=[],
                    help="真实 dsh 会话重建的 prompt JSON（可多次）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    effort = None if a.reasoning_effort in ("", "none", "None") else a.reasoning_effort

    corpus = ""
    if a.corpus and os.path.exists(a.corpus):
        corpus = open(a.corpus, encoding="utf-8", errors="ignore").read()
        print(f"[gate] 语料 {len(corpus)} 字符：{a.corpus}")
    else:
        print("[gate] 未提供语料，使用占位文本（长度扫描仍有效）")

    tools = [tool_read(), tool_bash()]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    # 字符数 → 目标 token 数（中文约 0.76 tok/char，见 README 记录）
    char_for = {"8k": 10500, "32k": 42000, "128k": 168000, "256k": 340000}

    results = []
    print(f"[gate] base={a.base} reps={a.reps} arms={arms} "
          f"reasoning_effort={effort!r} temperature={a.temperature}")
    for arm in arms:
        if arm == "canary":
            msgs = synth_messages(build_doc(corpus, 128000), tools)
            r = run_arm(a.base, "canary", msgs, tools, a.reps, a.max_tokens, "read",
                        a.temperature, effort)
        elif arm in char_for:
            msgs = synth_messages(build_doc(corpus, char_for[arm]), tools)
            r = run_arm(a.base, arm, msgs, tools, a.reps, a.max_tokens, "read",
                        a.temperature, effort)
        elif arm == "real":
            for p in a.real:
                if not os.path.exists(p):
                    print(f"[gate] 跳过 {p}（不存在）")
                    continue
                d = json.load(open(p))
                r = run_arm(a.base, f"real:{os.path.basename(p)}",
                            d["messages"], d["tools"], a.reps, a.max_tokens, None,
                            a.temperature, effort)
                results.append(r)
                print(f"  {r['arm']:<40} pass={r['pass']}/{r['valid']} "
                      f"trunc={r['truncated']} rate={r['pass_rate']}")
            continue
        else:
            print(f"[gate] 未知 arm：{arm}，跳过")
            continue
        results.append(r)
        print(f"  {r['arm']:<40} pass={r['pass']}/{r['valid']} "
              f"trunc={r['truncated']} rate={r['pass_rate']} "
              f"ci95={r['pass_rate_ci95']}")

    out = {
        "base": a.base,
        "reps": a.reps,
        "max_tokens": a.max_tokens,
        "temperature": a.temperature,
        "reasoning_effort": effort,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arms": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[gate] 写入 {a.out}")

    # 打印一个可贴进报告的汇总表
    print("\n| arm | prompt_tokens | pass/valid | pass_rate | ci95 | truncated | 主要失败签名 |")
    print("|---|---:|---:|---:|---|---:|---|")
    for r in results:
        sig = ", ".join(f"{k}×{v}" for k, v in list(r["fail_signatures"].items())[:3])
        print(f"| {r['arm']} | {r['prompt_tokens']} | {r['pass']}/{r['valid']} | "
              f"{r['pass_rate']} | {r['pass_rate_ci95']} | {r['truncated']} | {sig} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
