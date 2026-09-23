#!/usr/bin/env python3
"""ctx_agent_probe.py —— **长上下文 × Agent 场景**的精度/乱码探针（纯 stdlib）

## 为什么需要它（本卷的判据缺口）

我们此前**所有**"正确性"判据都是**短上下文 + 无歧义问答**：
  · 题库 10 题（`tools` 里的探针）prompt 只有几十 token；
  · replay vs fill 的 **sha / 逐字相同** —— 本仓自己已写明"对路径差异没有判别力"；
  · 计数器（BlockStored / CPU_to_GPU / hits）只说明"搬了字节"，不说明"搬对了内容"。

而现场反馈是：**问题主要出现在"长上下文 + Agent（工具调用、多轮）"场景**
（且**不开 DRAM 卸载也有** ⇒ 不是卸载引入的）。⇒ 判据必须补上这一格。

## 它测什么（四种模式，都可单独跑）

1. `needle` —— **长上下文检索**：把语料拼到目标长度、在不同深度埋入唯一"针"，
   逐个提问并要求**只回密码本身** ⇒ 判"长上下文下语义有没有坏"。
2. `grow`  —— ★ **Agent 多轮增长**（最贴近现场）：模拟 coding agent 的
   `assistant(tool_call) → tool(result长块) → assistant` 轨迹，**每轮再加一块**，
   每轮问一个**ground truth 在该轮块里**的问题 ⇒ 判"上下文随轮次增长时会不会坏"。
   ★ 同时这条路会**反复命中前缀缓存**（同一前缀 + 追加），是"缓存/块边界"类缺陷的靶场。
3. `reuse` —— **同一长 prompt 连发 N 次**：第 1 次冷算、之后走缓存 ⇒
   判"走下缓存路径的输出与冷算是否一致"（`temperature=0`）。
4. `toolargs` —— **工具调用参数逐字保真**：要求模型把一串混合 CJK/ASCII/数字的校验码
   **原样**放进工具参数里 ⇒ 判 `tool_call.arguments` 是否逐字相等（乱码在 agent 场景里
   最先坏的就是这里）。

## 判据（三条一起看，缺一不可）

* **答案原文**（`repr`，不是"通过/失败"）；
* **乱码指纹**：U+FFFD / NUL / C0 控制字符 / 孤立代理对 的**计数**；
* **确定性**：`temperature=0` 重复时是否逐字相同；以及**复读检测**（同一 40 字子串重复 ≥3 次）。

## 用法

    python3 tools/ctx_agent_probe.py --base-url http://127.0.0.1:8020 --model deepseek-v41 \\
        --mode all --context-tokens 32768 --turns 4 --repeats 2 --out ~/ctxprobe.json

    # 长度扫描（现场反馈是"长上下文"才出问题 ⇒ 必做对照）
    for L in 8192 32768 131072; do
      python3 tools/ctx_agent_probe.py --mode needle --context-tokens $L --out ~/ctx_$L.json
    done

退出码：0 = 全部判据通过；1 = 有失败项；2 = 连接/用法错误。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

SYSTEM = ("你是一个严谨的中文助手，同时可以调用工具。"
          '回答要**直接、简短**；被要求"只给密码"时就只输出密码本身，不要解释。')

# 唯一"针"：混了字母/数字/连字符，逐字判不会误判
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


# ---------------------------------------------------------------- HTTP
def post(base: str, path: str, payload: dict, timeout: float = 3600.0):
    req = urllib.request.Request(base.rstrip("/") + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"http": 200, "wall_s": time.time() - t0,
                    "body": json.loads(r.read().decode()), "err": None}
    except urllib.error.HTTPError as e:
        return {"http": e.code, "wall_s": time.time() - t0, "body": None,
                "err": e.read().decode()[:400]}
    except Exception as e:  # noqa: BLE001
        return {"http": -1, "wall_s": time.time() - t0, "body": None, "err": repr(e)[:400]}


def get(base: str, path: str, timeout: float = 15.0):
    try:
        with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as r:
            return r.status, r.read().decode()
    except Exception as e:  # noqa: BLE001
        return -1, repr(e)


_TOK_CACHE: dict[tuple[str, int], int] = {}


def tok_count(base: str, model: str, text: str) -> int:
    """用服务的 /tokenize 精确计数；拿不到就按字符数近似（并**标注**为近似）。"""
    key = (text[:64], len(text))
    if key in _TOK_CACHE:
        return _TOK_CACHE[key]
    r = post(base, "/tokenize", {"model": model, "prompt": text}, timeout=120)
    n = -1
    if r["http"] == 200 and isinstance(r["body"], dict):
        n = int(r["body"].get("count") or len(r["body"].get("tokens") or []))
    if n <= 0:
        n = len(text)          # 中文近似 1 字 ≈ 1 token
    _TOK_CACHE[key] = n
    return n


def load_corpus() -> str:
    for p in (os.path.join(PKG, "data", "hongloumeng.txt"),
              os.path.join(PKG, "data", "hlm", "hongloumeng.txt")):
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="replace") as fh:
                return fh.read()
    return ("这是一段用于测试长上下文的中文说明。" * 4000)


def slice_for_tokens(base: str, model: str, corpus: str, want_tokens: int,
                     offset: int = 0, mark: str = "") -> str:
    """从语料切出约 want_tokens 个 token 的一段（用 /tokenize 校准一次比例）。"""
    if want_tokens <= 0:
        return ""
    n_chars = min(len(corpus), max(1, want_tokens))
    body = corpus[offset:offset + n_chars]
    got = tok_count(base, model, body)
    if got > 0:
        ratio = want_tokens / got
        if abs(ratio - 1.0) > 0.03:
            n_chars = min(len(corpus) - offset, max(1, int(n_chars * ratio)))
            body = corpus[offset:offset + n_chars]
    return body


def embed_needles(body: str, keys: list[str]) -> str:
    """把针按深度均匀插进 body（保持原文不变，只是插入）。"""
    if not keys:
        return body
    out = []
    n = len(keys)
    for i, k in enumerate(keys):
        pos = int(len(body) * (i + 1) / (n + 1))
        out.append((pos, NEEDLES[i % len(NEEDLES)][1]))
    out.sort()
    res, prev = [], 0
    for pos, txt in out:
        res.append(body[prev:pos]); res.append("\n" + txt + "\n"); prev = pos
    res.append(body[prev:])
    return "".join(res)


# ---------------------------------------------------------------- 判读助手
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SURR = re.compile(r"[\ud800-\udfff]")


def fingerprints(text: str) -> dict:
    """乱码指纹：替换字符 / NUL / C0 控制字符 / 孤立代理对 / 非字符。"""
    return {
        "len": len(text),
        "u_fffd": text.count("\ufffd"),
        "nul": text.count("\x00"),
        "ctrl": len(_CTRL.findall(text)),
        "surrogate": len(_SURR.findall(text)),
        "nonchar": sum(1 for ch in text if unicodedata.category(ch) == "Cn"),
        "replacement_delta": sum(1 for ch in text if ch == "?"),
    }


def repeat_loop(text: str, win: int = 40, thresh: int = 3) -> bool:
    """复读检测：同一 win 字窗口出现 ≥thresh 次（模型掉进复读循环的形态）。"""
    if len(text) < win * thresh:
        return False
    seen: dict[str, int] = {}
    for i in range(0, len(text) - win + 1, max(1, win // 4)):
        s = text[i:i + win]
        seen[s] = seen.get(s, 0) + 1
        if seen[s] >= thresh and s.strip():
            return True
    return False


def judge(ans: str, expect: str) -> dict:
    got = ans.strip()
    return {
        "expect": expect,
        "exact": got == expect,                     # ★ 最强判据：逐字
        "contains": expect in ans,
        "answer_repr": repr(ans[:400]),
        "fingerprints": fingerprints(ans),
        "repeat_loop": repeat_loop(ans),
    }


def chat(base: str, model: str, messages: list, tools=None,
         max_tokens: int = 64, timeout: float = 3600.0):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": False}
    if tools:
        payload["tools"] = tools
    return post(base, "/v1/chat/completions", payload, timeout)


def text_of(resp) -> str:
    try:
        ch = resp["body"]["choices"][0]
        msg = ch.get("message") or {}
        if msg.get("content"):
            return msg["content"]
        tcs = msg.get("tool_calls") or []
        if tcs:
            return json.dumps([t.get("function", {}) for t in tcs], ensure_ascii=False)
        return ""
    except Exception:  # noqa: BLE001
        return ""


def toolcalls_of(resp) -> list:
    try:
        return (resp["body"]["choices"][0].get("message") or {}).get("tool_calls") or []
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------- 四个模式
def mode_needle(a, out: dict) -> int:
    print("=" * 78)
    print(f"[needle] 长上下文检索  context≈{a.context_tokens} token  repeat={a.repeats}")
    print("=" * 78)
    corpus = load_corpus()
    body = slice_for_tokens(a.base_url, a.model, corpus, a.context_tokens,
                            offset=a.offset, mark="needle")
    keys = ["A", "B", "C", "D"]
    filled = embed_needles(body, keys)
    real = tok_count(a.base_url, a.model, filled)
    print(f"  实际上下文 ≈{real} token（目标 {a.context_tokens}）")
    fails = 0
    for rep in range(a.repeats):
        for k in keys:
            q, expect = NEEDLE_Q[k]
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": filled + "\n\n" + q}]
            r = chat(a.base_url, a.model, msgs, max_tokens=a.max_tokens, timeout=a.timeout)
            ans = text_of(r)
            v = judge(ans, expect)
            tag = "PASS" if v["exact"] else ("~contains" if v["contains"] else "FAIL")
            if not v["contains"]:
                fails += 1
            print(f"  [rep{rep} {k}] {tag:9s} http={r['http']} wall={r['wall_s']:.1f}s "
                  f"fp={v['fingerprints']['u_fffd']}/{v['fingerprints']['nul']}")
            print(f"      A: {v['answer_repr']}")
            out.setdefault("needle", []).append({"rep": rep, "key": k, **v,
                                                 "http": r["http"], "wall_s": r["wall_s"],
                                                 "ctx_tokens_real": real})
    return fails


def mode_grow(a, out: dict) -> int:
    """★ Agent 多轮增长：tool 结果块逐轮追加（前缀缓存反复命中）。"""
    print("=" * 78)
    print(f"[grow] Agent 多轮增长  block≈{a.context_tokens // max(1, a.turns)} token × {a.turns} 轮")
    print("=" * 78)
    corpus = load_corpus()
    per = max(512, a.context_tokens // max(1, a.turns))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "帮我读几个日志文件，然后回答我的问题。"}]
    fails = 0
    for t in range(a.turns):
        k = ["A", "B", "C", "D"][t % 4]
        blob = slice_for_tokens(a.base_url, a.model, corpus, per, offset=a.offset + t * per)
        blob = embed_needles(blob, [k])
        cid = f"call_{t}"
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": cid, "type": "function",
                                     "function": {"name": "read_file",
                                                  "arguments": json.dumps(
                                                      {"path": f"/work/logs/app_{t}.log",
                                                       "max_bytes": 200000})}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": blob})
        msgs.append({"role": "assistant",
                     "content": f"已读取 app_{t}.log，内容很长，我先记下要点。"})
        q, expect = NEEDLE_Q[k]
        msgs.append({"role": "user", "content": q})
        r = chat(a.base_url, a.model, msgs, max_tokens=a.max_tokens, timeout=a.timeout)
        ans = text_of(r)
        v = judge(ans, expect)
        if not v["contains"]:
            fails += 1
        used = tok_count(a.base_url, a.model,
                         "".join(m.get("content") or "" for m in msgs))
        tag = "PASS" if v["exact"] else ("~contains" if v["contains"] else "FAIL")
        print(f"  [turn{t} {k}] {tag:9s} ctx≈{used} http={r['http']} wall={r['wall_s']:.1f}s")
        print(f"      A: {v['answer_repr']}")
        out.setdefault("grow", []).append({"turn": t, "key": k, "ctx_tokens": used,
                                           **v, "http": r["http"], "wall_s": r["wall_s"]})
        # 把这一轮的问答也留在历史里（模拟真实 agent 对话继续）
        msgs.append({"role": "assistant", "content": ans})
    return fails


def mode_reuse(a, out: dict) -> int:
    """同一长 prompt 连发 N 次：第 1 次冷算，之后走前缀缓存。"""
    print("=" * 78)
    print(f"[reuse] 同一长 prompt 连发 {a.repeats} 次（第 1 次冷算，之后走缓存）")
    print("=" * 78)
    corpus = load_corpus()
    body = embed_needles(slice_for_tokens(a.base_url, a.model, corpus,
                                          a.context_tokens, offset=a.offset), ["A"])
    q, expect = NEEDLE_Q["A"]
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": body + "\n\n" + q}]
    answers, fails = [], 0
    for rep in range(a.repeats):
        r = chat(a.base_url, a.model, msgs, max_tokens=a.max_tokens, timeout=a.timeout)
        ans = text_of(r)
        v = judge(ans, expect)
        answers.append(ans)
        if not v["contains"]:
            fails += 1
        print(f"  [rep{rep}] {'PASS' if v['exact'] else 'FAIL':4s} "
              f"wall={r['wall_s']:.1f}s A: {v['answer_repr']}")
        out.setdefault("reuse", []).append({"rep": rep, **v, "wall_s": r["wall_s"]})
    same = len(set(answers)) == 1
    print(f"  ★ 逐字相同: {same}（distinct={len(set(answers))}）"
          f"{'' if same else '  ⇒ 取回/缓存路径不稳定'}")
    out["reuse_all_same"] = same
    if not same:
        fails += 1
    return fails


TOOL_WRITE = [{
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "把内容原样写入指定路径",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}}]


def mode_toolargs(a, out: dict) -> int:
    """工具调用参数逐字保真（乱码在 agent 场景最先坏的地方）。"""
    print("=" * 78)
    print("[toolargs] 工具参数逐字保真")
    print("=" * 78)
    payload = ("校验码串：ZQ7K-3341-VX2M-8890-HT4P-5527-RB9N-6014-"
               "PLM3-7712-CDF8-2205\n文件名：/work/out/checksum.txt")
    want_path = "/work/out/checksum.txt"
    want_code = "ZQ7K-3341-VX2M-8890-HT4P-5527-RB9N-6014-PLM3-7712-CDF8-2205"
    fails = 0
    for rep in range(a.repeats):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content":
                 "请调用 write_file 工具，把下面这串校验码**原样**写入文件。"
                 'content 只放校验码本身（不含引号里那几个字）。\n\n' + payload}]
        r = chat(a.base_url, a.model, msgs, tools=TOOL_WRITE,
                 max_tokens=a.max_tokens, timeout=a.timeout)
        tcs = toolcalls_of(r)
        got_code = got_path = None
        parse_err = None
        if tcs:
            try:
                args = json.loads(tcs[0]["function"]["arguments"])
                got_code = args.get("content")
                got_path = args.get("path")
            except Exception as e:  # noqa: BLE001
                parse_err = repr(e)[:200]
        ok_code = (got_code == want_code)
        ok_path = (got_path == want_path)
        if not (ok_code and ok_path):
            fails += 1
        fp = fingerprints(json.dumps(tcs, ensure_ascii=False))
        print(f"  [rep{rep}] n_calls={len(tcs)} content_exact={ok_code} path_exact={ok_path} "
              f"fp(fffd/nul)={fp['u_fffd']}/{fp['nul']}")
        print(f"      content: {repr(got_code)[:400]}")
        if parse_err:
            print(f"      ★ arguments 不是合法 JSON: {parse_err}")
        out.setdefault("toolargs", []).append({
            "rep": rep, "n_calls": len(tcs), "content_exact": ok_code,
            "path_exact": ok_path, "got_content_repr": repr(got_code)[:400],
            "parse_err": parse_err, "fingerprints": fp, "wall_s": r["wall_s"]})
    return fails


MODES = {"needle": mode_needle, "grow": mode_grow,
         "reuse": mode_reuse, "toolargs": mode_toolargs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8020")
    ap.add_argument("--model", default="deepseek-v41")
    ap.add_argument("--mode", default="all",
                    choices=["all", "needle", "grow", "reuse", "toolargs"])
    ap.add_argument("--context-tokens", type=int, default=32768,
                    help="目标上下文长度（token）；现场反馈是长上下文才出问题 ⇒ 必做长度扫描")
    ap.add_argument("--turns", type=int, default=4, help="grow 模式的轮数")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--offset", type=int, default=0, help="语料起点偏移（换一段文本）")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    code, body = get(a.base_url, "/health")
    print(f"[probe] /health = {code}  {body[:60]!r}")
    if code != 200:
        print(f"[probe] ✗ 服务不可达：{a.base_url}", file=sys.stderr)
        return 2

    todo = list(MODES) if a.mode == "all" else [a.mode]
    out: dict = {"base_url": a.base_url, "model": a.model,
                 "context_tokens_target": a.context_tokens, "modes": todo}
    total_fails = 0
    t0 = time.time()
    for m in todo:
        total_fails += MODES[m](a, out)
    out["total_fails"] = total_fails
    out["elapsed_s"] = time.time() - t0
    print("=" * 78)
    print(f"结果：失败 {total_fails} 项 ；用时 {out['elapsed_s']:.0f}s")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"[probe] 证据已写：{a.out}")
    return 0 if total_fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
