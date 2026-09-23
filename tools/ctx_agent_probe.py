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
# ★ 键 → 针文本（**必须按 key 取**）。第一版 `embed_needles` 用的是**位置下标**取针
#   `NEEDLES[i % len(NEEDLES)]` ⇒ `grow` 模式里问 B/C/D 时插进去的仍是 **A**。
#   现场表现极好认：模型回答"app_1.log 里没有运维备忘 B，**只有重复出现的运维备忘 A**"
#   —— **模型是对的，判据是错的**。⇒ 现在按 key 取，并加 `--selfcheck` 把这类错钉死。
KEY_TEXT = {k: t for (k, t) in NEEDLES}


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


def build_context(corpus: str, want_chars: int, rotate: int = 0) -> str:
    """★ 按需长度**平铺**语料（不是"切一行算一行"）。

    为什么必须平铺（2026-09-23 实测踩到，属"静默降级"）：
      语料 `data/hongloumeng.txt` 是 2.47 MB 的 UTF-8 中文 ⇒ **只有 826,639 个字符**。
      而我要 520k token 的上下文时，代码写成 `corpus[offset:offset+n]`，
      一旦 `offset` 超过语料长度就**切出空串**；`embed_needles` 再把 4 条针插进去，
      于是实际上下文只有 **91 个 token** —— 而探针**照样报 PASS**（"答案对了"）。
      ⇒ 这是"判据自己骗自己"：拿一个 91 token 的请求冒充 520k 的请求。
    现在：先把语料按 `rotate` 旋转（让不同 lane 拿到**不同内容**），再平铺到目标长度。
    """
    if want_chars <= 0 or not corpus:
        return ""
    c = corpus[rotate:] + corpus[:rotate]
    need = want_chars // len(c) + 1
    return (c * need)[:want_chars]


def slice_for_tokens(base: str, model: str, corpus: str, want_tokens: int,
                     offset: int = 0, mark: str = "") -> str:
    """给出约 want_tokens 个 token 的上下文（用 /tokenize 校准一次比例）。

    ★ 长度不足时**响亮失败**：返回空串会让上层把"短上下文"当成"长上下文"来判。
    """
    if want_tokens <= 0:
        return ""
    # 第一版：按 token 数≈字符数起手，再校准
    n_chars = max(1, want_tokens)
    body = build_context(corpus, n_chars, rotate=offset % max(1, len(corpus)))
    got = tok_count(base, model, body)
    if got > 0:
        ratio = want_tokens / got
        if abs(ratio - 1.0) > 0.03:
            n_chars = max(1, int(n_chars * ratio))
            body = build_context(corpus, n_chars, rotate=offset % max(1, len(corpus)))
    return body


def ctx_ratio_ok(real_tokens: int, target_tokens: int, min_ratio: float) -> bool:
    """上下文长度是否真的到位（不达标 ⇒ 这次读数**不能用来判对错**）。"""
    if target_tokens <= 0:
        return True
    return real_tokens >= target_tokens * min_ratio


def embed_needles(body: str, keys: list[str]) -> str:
    """把针按深度均匀插进 body（保持原文不变，只是插入）。"""
    if not keys:
        return body
    out: list[tuple[int, str]] = []
    n = len(keys)
    for i, k in enumerate(keys):
        if k not in KEY_TEXT:
            raise KeyError(f"未知的针 key={k!r}（合法：{sorted(KEY_TEXT)}）")
        pos = int(len(body) * (i + 1) / (n + 1))
        out.append((pos, KEY_TEXT[k]))
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


def repeat_loop(text: str, win: int = 40, thresh: int = 3, cover: float = 0.5) -> bool:
    """复读检测：**同一非重叠窗口**重复 ≥thresh 次，**且占全文 ≥cover**。

    ★ 两个参数都是被自检"逼"出来的（`--selfcheck` 抓到的过敏感误报）：
      第一版用 `步长=win//4`（重叠采样）+ 只看"出现 ≥3 次" ⇒ 一句**正常**的中文
      重复 6 遍（156 字）就被判成复读循环（它每个 40 字窗口都出现多次）。
      ⇒ 现在：① **非重叠**采样（步长 = win，避免"同一段被数成多次"）；
              ② 还要求**重复内容覆盖半篇以上**（真复读会占满输出；列表/表格类
                 的规律性重复不会）。
    """
    if len(text) < win * thresh:
        return False
    seen: dict[str, int] = {}
    for i in range(0, len(text) - win + 1, win):
        s = text[i:i + win]
        if not s.strip():
            continue
        seen[s] = seen.get(s, 0) + 1
    if not seen:
        return False
    top = max(seen.values())
    return top >= thresh and top * win >= cover * len(text)


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


def chat_stream(base: str, model: str, messages: list, tools=None,
                max_tokens: int = 64, timeout: float = 3600.0):
    """★ **真流式**（SSE）请求 —— 真实 Agent 客户端用的就是这条（`stream: true`）。

    为什么必须单独测（2026-09-23）：我此前**所有**请求都是 `stream:false`，
    而现场是 Agent 场景 ⇒ 输出走的是**增量 SSE 路径**：每个 chunk 都要经过
    detokenizer + 增量解码 +（若开了工具）tool-call 参数拼接。
    把"非流式干净"当成"Agent 场景干净"就是**用错的判据**（本仓已栽多类同族）。

    返回：{"http", "wall_s", "text", "tool_calls", "n_chunks", "first_chunk_s",
           "finish_reason", "err", "raw_tail"}
    """
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True}
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream"})
    t0 = time.time()
    text_parts: list[str] = []
    tc_acc: dict[int, dict] = {}
    n_chunks = 0
    first = None
    finish = ""
    raw_tail: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return {"http": r.status, "wall_s": time.time() - t0, "text": "",
                        "tool_calls": [], "n_chunks": 0, "first_chunk_s": None,
                        "finish_reason": "", "err": r.read().decode()[:400],
                        "raw_tail": []}
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    obj = json.loads(line)
                except Exception:  # noqa: BLE001
                    raw_tail.append(line[:200])
                    if len(raw_tail) > 5:
                        raw_tail.pop(0)
                    continue
                n_chunks += 1
                if first is None:
                    first = time.time() - t0
                for ch in obj.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    d = ch.get("delta") or {}
                    if d.get("content"):
                        text_parts.append(d["content"])
                    # 注意：raw 模式下增量在 `text` 而不是 `delta.content`
                    if d.get("text"):
                        text_parts.append(d["text"])
                    for tc in d.get("tool_calls") or []:
                        i = int(tc.get("index") or 0)
                        slot = tc_acc.setdefault(i, {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
                if obj.get("usage"):
                    pass
        return {"http": 200, "wall_s": time.time() - t0,
                "text": "".join(text_parts),
                "tool_calls": [tc_acc[k] for k in sorted(tc_acc)],
                "n_chunks": n_chunks, "first_chunk_s": first,
                "finish_reason": finish, "err": None, "raw_tail": raw_tail}
    except Exception as e:  # noqa: BLE001
        return {"http": -1, "wall_s": time.time() - t0, "text": "".join(text_parts),
                "tool_calls": [tc_acc[k] for k in sorted(tc_acc)], "n_chunks": n_chunks,
                "first_chunk_s": first, "finish_reason": finish,
                "err": repr(e)[:400], "raw_tail": raw_tail}


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


def finish_of(resp) -> str:
    try:
        return resp["body"]["choices"][0].get("finish_reason") or ""
    except Exception:  # noqa: BLE001
        return ""


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
                 max_tokens=a.tool_max_tokens, timeout=a.timeout)
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
        raw_args = json.dumps([t.get("function", {}) for t in tcs],
                              ensure_ascii=False)[:800]
        print(f"  [rep{rep}] n_calls={len(tcs)} content_exact={ok_code} path_exact={ok_path} "
              f"fp(fffd/nul)={fp['u_fffd']}/{fp['nul']} finish={finish_of(r)!r}")
        print(f"      content: {repr(got_code)[:400]}")
        if not ok_code:
            print(f"      ★ 原始 arguments: {raw_args}")
        if parse_err:
            print(f"      ★ arguments 不是合法 JSON: {parse_err}")
        out.setdefault("toolargs", []).append({
            "rep": rep, "n_calls": len(tcs), "content_exact": ok_code,
            "path_exact": ok_path, "got_content_repr": repr(got_code)[:400],
            "raw_arguments_repr": raw_args, "finish_reason": finish_of(r),
            "parse_err": parse_err, "fingerprints": fp, "wall_s": r["wall_s"]})
    return fails


def _metrics(base: str, keys: tuple) -> dict:
    """抓几条引擎指标（用于证明"缓存真的被灌满/逐出"，不是靠猜）。"""
    code, txt = get(base, "/metrics", timeout=20)
    out = {}
    if code != 200:
        return out
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for k in keys:
            if line.startswith(k + " ") or line.startswith(k + "{"):
                try:
                    out[line.split()[0]] = float(line.split()[-1])
                except Exception:  # noqa: BLE001
                    pass
    return out


def mode_evict(a, out: dict) -> int:
    """★ 长会话被**逐出后重算**：先冷算一条长 prompt，再灌爆缓存，然后**重问同一条**。

    为什么单列（真机教训）：此前 `reuse` 模式两次都能命中缓存 ⇒ 只证明了"命中时一致"。
    真实 Agent 会话会**把缓存灌满**（多会话/长上下文），前缀块被逐出 ⇒ 之后再问同一条
    会走**部分重算**路径。`ENGRAM` 的 pad 历史降级、APC 边界对齐这类缺陷都藏在这条路上。
    """
    print("=" * 78)
    print(f"[evict] 逐出后重算：ctx≈{a.context_tokens} ×(1 冷算 + {a.fill_sessions} 灌爆 + 1 重问)")
    print("=" * 78)
    corpus = load_corpus()
    q, expect = NEEDLE_Q["A"]
    body = embed_needles(slice_for_tokens(a.base_url, a.model, corpus,
                                          a.context_tokens, offset=a.offset), ["A"])
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": body + "\n\n" + q}]

    mk = ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc",
          "vllm:prefix_cache_hit_rate", "vllm:prefix_cache_queries_total",
          "vllm:prefix_cache_hits_total")
    m0 = _metrics(a.base_url, mk)
    r1 = chat(a.base_url, a.model, msgs, max_tokens=a.max_tokens, timeout=a.timeout)
    ans1 = text_of(r1)
    v1 = judge(ans1, expect)
    m1 = _metrics(a.base_url, mk)
    print(f"  [cold ] {'PASS' if v1['exact'] else 'FAIL':4s} wall={r1['wall_s']:.1f}s "
          f"A: {v1['answer_repr']}")

    # 灌爆：N 段**互不相同**的长上下文（每段各自埋针，顺便验证没串味）
    keys = ["A", "B", "C", "D"]
    n_bad_fill = 0
    for i in range(a.fill_sessions):
        off = a.offset + 200000 + i * (a.context_tokens + 1000)
        k = keys[i % 4]
        b = embed_needles(slice_for_tokens(a.base_url, a.model, corpus,
                                           a.context_tokens, offset=off), [k])
        qq, ee = NEEDLE_Q[k]
        rr = chat(a.base_url, a.model,
                  [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": b + "\n\n" + qq}],
                  max_tokens=a.max_tokens, timeout=a.timeout)
        aa = text_of(rr)
        hit = ee in aa
        if not hit:
            n_bad_fill += 1
        print(f"  [fill{i}] {'PASS' if hit else 'FAIL':4s} wall={rr['wall_s']:.1f}s A: {repr(aa)[:120]}")
    m2 = _metrics(a.base_url, mk)

    r3 = chat(a.base_url, a.model, msgs, max_tokens=a.max_tokens, timeout=a.timeout)
    ans3 = text_of(r3)
    v3 = judge(ans3, expect)
    same = ans1.strip() == ans3.strip()
    print(f"  [again] {'PASS' if v3['exact'] else 'FAIL':4s} wall={r3['wall_s']:.1f}s "
          f"A: {v3['answer_repr']}")
    print(f"  ★ 与冷算逐字相同: {same}")
    print(f"  ★ 指标 冷算前={m0}")
    print(f"          冷算后={m1}")
    print(f"          灌爆后={m2}")

    fails = 0
    if not v3["contains"]:
        fails += 1
    if not same:
        fails += 1
    if n_bad_fill:
        fails += n_bad_fill
    out["evict"] = [{"phase": "cold", **v1, "wall_s": r1["wall_s"],
                     "metrics": m1},
                    {"phase": "again", **v3, "wall_s": r3["wall_s"],
                     "metrics": m2, "same_as_cold": same,
                     "fill_sessions": a.fill_sessions, "bad_fills": n_bad_fill,
                     "metrics_before": m0}]
    return fails


def mode_conc(a, out: dict) -> int:
    """★ 并发长上下文（真 Agent 会**并排**发多个工具请求）：C 路不同上下文同时打。"""
    import threading
    print("=" * 78)
    print(f"[conc] 并发 {a.conc} 路（各自 ctx≈{a.context_tokens}、各自的针）")
    print("=" * 78)
    corpus = load_corpus()
    keys = ["A", "B", "C", "D"]
    results: list = [None] * a.conc

    def worker(i: int):
        k = keys[i % 4]
        off = a.offset + i * (a.context_tokens + 1000)
        b = embed_needles(slice_for_tokens(a.base_url, a.model, corpus,
                                          a.context_tokens, offset=off), [k])
        qq, ee = NEEDLE_Q[k]
        rr = chat(a.base_url, a.model,
                  [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": b + "\n\n" + qq}],
                  max_tokens=a.max_tokens, timeout=a.timeout)
        aa = text_of(rr)
        results[i] = {"idx": i, "key": k, "expected": ee, "answer_repr": repr(aa)[:200],
                      "hit": ee in aa, "exact": aa.strip() == ee,
                      "http": rr["http"], "wall_s": rr["wall_s"],
                      "fingerprints": fingerprints(aa), "repeat_loop": repeat_loop(aa)}

    ths = [threading.Thread(target=worker, args=(i,)) for i in range(a.conc)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    el = time.time() - t0
    fails = 0
    for r in results:
        if not r or not r["hit"]:
            fails += 1
        if r:
            print(f"  [{r['idx']} {r['key']}] {'PASS' if r['exact'] else ('~hit' if r['hit'] else 'FAIL'):7s} "
                  f"wall={r['wall_s']:.1f}s fp(fffd/nul)={r['fingerprints']['u_fffd']}/{r['fingerprints']['nul']}")
            if not r["hit"]:
                print(f"      A: {r['answer_repr']}")
    print(f"  并发 {a.conc} 路总墙钟 {el:.1f}s")
    out["conc"] = results
    return fails


def mode_bigprefill(a, out: dict) -> int:
    """★★★ **用户现场的可复现场景**：一次 Agent 的 context ≈520k 直接进入，
    完整做一次 prefill（可能还有其它流同时请求），然后出现错误。

    为什么单列（2026-09-23 用户反馈）：
      · 我此前的所有扫描最大只到 131K，而现场是 **≈520k 单发**；
      · `--max-num-batched-tokens 8192` 会把 520k 切成 **~64 个 chunk** 依次做
        ⇒ 走的是**分块 prefill** 路径（与 128K 以下的块数完全不同）；
      · 现场还提到"可能还有其它流同时请求" ⇒ 所以本模式提供 `--conc` 并行臂。

    判据（与其它模式一致的三条）：逐字命中 + 乱码指纹 + 复现次数。
    **每次都用一段不同的语料**（offset 递增）⇒ 不是"同一个 prompt 的缓存效应"。
    """
    import threading
    print("=" * 78)
    print(f"[bigprefill] 单发 ≈{a.context_tokens} token 的**完整 prefill**"
          f"（并发 {a.conc} 路），重复 {a.repeats} 轮")
    print("=" * 78)
    corpus = load_corpus()
    keys = ["A", "B", "C", "D"]
    results: list = []
    lock = threading.Lock()

    def one(idx: int, rep: int, path_tag: str):
        """idx 决定语料偏移与针；每个线程跑完整一轮（含前置一次性大请求）。"""
        k = keys[idx % 4]
        off = a.offset + 1000000 + (rep * a.conc + idx) * (a.context_tokens + 2000)
        body = slice_for_tokens(a.base_url, a.model, corpus, a.context_tokens, offset=off)
        # 针埋 4 个位置（20/40/60/80%），只问其中一个 ⇒ 同时暴露"某段丢了"这类错
        body = embed_needles(body, keys)
        q, expect = NEEDLE_Q[k]
        real = tok_count(a.base_url, a.model, body)
        t0 = time.time()
        rr = chat(a.base_url, a.model,
                  [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": body + "\n\n" + q}],
                  max_tokens=a.max_tokens, timeout=a.timeout)
        ans = text_of(rr)
        v = judge(ans, expect)
        # ★★ 上下文长度门：不达标 ⇒ 这次读数**无效**（不许拿 91 token 冒充 520k）
        _ok_ctx = ctx_ratio_ok(real, a.context_tokens, a.min_ctx_ratio)
        row = {"rep": rep, "lane": idx, "key": k, "path": path_tag,
               "ctx_tokens_real": real, "ctx_ok": _ok_ctx,
               "ctx_target": a.context_tokens,
               "http": rr["http"], "wall_s": rr["wall_s"] if False else time.time() - t0,
               "err": rr.get("err"), **v}
        with lock:
            results.append(row)

    for rep in range(a.repeats):
        # ★ 先**串行**发一路（= 现场"一次 agent 的 context 直接进入"），
        #   然后再按 --conc 并发打（= 现场"可能还有其它流同时请求"）。
        print(f"\n  --- round {rep}：先单路大 prefill，再 {a.conc} 路并发 ---")
        one(rep % 4, rep, "single")
        r0 = results[-1]
        tag = "PASS" if r0["exact"] else ("~hit" if r0["contains"] else "FAIL")
        print(f"  [rep{rep} single {r0['key']}] {tag:7s} ctx≈{r0['ctx_tokens_real']} "
              f"wall={r0['wall_s']:.1f}s http={r0['http']} "
              f"fp={r0['fingerprints']['u_fffd']}/{r0['fingerprints']['nul']}")
        if not r0["contains"]:
            print(f"      A: {r0['answer_repr']}")

        ths = [threading.Thread(target=one, args=((rep + 1 + i) % 4, rep, f"conc{i}"))
               for i in range(a.conc)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        for r in results[-(a.conc):]:
            tag = "PASS" if r["exact"] else ("~hit" if r["contains"] else "FAIL")
            print(f"  [rep{rep} {r['path']} {r['key']}] {tag:7s} "
                  f"wall={r['wall_s']:.1f}s http={r['http']} "
                  f"fp={r['fingerprints']['u_fffd']}/{r['fingerprints']['nul']}")
            if not r["contains"]:
                print(f"      A: {r['answer_repr']}")

    fails = sum(1 for r in results if not r["contains"])
    bad_fp = sum(1 for r in results if r["fingerprints"]["u_fffd"] or r["fingerprints"]["nul"])
    loops = sum(1 for r in results if r["repeat_loop"])
    bad_ctx = sum(1 for r in results if not r.get("ctx_ok", True))
    print(f"\n  ★ 汇总：{len(results)} 次请求 / 未命中 {fails} / 带乱码指纹 {bad_fp} / 复读 {loops}"
          f" / ★上下文不达标 {bad_ctx}")
    if bad_ctx:
        print(f"  ⛔ 有 {bad_ctx} 次请求的上下文**没到目标长度**（判据无效，不是通过）"
              f" —— 检查语料长度与 build_context 的平铺")
    out["bigprefill"] = results
    out["bigprefill_summary"] = {"n": len(results), "fails": fails,
                                 "with_garbling_fp": bad_fp, "repeat_loops": loops,
                                 "bad_ctx": bad_ctx,
                                 "target_tokens": a.context_tokens, "conc": a.conc}
    return fails + bad_fp + loops + bad_ctx


def mode_biggrow(a, out: dict) -> int:
    """★★★ **大 prefill 之后再走多轮**（更贴近用户现场的第二半）。

    为什么单列：`bigprefill` 只判"那一次大 prefill 的答案对不对"，
    但现场描述是"**完成一次 prefill 之后**就会出现错误" ⇒ 错误可能出现在**后续轮次**，
    尤其当后续请求**复用那 520k 前缀**时（前缀缓存 + 分块 prefill 的边界）。
    本模式：先用一条 520k 的 Agent 轨迹做一次完整 prefill，然后**在同一会话里追问 3 轮**
    （每轮问一个埋在不同深度的针）⇒ 覆盖"复用大前缀"的路径。
    """
    import threading
    print("=" * 78)
    print(f"[biggrow] 大 prefill({a.context_tokens}) → 同会话追问 {a.followups} 轮"
          f"（并发 {a.conc} 路）")
    print("=" * 78)
    corpus = load_corpus()
    keys = ["A", "B", "C", "D"]
    fails = 0
    results: list = []
    lock = threading.Lock()

    def lane(idx: int, rep: int):
        k0 = keys[idx % 4]
        rot = a.offset + 3000000 + (rep * a.conc + idx) * 7919
        # ★ 必须用 slice_for_tokens（它会用 /tokenize **校准**到目标 token 数）；
        #   直接用 build_context(…, a.context_tokens) 是"字符数当 token 数"，
        #   实测这段语料 1 字 ≈ 0.73 token ⇒ 520k 字符只有 379k token（长度门会拦下来）。
        body = slice_for_tokens(a.base_url, a.model, corpus, a.context_tokens, offset=rot)
        body = embed_needles(body, keys)          # 四针埋在 20/40/60/80%
        # —— 造"一次 Agent 的 context"：system + user + tool 往返 + 最后提问
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "这是一次长时间的编码会话记录，请先读进去。"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c0", "type": "function",
                                 "function": {"name": "read_file",
                                              "arguments": json.dumps(
                                                  {"path": f"/work/session_{idx}.log"},
                                                  ensure_ascii=False)}}]},
                {"role": "tool", "tool_call_id": "c0", "content": body}]
        q0, e0 = NEEDLE_Q[k0]
        msgs.append({"role": "user", "content": q0})
        t0 = time.time()
        r0 = chat(a.base_url, a.model, msgs, max_tokens=a.max_tokens, timeout=a.timeout)
        a0 = text_of(r0)
        v0 = judge(a0, e0)
        real = tok_count(a.base_url, a.model, body)
        row0 = {"rep": rep, "lane": idx, "phase": "prefill", "key": k0,
                "ctx_tokens_real": real,
                "ctx_ok": ctx_ratio_ok(real, a.context_tokens, a.min_ctx_ratio),
                "ctx_target": a.context_tokens, "http": r0["http"],
                "wall_s": time.time() - t0, **v0}
        with lock:
            results.append(row0)
        # —— 追问：把答案也留在历史里（模拟真实会话），每轮问**另一个深度**的针
        msgs.append({"role": "assistant", "content": a0})
        for fi in range(a.followups):
            k = keys[(idx + 1 + fi) % 4]
            qq, ee = NEEDLE_Q[k]
            msgs.append({"role": "user", "content": qq})
            t1 = time.time()
            rr = chat(a.base_url, a.model, msgs, max_tokens=a.max_tokens, timeout=a.timeout)
            aa = text_of(rr)
            vv = judge(aa, ee)
            row = {"rep": rep, "lane": idx, "phase": f"followup{fi}", "key": k,
                   "ctx_tokens_real": real,
                   "ctx_ok": row0["ctx_ok"], "ctx_target": a.context_tokens,
                   "http": rr["http"], "wall_s": time.time() - t1, **vv}
            with lock:
                results.append(row)
            msgs.append({"role": "assistant", "content": aa})

    for rep in range(a.repeats):
        ths = [threading.Thread(target=lane, args=(i, rep)) for i in range(a.conc)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
    for r in results:
        tag = "PASS" if r["exact"] else ("~hit" if r["contains"] else "FAIL")
        if not r["contains"]:
            fails += 1
        print(f"  [rep{r['rep']} lane{r['lane']} {r['phase']:9s} {r['key']}] {tag:7s} "
              f"ctx≈{r['ctx_tokens_real']} wall={r['wall_s']:.1f}s http={r['http']} "
              f"fp={r['fingerprints']['u_fffd']}/{r['fingerprints']['nul']}")
        if not r["contains"]:
            print(f"      A: {r['answer_repr']}")
    bad_fp = sum(1 for r in results if r["fingerprints"]["u_fffd"] or r["fingerprints"]["nul"])
    bad_ctx = sum(1 for r in results if not r.get("ctx_ok", True))
    print(f"\n  ★ 汇总：{len(results)} 次 / 未命中 {fails} / 乱码 {bad_fp} / 上下文不达标 {bad_ctx}")
    out["biggrow"] = results
    out["biggrow_summary"] = {"n": len(results), "fails": fails,
                              "with_garbling_fp": bad_fp, "bad_ctx": bad_ctx,
                              "target_tokens": a.context_tokens, "conc": a.conc,
                              "followups": a.followups}
    return fails + bad_fp + bad_ctx


def mode_mixed(a, out: dict) -> int:
    """★★★ **混合负载**：一个 520k 大 prefill 在飞的同时，**其它流**在打短请求。

    为什么单列（用户原话："完整进行一次 prefill（**可能还要有其他的流同时请求**），
    然后就会出现错误"）：`bigprefill`/`biggrow` 里每条流都是"自己一个大上下文"，
    而现场更可能是——**一个 agent 的大 prefill 占着引擎**，同时**别的会话**在正常聊天。
    分块 prefill（520k / 8192 ≈ 64 个 chunk）与并发 decode 交错时，
    任何"块边界 / 状态共享"类缺陷都会先在这个短流上炸（它最容易看出乱码）。

    判据：① 大 prefill 那一路；② **每条短流**逐字命中 + 乱码指纹 + 复读；③ 短流在
    "大 prefill 期间 vs 空闲时"的对照（同一问题、同一 seed=0，逐字相同才算稳）。
    """
    import threading
    print("=" * 78)
    print(f"[mixed] 1 路 {a.context_tokens} 大 prefill ＋ {a.conc} 路短流并发")
    print("=" * 78)
    corpus = load_corpus()
    keys = ["A", "B", "C", "D"]
    results: list = []
    lock = threading.Lock()

    def big():
        rot = a.offset + 5000000
        body = slice_for_tokens(a.base_url, a.model, corpus, a.context_tokens, offset=rot)
        body = embed_needles(body, keys)
        q, e = NEEDLE_Q["A"]
        real = tok_count(a.base_url, a.model, body)
        t0 = time.time()
        r = chat(a.base_url, a.model,
                 [{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": body + "\n\n" + q}],
                 max_tokens=a.max_tokens, timeout=a.timeout)
        ans = text_of(r)
        with lock:
            results.append({"phase": "big", "key": "A", "ctx_tokens_real": real,
                            "ctx_ok": ctx_ratio_ok(real, a.context_tokens, a.min_ctx_ratio),
                            "ctx_target": a.context_tokens, "http": r["http"],
                            "wall_s": time.time() - t0, **judge(ans, e)})

    def short(i: int):
        k = keys[i % 4]
        q, e = NEEDLE_Q[k]
        # 短上下文（几百 token）—— 现场"别的会话在正常聊天"
        body = embed_needles(corpus[:600], [k])
        t0 = time.time()
        r = chat(a.base_url, a.model,
                 [{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": body + "\n\n" + q}],
                 max_tokens=a.max_tokens, timeout=a.timeout)
        ans = text_of(r)
        with lock:
            results.append({"phase": f"short{i}", "key": k,
                            "ctx_tokens_real": tok_count(a.base_url, a.model, body),
                            "ctx_ok": True, "ctx_target": 0, "http": r["http"],
                            "wall_s": time.time() - t0, **judge(ans, e)})

    # 先量一次"空闲时"的短流基线（对照锚）
    print("  --- 基线：空闲时（没有大 prefill）---")
    for i in range(a.conc):
        short(i)
    base_rows = [r for r in results if r["phase"].startswith("short")]
    for r in base_rows:
        print(f"  [idle {r['phase']}] {'PASS' if r['exact'] else 'FAIL':4s} A: {r['answer_repr'][:80]}")

    # ★ 混合：大 prefill 起一个线程，短流**立刻**跟上（不等它）
    print("  --- 混合：大 prefill 在飞 + 短流同时打 ---")
    tb = threading.Thread(target=big)
    tb.start()
    time.sleep(1.0)                      # 确保大 prefill 已经开始排队/在跑
    ths = [threading.Thread(target=short, args=(i,)) for i in range(a.conc)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    tb.join()

    for r in results[len(base_rows):]:
        tag = "PASS" if r["exact"] else ("~hit" if r["contains"] else "FAIL")
        print(f"  [{r['phase']:6s} {r['key']}] {tag:7s} wall={r['wall_s']:.1f}s "
              f"http={r['http']} fp={r['fingerprints']['u_fffd']}/{r['fingerprints']['nul']}")
        if not r["contains"]:
            print(f"      A: {r['answer_repr']}")
    # 短流对照：混合期的答案 vs 空闲期的答案（同一问题，应逐字相同）
    mixed_shorts = {r["phase"]: r for r in results if r["phase"].startswith("short")
                    and r is not None}
    diffs = []
    for i, br in enumerate(base_rows):
        mr = next((r for r in results if r["phase"] == f"short{i}"
                   and r["answer_repr"] != br["answer_repr"]), None)
    print(f"\n  ★ 汇总：{len(results)} 次请求；空闲基线 {len(base_rows)} 条；"
          f"失败 {sum(1 for r in results if not r['contains'])}")
    out["mixed"] = results
    fails = sum(1 for r in results if not r["contains"])
    bad_fp = sum(1 for r in results if r["fingerprints"]["u_fffd"] or r["fingerprints"]["nul"])
    bad_ctx = sum(1 for r in results if not r.get("ctx_ok", True))
    out["mixed_summary"] = {"n": len(results), "fails": fails,
                            "with_garbling_fp": bad_fp, "bad_ctx": bad_ctx,
                            "target_tokens": a.context_tokens, "conc": a.conc}
    return fails + bad_fp + bad_ctx


def mode_stream(a, out: dict) -> int:
    """★★★ **真流式（SSE）+ 工具调用** —— 真实 Agent 客户的走法，此前完全没测。

    用户现场："一次 agent 的 context ≈520k 直接进入，完整做一次 prefill
    （**可能还要有其他的流同时请求**）" —— "流"在这里极可能指 **streaming 请求**，
    而 Agent 还会**带 tools 定义**并期望**流式返回工具调用参数**。

    三条子臂（都在同一次运行里）：
      ① `big`      ：520k 上下文 + `stream:true`，问埋在中段的针 ⇒ 判**流式拼接后的全文**；
      ② `tools`    ：520k 上下文 + `tools` + `stream:true`，要求把校验码原样写入 ⇒
                     判**流式拼出来的 tool_call.arguments 是否逐字**
                     （增量拼接是最容易坏的地方：分片边界、多字节字符、JSON 转义）；
      ③ `concurrent`：大流在飞时，另起 N 路**短流**（同样 stream:true）⇒ 判并发下短流是否被污染。
    """
    import threading
    print("=" * 78)
    print(f"[stream] 真流式 + 工具：520k 单流 ＋ {a.conc} 路并发短流")
    print("=" * 78)
    corpus = load_corpus()
    keys = ["A", "B", "C", "D"]
    results: list = []
    lock = threading.Lock()
    fails = 0

    def rec(row):
        nonlocal fails
        with lock:
            results.append(row)
            if not row.get("ok", True):
                fails += 1

    # ① 大流：520k + stream（带 tools，模拟真实 agent）
    rot = a.offset + 7000000
    body = slice_for_tokens(a.base_url, a.model, corpus, a.context_tokens, offset=rot)
    body = embed_needles(body, keys)
    real = tok_count(a.base_url, a.model, body)
    ctx_ok = ctx_ratio_ok(real, a.context_tokens, a.min_ctx_ratio)
    print(f"  实测上下文 ≈{real} token（目标 {a.context_tokens}）ctx_ok={ctx_ok}")

    big_q, big_e = NEEDLE_Q["B"]
    big_msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": body + "\n\n" + big_q}]

    big_row: dict = {}

    def run_big():
        r = chat_stream(a.base_url, a.model, big_msgs, tools=TOOL_WRITE,
                        max_tokens=a.max_tokens, timeout=a.timeout)
        v = judge(r["text"], big_e)
        big_row.update({
            "phase": "big-stream", "ctx_tokens_real": real, "ctx_ok": ctx_ok,
            "ctx_target": a.context_tokens, "http": r["http"], "wall_s": r["wall_s"],
            "n_chunks": r["n_chunks"], "first_chunk_s": r["first_chunk_s"],
            "finish_reason": r["finish_reason"], "err": r["err"],
            "raw_tail": r["raw_tail"][:3], "ok": bool(v["contains"]) and ctx_ok, **v})
        rec(big_row)

    def run_short(i: int):
        k = keys[i % 4]
        q, e = NEEDLE_Q[k]
        b = embed_needles(corpus[:600], [k])
        r = chat_stream(a.base_url, a.model,
                        [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": b + "\n\n" + q}],
                        max_tokens=a.max_tokens, timeout=a.timeout)
        v = judge(r["text"], e)
        rec({"phase": f"short-stream{i}", "key": k, "ctx_tokens_real": -1, "ctx_ok": True,
             "ctx_target": 0, "http": r["http"], "wall_s": r["wall_s"],
             "n_chunks": r["n_chunks"], "first_chunk_s": r["first_chunk_s"],
             "finish_reason": r["finish_reason"], "err": r["err"],
             "ok": bool(v["contains"]) and bool(v["fingerprints"]["u_fffd"] == 0
                                              and v["fingerprints"]["nul"] == 0), **v})

    tb = threading.Thread(target=run_big)
    tb.start()
    time.sleep(1.0)
    ths = [threading.Thread(target=run_short, args=(i,)) for i in range(a.conc)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    tb.join()

    # ② 大流 + 工具：要求它把校验码原样写进工具参数（流式增量拼接）
    want_code = "ZQ7K-3341-VX2M-8890-HT4P-5527-RB9N-6014-PLM3-7712-CDF8-2205"
    want_path = "/work/out/checksum.txt"
    tool_msgs = [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content":
                  "这是本次会话的日志内容（很长）。读完请调用 write_file 工具："
                  "把下面的校验码**原样**写入文件，content 只放校验码本身。\n"
                  f"文件名：{want_path}\n校验码串：{want_code}\n\n" + body}]
    # ★★ **对照臂（必须）**：完全相同的工具请求，只把上下文换成**短**的。
    #   为什么必须有：若"520k 下没调工具"，有两种可能 ——
    #     ① 模型在长上下文下行为退化（= 我们要找的 bug）；
    #     ② 我的 prompt 本身就让模型不可靠地选择"文字回答 vs 工具调用"（= 探针的锅）。
    #   唯一能分开两者的办法 = **同一 prompt、只改上下文长度**做对照。
    short_tool_msgs = [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content":
                        "这是本次会话的日志内容（很短）。读完请调用 write_file 工具："
                        "把下面的校验码**原样**写入文件，content 只放校验码本身。\n"
                        f"文件名：{want_path}\n校验码串：{want_code}\n\n" + corpus[:600]}]
    sr = chat_stream(a.base_url, a.model, short_tool_msgs, tools=TOOL_WRITE,
                     max_tokens=a.tool_max_tokens, timeout=a.timeout)
    s_code = s_path = None
    if sr["tool_calls"]:
        try:
            _sa = json.loads(sr["tool_calls"][0].get("arguments") or "{}")
            s_code, s_path = _sa.get("content"), _sa.get("path")
        except Exception:  # noqa: BLE001
            pass
    rec({"phase": "short-stream+tools(对照)", "ctx_tokens_real": -1, "ctx_ok": True,
         "ctx_target": 0, "http": sr["http"], "wall_s": sr["wall_s"],
         "n_chunks": sr["n_chunks"], "first_chunk_s": sr["first_chunk_s"],
         "finish_reason": sr["finish_reason"], "err": sr["err"],
         "ok": bool(s_code == want_code and s_path == want_path),
         "expect": want_code, "exact": bool(s_code == want_code),
         "contains": bool(s_code == want_code),
         "answer_repr": repr(s_code)[:200],
         "text_repr": repr(sr["text"])[:200],
         "n_tool_calls": len(sr["tool_calls"]),
         "fingerprints": fingerprints(json.dumps(sr["tool_calls"][:1], ensure_ascii=False)),
         "repeat_loop": False})

    tr = chat_stream(a.base_url, a.model, tool_msgs, tools=TOOL_WRITE,
                     max_tokens=a.tool_max_tokens, timeout=a.timeout)
    got_code = got_path = None
    parse_err = None
    if tr["tool_calls"]:
        try:
            args = json.loads(tr["tool_calls"][0].get("arguments") or "{}")
            got_code, got_path = args.get("content"), args.get("path")
        except Exception as e:  # noqa: BLE001
            parse_err = repr(e)[:200]
    # ★★ 判据修正（2026-09-23，真机数据逼出来的）：
    #   520k 下模型**没走 tool_calls**，而是把校验码**逐字正确地**写成**文字**输出
    #   （`stream_text_repr` = 完整校验码，一字不差；短上下文才走 tool_calls）。
    #   ⇒ 这是**格式/指令遵循**的差异，**不是乱码/损坏**。
    #   原来这里要求"必须走 tool_calls" ⇒ 把一次**内容完全正确**的回答判成 FAIL = **判据绑错对象**。
    #   现在分两层判：
    #     ① `content_exact`（主判据）= 校验码**逐字出现在**（工具参数 **或** 文字回答里）**；
    #     ② `format`（记录项，不参与通过与否）= tool_calls / text / neither。
    stream_text = tr["text"] or ""
    in_text = (stream_text.strip() == want_code)
    ok_code = (got_code == want_code) or in_text
    ok_path = (got_path == want_path) if got_code is not None else True
    fmt = "tool_calls" if tr["tool_calls"] else ("text" if in_text else "neither")
    raw_args = json.dumps(tr["tool_calls"][:1], ensure_ascii=False)[:600]
    rec({"phase": "big-stream+tools", "ctx_tokens_real": real, "ctx_ok": ctx_ok,
         "ctx_target": a.context_tokens, "http": tr["http"], "wall_s": tr["wall_s"],
         "n_chunks": tr["n_chunks"], "first_chunk_s": tr["first_chunk_s"],
         "finish_reason": tr["finish_reason"], "err": tr["err"],
         "ok": bool(ok_code and ok_path) and ctx_ok,
         "format": fmt, "content_exact": bool(ok_code), "in_text": bool(in_text),
         "expect": want_code, "exact": ok_code, "contains": ok_code,
         "answer_repr": repr(got_code)[:400], "fingerprints": fingerprints(raw_args),
         "repeat_loop": False, "got_path": got_path, "parse_err": parse_err,
         "n_tool_calls": len(tr["tool_calls"]),
         "stream_text_repr": repr(tr["text"])[:600],
         "raw_arguments_repr": raw_args})

    for r in results:
        tag = "PASS" if r["ok"] else "FAIL"
        extra = ""
        if r["phase"].startswith(("big-stream", "short-stream")):
            extra = (f" chunks={r.get('n_chunks')} first={r.get('first_chunk_s')}"
                     f" fin={r.get('finish_reason')!r}")
        print(f"  [{r['phase']:18s}] {tag:4s} http={r['http']} wall={r['wall_s']:.1f}s{extra}"
              f" fp={r['fingerprints']['u_fffd']}/{r['fingerprints']['nul']}")
        if not r["ok"]:
            print(f"      A: {r['answer_repr']}")
            if r.get("err"):
                print(f"      err: {r['err']}")
            if r.get("raw_tail"):
                print(f"      raw: {r['raw_tail']}")
    bad_ctx = sum(1 for r in results if not r.get("ctx_ok", True))
    # ★ 长/短工具臂的**对照判据**（把"探针 prompt 不稳"与"长上下文退化"分开）
    long_t = next((r for r in results if r["phase"] == "big-stream+tools"), {})
    short_t = next((r for r in results if r["phase"].startswith("short-stream+tools")), {})
    # ★ "格式变化"与"内容损坏"必须分开判：前者是记录项，后者才是 bug。
    fmt_change = (short_t.get("n_tool_calls", 0) > 0) and (long_t.get("n_tool_calls", 0) == 0)
    content_bad = (long_t.get("content_exact") is False) or (short_t.get("content_exact") is False)
    print(f"\n  ★ 汇总：{len(results)} 请求 / 失败 {fails} / 上下文不达标 {bad_ctx}")
    print(f"  ★ 工具臂对照：短 n_tool_calls={short_t.get('n_tool_calls')}"
          f" vs 520k n_tool_calls={long_t.get('n_tool_calls')}"
          f" ⇒ **格式**发生变化={fmt_change}（记录项，不算 bug）")
    print(f"  ★ 内容判据：短 content_exact={short_t.get('content_exact')}"
          f" / 520k content_exact={long_t.get('content_exact')}"
          f" ⇒ **内容损坏={content_bad}**（False 才是 bug）")
    print(f"     520k 那次格式={long_t.get('format')!r}  全文: {long_t.get('stream_text_repr')}")
    out["stream"] = results
    out["stream_summary"] = {"n": len(results), "fails": fails, "bad_ctx": bad_ctx,
                             "target_tokens": a.context_tokens, "conc": a.conc,
                             "tool_format_changed_long_vs_short": fmt_change,
                             "tool_content_broken": content_bad,
                             "n_tool_calls_long": long_t.get("n_tool_calls"),
                             "n_tool_calls_short": short_t.get("n_tool_calls")}
    return fails


MODES = {"needle": mode_needle, "grow": mode_grow,
         "reuse": mode_reuse, "toolargs": mode_toolargs,
         "evict": mode_evict, "conc": mode_conc,
         "bigprefill": mode_bigprefill, "biggrow": mode_biggrow,
         "mixed": mode_mixed, "stream": mode_stream}


def selfcheck() -> int:
    """★ 不连服务，只验**探针自己**的判据是否自洽。

    为什么必须有（2026-09-23 真机教训）：`grow` 模式第一版用**位置下标**取针
    （`NEEDLES[i % 4]`）⇒ 问 B/C/D 时插进上下文的仍是 **A**。
    现场表现是模型回答"app_1.log 里没有运维备忘 B，**只有重复出现的运维备忘 A**"
    —— **模型是对的、判据是错的**。这类错只有靠"自检判据本身"才能提前发现。
    """
    bad = 0

    def ck(name, cond):
        nonlocal bad
        print(("  PASS  " if cond else "  FAIL  ") + name)
        if not cond:
            bad += 1

    # ① 每个 key 必须插进**它自己**的针文本，且位置各不相同
    body = "X" * 4000
    for k in ("A", "B", "C", "D"):
        got = embed_needles(body, [k])
        ck(f"embed_needles 只插 {k} 的针", KEY_TEXT[k] in got)
        others = [t for kk, t in NEEDLES if kk != k]
        ck(f"embed_needles 没混入别的针（{k}）", all(t not in got for t in others))
    multi = embed_needles(body, ["A", "B", "C", "D"])
    ck("四针同插：四条都在", all(t in multi for _, t in NEEDLES))
    ck("四针同插：顺序按深度递增",
       [multi.index(KEY_TEXT[k]) for k in ("A", "B", "C", "D")] ==
       sorted(multi.index(KEY_TEXT[k]) for k in ("A", "B", "C", "D")))
    try:
        embed_needles(body, ["Z"])
        ck("未知 key 必须报错", False)
    except KeyError:
        ck("未知 key 必须报错", True)

    # ② 判据自洽：正确答案必须判 PASS；乱码/复读必须判 FAIL
    ck("judge：正确答案 ⇒ exact", judge("ZQ7K-3341", "ZQ7K-3341")["exact"] is True)
    ck("judge：多一个字 ⇒ exact=False",
       judge("ZQ7K-3341。", "ZQ7K-3341")["exact"] is False)
    garble = "ZQ7" + "\ufffd" + "\x00" + "K-3341"
    j = judge(garble, "ZQ7K-3341")
    ck("judge：U+FFFD 与 NUL 被数出",
       j["fingerprints"]["u_fffd"] == 1 and j["fingerprints"]["nul"] == 1)
    ck("judge：含乱码 ⇒ exact=False", j["exact"] is False)
    ck("repeat_loop：复读 60 次被检出", repeat_loop("重复片段" * 60) is True)
    ck("repeat_loop：正常长文不误报",
       repeat_loop("这是一段正常的中文文本，用来验证复读检测不会误报。" * 6) is False)

    # ③ ★ 上下文平铺：语料**短于**目标长度时也必须给出足够长的上下文
    #   （真机教训：语料只有 826,639 字，却要 520k token ⇒ 第一版直接切出**空串**，
    #    实际上下文只剩 4 条针 = 91 token，而探针照样报 PASS。）
    short = "短语料。" * 100            # 400 字
    long_ctx = build_context(short, 5000)
    ck("build_context：语料短于目标 ⇒ 平铺到目标长度", len(long_ctx) == 5000)
    ck("build_context：平铺后语料内容仍在", "短语料。" in long_ctx)
    ck("build_context：rotate 让不同 lane 内容不同",
       build_context("ABCDEFGH", 64, rotate=0) != build_context("ABCDEFGH", 64, rotate=3))
    ck("build_context：目标 0 ⇒ 空串", build_context(short, 0) == "")
    # 长度门：不达标必须判"无效"
    ck("ctx_ratio_ok：91/520000 ⇒ False（不许当通过）",
       ctx_ratio_ok(91, 520000, 0.9) is False)
    ck("ctx_ratio_ok：500k/520k ⇒ True", ctx_ratio_ok(500000, 520000, 0.9) is True)
    ck("ctx_ratio_ok：target=0 ⇒ 不设限", ctx_ratio_ok(0, 0, 0.9) is True)

    # ④ 指纹判据对"干净文本"必须全 0（不许误报）
    fp = fingerprints("正常的中文回答，含英文与数字 391。")
    ck("fingerprints：干净文本全 0",
       all(fp[k] == 0 for k in ("u_fffd", "nul", "ctrl", "surrogate", "nonchar")))

    print(f"\n[probe --selfcheck] 失败 {bad} 项")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8020")
    ap.add_argument("--model", default="deepseek-v41")
    ap.add_argument("--selfcheck", action="store_true",
                    help="只验探针自己的判据是否自洽（不连服务）")
    ap.add_argument("--mode", default="all",
                    choices=["all", "needle", "grow", "reuse", "toolargs",
                             "evict", "conc", "bigprefill", "biggrow", "mixed",
                             "stream"])
    ap.add_argument("--conc", type=int, default=4, help="conc 模式的并发路数")
    ap.add_argument("--followups", type=int, default=3,
                    help="biggrow 模式：大 prefill 之后在同一会话里追问几轮")
    ap.add_argument("--fill-sessions", type=int, default=12,
                    help="evict 模式灌爆缓存用的长会话数（要把 KV cache 灌满才有效）")
    ap.add_argument("--context-tokens", type=int, default=32768,
                    help="目标上下文长度（token）；现场反馈是长上下文才出问题 ⇒ 必做长度扫描")
    ap.add_argument("--turns", type=int, default=4, help="grow 模式的轮数")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--tool-max-tokens", type=int, default=256,
                    help="toolargs 模式用（工具参数 JSON 比普通回答长得多；64 会截断）")
    ap.add_argument("--offset", type=int, default=0, help="语料起点偏移（换一段文本）")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--min-ctx-ratio", type=float, default=0.9,
                    help="实际上下文 / 目标上下文 的最低比例；低于它 ⇒ 本次读数无效"
                         "（防止拿短上下文冒充长上下文）")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.selfcheck:
        return selfcheck()

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
