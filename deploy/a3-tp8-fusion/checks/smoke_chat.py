#!/usr/bin/env python3
"""3 条中英短问答冒烟 —— 精度最小判据（看有无乱码 / 退化）。

用法：
    python3 smoke_chat.py --port 19210
    python3 smoke_chat.py --port 19210 --model deepseek-v41

判据（不需要标准答案，看"是否明显坏掉"）：
  * HTTP 200 且能解析出 choices[0].message.content
  * 回复非空、非重复字符垃圾、不含 U+FFFD 替换符
  * "1+1" 那条必须回答出数字 2（最强的单点判据）
退出码 0 = 全过；1 = 有任一失败。
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

CASES = [
    ("1+1等于几？只回答数字。", lambda s: "2" in s, "包含数字 2"),
    ("用一句话说明什么是KV cache。", lambda s: len(s) >= 20 and "KV" in s.upper(), "非空且提到 KV"),
    ("把下面这句话翻译成英文：今天天气很好。", lambda s: "weather" in s.lower(), "含 weather"),
]


def chat(base: str, model: str, prompt: str, timeout: int = 300) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 96,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]


def looks_broken(s: str) -> str | None:
    if not s.strip():
        return "回复为空"
    if "\ufffd" in s:
        return "含 U+FFFD 替换符（乱码）"
    # 单字符占比过高 ⇒ 重复退化的典型形态
    if len(s) > 20 and len(set(s)) / len(s) < 0.15:
        return "字符种类过少（疑似重复退化）"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=19210)
    ap.add_argument("--model", default="deepseek-v41")
    args = ap.parse_args()
    base = "http://%s:%d" % (args.host, args.port)

    bad = 0
    for prompt, check, desc in CASES:
        try:
            out = chat(base, args.model, prompt)
        except urllib.error.URLError as e:
            print("  x 请求失败: %s" % e)
            bad += 1
            continue
        except Exception as e:  # noqa: BLE001 - 冒烟脚本，任何异常都算失败
            print("  x 异常: %s" % str(e)[:160])
            bad += 1
            continue
        reason = looks_broken(out)
        hit = check(out)
        if reason or not hit:
            bad += 1
            print("  x %-28s -> %r  (%s%s)" % (prompt[:14], out[:90], reason or "", "" if reason else "判定: " + desc))
        else:
            print("  v %-28s -> %r" % (prompt[:14], out[:70]))
    print("----")
    if bad:
        print("  冒烟失败：%d/%d" % (bad, len(CASES)))
        return 1
    print("  冒烟通过：%d/%d" % (len(CASES), len(CASES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
