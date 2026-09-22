#!/usr/bin/env python3
"""自然语言文本正确性探针 —— 补 `071 §A1` / `072 §1` 那个"从未做过的语义判据"。

## 为什么要这个文件（本仓的原话）

`a2/logs/072` §1：
> **p2e 的 prompt 是随机 token id，不是自然语言。** `kv_offload_client.py: make_prompt()`
> 生成 `[1000 + ((base + i*7919) % 100000) ...]` ⇒ 生成的是**乱码**，**没有语义可判**。
> ⇒ **结论：整个 p2e 臂（以及此前所有 8 卡臂）从头到尾没有做过任何语义正确性判据。**

⇒ 于是所有"卸载/Engram/int8 都对"的结论，都建立在 **计数器** 与 **sha 相等** 上，而
`062` 已经证明"热 replay == 冷算参考"这类判据**对路径差异没有判别力**。
本脚本补的就是这一格：**用自然语言问有确定答案的问题，并逐字判对错**。

## 两个模式（可单独跑，也可一起跑）

1. `--mode questions`：**语义正确性**。问一组有确定答案/关键词的问题，按 `must_contain`
   逐条判。★ 这组题在 `066`（单卡）上有历史对照（10 条全过），所以有基线可比。
2. `--mode prefix-pair`：**取回路径一致性**。同一段**长前缀** + 同一个问题**连发两次**
   （第 1 发建立缓存、第 2 发触发取回），比较两次输出是否**逐字相同**。
   ★ 这是最贴近"卸载取回被读回去"的语义判据 —— 计数器说"搬了 21.5 GB"，
     这个模式说"搬回来的东西让模型答出了同一个答案"。

## 用法（在起好服务的机器上，例如 A3-node1 的 8 卡臂 localhost:8050）

    python3 text_correctness_probe.py --base-url http://127.0.0.1:8050 --model deepseek-v41
    python3 text_correctness_probe.py --mode prefix-pair --prefix-tokens 8192 --repeats 3 ...
    python3 text_correctness_probe.py --out /path/to/textprobe.json      # 落证据

退出码：0 = 全过；1 = 有失败项；2 = 用法/连接错误。
★ 判据自带"反假阳性"：`--mode questions` 会打印**原始回答**，不是只打印 PASS。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- 题库
# 选取原则：① 答案唯一、可机器判；② 覆盖"算术 / 常识 / 指令跟随 / 中文";
# ③ 与 066 的历史题库**同源**，好和单卡基线对照（见 072 引用的那组）。
QUESTIONS = [
    ("计算 17×23 等于多少？只给数字。", ["391"]),
    ("《红楼梦》的作者是谁？只给名字。", ["曹雪芹", "曹雪芹（清）", "曹霑"]),
    ("40×60%÷2 等于多少？只给数字。", ["12"]),
    ("把字符串 Hello, world! 反转输出。", ["!dlrow ,olleH", "!dlrow, olleH"]),
    ("天空为什么是蓝色的？一句话。", ["散射", "瑞利", "Rayleigh"]),
    ("中国的首都是哪座城市？只给城市名。", ["北京"]),
    ("水的化学式是什么？", ["H2O", "H₂O"]),
    ("9 的平方根是多少？只给数字。", ["3"]),
    ("用一句话解释什么是缓存命中。", ["缓存", "命中"]),
    ("100 减去 37 等于多少？只给数字。", ["63"]),
]

# 取回路径用的长前缀（自然语言、可复读，避免随机 token 的乱码问题）
PREFIX_SENTENCE = (
    "下面是一段用于测试的说明文字。缓存与卸载是现代推理系统里两个常见的优化手段："
    "缓存把已经算过的结果留下来复用，卸载把暂时用不到的状态搬到更便宜的存储上，"
    "等到需要时再搬回来。两者的共同前提是——搬回来的东西必须和当初搬走的一模一样，"
    "否则复用带来的就不是加速，而是错误。这段话会被重复很多次以便凑够长度。"
)

PREFIX_QUESTION = "上面这段说明文字里，缓存和卸载的共同前提是什么？用一句话回答。"


def post(base_url: str, path: str, payload: dict, timeout: float = 600.0):
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode())
        return {"http": 200, "wall_s": time.time() - t0, "body": body, "error": None}
    except urllib.error.HTTPError as e:
        return {"http": e.code, "wall_s": time.time() - t0, "body": None,
                "error": e.read().decode()[:400]}
    except Exception as e:  # noqa: BLE001
        return {"http": -1, "wall_s": time.time() - t0, "body": None, "error": repr(e)[:400]}


def gen(base_url: str, model: str, prompt: str | list[int], max_tokens: int,
        timeout: float = 600.0):
    return post(base_url, "/v1/completions", {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": False,
    }, timeout)


def text_of(resp) -> str:
    if not resp.get("body"):
        return ""
    try:
        return resp["body"]["choices"][0]["text"]
    except Exception:  # noqa: BLE001
        return ""


def finish_of(resp) -> str:
    if not resp.get("body"):
        return ""
    try:
        return resp["body"]["choices"][0].get("finish_reason") or ""
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------- 模式 1
def run_questions(a, out: dict) -> int:
    print("=" * 78)
    print(f"模式 1 · 语义正确性（{len(QUESTIONS)} 题，temperature=0，逐条判关键词）")
    print("=" * 78)
    rows = []
    n_pass = 0
    for i, (q, must) in enumerate(QUESTIONS):
        r = gen(a.base_url, a.model, q, a.max_tokens, a.timeout)
        ans = text_of(r)
        hit = any(m in ans for m in must)
        n_pass += int(hit)
        flag = "✓" if hit else "✗"
        print(f"[{i:2d}] {flag} http={r['http']} wall={r['wall_s']:.2f}s "
              f"fin={finish_of(r)!r}")
        print(f"     Q: {q}")
        print(f"     A(原文): {ans!r}")
        if not hit:
            print(f"     ★ 期望包含其中之一: {must}")
            if r.get("error"):
                print(f"     ★ http 错误: {r['error']}")
        rows.append({"q": q, "must_contain": must, "answer": ans, "pass": hit,
                     "http": r["http"], "wall_s": r["wall_s"],
                     "finish_reason": finish_of(r), "error": r.get("error")})
    out["questions"] = {"rows": rows, "n_pass": n_pass, "n": len(QUESTIONS)}
    print("-" * 78)
    print(f"模式 1 结果：{n_pass}/{len(QUESTIONS)} 通过")
    # ★ 反假阳性：全 0 通过时要能区分"模型不行"与"服务挂了"
    if n_pass == 0 and all(r["http"] != 200 for r in rows):
        print("★ 全部 http 非 200 ⇒ 这**不是**模型答错，是服务不可用/请求失败")
    return 0 if n_pass == len(QUESTIONS) else 1


# ---------------------------------------------------------------- 模式 2
def run_prefix_pair(a, out: dict) -> int:
    """同一段长前缀 + 同一问题，连发 `--repeats` 次，比较输出是否逐字相同。"""
    unit = PREFIX_SENTENCE
    need = max(1, a.prefix_tokens * 2 // max(1, len(unit)))
    prefix = unit * need
    print("=" * 78)
    print(f"模式 2 · 取回路径一致性（前缀≈{len(prefix)} 字，连发 {a.repeats} 次，逐字比对）")
    print("=" * 78)
    outs = []
    rows = []
    for k in range(a.repeats):
        r = gen(a.base_url, a.model, prefix + "\n\n" + PREFIX_QUESTION, a.max_tokens, a.timeout)
        t = text_of(r)
        outs.append(t)
        rows.append({"round": k, "answer": t, "http": r["http"],
                     "wall_s": r["wall_s"], "finish_reason": finish_of(r),
                     "error": r.get("error")})
        print(f"[round {k}] http={r['http']} wall={r['wall_s']:.2f}s fin={finish_of(r)!r}")
        print(f"          A: {t!r}")
    same_all = len(set(outs)) == 1 and outs[0] != ""
    # ★ 允许"第 1 发与后续不同"（第 1 发是冷算/fill，后续才是取回）——
    #   但**后续各发之间必须逐字相同**，否则说明取回路径不稳定。
    tail_same = len(set(outs[1:])) == 1 if len(outs) >= 3 else None
    out["prefix_pair"] = {"rows": rows, "same_all": same_all, "tail_same": tail_same,
                          "n_distinct": len(set(outs))}
    print("-" * 78)
    print(f"全部 {a.repeats} 发逐字相同: {same_all}（distinct={len(set(outs))}）")
    if tail_same is not None:
        print(f"第 2 发起逐字相同: {tail_same}  ← ★ 这才是'取回路径'的判据（第 1 发是冷算）")
    ok = bool(tail_same) if tail_same is not None else same_all
    if not ok:
        print("★ 后续几发不一致 ⇒ 取回路径**不稳定**（或服务有抖动，见 logs/037）")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8050")
    ap.add_argument("--model", default="deepseek-v41")
    ap.add_argument("--mode", default="all",
                    choices=["all", "questions", "prefix-pair"])
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prefix-tokens", type=int, default=2048,
                    help="模式 2 的近似前缀长度（按字符粗算，中文≈1 token/字）")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    out: dict = {"base_url": a.base_url, "model": a.model, "started_at": time.time(),
                 "args": vars(a)}
    rc = 0
    # 先探活：连不上就别浪费 10 分钟
    h = post(a.base_url, "/health", {})
    if h["http"] != 200:
        try:
            with urllib.request.urlopen(a.base_url.rstrip("/") + "/health", timeout=10) as r:
                h["http"] = r.status
        except Exception as e:  # noqa: BLE001
            print(f"[probe] ✗ 服务不可达：{a.base_url} （{e!r}）", file=sys.stderr)
            return 2
    print(f"[probe] /health = {h['http']}")

    if a.mode in ("all", "questions"):
        rc |= run_questions(a, out)
    if a.mode in ("all", "prefix-pair"):
        rc |= run_prefix_pair(a, out)

    out["finished_at"] = time.time()
    out["rc"] = rc
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"[probe] 证据已写：{a.out}")
    print("=" * 78)
    print("✓ 全部通过" if rc == 0 else "✗ 有失败项（见上面原文）")
    return rc


if __name__ == "__main__":
    sys.exit(main())
