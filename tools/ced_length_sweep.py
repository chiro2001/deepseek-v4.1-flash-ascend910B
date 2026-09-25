#!/usr/bin/env python3
"""在**同一个 D 实例**上扫多个 prompt 长度，记录"首次失败发生在第几个请求"。

目的：区分三种候选机制（C = D 的池块数，R = 每请求占用的池块数）：
  A) 块号上界：失败发生在"块号首次触及某个常数上界"的那个请求
     ⇒ 首次失败序号 ≈ floor(B/R)，随 R 变化而非 4；
  B) 绕回：失败发生在"该请求必须绕回池尾"的那个请求
     ⇒ 首次失败序号 = floor(C/R) + 1；
  C) 固定周期（例如 MAX_SEQS 或别的常数）
     ⇒ 首次失败序号与 R 无关（恒为同一个数）。

同一个 D 跑完所有长度，避免换实例带来的混淆（本次调查已经吃过一次
跨实例比较的亏：900K 那轮其实跑在另一个 P 上）。

R 由 prompt 长度推出：R = ceil((N-1)/128) + 21
（g0 的 ceil((N-1)/128) 块 + g1 的 1 块 + g2..g11 各 2 块）。

用法：
  python3 tools/ced_length_sweep.py --corpus data/hongloumeng.txt \
      --url http://127.0.0.1:18992/v1/chat/completions \
      --tokenize http://127.0.0.1:18990/tokenize --model deepseek-v41-ced-pd \
      --targets 200000,400000,700000,900000 --count 8 \
      --out /path/outdir

★★ 修过一个会让整轮结果作废的 bug：初版用"从 1M 模板**头部**截断用户内容"来造短
prompt，而针与提问都在内容**末尾**，于是短 prompt 里既没有问题也没有针 —— 模型答出
了另一个针的码（`ZQ7K-3341`），全部 FAIL 都是假故障。现在改为**从语料重新构造**：
切片语料 → 在 ~80% 深度插入针 D → 追加提问，并在发请求前断言两者都在。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

EXPECTED = "RB9N-6014"
BLOCK = 128
EXTRA_GROUPS = 21  # g1 + (g2..g11) 各 2 块

NEEDLE = "【运维备忘 D】构建机临时令牌是 RB9N-6014，用完即废。"
QUESTION = "运维备忘 D 里的构建机临时令牌是什么？只给令牌本身。"
SYSTEM = ("你是一个严谨的中文助手。回答要直接、简短；"
          '被要求"只给密码/口令/访问码/令牌"时就只输出它本身，不要解释。')


def post(url: str, payload: dict, timeout: float):
    body = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.load(resp), body
    except urllib.error.HTTPError as err:
        try:
            return err.code, json.loads(err.read()), body
        except Exception:
            return err.code, {}, body
    except Exception as err:  # noqa: BLE001
        return -1, {"transport_error": repr(err)}, body


def count_tokens(tokenize_url: str, model: str, text: str, timeout: float) -> int:
    status, data, _ = post(tokenize_url, {"model": model, "prompt": text}, timeout)
    if status != 200:
        raise SystemExit(f"/tokenize 失败 status={status} data={str(data)[:200]}")
    return int(data.get("count") or len(data.get("tokens") or []))


def build_prompt(corpus: str, tokenize_url: str, model: str, target: int,
                 timeout: float):
    """用**语料**构造目标长度的 prompt：切片 → 80% 处插针 → 追加提问。

    长度校准针对**最终 prompt**（含针与提问）做，这样 R 的推算才准确。
    """
    chars = max(1, target)          # 初始猜测：中文约 1 token/字
    prompt, got = "", 0
    for _ in range(6):
        body = (corpus * (chars // max(1, len(corpus)) + 1))[:chars]
        # 针插在 80% 深度（与既有四针语料同口径）
        cut = int(len(body) * 0.8)
        user = body[:cut] + "\n" + NEEDLE + "\n" + body[cut:] + "\n\n" + QUESTION
        prompt = user
        got = count_tokens(tokenize_url, model, prompt, timeout)
        if abs(got - target) <= max(200, target // 500):
            break
        chars = max(1, round(chars * target / max(1, got)))
    if NEEDLE not in prompt or QUESTION not in prompt:
        raise SystemExit("构造失败：prompt 里缺少针或提问（拒绝发出无效请求）")
    payload = {"model": model,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt}],
               "max_tokens": 64, "temperature": 0.0, "stream": False}
    return payload, got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="长语料文件（如 data/hongloumeng.txt）")
    ap.add_argument("--url", required=True)
    ap.add_argument("--tokenize", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True, help="逗号分隔的 token 目标")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corpus = open(args.corpus, encoding="utf-8", errors="replace").read()
    if len(corpus) < 10000:
        raise SystemExit(f"语料太短：{args.corpus}")
    os.makedirs(args.out, exist_ok=True)
    summary = []

    for target in [int(x) for x in args.targets.split(",") if x.strip()]:
        req_template, got = build_prompt(corpus, args.tokenize, args.model, target, 300.0)
        r_blocks = -(-(got - 1) // BLOCK) + EXTRA_GROUPS
        print(f"\n=== 目标 {target} → 实际 {got} tokens，R≈{r_blocks} 块 ===", flush=True)

        first_fail = None
        rows = []
        for i in range(1, args.count + 1):
            started = time.strftime("%H:%M:%S")
            status, data, body = post(args.url, req_template, args.timeout)
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content")
            usage = data.get("usage") or {}
            passed = isinstance(content, str) and EXPECTED in content
            if not passed and first_fail is None:
                first_fail = i
            rows.append({"idx": i, "status": status, "prompt": usage.get("prompt_tokens"),
                         "completion": usage.get("completion_tokens"),
                         "content": content, "passed": passed, "start": started})
            req_path = os.path.join(args.out, f"L{target}_r{i}.request.json")
            open(req_path, "wb").write(body)
            print(f"  r{i}: {'PASS' if passed else 'FAIL'} completion={usage.get('completion_tokens')} "
                  f"content={str(content)[:20]!r} at {started}", flush=True)

        entry = {"target": target, "actual_prompt_tokens": got, "r_blocks_estimate": r_blocks,
                 "count": args.count, "first_fail": first_fail, "rows": rows,
                 "request_sha256": hashlib.sha256(
                     json.dumps(req_template, ensure_ascii=False).encode()).hexdigest()}
        summary.append(entry)
        json.dump(summary, open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    print("\n=== 汇总：首次失败序号 vs R ===")
    print(f"{'target':>9} {'actual':>9} {'R≈':>6} {'首次失败':>8}")
    for e in summary:
        print(f"{e['target']:>9} {e['actual_prompt_tokens']:>9} {e['r_blocks_estimate']:>6} "
              f"{str(e['first_fail']):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
