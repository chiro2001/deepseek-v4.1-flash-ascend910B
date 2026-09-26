#!/usr/bin/env python3
"""同运行内的 replay 对判据 —— 替代**已证无效**的跨运行 sha 比对。

## 为什么不能用跨运行 sha（`a2/logs/083` §2，实测）

`p2e` 与 `p2h` 是**同一配置**（文件逐字节相同、env 相同、`salt=20260922` 相同）的**两次独立运行**，
而它们的 `fill_out_sha256_all` 分别是 `1fc2a9ce…` 与 `a98f0645…`。
⇒ 原因是 prompt 由随机 token id 构成 ⇒ 输出分布近均匀 ⇒ 采样/规约噪声一翻就翻
⇒ **sha 只是"这次抽签的签名"，不是内容指纹**。
⇒ 因此"用 A 臂的 sha 比 B 臂的 sha"**在这套几何下没有判别力**。

## 本脚本用的判据（**同一次运行内**）

`kv_offload_client.py --rounds N` 的语义（`bench/kv_offload_client.py:348-404`）：
  `rounds[0]`   = fill（完整 prefill）
  `rounds[1..]` = 每轮之间 `POST /reset_prefix_cache` 之后的重放 ⇒ **每一轮都走池取回**
⇒ `rounds[1]` 与 `rounds[2]` 是**同进程、同 salt、同输入、都经过取回路径**的两次测量
⇒ 它们**逐字相同**才是"取回路径稳定且可复现"的判据。

★ 与 `chat` 的 `--repeats`（`text_correctness_probe.py --mode prefix-pair`）互补：
  那个用**自然语言**判"答得出同一个答案"，这个用**同一批随机 prompt**判逐字可复现。

用法：
    python3 check_same_run_replay.py <arm>.client.json [more.json ...]
    python3 check_same_run_replay.py --glob 'out/*/*.client.json'

退出码：0 = 全部通过；1 = 有失败；2 = 用法/数据不足。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys


def hashes(round_obj):
    """取该轮的逐 prompt 输出哈希（dict 或 list 都兼容）。"""
    h = round_obj.get("out_sha256_all")
    if h is None:
        h = round_obj.get("per_prompt_token_evidence")
    return h


def judge(path, need_rounds=3):
    with open(path) as fh:
        d = json.load(fh)
    rounds = d.get("rounds") or []
    tag = d.get("tag") or path
    out = {"path": path, "tag": tag, "n_rounds": len(rounds), "checks": [], "ok": True}

    def rec(name, ok, detail):
        out["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            out["ok"] = False

    # 每一轮都必须 0 失败（否则这一臂本身无效，谈不上比 sha）
    for i, r in enumerate(rounds):
        f = r.get("requests_failed")
        ok = r.get("requests_ok")
        rec(f"round{i+1} 无失败",
            (f == 0),
            f"ok={ok} failed={f} wall={r.get('wall_s')}")

    if len(rounds) < 2:
        rec("轮数 >= 2", False, f"只有 {len(rounds)} 轮 ⇒ 无法做同运行内比对（需要 --rounds 3）")
        return out

    # ★ 核心：最后两轮（都经过取回）逐字相同
    if len(rounds) >= 3:
        a, b = rounds[-2], rounds[-1]
        ha, hb = hashes(a), hashes(b)
        same = (ha is not None and ha == hb)
        rec("★ replay1 == replay2（同运行、都走取回）逐字相同",
            same,
            f"replay1={str(ha)[:24]} replay2={str(hb)[:24]}")
    else:
        rec("轮数 >= 3（要有两轮 replay 才能比对）", False,
            "只有 2 轮（fill+replay1）⇒ 请用 --rounds 3 重跑，别用跨运行 sha 代替（logs/083 §2）")

    # 顺带记下 fill vs replay 的关系 —— ★ 只作观察，**不作判据**（长度/路径都不同）
    hf, hr = hashes(rounds[0]), hashes(rounds[-1])
    out["fill_vs_replay_note"] = (
        f"fill={str(hf)[:24]} replay={str(hr)[:24]} "
        "（★ 不要拿这个当判据：fill 与 replay 的输入长度不同、路径不同，见 logs/083 §2）"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--glob", default="")
    a = ap.parse_args()
    paths = list(a.paths)
    if a.glob:
        paths += sorted(glob.glob(a.glob))
    if not paths:
        print("用法：check_same_run_replay.py <arm>.client.json [--glob 'out/*/*.client.json']",
              file=sys.stderr)
        return 2

    allok = True
    for p in paths:
        try:
            r = judge(p)
        except Exception as e:  # noqa: BLE001
            print(f"✗ {p}: 读不了（{e!r}）")
            allok = False
            continue
        print("=" * 78)
        print(f"{'✓' if r['ok'] else '✗'} {r['tag']}   rounds={r['n_rounds']}")
        print(f"   {r['path']}")
        for c in r["checks"]:
            print(f"   [{'✓' if c['ok'] else '✗'}] {c['name']}  —  {c['detail']}")
        if r.get("fill_vs_replay_note"):
            print(f"   注：{r['fill_vs_replay_note']}")
        allok = allok and r["ok"]
    print("=" * 78)
    print("✓ 同运行内 replay 对逐字相同" if allok else "✗ 有失败项（见上面）")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
