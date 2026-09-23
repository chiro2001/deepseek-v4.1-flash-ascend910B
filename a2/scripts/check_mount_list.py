#!/usr/bin/env python3
"""check_mount_list.py —— 校验一份 `docker run` 的挂载清单是否**成对完整**且**无重复目标**。

为什么需要（2026-09-23 真机两次踩到，都是同一类错）：
  ① `MOUNTS` 里每个挂载占**两个元素**（`-v` 与 `路径:目标:mode`）；
     "摘掉一条"只删路径 ⇒ 留下孤立的 `-v` ⇒ docker 把后面的挂载串当成 `-v` 的值，
     再把更后面顶到镜像位置 ⇒ **`docker: invalid reference format.`**
  ② 两个来源挂到**同一个容器目标** ⇒ docker 直接报 **`Duplicate mount point`**。
  两者 `bash -n` 都查不出来（语法合法），只能在**生成物**上验。

用法：
  python3 check_mount_list.py --from-log <含 [a2-dry] MOUNTS(N): 行的日志>
  python3 check_mount_list.py --tokens -v A:/t:ro -v B:/t2:ro ...     # 直接给元素

退出码：0 = 通过；1 = 有问题（打印问题清单）；2 = 用法/输入错
"""
from __future__ import annotations

import argparse
import re
import shlex
import sys


def tokens_from_log(text: str) -> list[str]:
    for line in text.splitlines():
        if "[a2-dry] MOUNTS(" in line:
            payload = line.split("):", 1)[1].strip() if "):" in line else ""
            return shlex.split(payload)
    return []


def check(tokens: list[str]) -> list[str]:
    problems: list[str] = []
    if not tokens:
        return ["挂载清单为空（没找到 MOUNTS 行，或它真的是空的）"]
    # ① 成对性：每个 `-v` 后面必须紧跟一个"含冒号"的挂载串
    i = 0
    pairs = 0
    targets: dict[str, int] = {}
    while i < len(tokens):
        if tokens[i] == "-v":
            if i + 1 >= len(tokens):
                problems.append(f"第 {i} 个元素是 `-v`，后面**没有**值（孤立的 -v ⇒ invalid reference format）")
                break
            spec = tokens[i + 1]
            if spec.startswith("-") or ":" not in spec:
                problems.append(f"`-v` 的值不像挂载串：{spec!r}（孤立的 -v ⇒ invalid reference format）")
            else:
                parts = spec.rsplit(":", 2)
                tgt = parts[1] if len(parts) >= 2 else spec
                targets[tgt] = targets.get(tgt, 0) + 1
                pairs += 1
            i += 2
        else:
            problems.append(f"第 {i} 个元素不是 `-v` 而是 {tokens[i]!r}"
                            "（挂载清单必须严格成对；说明有一半被删掉了）")
            i += 1
    # ② 目标唯一
    for tgt, n in sorted(targets.items()):
        if n > 1:
            problems.append(f"容器目标被挂 {n} 次（docker 会报 Duplicate mount point）：{tgt}")
    # ③ 计数自洽（`-v` 数应等于挂载数）
    n_v = sum(1 for t in tokens if t == "-v")
    if n_v != pairs:
        problems.append(f"`-v` 出现 {n_v} 次，但成对解析出 {pairs} 条 ⇒ 有成对性错误")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-log", default="")
    ap.add_argument("--tokens", nargs="*", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.from_log:
        with open(a.from_log, encoding="utf-8", errors="replace") as fh:
            toks = tokens_from_log(fh.read())
    elif a.tokens is not None:
        toks = a.tokens
    else:
        print("用法：check_mount_list.py --from-log <日志> | --tokens ...", file=sys.stderr)
        return 2

    probs = check(toks)
    n = len([t for t in toks if not t.startswith("-") or ":" in t])
    if probs:
        print("FAIL 挂载清单有问题：")
        for p in probs:
            print("  - " + p)
        return 1
    if not a.quiet:
        print(f"OK   挂载清单成对完整、目标唯一（元素 {len(toks)} 个）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
