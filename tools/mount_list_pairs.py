#!/usr/bin/env python3
"""mount_list_pairs.py —— 把 serve_a2/serve_a3 的 `[a2-dry] MOUNTS(...)` 行
解析成 `源<TAB>容器内目标<TAB>模式` 三列（**成对解析**）。

为什么单独一个文件（而不是在 shell 里 heredoc 嵌 python）：
  `MOUNTS` 里**每条挂载占两个元素**（`-v` 与 `路径:目标:mode`）；
  在 shell heredoc 里写多行 python 极易被引号/续行吃掉，
  而且写错时脚本看起来还是对的（本仓已栽过多次）。拆开后：
  ① 可用 py_compile 验；② 可单测；③ shell 侧只剩一行调用。

用法：
  python3 mount_list_pairs.py <含 MOUNTS 行的文件>          # -> TSV 到 stdout
  python3 mount_list_pairs.py --check <文件>                # 只校验（成对/唯一/计数）
退出码：0 = 解析成功；2 = 找不到 MOUNTS 行或成对性有问题
"""
from __future__ import annotations

import argparse
import shlex
import sys


def find_mounts_line(text: str) -> str:
    for line in text.splitlines():
        if "[a2-dry] MOUNTS" in line:
            return line
    return ""


def parse_pairs(line: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """返回 (pairs, problems)。pairs = [(src, dst, mode), ...]"""
    problems: list[str] = []
    if not line:
        return [], ["没找到 MOUNTS 行"]
    i = line.find("):")
    payload = line[i + 2:] if i >= 0 else ""
    toks = shlex.split(payload)
    pairs: list[tuple[str, str, str]] = []
    k = 0
    while k < len(toks):
        if toks[k] != "-v":
            problems.append("第 %d 个元素不是 -v 而是 %r（挂载清单必须严格成对）"
                            % (k, toks[k]))
            k += 1
            continue
        if k + 1 >= len(toks):
            problems.append("末位是孤立的 -v（缺挂载值）")
            break
        spec = toks[k + 1]
        if spec.startswith("-") or ":" not in spec:
            problems.append("-v 的值不像挂载串：%r" % spec)
        else:
            parts = spec.rsplit(":", 2)
            src = parts[0]
            dst = parts[1] if len(parts) > 1 else ""
            mode = parts[2] if len(parts) > 2 else ""
            pairs.append((src, dst, mode))
        k += 2
    # 目标唯一性（docker 会报 Duplicate mount point）
    seen: dict[str, int] = {}
    for _, dst, _ in pairs:
        seen[dst] = seen.get(dst, 0) + 1
    for dst, n in seen.items():
        if n > 1:
            problems.append("容器目标被挂 %d 次（docker 会报 Duplicate mount point）：%s"
                            % (n, dst))
    return pairs, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    try:
        with open(a.path, encoding="utf-8", errors="replace") as fh:
            line = find_mounts_line(fh.read())
    except OSError as e:
        print("读不了 %s: %r" % (a.path, e), file=sys.stderr)
        return 2
    pairs, problems = parse_pairs(line)
    if problems:
        print("FAIL 挂载清单有问题：", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        return 2
    if not pairs:
        print("FAIL 解析出 0 条挂载", file=sys.stderr)
        return 2
    if a.check:
        print("OK   %d 条挂载，成对完整、目标唯一" % len(pairs))
        return 0
    for src, dst, mode in pairs:
        print("%s\t%s\t%s" % (src, dst, mode))
    return 0


if __name__ == "__main__":
    sys.exit(main())
