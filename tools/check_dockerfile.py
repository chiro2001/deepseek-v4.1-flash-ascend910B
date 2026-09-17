#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_dockerfile.py —— Dockerfile 续行链合法性检查（10 秒，不需要 docker）。

    python3 tools/check_dockerfile.py [Dockerfile]

## 为什么需要（真实故障）

Docker 会先把 `\\`+换行拼成**一整行**再交给 shell，所以：

  1. **除最后一行外，续行链里每一行都必须以 `\\` 结尾。**
     漏一个 `\\` ⇒ 指令**提前结束**，后面的 `local` / `test` / `cp` 会被当成
     Dockerfile 指令解析并报 `unknown instruction`。
     （本文件踩过：`inst() { \\n local tgt=...` 里 `inst() {` 那行漏了 `\\`。）

  2. **续行链里绝对不能出现行内 `#` 注释。**
     拼接成一整行后，`#` 会把**它后面的一切**都注释掉（包括还没执行的命令）；
     而如果把 `\\` 写在注释后面，那个 `\\` 本身也在注释里 ⇒ 等于没写。
     （本文件踩过：`inst() { # src_in_tmp  target_rel`。）

## 检查项

  A. 续行链中出现行内 `#`（本行注释或上一行延续下来的）→ ERROR
  B. 续行链中某行以 `#` 注释结尾、且注释里含 `\\` → ERROR（`\\` 失效）
  C. 链中行尾的 `\\` 前面是空格且被注释吞掉 → ERROR
  D. 报告每条 RUN 的续行行数与行号，便于人工复核

退出码：0 = 通过；1 = 有问题
"""
from __future__ import annotations

import re
import sys


def is_comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def inline_hash(line: str) -> int:
    """返回行内 `#` 的位置（跳过注释行与引号内的 `#`），没有则 -1。"""
    if is_comment(line):
        return -1
    q = None
    i = 0
    while i < len(line):
        ch = line[i]
        if q:
            if ch == "\\":
                i += 2
                continue
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "#":
            # shell 里 `#` 要成为注释，前面必须是空白或行首
            if i == 0 or line[i - 1] in " \t":
                return i
        i += 1
    return -1


def main(path: str) -> int:
    try:
        raw = open(path, encoding="utf-8").read().splitlines()
    except OSError as exc:
        print("无法读取 %s: %r" % (path, exc))
        return 1

    errors: list[str] = []
    n_runs = 0

    i = 0
    while i < len(raw):
        line = raw[i]
        if not re.match(r"^\s*RUN\b", line, re.I):
            i += 1
            continue
        n_runs += 1
        start = i + 1
        chain = [line]
        # 收集续行
        while chain[-1].rstrip().endswith("\\"):
            i += 1
            if i >= len(raw):
                errors.append("第 %d 行起的 RUN 以 `\\` 结尾但文件已结束" % start)
                break
            chain.append(raw[i])

        for off, cl in enumerate(chain):
            lno = start + off
            stripped = cl.rstrip()
            # 跳过纯注释行（Docker 允许链中插注释行？—— 实际上不允许，
            # 因为拼行后注释会吞掉后续内容，这里也报出来）
            if is_comment(cl):
                errors.append(
                    "第 %d 行：续行链中出现**整行注释**；拼接后会把后续命令全部注释掉"
                    "（请把说明移到 RUN 之外）" % lno
                )
                continue
            h = inline_hash(cl)
            if h >= 0:
                tail = cl[h:]
                errors.append(
                    "第 %d 行：续行链中出现行内 `#` 注释（%r）；拼接后 `#` 之后的一切"
                    "（含后续命令）都会被注释掉" % (lno, tail[:50])
                )
                if "\\" in tail:
                    errors.append(
                        "第 %d 行：`\\` 出现在注释里 ⇒ 续行失效，指令会在此提前结束"
                        % lno
                    )
            # 链中非最后一行必须以 \ 结尾
            is_last = off == len(chain) - 1
            if not is_last and not stripped.endswith("\\"):
                errors.append(
                    "第 %d 行：续行链中间的行没有以 `\\` 结尾（链断在这里）" % lno
                )

        if len(chain) > 1:
            print("  RUN @%d-%d: %d 行续行链" % (start, start + len(chain) - 1, len(chain)))
        i += 1

    print("[check_dockerfile] %s：共 %d 条 RUN" % (path, n_runs))
    if errors:
        print()
        for e in errors:
            print("  \033[31mERROR\033[0m %s" % e)
        print()
        print("[check_dockerfile] %d 个问题 ❌" % len(errors))
        return 1
    print("[check_dockerfile] 续行链全部合法 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Dockerfile"))
