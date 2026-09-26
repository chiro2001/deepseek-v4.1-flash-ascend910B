#!/usr/bin/env python3
"""回归守卫：`curl -w '%{http_code}' ... || echo 000` 会拼出 "000000"。

## 为什么是 bug（真踩过）

`curl -w '%{http_code}'` 在**连接失败时也会输出**（打 `000`），退出码非 0；
于是 `|| echo 000` 又追加一个 ⇒ 变量变成 **`000000`**。
后续所有 `[ "$code" = "200" ]` 判断**全部失效** —— 而且方向是"永远判不就绪"，
看起来像"服务起不来"，把排查带偏。

与 `AGENTS.md §3.2` 的 `grep -c ... || echo 0` 是**同一族**：
命令自身已经把"失败"写进 stdout，再 `|| echo 默认值` 就会重复。

## 正确写法

```bash
code=$(curl -s -o /dev/null -w '%{http_code}' "$URL" 2>/dev/null)
code=${code:-000}          # 空 ⇒ 补默认；绝不会拼成 000000
```

## 用法

    python3 tools/check_http_code_pattern.py     # 0 = 干净，1 = 发现旧写法
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".layer-cache", "payload", "node_modules"}

# 匹配：curl ... -w '...http_code...' ... || echo <digits>
PAT = re.compile(r"curl\b[^\n]*http_code[^\n]*\|\|\s*echo\s+[0-9]+")
# 负控本身**必须**保留旧写法（它要证明这个 bug 真的会发生）。
# 用显式标记放行，且要求标记出现在**同一行或上一行** —— 不能靠窗口大小蒙过去。
ALLOW_MARK = "http-code-guard:allow"


def main() -> int:
    bad: list[tuple[str, int, str]] = []
    for p in ROOT.rglob("*.sh"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            if line.lstrip().startswith("#"):
                continue
            if not PAT.search(line):
                continue
            if ALLOW_MARK in line or (i >= 2 and ALLOW_MARK in lines[i - 2]):
                continue
            bad.append((str(p.relative_to(ROOT)), i, line.strip()))

    if not bad:
        print("[http-code] ✓ 没有 `curl ... http_code ... || echo N` 旧写法")
        return 0
    print(f"[http-code] ✗ 发现 {len(bad)} 处旧写法（会拼出 000000）：", file=sys.stderr)
    for f, i, line in bad:
        print(f"  {f}:{i}  {line[:110]}", file=sys.stderr)
    print("\n改成：code=$(curl ... -w '%{http_code}' URL 2>/dev/null); code=${code:-000}",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
