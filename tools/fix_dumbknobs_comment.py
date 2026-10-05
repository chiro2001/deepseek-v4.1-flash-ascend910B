#!/usr/bin/env python3
"""把 serve_a2.sh 里 [DUMB-KNOBS] 注释块中的**反引号**去掉。

为什么必须去掉：那块注释位于**未加引号的 heredoc**（生成 inner.sh）内，
反引号会被 shell 当**命令替换**执行 —— `set -e` 下命令失败会让生成的 inner.sh 变成 0 字节
（本仓已有"往该 heredoc 插块 ⇒ inner.sh 0 字节"的历史记录，疑似就是这个机制）。

用法（在 a3-21 的 ~/cedpd-repo 下）: python3 fix_dumbknobs_comment.py [--check]
"""

from __future__ import annotations

import io
import sys

PATH = "scripts/serve_a2.sh"
BAD_START = "# ★ [DUMB-KNOBS 2026-10-05]"
GOOD = (
    "# ★ [DUMB-KNOBS 2026-10-05] 这 5 个开关 serve_v2.sh 会读、但本脚本从不设置 ⇒ 从启动器设它们\n"
    "#   是静默无效的（实测：FORCE_EPLB=1 起的臂，additional-config 里根本没有 enable_force_eplb，\n"
    "#   于是那次 A/B 其实是同配置对同配置）。这里补齐透传；默认值与 serve_v2.sh 的内建默认一致。\n"
    "#   ⚠️ 本 heredoc 不加引号：**禁止在此块内使用反引号或 $() 形式**（会被当命令替换执行，set -e 下会毁掉 inner.sh）。\n"
)


def main() -> int:
    check = "--check" in sys.argv
    s = io.open(PATH, encoding="utf-8").read()
    lines = s.splitlines(True)
    idx = next((i for i, ln in enumerate(lines) if ln.startswith(BAD_START)), None)
    if idx is None:
        print("找不到 DUMB-KNOBS 注释块")
        return 1
    # 旧块 = 从 BAD_START 起、直到以 "内建默认一致。" 结尾的那一行
    end = idx
    while end < len(lines) and "内建默认一致" not in lines[end]:
        end += 1
    old_block = "".join(lines[idx : end + 1])
    n_backtick = old_block.count("`")
    print("旧块 %d 行，含反引号 %d 个" % (end - idx + 1, n_backtick))
    if check:
        return 0 if n_backtick == 0 else 2
    if n_backtick == 0:
        print("已经是干净的，不动")
        return 0
    new = s.replace(old_block, GOOD, 1)
    assert "`" not in GOOD
    io.open(PATH, "w", encoding="utf-8").write(new)
    print("已替换为无反引号版本")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
