#!/usr/bin/env python3
"""[A2] 判定两份 Python 文件是否**语义等价**（只差注释与字符串字面量）。

为什么需要它：交付件的身份台账要求"发布件的 md5 必须在某条 PASS 臂上跑过"。
但有两种合法情形会让 md5 不同：
  * 只有**注释**不同；
  * 只有**日志/报错文案**（普通字符串 + f-string 文本段）不同。
这两种情形**语义零差异**，不该被当成"没跑过的新版本"挡发布，也不该被当成"同一个文件"放行。
⇒ 用本脚本给出**机械判据**（token 级），而不是靠肉眼看 diff。

用法：
    python3 a2/scripts/check_semantic_nodiff.py <文件A> <文件B>
退出码：
    0 = 语义零差异（忽略注释/字符串/f-string 文本段后逐 token 相同）
    2 = 有真实差异（会打印前 8 处）
    64 = 用法错误 / 读不到文件

★ 忽略的 token：COMMENT / NL / NEWLINE / INDENT / DEDENT / ENCODING / ENDMARKER /
  STRING（普通字符串字面量）/ FSTRING_MIDDLE（f-string 里的文本段）。
★ **没有**忽略 NUM / NAME / OP / 关键字 / FSTRING_START / FSTRING_END ⇒
  变量名、数字、运算符、控制流、f-string 里的**表达式**都必须逐一相同。
"""

from __future__ import annotations

import itertools
import sys
import tokenize

_IGNORED = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.STRING,
        getattr(tokenize, "FSTRING_MIDDLE", -1),
    }
)


def tokens(path: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in _IGNORED:
                continue
            out.append((tokenize.tok_name[tok.type], tok.string))
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.split("用法：")[1].strip().splitlines()[0], file=sys.stderr)
        return 64
    a, b = argv[1], argv[2]
    try:
        ta, tb = tokens(a), tokens(b)
    except OSError as exc:
        print(f"读文件失败：{exc}", file=sys.stderr)
        return 64

    print(f"非注释/非字符串 token 数：{len(ta)} vs {len(tb)}")
    if ta == tb:
        print("★ 逐 token 相同 ⇒ 两份文件语义零差异（只差注释与字符串字面量）")
        return 0

    print("⛔ token 序列不同，前 8 处差异：")
    shown = 0
    for idx, (x, y) in enumerate(itertools.zip_longest(ta, tb)):
        if x != y:
            print(f"  #{idx}: {x!r} vs {y!r}")
            shown += 1
            if shown >= 8:
                break
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
