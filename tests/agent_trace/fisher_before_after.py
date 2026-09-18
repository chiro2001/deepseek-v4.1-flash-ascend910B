#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复前/后的 Fisher 精确检验 —— 为"统计显著的前后对比"提供可复算的判据。

**为什么需要它**：单看"修复后 10/10"没有说服力 —— 10 次全过也可能只是运气。
本脚本对"修复前 vs 修复后"做 Fisher 精确检验，给出 p 值。

不依赖 scipy：用超几何分布精确求和。

用法：
    python3 fisher_before_after.py
    python3 fisher_before_after.py --before 5/10 --after 10/10

判据：p < 0.05 即"修复前失败率显著高于修复后"。
"""

from __future__ import annotations

import argparse
from math import comb


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """表 = [[a, b], [c, d]]，检验"第一组（修复前）失败率更高"。

    a = 修复前 pass, b = 修复前 fail
    c = 修复后 pass, d = 修复后 fail

    做法：固定边际后，第一组 pass 数 X ~ Hypergeom(N, K=总pass, n=第一组样本数)，
    单侧 p = P(X <= a)（即"第一组这么少 pass"的概率）。
    """
    n = a + b + c + d
    r1 = a + b            # 修复前样本数
    k = a + c             # 总 pass 数
    lo = max(0, r1 - (n - k))
    denom = comb(n, r1)
    return sum(comb(k, x) * comb(n - k, r1 - x) for x in range(lo, a + 1)) / denom


def parse_ratio(s: str) -> tuple[int, int]:
    a, b = s.split("/")
    return int(a), int(b)


# ---------------------------------------------------------------- 默认数据集
#
# 全部来自 8×910C、TP8、同一探针（needle 埋在长文档 50% 处、只问答案唯一的问题、
# 字符串精确匹配判分），每档 N=10 个内容不同的样本（不同 nonce）。
#
# 修复前：BAT_TOKENS=2048（`--max-num-batched-tokens 2048`）
# 修复后：BAT_TOKENS=8192
#
# 说明：改 `BAT_TOKENS` 必须重启，所以"修复前/后"天然跨会话。
# 为把会话差异压到最小，这里优先用**逐 token 对齐**的配对
# （130,538 那一条前后是同一次探针、同一个 prompt 长度），
# 并附上 60K 的三次独立会话汇总。
DEFAULT_CASES = [
    # (标签, 修复前 pass, 修复前 n, 修复后 pass, 修复后 n)
    ("130,538（逐token对齐）", 5, 10, 10, 10),
    ("60K（三次会话汇总）",   22, 36, 30, 30),
    ("60K（单次最保守）",      5, 10, 10, 10),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", help="形如 5/10")
    ap.add_argument("--after", help="形如 10/10")
    a = ap.parse_args()

    if a.before and a.after:
        bp, bn = parse_ratio(a.before)
        qp, qn = parse_ratio(a.after)
        cases = [("自定义", bp, bn, qp, qn)]
    else:
        cases = DEFAULT_CASES

    print(f"{'对比点':<16}{'修复前':>12}{'修复后':>12}{'Fisher p(单侧)':>18}  判定")
    print("-" * 72)
    for name, bp, bn, qp, qn in cases:
        p = fisher_one_sided(bp, bn - bp, qp, qn - qp)
        verdict = "显著" if p < 0.05 else "不显著"
        print(f"{name:<16}{f'{bp}/{bn}':>12}{f'{qp}/{qn}':>12}{p:>18.2e}  {verdict}")
    print()
    print("判据：p < 0.05 ⇒ 修复前失败率显著高于修复后（单侧 Fisher 精确检验）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
