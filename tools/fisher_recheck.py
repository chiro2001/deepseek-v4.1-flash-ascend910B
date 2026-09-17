#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fisher_recheck.py —— Fisher 精确检验（2x2）**带教科书自检**。

### 为什么包里有这个
2026-09-16 我们自己的 `exp_tools/interleave_ab.py` 的 `fisher()` 在**两行计数完全相同**时
打印 `p = 0.0000`（相同的表必须 p=1）。根因是把列和固定住却没有在边际固定的条件下
遍历所有可能的 `x`，且"概率不超过观测表"的判据在 `obs` 恰为最大概率时退化成 0/0。
⇒ **任何统计工具上线前必须用已知答案自检。** 本脚本内置 4 个用例（含我们踩过的那个坑）。

### 三种 p（**先看定义再引用数字，否则会互相打架**）
表 = `[[a, b], [c, d]]`：`a/c` = 两臂"成功"数（如 clean 轮数），`b/d` = 失败数。
`X` = 第 1 臂的成功数，其分布是**边际固定**的超几何分布。

| 记号 | 定义 | 本项目里对应 |
|---|---|---|
| `p_right` | `P(X ≥ a)` —— "第 1 臂更好"的单侧 p | `correctness-line.md:371` 的 **0.0256** |
| `p_left` | `P(X ≤ a)` | 「反方向」 |
| `p_two` | 所有 `P(表) ≤ P(观测表)` 的表之和（= R `fisher.test` 默认口径） | `session-attractor-and-clean-rate.md` §6.4 的 **0.0042** |

⚠️ 同一张表 `p_right` 与 `p_two` **不相等**（例：`[[9,7],[2,18]]` → 0.0039 vs 0.0042）。
报告里"单侧/双侧"必须与上述定义对齐后再引用。

### 用法
```
python3 tools/fisher_recheck.py                 # 只跑自检
python3 tools/fisher_recheck.py 5 5 2 18        # 三种 p 全打印
```
"""
import argparse
import math
import sys


def _log_comb(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher(a, b, c, d):
    """返回 (p_right, p_left, p_two, p_obs)；table = [[a,b],[c,d]]。"""
    n = a + b + c + d
    r1, r2 = a + b, c + d
    c1 = a + c
    log_den = _log_comb(n, c1)
    lo = max(0, c1 - r2)
    hi = min(r1, c1)

    def log_p(x):
        return _log_comb(r1, x) + _log_comb(r2, c1 - x) - log_den

    log_obs = log_p(a)
    p_obs = math.exp(log_obs)
    p_right = sum(math.exp(log_p(x)) for x in range(a, hi + 1))
    p_left = sum(math.exp(log_p(x)) for x in range(lo, a + 1))
    p_two = sum(math.exp(log_p(x)) for x in range(lo, hi + 1)
                if log_p(x) <= log_obs + 1e-12)
    return min(p_right, 1.0), min(p_left, 1.0), min(p_two, 1.0), p_obs


# (a, b, c, d), 期望的 p_two（教科书 / R 的标准值）
SELFTEST = [
    ((3, 1, 1, 3), 0.4857142857),      # 经典"女士品茶"
    ((10, 0, 0, 10), 0.0000108250),    # 教科书极端表
    ((1, 9, 11, 3), 0.0027594),        # 教科书小样本
    ((2, 22, 2, 22), 1.0),             # ★ 我们踩过的坑：相同表必须 p=1（不是 0.0000）
]


def selftest():
    ok = True
    print("== Fisher 实现自检（标准值 = 教科书 / R `fisher.test`）==")
    for (a, b, c, d), want in SELFTEST:
        _, _, p2, _ = fisher(a, b, c, d)
        good = abs(p2 - want) < 1e-6 or (want >= 1.0 and p2 > 1 - 1e-9)
        ok &= good
        print(f"  [[{a},{b}],[{c},{d}]] -> p_two={p2:.7f} (期望 {want:.7f}) "
              f"{'OK' if good else 'FAIL'}")
    print(f"== 自检{'通过' if ok else '失败'} ==")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", type=int, help="a b c d（表 [[a,b],[c,d]]）")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if len(a.cells) != 4:
        return selftest()
    rc = 0 if a.quiet else selftest()
    a_, b_, c_, d_ = a.cells
    p_r, p_l, p_2, p_obs = fisher(a_, b_, c_, d_)
    n1, n2 = a_ + b_, c_ + d_
    print(f"\n表 [[{a_},{b_}],[{c_},{d_}]]  "
          f"臂1 = {a_}/{n1} = {a_/n1:.4f}，臂2 = {c_}/{n2} = {c_/n2:.4f}")
    print(f"P(观测表)        = {p_obs:.6g}")
    print(f"p_right = P(X≥a) = {p_r:.6f}   ← '第 1 臂更好'的单侧")
    print(f"p_left  = P(X≤a) = {p_l:.6f}")
    print(f"p_two   = Σ P(表)≤P(观测) = {p_2:.6f}   ← R fisher.test 默认口径")
    return rc


if __name__ == "__main__":
    sys.exit(main())
