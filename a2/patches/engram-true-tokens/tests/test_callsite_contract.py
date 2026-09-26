#!/usr/bin/env python3
"""调用点契约测试 —— 专抓"模块各自都对、接起来就崩"这一类。

## 为什么需要这个文件（2026-09-22 22:1x 实测踩到）

交付件里 `model.py` 的 `_engram_build_prev_tok()` 负责把 runner 发布的 host token 表
转成 `prev_tok[n, lookback]`，再交给 `engram_repair.apply_repairs()`。

而 `engram_repair.build_prev_tok()` 返回的是 **`(prev_tok, stats)` 二元组**。
第一版调用点直接 `return build_prev_tok(...)`（忘了取 `[0]`）⇒ 下游
`prev_tok[row, sh]` 抛：

    TypeError: tuple indices must be integers or slices, not tuple

⇒ **worker 起服后第一个请求就死**（A3 实测：`p3b-true1-rowids1`，
`serve.log:2143`，`True_Tokens=1` 首次 `apply_repairs` 即崩）。

★ 为什么既有单测抓不到：`test_true_tokens_repair.py` 是**隔离**测
  `apply_repairs`（自己造一个规范的 2-D 数组喂进去），**从不经过调用点**。
  ⇒ 单测全绿、真实链路一跑就死。**这类"接口形状不匹配"必须用调用点契约测试兜住。**
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import repair_helpers  # noqa: E402

FAILS: list[str] = []


def chk(cond: bool, msg: str) -> None:
    print(f"  [{'✓' if cond else '✗'}] {msg}")
    if not cond:
        FAILS.append(msg)


def _model_patched_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "..", "patches", "model.patched.py"),
              os.path.join(here, "..", "model.patched.py")):
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError("找不到 model.patched.py（试过 patches/ 与同级目录）")


def _hash_patched_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "..", "patches", "engram_hash.patched.py"),
              os.path.join(here, "..", "engram_hash.patched.py")):
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError("找不到 engram_hash.patched.py（试过 patches/ 与同级目录）")


def main() -> int:
    print("=" * 78)
    print("调用点契约测试：build_prev_tok 的返回形状 vs apply_repairs 的期望")
    print("=" * 78)

    R = repair_helpers()

    # ---- 1b. ★ 计数上报必须"能落到日志"（本仓实测踩过：只打一次的提示承载判据）
    print("\n① b 计数上报口径（logs/083：一次性提示不能承载判据）")
    src_hash = open(_hash_patched_path(), encoding="utf-8").read()
    # ★ 用 AST 判"函数体里真的没引用它"，**不要**用 `grep 字符串` ——
    #   我第一版就是 grep，结果被我自己的注释（"旧实现把 `_PAGELESS_WARNED[0]` 共用…"）
    #   判成"仍在共用" ⇒ **假失败**。这与 logs/079 §3 的"判据被自己的文本污染"是同一个坑，
    #   一天之内第二次出现 —— 结论：**判代码要用 AST/语法树，不要用字符串搜索**。
    import ast as _ast
    _tree = _ast.parse(src_hash)
    _fn = next((n for n in _ast.walk(_tree)
                if isinstance(n, _ast.FunctionDef) and n.name == "_engram_true_tokens_note"), None)
    chk(_fn is not None, "找得到 `_engram_true_tokens_note` 函数定义")
    _uses = set()
    if _fn is not None:
        for n in _ast.walk(_fn):
            if isinstance(n, _ast.Name):
                _uses.add(n.id)
            elif isinstance(n, _ast.Attribute):
                _uses.add(n.attr)
    chk("_PAGELESS_WARNED" not in _uses,
        f"★ 函数体（AST）里不再引用 `_PAGELESS_WARNED`；实际引用={sorted(_uses)[:8]}")
    chk("_TT_CUM" in src_hash and "mismatch" in src_hash,
        "存在按键累计 `_TT_CUM`（含 mismatch）")
    chk("_TT_LOG_EVERY" in src_hash,
        "有 `V41_ENGRAM_TRUE_TOKENS_LOG_EVERY` 节流（避免刷屏又不丢判据）")

    # ---- 1. 出货模块的返回形状必须就是 (arr, stats)，且 arr 可按 [row, sh] 索引
    print("\n① 出货模块 `build_prev_tok` 的返回形状")
    n, lb = 3, 4
    pos = np.array([10, 20, 30], np.int64)
    req = np.array([0, 1, 2], np.int64)
    ntok = np.array([64, 64, 64], np.int32)
    ids = np.zeros((8, 128), np.int32)
    for r in range(3):
        ids[r, :40] = np.arange(40, dtype=np.int32) + 1
    ret = R.build_prev_tok(ntok, pos, req, lb, ids)
    chk(isinstance(ret, tuple) and len(ret) == 2,
        f"返回二元组 (prev_tok, stats) —— 实际 type={type(ret).__name__}")
    arr, stats = ret
    chk(isinstance(arr, np.ndarray) and arr.ndim == 2,
        f"[0] 是 2-D ndarray（shape={getattr(arr, 'shape', None)}）")
    ok_idx = True
    try:
        _ = int(arr[0, 1])
    except Exception as e:  # noqa: BLE001
        ok_idx = False
        print(f"      arr[0,1] 抛：{e!r}")
    chk(ok_idx, "arr 可按 [row, sh] 整数索引（apply_repairs 的核心假设）")

    # ---- 2. 调用点必须取 [0]（这是本次真事故）
    print("\n② 调用点 `model.patched.py::_engram_build_prev_tok` 必须取 [0]")
    src = open(_model_patched_path(), encoding="utf-8").read()
    # ★ 必须按**整行**判断，不能用 `return\s+build_prev_tok\([^\n]*\)` —— 那是贪心匹配，
    #   会停在行内最后一个 `)` 上（也就是 `[0]` 之前的那个），从而把 `...[0]` 误判成缺 `[0]`。
    #   （这个正则我自己刚踩过：测试报 ✗ 而源码是对的。）
    lines = [ln.rstrip() for ln in src.splitlines() if "return build_prev_tok(" in ln]
    chk(len(lines) == 1, f"源里恰有 1 处 `return build_prev_tok(...)`（实际 {len(lines)} 处）")
    if lines:
        line = lines[0]
        print(f"      原文：{line.strip()}")
        chk(line.endswith("[0]"),
            "★ 该 return 以 `[0]` 结尾（否则会把整个元组当 prev_tok 传下去）")

    # ---- 3. 反向验证：把元组喂给 apply_repairs 必须"响亮地"失败
    #        （证明这条 bug 的形状就是致命的，不是无害的）
    print("\n③ 反向验证：元组直接喂 apply_repairs ⇒ 应当抛 TypeError")
    pages = np.full((8, 4), -1, np.int64)
    present = np.zeros(8, np.uint8)
    tm = np.arange(64, dtype=np.int64)
    bt = np.zeros((2, 4), np.int64)
    pos2 = np.array([4], np.int64)
    req2 = np.array([0], np.int64)
    threw = None
    try:
        R.apply_repairs(pages, present, tm, bt, req2, pos2, ret, 4, lb,
                        -1, -2, 1, {})     # ← 故意传元组
    except Exception as e:  # noqa: BLE001
        threw = e
    chk(isinstance(threw, TypeError),
        f"抛 TypeError（实际 {type(threw).__name__}: {threw}）")

    # ---- 4. 正确形状必须能跑通（对照，证明③的失败不是别的原因）
    print("\n④ 对照：正确的 2-D 数组喂进去应当正常返回")
    pages2 = np.full((8, 4), -1, np.int64)
    present2 = np.zeros(8, np.uint8)
    st: dict[str, int] = {}
    oob = R.apply_repairs(pages2, present2, tm, bt, req2, pos2, arr, 4, lb,
                          -1, -2, 1, st)
    chk(oob == -1 and isinstance(st, dict),
        f"正常返回 oob={oob} stats={st}")

    print("\n" + "=" * 78)
    if FAILS:
        print(f"✗ 调用点契约测试：{len(FAILS)} 项失败")
        for f in FAILS:
            print(f"    - {f}")
        return 1
    print("✓ 调用点契约测试：全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
