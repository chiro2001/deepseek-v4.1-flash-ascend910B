"""按源码 exec 一个模块，**不写任何字节码缓存**（保护只读的发布树）。

另：在 import 任何 kernel 模块**之前**设 `NUMBA_DISABLE_JIT=1`，
这样 `patches/files/engram_jit_kernel.py` 可以原样只读使用（不编译、不落 cache）。
"""
from __future__ import annotations

import os
import sys
import types

# 必须在 import numba / kernel 之前
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
AGENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _find(*cands: str) -> str:
    """★ 在**多个候选布局**里找同一个文件，找不到就报出所有试过的路径。

    为什么需要：本目录会被**原样复制到发布仓**（`a2/patches/engram-true-tokens/tests/`），
    而发布布局与开发布局**差一层**：
        开发  : a2/agents/Engram_exactfix/{tests,patches}/…
        发布  : a2/patches/engram-true-tokens/{tests,*.py}      ← 没有那层 `patches/`
    ⇒ 写死一个相对路径的话，**开发机上全绿、发布树上全红**（2026-09-22 实测踩到过一次）。
    """
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "找不到文件；试过：\n  " + "\n  ".join(str(c) for c in cands if c)
    )


# `RELEASE`：优先环境变量（在 A3 上可由调用方指定），否则向上找 dsv41-release
def _release_kernel_path() -> str:
    env = os.environ.get("A2_RELEASE_ROOT")
    cands = []
    if env:
        cands.append(os.path.join(env, "patches", "files", "engram_jit_kernel.py"))
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(1, 7):
        root = os.path.abspath(os.path.join(here, *([".."] * up)))
        cands.append(os.path.join(root, "dsv41-release", "patches", "files", "engram_jit_kernel.py"))
        cands.append(os.path.join(root, "patches", "files", "engram_jit_kernel.py"))
    return _find(*cands)


def load_source(path: str, name: str) -> types.ModuleType:
    """exec 源码成一个模块（零 .pyc 写入）。"""
    with open(path, "r") as fh:
        src = fh.read()
    mod = types.ModuleType(name)
    mod.__file__ = path
    mod.__dict__["__name__"] = name
    sys.modules[name] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def release_kernel():
    """生产里真正跑的那个 numba kernel（只读；JIT 已禁用）。"""
    return load_source(_release_kernel_path(), "v41_release_engram_jit_kernel")


def repair_helpers():
    """本次交付的**出货件本身**（影子包会把它挂成 models/deepseek_v41/engram_repair.py）。"""
    return load_source(
        _find(
            os.path.join(AGENT, "patches", "engram_repair.py"),   # 开发布局
            os.path.join(AGENT, "engram_repair.py"),              # 发布布局（同目录）
        ),
        "v41_engram_repair_helpers",
    )
