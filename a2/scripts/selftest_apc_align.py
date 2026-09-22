#!/usr/bin/env python3
"""`[APC_ALIGN]` 的离线单元自检（不占卡、不 import vllm）。

验证三件事（`logs/047` 判据 ⑪ 的本地版）：
  1. **默认关闭**（`VLLM_V41_APC_ALIGN` 未设 / =0）⇒ 命中长度逐字原样返回；
  2. **V4.1 语义**（ratio=2 + 段栅格 1024）⇒ 4095→3072、4096→4096、2047→1024；
  3. ★ **普通模型**（ratio=1，即使有 full-attention 组、段栅格=1024）⇒ **逐字 no-op**
     —— 这是 `_apc_has_compressed_group` 那道门的作用。

用法：`python3 a2/scripts/selftest_apc_align.py`
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_PUB = os.path.join(HERE, "..", "publish")
# ★ 两个版本都要过：普通链（0001）与 8 卡链（0001-8card）
PATCHES = [
    os.path.join(_PUB, "0001-offload-scheduler.patch.py"),
    os.path.join(_PUB, "0001-8card-offload-scheduler.patch.py"),
]
PATCH = PATCHES[0]  # 兼容旧引用


def _load_pure_funcs(src: str):
    """从补丁文件里抽出三个纯函数并 exec（不 import vllm）。"""
    ns: dict = {
        "os": os,
        "_env_int": lambda n, d, m: int(os.environ.get(n, d) or d),
        # 纯注解用的占位（真模块里由 imports 提供；`from __future__ import annotations`
        # 让注解不求值，但 exec 出来的函数体里若被引用就仍需存在）
        "KVCacheConfig": object,
    }
    for name in ("_apc_align_mode", "_apc_align_hit", "_apc_has_compressed_group"):
        m = re.search(
            r"\ndef " + name + r"\(.*?(?=\n\ndef |\n\nclass |\n\n# ---)", src, re.S
        )
        assert m, f"找不到 {name}"
        exec(m.group(0).strip(), ns)  # noqa: S102
    return ns


class _Spec:
    def __init__(self, compress_ratio=1):
        self.compress_ratio = compress_ratio


class _Group:
    def __init__(self, compress_ratio=1):
        self.kv_cache_spec = _Spec(compress_ratio)


class _Cfg:
    def __init__(self, unit):
        self.apc_align_unit = unit


class _KVCfg:
    def __init__(self, ratios):
        self.kv_cache_groups = [_Group(r) for r in ratios]


def main() -> int:
    npass = nfail = 0

    def ck(name, ok, extra=""):
        nonlocal npass, nfail
        if ok:
            npass += 1
            print(f"  PASS  {name}")
        else:
            nfail += 1
            print(f"  FAIL  {name}   {extra}")

    for patch in PATCHES:
        tag = os.path.basename(patch)
        ns = _load_pure_funcs(open(patch, encoding="utf-8").read())
        align_hit = ns["_apc_align_hit"]
        has_comp = ns["_apc_has_compressed_group"]
        print(f"\n========== {tag} ==========")

        print("[1] 默认关闭 ⇒ 逐字 no-op")
        os.environ.pop("VLLM_V41_APC_ALIGN", None)
        ck("未设 env 时 mode=0", ns["_apc_align_mode"]() == 0)
        for n in (1, 255, 1024, 2048, 4095, 4096, 4097):
            ck(f"unit=0 时 {n} 原样返回", align_hit(_Cfg(0), n) == n)

        print("[2] V4.1 语义（ratio=2 + 段栅格 1024 ⇒ unit=1024）")
        for n, want in ((4095, 3072), (4096, 4096), (2047, 1024), (1024, 1024), (1023, 0)):
            got = align_hit(_Cfg(1024), n)
            ck(f"{n} -> {want}", got == want, f"got={got}")

        print("[3] ★ 普通模型（ratio=1）⇒ 门控必须挡住")
        ck("ratio=1 无压缩组 ⇒ False", has_comp(_KVCfg([1, 1, 1])) is False)
        ck("有 ratio=2 组 ⇒ True", has_comp(_KVCfg([2, 1, 1])) is True)
        unit_plain = 1024 if has_comp(_KVCfg([1, 1])) else 0
        ck("普通模型 unit 被压回 0", unit_plain == 0)
        for n in (4095, 4096, 2047):
            ck(f"普通模型 {n} 原样返回", align_hit(_Cfg(unit_plain), n) == n)

    print(f"\n[selftest_apc_align] {npass} PASS / {nfail} FAIL")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
