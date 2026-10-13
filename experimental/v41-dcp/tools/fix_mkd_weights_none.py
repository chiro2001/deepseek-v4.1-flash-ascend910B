#!/usr/bin/env python3
"""修复：`_merge_kd`（AscendC 融合路径）下 `weights=None` 导致诊断崩溃。

## 症状
`V41_DCP_MERGE_KERNEL=1` 时**起服失败**（tiny 与 TP8 都复现）：
```
File ".../dsa_v41.py", line 3728, in _native_attention
File ".../dsa_v41.py", line 609, in _v41_dcp_merge_attention
AttributeError: 'NoneType' object has no attribute 'sum'
RuntimeError: NPUModelRunner failed, error is 'NoneType' object has no attribute 'sum'
```

## 根因
`_merge_kd=True` 时：
```python
weights = (None if _merge_kd else torch.nan_to_num(torch.exp(_delta.clamp(max=60.0))))
```
（融合算子内部自己算 weights ⇒ Python 侧省掉）

但多处**诊断**仍用 `weights.*`，而它们的守卫是 `not _is_capturing()`。
崩溃发生在 `_warmup_and_capture → _dummy_run` 的**预热阶段** —— 此时
**还没进入 ACL graph 捕获**，`_is_capturing()` 为 False ⇒ 诊断跑 ⇒ 崩。

⇒ `not _is_capturing()` **不能**作为 `_merge_kd` 的保护。

## 修法
给所有用 `weights` 的诊断加 `weights is not None`（等价于 `not _merge_kd`，
但这些诊断本来就是给**非融合路径**做数值核对的，融合路径下无意义）。

## 覆盖的 4 处
| abs 行 | 诊断 | 原守卫 |
|---|---|---|
| 594 | `_ORI_REF_DIAG`（默认每 rank 前 2 次会打印） | `not _is_capturing() and n<2 and numel` |
| 680 | `[V41-LSE]` | `V41_DCP_LSE_DIAG=1 and not _is_capturing()` |
| 984 | `prew` | `_dcp_diag_on("prew") and not _is_capturing()` |
| 1073 | `mdiag` | `_dcp_diag_on("mdiag") and not _is_capturing()` |
"""
import hashlib
import pathlib
import sys

P = pathlib.Path.home() / "dcpw/vllm_ascend/attention/dsa_v41.py"

E1_OLD = """        if not _is_capturing() and _ORI_REF_DIAG["n"] < 2 and _delta.numel():
"""
E1_NEW = """        # ★ [V41-MKD-NONE 2026-10-01] 必须加 `weights is not None`：
        #   `_merge_kd`（AscendC 融合路径）下 `weights` 被置为 None（改由 kernel 内部算），
        #   而本诊断的守卫只有 `not _is_capturing()` —— 崩溃发生在
        #   `_warmup_and_capture → _dummy_run` 的**预热阶段**（此时还没进 capture，
        #   `_is_capturing()` 为 False）⇒ 诊断跑 ⇒ `weights.sum()` 抛
        #   `AttributeError: 'NoneType' object has no attribute 'sum'`
        #   ⇒ 8 worker 全挂、**起服失败**（实测 tiny 与 TP8 都复现，同一行 609）。
        if (not _is_capturing() and weights is not None
                and _ORI_REF_DIAG["n"] < 2 and _delta.numel()):
"""

E2_OLD = """        if _n > 129:
            _bump_lse_diag()
"""
E2_NEW = """        if _n > 129 and weights is not None:   # [V41-MKD-NONE] 同因：融合路径下 weights=None
            _bump_lse_diag()
"""

E3_OLD = """    if (
        _dcp_diag_on("prew", "V41_DCP_PREW")
        and not _is_capturing()
        and diag_seq_lens is not None
        and int(diag_seq_lens.max()) > 1
    ):
"""
E3_NEW = """    if (
        _dcp_diag_on("prew", "V41_DCP_PREW")
        and not _is_capturing()
        and weights is not None        # [V41-MKD-NONE] 融合路径下 weights=None
        and diag_seq_lens is not None
        and int(diag_seq_lens.max()) > 1
    ):
"""

E4_OLD = """        _mdiag_here = _dcp_diag_on("mdiag", "V41_DCP_MERGE_DIAG") and not _is_capturing()
"""
E4_NEW = """        _mdiag_here = (
            _dcp_diag_on("mdiag", "V41_DCP_MERGE_DIAG")
            and not _is_capturing()
            and weights is not None     # [V41-MKD-NONE] 融合路径下 weights=None
        )
"""


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    print("[before] dsa_v41.py md5=%s" % md5(P))
    s = P.read_text()
    n_done = 0
    for tag, old, new in (("_ORI_REF_DIAG", E1_OLD, E1_NEW),
                          ("[V41-LSE]", E2_OLD, E2_NEW),
                          ("prew", E3_OLD, E3_NEW),
                          ("mdiag", E4_OLD, E4_NEW)):
        if new.strip().splitlines()[0].strip() in s and "V41-MKD-NONE" in s:
            pass
        cnt = s.count(old)
        if cnt != 1:
            print("[FAIL] %s 锚点 %d 次" % (tag, cnt))
            return 2
        s = s.replace(old, new, 1)
        n_done += 1
        print("[patch] %s" % tag)
    P.write_text(s)
    print("[after ] dsa_v41.py md5=%s  (改了 %d 处)" % (md5(P), n_done))
    import ast
    ast.parse(P.read_text())
    print("[ok] 语法通过")
    # 复核：不应再有未保护的 weights.
    for i, ln in enumerate(s.splitlines(), 1):
        if "weights." in ln and "weight" in ln:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
