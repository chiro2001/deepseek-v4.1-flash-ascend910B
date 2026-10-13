#!/usr/bin/env python3
"""修复 RS_MERGE 路径：reduce_scatter 之后的 permute 视图必须 .contiguous()。

## 症状
`V41_DCP_RS_MERGE=1` 时 A（平均接受长度）从 2.5–3.0 掉到 **1.00**
⇒ 所有草稿被 verify 拒绝 ⇒ merge 结果算错。

## 根因
`_v41_dcp_merge_attention` 的 RS 分支：
```python
_pack = _rs_out.view(_rows, T, _W).permute(1, 0, 2)   # [T,8,W] 非连续
head_slice = None
```
下游立刻在 `_pack` 上做**逐元素减法 + 除法**：
```python
scaled = _pack[..., :_out_dim] - _onum
wsum   = _pack[..., _out_dim:_out_dim+1] - dcp*_keep
```
而**同一文件的 `V41-DENFIX` 注释精确记录过这个失败模式**：
> 设备侧那次「`[T,H,1]` 视图 + 标量减法」**没有读到真实数据**（读到的等价于 padding 的 0）

代码库另有三处记录："非连续视图上的逐元素算子在本平台不可信"
（`PACKDIRECT-ABORT`、`SUBALPHA-ABORT`、`contigw`）。

## 修法
permute 之后加 `.contiguous()`。代价：一次 `[T,8,640]` fp32 拷贝（164 KB），
远小于 reduce_scatter 省下的搬运量（2.29 MB → 1.15 MB）。
"""
import hashlib
import pathlib
import sys

P = pathlib.Path.home() / "dcpw/vllm_ascend/attention/dsa_v41.py"

OLD = """            _pack = _rs_out.view(_rows, int(_pack.shape[0]), _W).permute(1, 0, 2)
            head_slice = None
            _rs_applied = True
"""

NEW = """            # ★★★★★★ [V41-RSCONTIG 2026-10-01] **必须 .contiguous()**。
            #   症状：不加时 `V41_DCP_RS_MERGE=1` 的 A 从 2.5-3.0 掉到 **1.00**
            #   （所有草稿被 verify 拒绝）⇒ merge 结果算错。
            #   根因：`permute` 产生**非连续视图**，而下游立刻做逐元素减法/除法
            #   （`scaled = _pack[..., :D] - _onum`、`wsum = _pack[...] - dcp*_keep`）。
            #   本文件 `V41-DENFIX` 的注释精确记录过同一失败模式：
            #     「设备侧那次「[T,H,1] 视图 + 标量减法」**没有读到真实数据**」。
            #   代码库另有三次同类记录（PACKDIRECT-ABORT / SUBALPHA-ABORT / contigw）
            #   ⇒ 「非连续视图 + 逐元素算子」在本平台一律不可信。
            #   代价：一次 [T,8,640] fp32 拷贝（164 KB），远小于 reduce_scatter
            #   省下的搬运（2.29 MB → 1.15 MB）。
            _pack = _rs_out.view(_rows, int(_pack.shape[0]), _W).permute(1, 0, 2).contiguous()
            head_slice = None
            _rs_applied = True
"""


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    print("[before] dsa_v41.py md5=%s" % md5(P))
    s = P.read_text()
    n = s.count(OLD)
    if n == 0 and "V41-RSCONTIG" in s:
        print("[ok] 已应用")
        return 0
    if n != 1:
        raise SystemExit("[FAIL] 锚点 %d 次" % n)
    P.write_text(s.replace(OLD, NEW, 1))
    print("[after ] dsa_v41.py md5=%s" % md5(P))
    import ast
    ast.parse(P.read_text())
    print("[ok] 语法通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
