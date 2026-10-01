#!/usr/bin/env python3
"""给 merge_pre 也加连续性防御（与 merge_post 同因）。

`output` / `lse` / `ori_lse` 都走 `data_ptr()` 裸指针：
  · `output` 来自 SMLA 算子，通常连续；
  · `_ori_lse_f32` 是 `_ori_lse.reshape(T,H,1)` —— reshape 通常是视图，
    内存序与源一致（连续）；
  · 但**不能假设**：`lse` 在本函数里已被 `reshape` 过（见调用方 :489-497），
    若上游给的是非连续张量，reshape 可能产出非连续视图。
⇒ 统一在 wrapper 层加 `.contiguous()` 防御（连续时是 no-op）。
"""
import hashlib
import pathlib
import sys

P = pathlib.Path.home() / "dcpw/vllm_ascend/attention/v41_merge_kernel.py"

OLD = """    T, H, _Dd = (int(v) for v in output.shape)
    # alpha/subw 在 pre 里不用，但 tiling 是共用的 ⇒ 传 0 即可（kernel 不读）
    tt = _tiling(T, H, int(ori_lse.shape[1]) if ori_lse.dim() == 3 else H,
                 0, 0.0, 0.0, output.device)
    s = torch.npu.current_stream().npu_stream
    rc = lib.v41_merge_pre_launch(
        _GRID, ctypes.c_void_p(s),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(lse.data_ptr()),
        ctypes.c_void_p(ori_lse.data_ptr()),
        ctypes.c_void_p(pack.data_ptr()),
        ctypes.c_void_p(tt.data_ptr()),
    )
"""

NEW = """    T, H, _Dd = (int(v) for v in output.shape)
    # alpha/subw 在 pre 里不用，但 tiling 是共用的 ⇒ 传 0 即可（kernel 不读）
    tt = _tiling(T, H, int(ori_lse.shape[1]) if ori_lse.dim() == 3 else H,
                 0, 0.0, 0.0, output.device)
    # ★ [V41-MKC-CONTIG] 同 merge_post：裸指针 ⇒ 必须先保证连续
    output = _as_contig(output, "output")
    lse = _as_contig(lse, "lse")
    ori_lse = _as_contig(ori_lse, "ori_lse")
    pack = _as_contig(pack, "pack")
    s = torch.npu.current_stream().npu_stream
    rc = lib.v41_merge_pre_launch(
        _GRID, ctypes.c_void_p(s),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(lse.data_ptr()),
        ctypes.c_void_p(ori_lse.data_ptr()),
        ctypes.c_void_p(pack.data_ptr()),
        ctypes.c_void_p(tt.data_ptr()),
    )
"""


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    print("[before] md5=%s" % md5(P))
    s = P.read_text()
    if 'output = _as_contig(output, "output")' in s:
        print("[ok] 已应用")
        return 0
    if s.count(OLD) != 1:
        raise SystemExit("[FAIL] 锚点 %d" % s.count(OLD))
    P.write_text(s.replace(OLD, NEW, 1))
    print("[after ] md5=%s" % md5(P))
    import ast
    ast.parse(P.read_text())
    print("[ok] 语法通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
