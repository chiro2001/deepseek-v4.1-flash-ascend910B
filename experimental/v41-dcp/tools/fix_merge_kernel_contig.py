#!/usr/bin/env python3
"""修复 AscendC merge kernel 的**非连续输入**问题（乱码根因）。

## 症状
`V41_DCP_MERGE_KERNEL=1` 在 tiny 上 5/6 相同、**TP8+DCP8 真权重下 5/5 全乱码**
（`17×23` 输出 `'6. false;  fight; 0;'`）。

## 根因
wrapper 用 `tensor.data_ptr()` 把裸指针交给 kernel，**不检查 strides**：

```python
ctypes.c_void_p(ori_out.data_ptr()),      # merge_post
ctypes.c_void_p(output.data_ptr()),       # merge_pre
```

而调用方传进来的 `ori_out` 是 **非连续切片**：

```python
_oi = ori_out[:, head_slice[0]:head_slice[1], :]      # [T,8,512]，stride=(64*512,512,1)
```

kernel 按**连续布局** `[T,Hout,D]` 读 ⇒ 行 stride 用 `Hout*D`（实际是 `H*D`）
⇒ 从第 1 行起全部错位 ⇒ 输出乱码。

同样的失败模式在本项目已记录三次（`V41-SUBALPHA-ABORT` 的注释精确写过
「两个输入都是 strided 切片」），但那三次是 **torch 逐元素算子**；
本次是 **ctypes 直调 kernel**，问题更严重（torch 至少还看 strides）。

## 修法
在 wrapper 里对**所有**交给 kernel 的输入做 `.contiguous()` 防御
（连续时是 no-op，零开销；非连续时才拷贝）。

## 影响
- tiny（DCP=2）：`_oi` 同样非连续 ⇒ 也会错，只是 dummy 权重看不出
- TP8（DCP=8）：真权重下立刻暴露
"""
import hashlib
import pathlib
import sys

P = pathlib.Path.home() / "dcpw/vllm_ascend/attention/v41_merge_kernel.py"

HELPER = '''

def _as_contig(t: torch.Tensor, tag: str) -> torch.Tensor:
    """kernel 只认连续布局（走 data_ptr 裸指针）⇒ 非连续输入必须先拷成连续。

    ★ [V41-MKC-CONTIG 2026-10-01] 为什么必须：
      调用方传进来的 `ori_out` 常常是**切片**：
          _oi = ori_out[:, head_slice[0]:head_slice[1], :]   # [T,8,512]
      其 stride 是 `(H*D, D, 1)` 而非连续的 `(Hout*D, D, 1)`。
      kernel 按 `[T,Hout,D]` 连续读 ⇒ 从第 1 行起错位 ⇒ 输出乱码。
      实测：TP8+DCP8 真权重下 `17x23` 输出 `'6. false;  fight; 0;'`（5/5 全错）。
    连续输入时 `.contiguous()` 是 no-op（零开销）。
    """
    if t is None:
        return t
    if not t.is_contiguous():
        return t.contiguous()
    return t


'''

M1_OLD = """    T, H, _Ww = (int(v) for v in pack.shape)
    Hout = int(ori_out.shape[1])
    tt = _tiling(T, H, Hout, int(h0), float(alpha), float(subw), pack.device)
    s = torch.npu.current_stream().npu_stream
    rc = lib.v41_merge_post_launch(
        _GRID, ctypes.c_void_p(s),
        ctypes.c_void_p(pack.data_ptr()),
        ctypes.c_void_p(ori_out.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(tt.data_ptr()),
    )
"""
M1_NEW = """    T, H, _Ww = (int(v) for v in pack.shape)
    Hout = int(ori_out.shape[1])
    tt = _tiling(T, H, Hout, int(h0), float(alpha), float(subw), pack.device)
    # ★ [V41-MKC-CONTIG] kernel 走裸指针 ⇒ 必须先保证连续
    pack = _as_contig(pack, "pack")
    ori_out = _as_contig(ori_out, "ori_out")
    out = _as_contig(out, "out")
    s = torch.npu.current_stream().npu_stream
    rc = lib.v41_merge_post_launch(
        _GRID, ctypes.c_void_p(s),
        ctypes.c_void_p(pack.data_ptr()),
        ctypes.c_void_p(ori_out.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(tt.data_ptr()),
    )
"""


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    print("[before] %s md5=%s" % (P.name, md5(P)))
    s = P.read_text()
    if "V41-MKC-CONTIG" in s:
        print("[ok] 已应用")
        return 0

    # 1) 插入 helper（放在第一次 def 之前）
    anchor = "def merge_pre("
    if s.count(anchor) != 1:
        raise SystemExit("[FAIL] merge_pre 锚点 %d" % s.count(anchor))
    s = s.replace(anchor, HELPER.lstrip("\n") + "\n" + anchor, 1)

    # 2) merge_post 加防御
    if s.count(M1_OLD) != 1:
        raise SystemExit("[FAIL] merge_post 锚点 %d" % s.count(M1_OLD))
    s = s.replace(M1_OLD, M1_NEW, 1)

    P.write_text(s)
    print("[after ] %s md5=%s" % (P.name, md5(P)))
    import ast
    ast.parse(P.read_text())
    print("[ok] 语法通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
