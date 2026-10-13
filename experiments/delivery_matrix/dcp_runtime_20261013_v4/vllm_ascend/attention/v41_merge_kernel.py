# SPDX-License-Identifier: Apache-2.0
"""V4.1 DCP merge 的 **AscendC 融合算子**（ctypes 直调，可被 ACL graph 捕获）。

## 为什么需要它

真实 8 卡 profiler（run `dcpcap_1001_070358`，20 forwards 口径）显示：
DCP8 的 decode 比 DCP1 多出 **1213 个图节点/step**，其中 **573 个**来自
`_v41_dcp_merge_attention` 的逐元素前/后处理（每层约 15 个小算子 × 38 层）。
实测**每个图节点约值 3 µs 墙钟**（删 76 节点 → −0.23 ms/step）⇒ 这 573 个节点
约值 **1.72 ms/step**。

本模块把那 15 个算子压成 **2 个 kernel**（all_reduce 必须在中间，所以不能合成一个）：

| kernel | 替代的 Python 算子（每层） |
|---|---|
| `pre`  | `lse−ori_lse` / `clamp(60)` / `exp` / `nan_to_num` / `output.to(fp32)` / `*w` / 写 pack |
| `post` | `pack[:D]−alpha·ori_out` / `pack[D]−subw` / `clamp_min` / 除法 / `to(bf16)` |

## 实测（chip6，单卡，`bench_merge.py`）

| T | pre | post | 合计/层 | ×38 层 |
|---:|---:|---:|---:|---:|
| **1（生产 decode）** | 7.08 µs | 6.61 µs | **13.68 µs** | **0.520 ms/step** |
| 4 | 10.41 | 6.41 | 16.82 | 0.639 |
| 16 | 19.99 | 6.39 | 26.38 | 1.002 |
| 96 | 79.02 | 8.97 | 88.00 | 3.344 |

**数值**：T=1–16（生产 decode 的全部尺寸）与 Python 参考**逐位一致**（`max|d|=0`）；
T=20–96 差 **恰好 1 个 bf16 ULP**（0.000488281），原因是本 CANN 上 **fp32 没有可用的
除法指令**（`Divs` 链接期 `undefined symbol`；`Div` 运行期 vector core exception）
⇒ 只能用 `Muls(num, 1/den)`，而 torch 的 `x/den` 是真正的除法。

## 安全门

* 默认 **关闭**（env `V41_DCP_MERGE_KERNEL=1` 打开）；
* `.so` 不存在时自动退回 Python 路径（不会让引擎起不来）；
* `pre` 的数值在 T=1–16 与 Python **逐位一致**，可作为回归判据。

## 图捕获注意事项

* ctypes 调用发生在**捕获期**（每次捕获一次），replay 时由图直接回放 ⇒ 无 host 开销；
* 因此**所有张量地址必须在 replay 间稳定** ⇒ 本模块对 tiling / pack / out 全部按
  `(T,H,D,dtype,device)` **缓存常驻张量**。
"""
from __future__ import annotations

import ctypes
import os

import torch

# ---------------------------------------------------------------------------
# 常量（与 kernel `v41_merge.asc` 的 `MergeTiling` 严格一致）
# ---------------------------------------------------------------------------
_D = 512
_W = 640                 # ((512 + 1 + 127) // 128) * 128，与 _v41_pack_for_reduce 一致
_GRID = 48               # AIV 核数（Ascend910 一个 die 48 个 vector 核）
_MODE = 0                # 0 = den = max(w, eps)；与当前 Python 默认一致
_EPS = 1e-30
_MAXD = 60.0

_LIB_NAME = "libv41merge_ops.so"


class _Tiling(ctypes.Structure):
    _fields_ = [
        ("T", ctypes.c_uint32), ("H", ctypes.c_uint32),
        ("Hout", ctypes.c_uint32), ("h0", ctypes.c_uint32),
        ("D", ctypes.c_uint32), ("W", ctypes.c_uint32),
        ("grid", ctypes.c_uint32), ("mode", ctypes.c_uint32),
        ("alpha", ctypes.c_float), ("subw", ctypes.c_float),
        ("eps", ctypes.c_float), ("maxd", ctypes.c_float),
    ]


_LIB = None
_LIB_TRIED = False
_TILING_CACHE: dict = {}
_BUF_CACHE: dict = {}


def _lib_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), _LIB_NAME)


def available() -> bool:
    """`.so` 是否存在且能加载。

★ 文件名故意与 .py 不同（`libv41merge_ops.so`）：
  Python 导入优先级是 扩展模块(.so) > 源文件(.py)，
  同名会让 `import v41_merge_kernel` 命中 .so 并报
  ImportError: dynamic module does not define module export function
  ⇒ 而调用方吞掉异常 ⇒ 融合算子静默失效（实测踩过）。
"""
    return _get_lib() is not None


def _get_lib():
    global _LIB, _LIB_TRIED
    if _LIB_TRIED:
        return _LIB
    _LIB_TRIED = True
    path = _lib_path()
    if not os.path.exists(path):
        return None
    try:
        lib = ctypes.CDLL(path)
        lib.v41_merge_pre_launch.restype = ctypes.c_int
        lib.v41_merge_pre_launch.argtypes = [
            ctypes.c_uint32, ctypes.c_void_p,
        ] + [ctypes.c_void_p] * 5
        lib.v41_merge_post_launch.restype = ctypes.c_int
        lib.v41_merge_post_launch.argtypes = [
            ctypes.c_uint32, ctypes.c_void_p,
        ] + [ctypes.c_void_p] * 4
        _LIB = lib
    except Exception:  # noqa: BLE001
        _LIB = None
    return _LIB


def _tiling(T, H, Hout, h0, alpha, subw, device):
    """按形状缓存 tiling（uint8 张量，地址跨 replay 稳定）。"""
    key = (int(T), int(H), int(Hout), int(h0), float(alpha), float(subw), str(device))
    t = _TILING_CACHE.get(key)
    if t is None:
        tt = _Tiling(int(T), int(H), int(Hout), int(h0), _D, _W, _GRID, _MODE,
                     float(alpha), float(subw), _EPS, _MAXD)
        raw = ctypes.string_at(ctypes.byref(tt), ctypes.sizeof(tt))
        t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone().to(device)
        _TILING_CACHE[key] = t
    return t


def _buf(key, shape, dtype, device):
    b = _BUF_CACHE.get(key)
    if b is None or tuple(b.shape) != tuple(shape):
        b = torch.zeros(shape, dtype=dtype, device=device)
        _BUF_CACHE[key] = b
    return b


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



def merge_pre(output, lse, ori_lse, pack):
    """`pack[T,H,W]` ← (output, lse, ori_lse)。

    语义等价于 `_v41_dcp_merge_attention` 归约前三步：
        ``w = nan_to_num(exp(clamp(lse - ori_lse, max=60)))``
        ``scaled = fp32(output) * w``
    """
    lib = _get_lib()
    if lib is None:
        return False
    T, H, _Dd = (int(v) for v in output.shape)
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
    return rc == 0


def merge_post(pack, ori_out, out, h0, alpha, subw):
    """`out[T,Hout,D]` ← (pack[T,H,W] 已 all_reduce, ori_out[T,Hout,D])。"""
    lib = _get_lib()
    if lib is None:
        return False
    T, H, _Ww = (int(v) for v in pack.shape)
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
    return rc == 0


def pack_buffer(T, H, device):
    """常驻 pack 缓冲（**必须先由 host 置零**：kernel 只写 `[0:D+8]`，
    `[D+8:W]` 的 padding 必须保持 0，否则 all_reduce 会把垃圾加进去）。"""
    return _buf(("pack", int(T), int(H), str(device)), (int(T), int(H), _W),
                torch.float32, device)


def out_buffer(T, Hout, device):
    """常驻输出缓冲（bf16，`[T,Hout,D]`）。"""
    return _buf(("out", int(T), int(Hout), str(device)), (int(T), int(Hout), _D),
                torch.bfloat16, device)
