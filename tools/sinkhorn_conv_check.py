#!/usr/bin/env python3
"""HcPre 的 Sinkhorn 迭代次数收敛性检查。

背景：`HcPre` 真实暴露 1.987 ms/步（占步长 8.1%），文档称其受"20 次 Sinkhorn 串行"限制。
但 Sinkhorn 只作用在 **4×4** 矩阵上（hc_mult=4）⇒ 收敛极快。
若 8~10 次已达 fp32 机器精度，则剩余迭代是纯浪费。
"""
import numpy as np

rng = np.random.default_rng(0)


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def sink(comb_raw, iters, eps=1e-6, dtype=np.float64):
    """忠实复刻 hc_split_sinkhorn_torch（hc_mult=4）。"""
    c = comb_raw.astype(dtype)                      # [4,4]
    c = _softmax(c).astype(dtype)                   # softmax(-1)
    c = c + dtype(eps)
    cs = c.sum(-2, keepdims=True)
    c = c / (cs + dtype(eps))
    for _ in range(iters - 1):
        rs = c.sum(-1, keepdims=True)
        c = c / (rs + dtype(eps))
        cs = c.sum(-2, keepdims=True)
        c = c / (cs + dtype(eps))
    return c


print("4x4 Sinkhorn (faithful): max |X_iters - X_20| over N random matrices")
print("%6s %14s %14s %14s" % ("iters", "fp32", "fp64", "bf16-in"))

N = 300
# comb 原始值域：hc_scale[2]*softmax 前的 logits 经线性变换 ⇒ 取 [-12,12] 覆盖极端
mats = rng.uniform(-12, 12, (N, 4, 4))

for t in (4, 6, 8, 10, 12, 16, 20):
    d32 = max(float(np.abs(sink(m, t, dtype=np.float32) - sink(m, 20, dtype=np.float32)).max()) for m in mats)
    d64 = max(float(np.abs(sink(m, t) - sink(m, 20)).max()) for m in mats)
    # bf16 输入：先把 logits 截断到 bf16 再算（复刻真机精度）
    def to_bf16(a):
        import struct

        def r(x):
            b = struct.pack(">f", float(x))
            i = int.from_bytes(b, "big")
            i = (i + 0x8000) & 0xFFFF0000
            return struct.unpack(">f", i.to_bytes(4, "big"))[0]

        return np.vectorize(r)(a)

    dbf = max(float(np.abs(sink(to_bf16(m), t, dtype=np.float32) - sink(to_bf16(m), 20, dtype=np.float32)).max())
              for m in mats)
    print("%6d %14.3e %14.3e %14.3e" % (t, d32, d64, dbf))

print()
print("fp32 eps = %.3e ; 典型 comb 量级 (softmax over 4 rows) ~0.25" % np.finfo(np.float32).eps)
