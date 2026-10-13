"""检查 dump 的 metadata 缓冲区是否含"地址样"的值（> 2^30）——
若含，说明 metadata 是**进程相关**的，单卡重放天然不忠实。"""
import sys

import torch

for path in sys.argv[1:]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = d.get("metadata")
    if m is None:
        print("%s: 无 metadata" % path.split("/")[-1]); continue
    v = m.to(torch.int64)
    nz = v != 0
    big = v.abs() > (1 << 30)
    neg = v < 0
    print("=== %s" % path.split("/")[-1])
    print("  shape=%s dtype=%s | 非零=%d/%d | >2^30 的=%d | 负数=%d | max=%d"
          % (tuple(m.shape), m.dtype, int(nz.sum()), v.numel(), int(big.sum()),
             int(neg.sum()), int(v.abs().max())))
    bigidx = torch.nonzero(big).flatten()[:12].tolist()
    print("  大值位置(前12)=%s 值=%s" % (bigidx, [int(v[i]) for i in bigidx]))
    print("  前 32 个 int: %s" % v[:32].tolist())
