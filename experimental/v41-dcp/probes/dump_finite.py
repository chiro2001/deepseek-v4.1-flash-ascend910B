"""检查 dump 里**每个张量**是否含非有限值 —— 若 KV 本身有 NaN，
那么"算子非确定"的结论就不成立（NaN 是输入带进来的）。"""
import sys

import torch

for path in sys.argv[1:]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    print("=== %s" % path.split("/")[-1])
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            f = v.to(torch.float32)
            nf = int((~torch.isfinite(f)).sum())
            print("  %-20s shape=%-24s nonfinite=%-8d absmax=%.6g"
                  % (k, tuple(v.shape), nf, float(f.abs().max()) if f.numel() else 0.0))
        elif isinstance(v, dict):
            pass
    for k in ("sinks",):
        if d.get(k) is not None:
            f = d[k].to(torch.float32)
            print("  %-20s shape=%-24s nonfinite=%-8d absmax=%.6g"
                  % (k, tuple(d[k].shape), int((~torch.isfinite(f)).sum()),
                     float(f.abs().max())))
    print("  scalars=%s" % d["scalars"])
