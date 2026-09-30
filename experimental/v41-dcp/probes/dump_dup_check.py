"""查 dump 的 cmp_sparse_indices 是否**行内有重复索引**（top-k 语义应为集合）。"""
import sys

import torch

for path in sys.argv[1:]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    flat = d["cmp_indices"].to(torch.int64).squeeze(1)   # [T, K]
    s = d["scalars"]
    T, K = flat.shape
    valid = flat >= 0
    nvalid = valid.sum(dim=1)
    ndup_rows = 0
    dup_total = 0
    samples = []
    for i in range(T):
        v = flat[i][valid[i]]
        u = torch.unique(v)
        if u.numel() != v.numel():
            ndup_rows += 1
            dup_total += int(v.numel() - u.numel())
            if len(samples) < 5:
                samples.append((i, int(v.numel()), int(u.numel()), v[:8].tolist()))
    print("=== %s layer=%d rank=%d T=%d cseq=%s" %
          (path.split("/")[-1], s["layer_idx"], s["dcp_rank"], T,
           d["seqused_cmp_kv"].tolist() if d.get("seqused_cmp_kv") is not None else None))
    print("  有效项总数=%d 去重后=%d ⇒ 重复项=%d" %
          (int(nvalid.sum()), int(sum(len(set(flat[i][valid[i]].tolist())) for i in range(T))), dup_total))
    print("  含重复的行数=%d/%d | 样例行(行号, 有效数, 去重数, 前8个)=%s" %
          (ndup_rows, T, samples))
