"""看 dump 里的索引布局：-1 是否紧凑、值域、各行的有效个数分布。"""
import sys

import torch

for path in sys.argv[1:]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    idx = d["cmp_indices"].to(torch.int64)      # [T,1,K]
    s = d["scalars"]
    flat = idx.squeeze(1)                        # [T,K]
    valid = flat >= 0
    nvalid = valid.sum(dim=1)
    # 是否"紧凑"：每行 -1 都出现在有效项之后？
    first_neg = torch.where(~valid, torch.arange(flat.shape[1]).expand_as(valid), flat.shape[1])
    compact = bool((first_neg.min(dim=1).values == nvalid).all())
    uniq_per_row = [int(torch.unique(flat[i][valid[i]]).numel()) for i in range(min(6, flat.shape[0]))]
    print("=== %s layer=%d rank=%d T=%d cseq=%s K=%d"
          % (path.split("/")[-1], s["layer_idx"], s["dcp_rank"],
             flat.shape[0], (d["seqused_cmp_kv"].tolist() if d.get("seqused_cmp_kv") is not None else None),
             flat.shape[1]))
    print("  值域: min=%d max=%d | 每行有效数 min=%d max=%d mean=%.2f | -1 总数=%d"
          % (int(flat[valid].min()), int(flat[valid].max()),
             int(nvalid.min()), int(nvalid.max()), float(nvalid.float().mean()),
             int((~valid).sum())))
    print("  -1 紧跟在有效项之后(紧凑)=%s | 前 6 行去重后个数=%s"
          % (compact, uniq_per_row))
    print("  ori_bt 非零列=%d / %d | cmp_bt 非零列=%d / %d"
          % (int((d["ori_block_table"] > 0).sum()), int(d["ori_block_table"].numel()),
             int((d["cmp_block_table"] > 0).sum()), int(d["cmp_block_table"].numel())))
