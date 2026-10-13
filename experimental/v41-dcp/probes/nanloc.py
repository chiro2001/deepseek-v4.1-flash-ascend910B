"""报 NaN 落在哪些 query 行（区分"开头的边界行"与"结尾的关键行"）。"""
import sys

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

DEV = "npu:0"


def run(path, reps=3):
    d = torch.load(path, map_location="cpu", weights_only=False)
    s = d["scalars"]
    H = int(s["num_heads_q"])
    outs = []
    for _ in range(reps):
        o, l = torch.ops._C_ascend.npu_sparse_flash_mla(
            d["q"].to(DEV), ori_kv=d["ori_pages"].to(DEV), cmp_kv=d["cmp_pages"].to(DEV),
            cmp_sparse_indices=d["cmp_indices"].to(DEV).to(torch.int32),
            ori_block_table=d["ori_block_table"].to(DEV).to(torch.int32),
            cmp_block_table=d["cmp_block_table"].to(DEV).to(torch.int32),
            cu_seqlens_q=d["cu_seqlens_q"].to(DEV).to(torch.int32),
            seqused_ori_kv=d["seqused_ori_kv"].to(DEV).to(torch.int32),
            seqused_cmp_kv=d["seqused_cmp_kv"].to(DEV).to(torch.int32),
            cmp_residual_kv=(d["cmp_residual_kv"].to(DEV).to(torch.int32)
                             if d.get("cmp_residual_kv") is not None else None),
            sinks=d["sinks"].to(DEV), metadata=d["metadata"].to(DEV).to(torch.int32),
            softmax_scale=float(s["softmax_scale"]), cmp_ratio=int(s["cmp_ratio"]),
            ori_mask_mode=int(s["ori_mask_mode"]), cmp_mask_mode=int(s["cmp_mask_mode"]),
            ori_win_left=int(s["ori_win_left"]), ori_win_right=int(s["ori_win_right"]),
            layout_q="TND", layout_kv="PA_BBND",
            topk_value_mode=int(s["topk_value_mode"]), return_softmax_lse=True)
        torch.npu.synchronize()
        outs.append(l.to(torch.float32).cpu())
    l = outs[-1]
    T = int(l.shape[1]) if l.ndim == 3 else int(l.shape[0])
    l2 = l.reshape(-1, T, H).squeeze(0)
    bad = ~torch.isfinite(l2)
    rows = torch.nonzero(bad.any(dim=1)).flatten().tolist()
    print("=== %s  T=%d H=%d  NaN 元素=%d" % (path.split("/")[-1], T, H, int(bad.sum())))
    print("    坏行数=%d  首行=%s  末行=%s  是否集中在前段=%s  是否集中在后段=%s"
          % (len(rows), rows[0] if rows else None, rows[-1] if rows else None,
             bool(rows) and rows[-1] < T // 4, bool(rows) and rows[0] > T * 3 // 4))
    if rows:
        print("    坏行列表（前 20）=%s" % rows[:20])


for p in sys.argv[1:]:
    run(p)
