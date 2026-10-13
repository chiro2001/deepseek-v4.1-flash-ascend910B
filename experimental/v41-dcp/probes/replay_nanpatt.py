"""精确区分：① NaN 位置是否漂移；② 有限值本身是否非确定。
并统计"每次都是 NaN"的行（顽固坏行）与"时好时坏"的行（间歇坏行）。
这决定"重试坏行"是否可行。
"""
import sys

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

DEV = "npu:0"


def main():
    path = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    d = torch.load(path, map_location="cpu", weights_only=False)
    s = d["scalars"]
    H = int(s["num_heads_q"])
    T = d["q"].shape[0]
    ls = []
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
        ls.append(l.to(torch.float32).cpu().reshape(T, H))

    nanmask = torch.stack([~torch.isfinite(x) for x in ls])      # [R,T,H]
    always = nanmask.all(dim=0); never = (~nanmask).any(dim=0)
    inter = nanmask.any(dim=0) & (~nanmask.all(dim=0))
    print("reps=%d | 总是 NaN 的元素=%d | 间歇 NaN=%d | 从不 NaN=%d"
          % (reps, int(always.sum()), int(inter.sum()), int(never.sum())), flush=True)
    print("  NaN 位置是否漂移：始终NaN的行=%d 间歇行=%d 无NaN行=%d"
          % (int(always.any(dim=1).sum()), int(inter.any(dim=1).sum()),
             int((~nanmask.any(dim=0).any(dim=1)).sum())), flush=True)

    fin = torch.stack([torch.isfinite(x) for x in ls])
    both = fin.all(dim=0)
    diffs = []
    for i in range(1, reps):
        m = both
        if int(m.sum()) == 0:
            diffs.append(float("nan")); continue
        diffs.append(float((ls[0][m] - ls[i][m]).abs().max()))
    print("  有限值处 第1次 vs 其余次 max|Δ| = %s" % [round(x, 8) for x in diffs], flush=True)
    # 只在"两次都有限"的位置比较第2次与第3次
    fin2 = fin[1] & fin[2]
    if int(fin2.sum()):
        print("  第2次 vs 第3次（都在有限位置）max|Δ| = %.8g"
              % float((ls[1][fin2] - ls[2][fin2]).abs().max()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
