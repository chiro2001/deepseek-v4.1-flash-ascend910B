"""零重编验证：ori（SWA）block table 只填第 0 列 是否就是 NaN 的来源。

背景：dump 里 ori_pages 只有 1 页（128 行，SWA 是环形缓冲），
`seqused_ori_kv=904`，而 `ori_block_table` 只有第 0 列非零（其余为 0 = null 块）。
若内核按 `pos / paOriBlockSize`（128）查表，则 pos>=128 会查到 null 块。

对照臂：把 ori block table 的**所有列**都指向那一页（环形语义：slot = pos % 128）。
判据：若 NaN 归零 ⇒ 问题在我们的 block table / remap，而不是算子。
"""
import sys

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

DEV = "npu:0"


def run(d, ori_bt, reps=6, tag=""):
    s = d["scalars"]
    res = []
    for _ in range(reps):
        o, l = torch.ops._C_ascend.npu_sparse_flash_mla(
            d["q"].to(DEV), ori_kv=d["ori_pages"].to(DEV), cmp_kv=d["cmp_pages"].to(DEV),
            cmp_sparse_indices=d["cmp_indices"].to(DEV).to(torch.int32),
            ori_block_table=ori_bt,
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
        res.append((o.to(torch.float32).cpu(), l.to(torch.float32).cpu()))
    same = all(bool(torch.equal(res[0][1], r[1])) for r in res[1:])
    dmax = max(float((res[0][1] - r[1]).abs().max()) for r in res[1:])
    nan = int((~torch.isfinite(res[0][1])).sum())
    print("  [%-22s] bit_identical=%-5s max|dlse|=%-12.6g lse_nan=%d" % (tag, same, dmax, nan), flush=True)
    return same, dmax, nan


for path in sys.argv[1:]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    obt = d["ori_block_table"].to(DEV).to(torch.int32)
    nz = int((obt > 0).sum())
    print("=== %s  ori_bt 非零列数=%d  前 10 列=%s"
          % (path.split("/")[-1], nz, obt[0, :10].tolist()), flush=True)
    run(d, obt, tag="asis (原样)")
    # 环形语义：所有列都指向第 1 页（= 唯一那一页）
    filled = torch.ones_like(obt)
    run(d, filled, tag="所有列->第1页(环形)")
    # 只把前 N 列填满（保留原第 0 列的页号）
    p0 = int(obt[0, 0])
    filled2 = obt.clone(); filled2[0, :] = p0
    run(d, filled2, tag="所有列->原第0页号")
