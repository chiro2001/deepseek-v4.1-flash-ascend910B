"""分别统计 attn_out 与 lse 的非有限元素 —— 区分"输出真的坏了"与"LSE 缓冲的边缘噪声"。"""
import sys

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

DEV = "npu:0"


def run(path, reps=6):
    d = torch.load(path, map_location="cpu", weights_only=False)
    s = d["scalars"]
    H = int(s["num_heads_q"])
    res = []
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
        res.append((o.to(torch.float32).cpu(), l.to(torch.float32).cpu()))
    o0, l0 = res[0]
    onan = int((~torch.isfinite(o0)).sum())
    lnan = int((~torch.isfinite(l0)).sum())
    osame = all(bool(torch.equal(o0, r[0])) for r in res[1:])
    odiff = max(float((o0 - r[0]).abs().max()) for r in res[1:])
    # 输出里 NaN 的行
    T = int(d["q"].shape[0])
    orows = torch.nonzero((~torch.isfinite(o0)).any(dim=1)).flatten().tolist() if o0.ndim == 2 else []
    print("=== %s  T=%d H=%d" % (path.split("/")[-1], T, H))
    print("    attn_out: 非有限=%d  out_bit_identical=%-5s  out max|d|=%.6g"
          % (onan, osame, odiff))
    print("    lse     : 非有限=%d" % lnan)
    print("    out 坏行数=%d 首=%s 末=%s" % (len(orows), orows[0] if orows else None,
                                          orows[-1] if orows else None))


for p in sys.argv[1:]:
    run(p)
