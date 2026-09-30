"""验证假设：把 seqused_cmp_kv 按**全局长度**传入，即可解除内核的全局坐标界约束。

依据（源码）：
    cmpMaskS2Size = actualCmpS2Size * cmpRatio + residual
    cmpMaskRight  = cmpMaskS2Size - actS1Size
    thresHold(t)  = (cmpMaskRight + s1EndIdx + 1) / cmpRatio
    bound         = min(actCmpS2Size, min(512, max(thresHold,0)))
    actCmpS2Size  = min(bound, CountValidCmpSparseLen(bound))

DCP8 下若传**本地长度** 128：cmpMaskRight = 128 - 904 = -776
  ⇒ thresHold(t) = t - 775 ⇒ 大量行的界小于真实有效项数 ⇒ 索引被丢弃 ⇒ 洞 ⇒ NaN。
若改传**全局长度** 904：cmpMaskRight = 0 ⇒ thresHold(t) = t + 1
  ⇒ bound = min(512, t+1) 恒 >= 本行有效项数（≤128）
  ⇒ CountValidCmpSparseLen(bound) = 本行有效项数（因为我们的索引已紧致压缩）
  ⇒ actCmpS2Size 恢复正确；且因果 mask 变宽松（索引集由我们给定，不影响正确性）。

本脚本在**单卡**上对比三种传参，判据：ALL_bit_identical=True 且 max|dlse|=0。
"""
import sys

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

DEV = "npu:0"


def build_meta(d, cseq, topk=512):
    s = d["scalars"]
    return torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        int(s["num_heads_q"]), 1, int(s["head_dim"]),
        cu_seqlens_q=d["cu_seqlens_q"].to(DEV).to(torch.int32),
        seqused_ori_kv=d["seqused_ori_kv"].to(DEV).to(torch.int32),
        seqused_cmp_kv=torch.tensor([cseq], dtype=torch.int32, device=DEV),
        cmp_residual_kv=(d["cmp_residual_kv"].to(DEV).to(torch.int32)
                         if d.get("cmp_residual_kv") is not None else None),
        batch_size=1,
        max_seqlen_q=int(s["max_seqlen_q"]),
        max_seqlen_ori_kv=int(s["max_seqlen_ori_kv"]),
        max_seqlen_cmp_kv=int(cseq),
        ori_topk=0, cmp_topk=int(topk), cmp_ratio=int(s["cmp_ratio"]),
        ori_mask_mode=int(s["ori_mask_mode"]), cmp_mask_mode=int(s["cmp_mask_mode"]),
        ori_win_left=int(s["ori_win_left"]), ori_win_right=0,
        layout_q="TND", layout_kv="PA_BBND", has_ori_kv=True, has_cmp_kv=True)


def call(d, meta, cseq, reps=8):
    s = d["scalars"]
    res = []
    for _ in range(reps):
        o, l = torch.ops._C_ascend.npu_sparse_flash_mla(
            d["q"].to(DEV), ori_kv=d["ori_pages"].to(DEV), cmp_kv=d["cmp_pages"].to(DEV),
            cmp_sparse_indices=d["cmp_indices"].to(DEV).to(torch.int32),
            ori_block_table=d["ori_block_table"].to(DEV).to(torch.int32),
            cmp_block_table=d["cmp_block_table"].to(DEV).to(torch.int32),
            cu_seqlens_q=d["cu_seqlens_q"].to(DEV).to(torch.int32),
            seqused_ori_kv=d["seqused_ori_kv"].to(DEV).to(torch.int32),
            seqused_cmp_kv=torch.tensor([cseq], dtype=torch.int32, device=DEV),
            cmp_residual_kv=(d["cmp_residual_kv"].to(DEV).to(torch.int32)
                             if d.get("cmp_residual_kv") is not None else None),
            sinks=d["sinks"].to(DEV), metadata=meta,
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
    return same, dmax, nan


def main():
    path = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    d = torch.load(path, map_location="cpu", weights_only=False)
    local = int(d["seqused_cmp_kv"].max())
    T = int(d["scalars"]["max_seqlen_q"])
    print("dump=%s T=%d 本地cseq=%d 全局(compressed)=%d"
          % (path.split("/")[-1], T, local, T // int(d["scalars"]["cmp_ratio"])), flush=True)
    for tag, cseq in (("A 本地长度(现状)", local),
                      ("B 全局长度", T // int(d["scalars"]["cmp_ratio"]))):
        try:
            meta = build_meta(d, cseq)
            same, dmax, nan = call(d, meta, cseq, reps)
            print("[%s] cseq=%-6d bit_identical=%-5s max|dlse|=%-12.6g lse_nan=%d"
                  % (tag, cseq, same, dmax, nan), flush=True)
        except Exception as e:  # noqa: BLE001
            print("[%s] 失败：%r" % (tag, e), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
