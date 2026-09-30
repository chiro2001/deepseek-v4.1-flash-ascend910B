"""零重编验证：只调 `cmp_residual_kv` 抬高内核的因果界。

内核（sparse_flash_mla_csa_kernel.h）：
    cmpMaskS2Size = actualCmpS2Size * cmpRatio + residual     // GetCmpMaskS2Size
    cmpMaskRight  = cmpMaskS2Size - actS1Size
    thresHold(t)  = (cmpMaskRight + s1EndIdx + 1) / cmpRatio
    bound         = min(actCmpS2Size, min(512, max(thresHold,0)))
    actCmpS2Size  = min(bound, CountValidCmpSparseLen(bound))

DCP8 下 local cseq=128、actS1Size=904 ⇒ cmpMaskRight = -776 ⇒ 界整体平移。
若把 **residual 设为 T − cseq（=776）**，则：
    cmpMaskS2Size = 128*1 + 776 = 904 ⇒ cmpMaskRight = 0 ⇒ thresHold(t) = t + 1
界被抬回，**而 seqused_cmp_kv 保持本地 128**（元数据/块范围不受影响）。
"""
import sys

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

DEV = "npu:0"


def build_meta(d, cseq, residual, topk=512):
    s = d["scalars"]
    return torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        int(s["num_heads_q"]), 1, int(s["head_dim"]),
        cu_seqlens_q=d["cu_seqlens_q"].to(DEV).to(torch.int32),
        seqused_ori_kv=d["seqused_ori_kv"].to(DEV).to(torch.int32),
        seqused_cmp_kv=torch.tensor([cseq], dtype=torch.int32, device=DEV),
        cmp_residual_kv=torch.tensor([residual], dtype=torch.int32, device=DEV),
        batch_size=1, max_seqlen_q=int(s["max_seqlen_q"]),
        max_seqlen_ori_kv=int(s["max_seqlen_ori_kv"]), max_seqlen_cmp_kv=int(cseq),
        ori_topk=0, cmp_topk=int(topk), cmp_ratio=int(s["cmp_ratio"]),
        ori_mask_mode=int(s["ori_mask_mode"]), cmp_mask_mode=int(s["cmp_mask_mode"]),
        ori_win_left=int(s["ori_win_left"]), ori_win_right=0,
        layout_q="TND", layout_kv="PA_BBND", has_ori_kv=True, has_cmp_kv=True)


def call(d, meta, cseq, residual, reps=8):
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
            cmp_residual_kv=torch.tensor([residual], dtype=torch.int32, device=DEV),
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
    return same, dmax, nan, res[0]


def main():
    path = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    d = torch.load(path, map_location="cpu", weights_only=False)
    local = int(d["seqused_cmp_kv"].max())
    T = int(d["scalars"]["max_seqlen_q"])
    ratio = int(d["scalars"]["cmp_ratio"])
    resid = T - local * ratio
    print("dump=%s T=%d local_cseq=%d ratio=%d ⇒ 拟用 residual=%d" % (path.split("/")[-1], T, local, ratio, resid), flush=True)
    base = None
    for tag, cseq, r in (("A 现状(resid=0)", local, 0),
                         ("B residual=T-cseq", local, resid),
                         ("C residual=T-cseq+8(余量)", local, resid + 8)):
        try:
            meta = build_meta(d, cseq, r)
            same, dmax, nan, first = call(d, meta, cseq, r, reps)
            extra = ""
            if base is not None:
                extra = " | vs A max|dlse|=%.6g" % float((base[1] - first[1]).abs().max())
            print("[%-24s] cseq=%-4d resid=%-5d bit_identical=%-5s max|dlse|=%-12.6g nan=%-6d%s"
                  % (tag, cseq, r, same, dmax, nan, extra), flush=True)
            if base is None:
                base = first
        except Exception as e:  # noqa: BLE001
            print("[%s] 失败：%r" % (tag, e), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
