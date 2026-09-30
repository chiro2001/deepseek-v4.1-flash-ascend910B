"""测试 **dense 路径**（不传 cmp_sparse_indices）是否确定。

动机：所有历史实验都走稀疏索引路径（`cmp_sparse_indices` 非空），
而从没测过 dense（`cmp_sparse_indices=None`）。若 dense 确定，那么
"把 ratio=1 层的 KV 复制到每个 rank + dense 因果"就是一条**正确的规避路线**。
"""
import sys

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

DEV = "npu:0"


def build_meta(d, max_q, topk, has_cmp=True):
    s = d["scalars"]
    return torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        int(s["num_heads_q"]), 1, int(s["head_dim"]),
        cu_seqlens_q=d["cu_seqlens_q"].to(DEV).to(torch.int32),
        seqused_ori_kv=d["seqused_ori_kv"].to(DEV).to(torch.int32),
        seqused_cmp_kv=d["seqused_cmp_kv"].to(DEV).to(torch.int32),
        cmp_residual_kv=(d["cmp_residual_kv"].to(DEV).to(torch.int32)
                         if d.get("cmp_residual_kv") is not None else None),
        batch_size=1, max_seqlen_q=int(max_q), max_seqlen_ori_kv=int(max_q),
        max_seqlen_cmp_kv=int(d["seqused_cmp_kv"].max()),
        ori_topk=0, cmp_topk=int(topk), cmp_ratio=int(s["cmp_ratio"]),
        ori_mask_mode=int(s["ori_mask_mode"]), cmp_mask_mode=int(s["cmp_mask_mode"]),
        ori_win_left=int(s["ori_win_left"]), ori_win_right=0,
        layout_q="TND", layout_kv="PA_BBND",
        has_ori_kv=True, has_cmp_kv=bool(has_cmp))


def call(d, meta, idx, reps=6):
    s = d["scalars"]
    res = []
    for _ in range(reps):
        o, l = torch.ops._C_ascend.npu_sparse_flash_mla(
            d["q"].to(DEV), ori_kv=d["ori_pages"].to(DEV), cmp_kv=d["cmp_pages"].to(DEV),
            cmp_sparse_indices=idx,
            ori_block_table=d["ori_block_table"].to(DEV).to(torch.int32),
            cmp_block_table=d["cmp_block_table"].to(DEV).to(torch.int32),
            cu_seqlens_q=d["cu_seqlens_q"].to(DEV).to(torch.int32),
            seqused_ori_kv=d["seqused_ori_kv"].to(DEV).to(torch.int32),
            seqused_cmp_kv=d["seqused_cmp_kv"].to(DEV).to(torch.int32),
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
    return same, dmax, nan, res[0]


def main():
    d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    print("T=%d cseq=%s K=%d" % (d["q"].shape[0],
                                 d["seqused_cmp_kv"].tolist(), d["cmp_indices"].shape[-1]), flush=True)

    # 臂 A：稀疏（基线）
    meta_s = build_meta(d, d["q"].shape[0], 512)
    same, dmax, nan, base = call(d, meta_s, d["cmp_indices"].to(DEV).to(torch.int32), reps)
    print("[A 稀疏 512]        bit_identical=%-5s max|dlse|=%-12.6g nan=%d" % (same, dmax, nan), flush=True)

    # 臂 B：dense（不传索引）
    try:
        same, dmax, nan, dense = call(d, meta_s, None, reps)
        print("[B dense(无索引)]   bit_identical=%-5s max|dlse|=%-12.6g nan=%d" % (same, dmax, nan), flush=True)
        dmax2 = float((base[1] - dense[1]).abs().max())
        print("    A vs B max|dlse|=%.6g   （若 B 确定且 A/B 接近 ⇒ dense 是可用替代）" % dmax2, flush=True)
    except Exception as e:  # noqa: BLE001
        print("[B dense(无索引)] 调用失败：%r" % (e,), flush=True)

    # 臂 C：dense + topk=1024
    try:
        meta_l = build_meta(d, d["q"].shape[0], 1024)
        same, dmax, nan, _ = call(d, meta_l, None, reps)
        print("[C dense topk=1024] bit_identical=%-5s max|dlse|=%-12.6g nan=%d" % (same, dmax, nan), flush=True)
    except Exception as e:  # noqa: BLE001
        print("[C dense topk=1024] 失败：%r" % (e,), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
