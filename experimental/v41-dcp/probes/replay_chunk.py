"""① 精确 NaN 分布（token×head）；② **分块调用**规避尝试。

分块为什么可能有效：算子的批次被切成 tile 处理，非确定可能只发生在某些 tile 形状。
把 T 拆成若干小块（每块 C 行）分别调用在**语义上等价**（注意力是逐 query 独立的），
只是把单次 batch 变小。若小块全部确定 ⇒ 这就是可交付的规避方案。
"""
import os
import sys

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

DEV = "npu:0"


def build_meta(d, max_q, topk):
    s = d["scalars"]
    return torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        int(s["num_heads_q"]), 1, int(s["head_dim"]),
        cu_seqlens_q=torch.tensor([0, max_q], dtype=torch.int32, device=DEV),
        seqused_ori_kv=torch.tensor([max_q], dtype=torch.int32, device=DEV),
        seqused_cmp_kv=d["seqused_cmp_kv"].to(DEV).to(torch.int32),
        cmp_residual_kv=(d["cmp_residual_kv"].to(DEV).to(torch.int32)
                         if d.get("cmp_residual_kv") is not None else None),
        batch_size=1, max_seqlen_q=int(max_q), max_seqlen_ori_kv=int(max_q),
        max_seqlen_cmp_kv=int(d["seqused_cmp_kv"].max()),
        ori_topk=0, cmp_topk=int(topk), cmp_ratio=int(s["cmp_ratio"]),
        ori_mask_mode=int(s["ori_mask_mode"]), cmp_mask_mode=int(s["cmp_mask_mode"]),
        ori_win_left=int(s["ori_win_left"]), ori_win_right=0,
        layout_q="TND", layout_kv="PA_BBND", has_ori_kv=True, has_cmp_kv=True)


def raw(d, q, idx, meta, s2_ori, s2_cmp, K=None):
    s = d["scalars"]
    o, l = torch.ops._C_ascend.npu_sparse_flash_mla(
        q, ori_kv=d["ori_pages"].to(DEV), cmp_kv=d["cmp_pages"].to(DEV),
        cmp_sparse_indices=idx,
        ori_block_table=d["ori_block_table"].to(DEV).to(torch.int32),
        cmp_block_table=d["cmp_block_table"].to(DEV).to(torch.int32),
        cu_seqlens_q=torch.tensor([0, q.shape[0]], dtype=torch.int32, device=DEV),
        seqused_ori_kv=torch.tensor([s2_ori], dtype=torch.int32, device=DEV),
        seqused_cmp_kv=torch.tensor([s2_cmp], dtype=torch.int32, device=DEV),
        cmp_residual_kv=(d["cmp_residual_kv"].to(DEV).to(torch.int32)
                         if d.get("cmp_residual_kv") is not None else None),
        sinks=d["sinks"].to(DEV), metadata=meta,
        softmax_scale=float(s["softmax_scale"]), cmp_ratio=int(s["cmp_ratio"]),
        ori_mask_mode=int(s["ori_mask_mode"]), cmp_mask_mode=int(s["cmp_mask_mode"]),
        ori_win_left=int(s["ori_win_left"]), ori_win_right=int(s["ori_win_right"]),
        layout_q="TND", layout_kv="PA_BBND",
        topk_value_mode=int(s["topk_value_mode"]), return_softmax_lse=True)
    torch.npu.synchronize()
    return o.to(torch.float32).cpu(), l.to(torch.float32).cpu()


def main():
    d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    q = d["q"].to(DEV)
    idx = d["cmp_indices"].to(DEV).to(torch.int32)
    T = q.shape[0]
    K = idx.shape[-1]
    cseq = int(d["seqused_cmp_kv"].max())
    H = int(d["scalars"]["num_heads_q"])
    meta_full = build_meta(d, T, K)

    # ① 精确 NaN 分布
    print("=== ① NaN 分布（T=%d, H=%d, cseq=%d）===" % (T, H, cseq), flush=True)
    for rep in range(2):
        o, l = raw(d, q, idx, meta_full, T, cseq)
        l2 = l.reshape(T, H)
        badtok = (~torch.isfinite(l2)).any(dim=1)
        toks = torch.nonzero(badtok).flatten().tolist()
        print("  rep%d: 坏 token 数=%d 首个=%s 末个=%s | 每 token 坏 head 数 取值=%s"
              % (rep + 1, len(toks), toks[0] if toks else None, toks[-1] if toks else None,
                 sorted(set((~torch.isfinite(l2)).sum(dim=1)[badtok].tolist()))[:6]), flush=True)

    # ② 分块
    print("=== ② 分块调用（语义等价：注意力逐 query 独立）===", flush=True)
    for C in [int(x) for x in os.environ.get("PROBE_CHUNKS", "128,256,512").split(",")]:
        nch = (T + C - 1) // C
        ok = True
        worst = 0.0
        tot_nan = 0
        for ci in range(nch):
            a, b = ci * C, min(T, (ci + 1) * C)
            qc = q[a:b].contiguous()
            ic = idx[a:b].contiguous()
            mc = build_meta(d, b - a, K)
            # 关键：保持"query 行 i ↔ 全局位置 a+i"对齐 ⇒ seqused_ori = 全局末位 b
            r1 = raw(d, qc, ic, mc, b, cseq)
            r2 = raw(d, qc, ic, mc, b, cseq)
            if not bool(torch.equal(r1[1], r2[1])):
                ok = False
            tot_nan += int((~torch.isfinite(r1[1])).sum())
            worst = max(worst, float((r1[1] - r2[1]).abs().max()))
        print("  块大小 C=%-4d 块数=%-3d 每块两次调用逐位相同=%-5s 最大差=%.6g NaN 总数=%d"
              % (C, nch, ok, worst, tot_nan), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
