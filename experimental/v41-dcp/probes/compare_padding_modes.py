"""对比三种索引填充方式的**输出**：精确均匀(floor) vs 填满(ceil) vs 原样。

关键问题：填满槽位（0 个 -1）能消除算子非确定，但会因截断导致某个键多出现一次，
引入轻微语义偏差。本脚本量化该偏差。
"""
import os
import sys

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

DEV = "npu:0"


def build_meta(d, topk):
    return torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        int(d["scalars"]["num_heads_q"]), 1, int(d["scalars"]["head_dim"]),
        cu_seqlens_q=d["cu_seqlens_q"].to(DEV).to(torch.int32),
        seqused_ori_kv=d["seqused_ori_kv"].to(DEV).to(torch.int32),
        seqused_cmp_kv=d["seqused_cmp_kv"].to(DEV).to(torch.int32),
        cmp_residual_kv=(d["cmp_residual_kv"].to(DEV).to(torch.int32)
                         if d.get("cmp_residual_kv") is not None else None),
        batch_size=1, max_seqlen_q=int(d["scalars"]["max_seqlen_q"]),
        max_seqlen_ori_kv=int(d["scalars"]["max_seqlen_ori_kv"]),
        max_seqlen_cmp_kv=int(d["seqused_cmp_kv"].max()),
        ori_topk=0, cmp_topk=int(topk), cmp_ratio=int(d["scalars"]["cmp_ratio"]),
        ori_mask_mode=int(d["scalars"]["ori_mask_mode"]),
        cmp_mask_mode=int(d["scalars"]["cmp_mask_mode"]),
        ori_win_left=int(d["scalars"]["ori_win_left"]), ori_win_right=0,
        layout_q="TND", layout_kv="PA_BBND", has_ori_kv=True, has_cmp_kv=True)


def run(d, idx, meta, reps=1):
    s = d["scalars"]
    outs = []
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
        outs.append((o.to(torch.float32).cpu(), l.to(torch.float32).cpu()))
    return outs


def mk(idx_flat, K, mode):
    rows = []
    for t in range(idx_flat.shape[0]):
        v = idx_flat[t][idx_flat[t] >= 0]
        if v.numel() == 0:
            rows.append(torch.full((K,), -1, dtype=torch.int64)); continue
        if mode == "floor":
            k = max(1, K // v.numel())
            r = v.repeat(k)
        else:
            k = max(1, (K + v.numel() - 1) // v.numel())
            r = v.repeat(k)[:K]
        rows.append(r)
    out = torch.full((idx_flat.shape[0], 1, K), -1, dtype=torch.int64)
    for t, r in enumerate(rows):
        out[t, 0, : r.numel()] = r
    return out.to(torch.int32).to(DEV)


def main():
    d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    flat = d["cmp_indices"].to(torch.int64).squeeze(1)
    for K in (512, 1024):
        meta = build_meta(d, K)
        res = {}
        for mode in ("floor", "ceil"):
            idx = mk(flat, K, mode)
            nneg = int((idx < 0).sum())
            outs = run(d, idx, meta, reps=4 if mode == "floor" else 4)
            same = all(bool(torch.equal(outs[0][1], x[1])) for x in outs[1:])
            res[mode] = (outs, nneg, same)
        o_f, n_f, s_f = res["floor"]
        o_c, n_c, s_c = res["ceil"]
        dl = float((o_f[0][1] - o_c[0][1]).abs().max())
        do = float((o_f[0][0] - o_c[0][0]).abs().max())
        rel = do / max(1e-9, float(o_f[0][0].abs().max()))
        print("K=%d | floor: -1=%d bit_identical=%s | ceil: -1=%d bit_identical=%s | "
              "floor vs ceil: max|dlse|=%.6g max|dout|=%.6g 相对=%.3e"
              % (K, n_f, s_f, n_c, s_c, dl, do, rel), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
