"""对比不同"索引填充模式"对算子确定性的影响（真实 dump）。

动机：合成探针显示"有效项 ≥256 即确定"，但生产的 `ceil` 填充（512 项）仍非确定。
差别在于填充后的**排列**：
  block  = v.repeat(k)          —— 连续重复 [a,a,a,b,b,b,...]（生产 uniqpad=ceil 用的）
  cyclic = v 循环 j % n          —— 均匀散布 [a,b,c,a,b,c,...]
  asis   = 原样（大量 -1）
若 cyclic 变确定 ⇒ "避免相邻重复"是触发条件，可作为规避。
"""
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
        layout_q="TND", layout_kv="PA_BBND", has_ori_kv=True, has_cmp_kv=True)


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
    return same, dmax, nan


def fill(flat, K, mode):
    rows = []
    for t in range(flat.shape[0]):
        v = flat[t][flat[t] >= 0]
        if v.numel() == 0:
            rows.append(torch.full((K,), -1, dtype=torch.int64)); continue
        n = v.numel()
        if mode == "asis":
            r = v
        elif mode == "block":
            k = max(1, (K + n - 1) // n)
            r = v.repeat(k)[:K]
        else:  # cyclic
            j = torch.arange(K)
            r = v[j % n]
        rows.append(r)
    out = torch.full((flat.shape[0], 1, K), -1, dtype=torch.int64)
    for t, r in enumerate(rows):
        out[t, 0, : r.numel()] = r
    return out.to(torch.int32).to(DEV)


def main():
    d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    flat = d["cmp_indices"].to(torch.int64).squeeze(1)
    K = flat.shape[-1]
    meta = build_meta(d, d["q"].shape[0], K)
    for mode in ("asis", "block", "cyclic"):
        idx = fill(flat, K, mode)
        neg = int((idx < 0).sum())
        same, dmax, nan = call(d, meta, idx, reps)
        # 检查是否还有相邻重复
        f = idx.squeeze(1)
        adj = int(((f[:, 1:] == f[:, :-1]) & (f[:, 1:] >= 0)).sum())
        print("[%-7s] -1=%-6d 相邻重复对=%-7d bit_identical=%-5s max|dlse|=%-12.6g nan=%d"
              % (mode, neg, adj, same, dmax, nan), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
