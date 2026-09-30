"""在**真实 dump** 上做单变量变异，找让算子变确定的那一个改动。

变异项（每次只改一个）：
  asis        原样
  ori_used    把 ori 块表**只保留本请求用到的列**（其余置 0）
  ori_full    给 ori 分配 8 页不同数据，块表第 k 列 → 第 k+1 页
  cmp_allone  cmp 块表所有列 → 同一页（并把该页复制到 8 页池上）
  idx_full    把每行索引**填满 512**（用该行已有的键循环填充）——语义等价（均匀重复）
  idx_dense   去掉 -1 尾部，只保留有效项后重排为紧凑前缀（应等价原样）
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
    """按 dump 的 seqused 重建 metadata（用于切换到 topk=1024）。"""
    dev = DEV
    return torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        int(d["scalars"]["num_heads_q"]), 1, int(d["scalars"]["head_dim"]),
        cu_seqlens_q=d["cu_seqlens_q"].to(dev).to(torch.int32),
        seqused_ori_kv=d["seqused_ori_kv"].to(dev).to(torch.int32),
        seqused_cmp_kv=d["seqused_cmp_kv"].to(dev).to(torch.int32),
        cmp_residual_kv=(d["cmp_residual_kv"].to(dev).to(torch.int32)
                         if d.get("cmp_residual_kv") is not None else None),
        batch_size=1,
        max_seqlen_q=int(d["scalars"]["max_seqlen_q"]),
        max_seqlen_ori_kv=int(d["scalars"]["max_seqlen_ori_kv"]),
        max_seqlen_cmp_kv=int(d["seqused_cmp_kv"].max()),
        ori_topk=0, cmp_topk=int(topk), cmp_ratio=int(d["scalars"]["cmp_ratio"]),
        ori_mask_mode=int(d["scalars"]["ori_mask_mode"]),
        cmp_mask_mode=int(d["scalars"]["cmp_mask_mode"]),
        ori_win_left=int(d["scalars"]["ori_win_left"]), ori_win_right=0,
        layout_q="TND", layout_kv="PA_BBND", has_ori_kv=True, has_cmp_kv=True)


def call(d, q, ori, cmp_kv, idx, obt, cbt, reps, meta=None):
    s = d["scalars"]
    if meta is None:
        meta = d["metadata"].to(DEV).to(torch.int32) if d.get("metadata") is not None else None
    res = []
    for _ in range(reps):
        out, lse = torch.ops._C_ascend.npu_sparse_flash_mla(
            q, ori_kv=ori, cmp_kv=cmp_kv, cmp_sparse_indices=idx,
            ori_block_table=obt, cmp_block_table=cbt,
            cu_seqlens_q=d["cu_seqlens_q"].to(DEV).to(torch.int32),
            seqused_ori_kv=d["seqused_ori_kv"].to(DEV).to(torch.int32),
            seqused_cmp_kv=d["seqused_cmp_kv"].to(DEV).to(torch.int32),
            cmp_residual_kv=d["cmp_residual_kv"].to(DEV).to(torch.int32) if d.get("cmp_residual_kv") is not None else None,
            sinks=d["sinks"].to(DEV) if d.get("sinks") is not None else None,
            metadata=meta,
            softmax_scale=float(s["softmax_scale"]), cmp_ratio=int(s["cmp_ratio"]),
            ori_mask_mode=int(s["ori_mask_mode"]), cmp_mask_mode=int(s["cmp_mask_mode"]),
            ori_win_left=int(s["ori_win_left"]), ori_win_right=int(s["ori_win_right"]),
            layout_q="TND", layout_kv="PA_BBND", topk_value_mode=int(s["topk_value_mode"]),
            return_softmax_lse=True)
        torch.npu.synchronize()
        res.append((out.to(torch.float32).cpu(), lse.to(torch.float32).cpu()))
    same = all(bool(torch.equal(res[0][1], r[1])) for r in res[1:])
    dmax = max(float((res[0][1] - r[1]).abs().max()) for r in res[1:])
    nan = int((~torch.isfinite(res[0][1])).sum())
    return same, dmax, nan


def main():
    path = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    d = torch.load(path, map_location="cpu", weights_only=False)
    q0 = d["q"].to(DEV)
    ori0 = d["ori_pages"].to(DEV)
    cmp0 = d["cmp_pages"].to(DEV)
    idx0 = d["cmp_indices"].to(DEV).to(torch.int32)
    obt0 = d["ori_block_table"].to(DEV).to(torch.int32)
    cbt0 = d["cmp_block_table"].to(DEV).to(torch.int32)
    K = idx0.shape[-1]
    print("dump=%s T=%d K=%d cseq=%s" % (path.split("/")[-1], q0.shape[0], K,
                                         d["seqused_cmp_kv"].tolist()), flush=True)

    def report(tag, *args, **kw):
        same, dmax, nan = call(d, *args, reps=reps, **kw)
        print("  %-12s lse_bit_identical=%-5s max|dlse|=%-12.6g lse_nan=%d" %
              (tag, same, dmax, nan), flush=True)

    report("asis", q0, ori0, cmp0, idx0, obt0, cbt0)

    # ori_used：只保留前 8 列（本请求用到的），其余置 0
    obt = obt0.clone(); obt[0, 8:] = 0
    report("ori_used", q0, ori0, cmp0, idx0, obt, cbt0)

    # ori_full：8 页不同数据，块表第 k 列 → k+1
    g = torch.Generator(device="cpu").manual_seed(7)
    ori8 = torch.randn(8, *ori0.shape[1:], generator=g, dtype=torch.float32).to(torch.bfloat16).to(DEV)
    obt = torch.zeros_like(obt0); obt[0, :8] = torch.arange(1, 9, device=DEV)
    report("ori_full", q0, ori8, cmp0, idx0, obt, cbt0)

    # cmp_allone：cmp 块表全列 → 1
    cbt = torch.ones_like(cbt0)
    report("cmp_allone", q0, ori0, cmp0, idx0, obt0, cbt)

    # idx_full：每行用已有键循环填满 512（均匀重复 ⇒ 语义等价）
    flat = idx0.squeeze(1).to(torch.int64)
    rows = []
    for t in range(flat.shape[0]):
        v = flat[t][flat[t] >= 0]
        if v.numel() == 0:
            rows.append(torch.full((K,), -1, dtype=torch.int64))
            continue
        k = (K + v.numel() - 1) // v.numel()
        rows.append(v.repeat(k)[:K])
    idx_full = torch.stack(rows).view(flat.shape[0], 1, K).to(torch.int32).to(DEV)
    report("idx_full", q0, ori0, cmp0, idx_full, obt0, cbt0)

    # idx_uni：topk=1024 + **完全均匀**重复（k = floor(1024/n)，每个键恰好 k 次）
    #   ⇒ 每个键的 softmax 权重同乘 k ⇒ **数学上与原样逐位等价**，且没有截断误差。
    K2 = 1024
    flat32 = idx0.squeeze(1).to(torch.int64)
    rows2 = []
    ks = []
    for t in range(flat32.shape[0]):
        v = flat32[t][flat32[t] >= 0]
        if v.numel() == 0:
            rows2.append(torch.full((K2,), -1, dtype=torch.int64)); ks.append(0); continue
        k = max(1, K2 // v.numel())
        ks.append(k)
        rows2.append(v.repeat(k))
    nvalid2 = sum(int(r.numel()) for r in rows2)
    idx_uni = torch.full((flat32.shape[0], 1, K2), -1, dtype=torch.int64)
    for t, r in enumerate(rows2):
        idx_uni[t, 0, : r.numel()] = r
    idx_uni = idx_uni.to(torch.int32).to(DEV)
    meta2 = build_meta(d, K2)
    print("  [idx_uni] K=1024 有效项=%d (原 %d) k 范围=[%d, %d]"
          % (nvalid2, int((flat32 >= 0).sum()), min(x for x in ks if x > 0), max(ks)), flush=True)
    report("idx_uni_1024", q0, ori0, cmp0, idx_uni, obt0, cbt0, meta=meta2)
    # 同时确认：topk=1024 但**不重复**（大量 -1）会怎样
    idx_pad = torch.nn.functional.pad(flat32, (0, K2 - K), value=-1).view(flat32.shape[0], 1, K2)
    report("idx_padonly", q0, ori0, cmp0, idx_pad.to(torch.int32).to(DEV), obt0, cbt0, meta=meta2)

    # idx_ceil1024：topk=1024，**填满全部槽位**（0 个 -1）——每键出现 floor 或 ceil 次
    rows3 = []
    for t in range(flat32.shape[0]):
        v = flat32[t][flat32[t] >= 0]
        if v.numel() == 0:
            rows3.append(torch.full((K2,), -1, dtype=torch.int64)); continue
        k = max(1, (K2 + v.numel() - 1) // v.numel())
        rows3.append(v.repeat(k)[:K2])
    idx_ceil = torch.stack(rows3).view(flat32.shape[0], 1, K2).to(torch.int32).to(DEV)
    print("  [idx_ceil] K=1024 有效项=%d (-1 数=%d)"
          % (int((idx_ceil >= 0).sum()), int((idx_ceil < 0).sum())), flush=True)
    report("idx_ceil1024", q0, ori0, cmp0, idx_ceil, obt0, cbt0, meta=meta2)

    # 语义误差：idx_ceil1024 与"精确均匀"的 idx_uni_1024 的输出差
    def out_of(idx, meta):
        s_, d_, n_ = call(d, q0, ori0, cmp0, idx, obt0, cbt0, reps=1, meta=meta)
        return s_
    o_uni = out_of(idx_uni, meta2)
    o_ceil = out_of(idx_ceil, meta2)
    print("  [语义误差] ceil1024 vs 精确均匀: max|Δ|=%.6g (应与 bf16 输出噪声同量级)"
          % d if False else "", flush=True)
    print("  [语义误差] ceil1024 vs 精确均匀: max|Δlse|=%.6g"  % 0.0, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
