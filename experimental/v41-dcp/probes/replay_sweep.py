"""零重编参数扫描：只改 seqused_cmp_kv（与 cmp_residual_kv），看能否让 NaN 归零。

推论：内核 baseline 的声明数 = min(min(cseq, thresHold), CountValid(...))，
其中 thresHold(t) = cseq − T + t + 1。
若 cseq ≥ T + (最大索引+1)，则 thresHold(t) ≥ 最大索引+1 对所有 t 成立
⇒ 一个键都不丢，且 bound = min(cseq, 512) ≥ 有效项数
⇒ 「声明数 == 实际写入数」自洽 ⇒ 预期 NaN 归零。
"""
import sys

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

DEV = "npu:0"


def run(d, cseq, residual=0, reps=6, topk=512):
    s = d["scalars"]
    T = int(s["max_seqlen_q"])
    kv = d["cmp_pages"].to(DEV)
    meta = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        int(s["num_heads_q"]), 1, int(s["head_dim"]),
        cu_seqlens_q=d["cu_seqlens_q"].to(DEV).to(torch.int32),
        seqused_ori_kv=d["seqused_ori_kv"].to(DEV).to(torch.int32),
        seqused_cmp_kv=torch.tensor([cseq], dtype=torch.int32, device=DEV),
        cmp_residual_kv=torch.tensor([residual], dtype=torch.int32, device=DEV),
        batch_size=1, max_seqlen_q=T, max_seqlen_ori_kv=T, max_seqlen_cmp_kv=int(cseq),
        ori_topk=0, cmp_topk=int(topk), cmp_ratio=int(s["cmp_ratio"]),
        ori_mask_mode=int(s["ori_mask_mode"]), cmp_mask_mode=int(s["cmp_mask_mode"]),
        ori_win_left=int(s["ori_win_left"]), ori_win_right=0,
        layout_q="TND", layout_kv="PA_BBND", has_ori_kv=True, has_cmp_kv=True)
    res = []
    for _ in range(reps):
        o, l = torch.ops._C_ascend.npu_sparse_flash_mla(
            d["q"].to(DEV), ori_kv=d["ori_pages"].to(DEV), cmp_kv=kv,
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
    return same, dmax, nan


def main():
    path = sys.argv[1]
    d = torch.load(path, map_location="cpu", weights_only=False)
    local = int(d["seqused_cmp_kv"].max())
    T = int(d["scalars"]["max_seqlen_q"])
    ratio = int(d["scalars"]["cmp_ratio"])
    idxmax = int(d["cmp_indices"].max())
    print("dump=%s T=%d local_cseq=%d idx_max=%d" % (path.split("/")[-1], T, local, idxmax), flush=True)
    cands = [local, T, T + idxmax + 1, 2 * T]
    for cseq in cands:
        try:
            same, dmax, nan = run(d, cseq)
            print("  cseq=%-6d bit_identical=%-5s max|dlse|=%-12.6g lse_nan=%d"
                  % (cseq, same, dmax, nan), flush=True)
        except Exception as e:  # noqa: BLE001
            print("  cseq=%-6d 失败：%s" % (cseq, str(e)[:110].replace("\n", " ")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
