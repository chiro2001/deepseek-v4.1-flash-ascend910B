"""把生产 dump 的 SMLA 输入在单卡上原样重放，检验算子是否非确定。

用法：python3 replay_dump.py <dump.pt> [reps]

dump 由生产侧 `dsa_v41.py` 的 `[V41-DUMPREPLAY]` 落盘，包含：
  q / cmp_indices / ori_block_table(已重映射) / cmp_block_table(已重映射)
  / cu_seqlens_q / seqused_* / cmp_residual_kv / sinks / metadata(1024,)
  / 用到的 ori 与 long_kv 页 / 标量参数
**页号已被重映射到 1..N，只搬运本请求实际读到的页** —— 数据与生产逐位相同。
"""
import sys

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

DEV = "npu:0"


def main():
    path = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    d = torch.load(path, map_location="cpu", weights_only=False)
    s = d["scalars"]
    # ===== [V41-DUMP-FAITHFUL-2] 核对 stride(0)：内核的 KvStride0 由它决定，
    # 不一致会让地址整体偏移。若保存/加载没能保持 stride，就按记录重建。
    def _fix_stride(t, want):
        if t is None or t.numel() == 0 or not want:
            return t
        if int(t.stride(0)) == int(want):
            return t
        flat = t.reshape(-1)
        shape = (t.shape[0],) + tuple(t.shape[1:])
        stride = (int(want),) + tuple(t.stride()[1:])
        need = (shape[0] - 1) * stride[0] + 1
        if need > flat.numel():
            buf = torch.zeros((shape[0] - 1) * stride[0] + 1, dtype=flat.dtype)
            for i in range(shape[0]):
                buf[i * stride[0] : i * stride[0] + max(1, t[0].numel())] = t[i].reshape(-1)
            return torch.as_strided(buf, shape, stride)
        return torch.as_strided(flat, shape, stride)

    q = d["q"].to(DEV)
    ori = _fix_stride(d["ori_pages"], d.get("ori_stride0"))
    cmp_kv = _fix_stride(d["cmp_pages"], d.get("cmp_stride0"))
    ori = ori.to(DEV) if ori is not None else ori
    cmp_kv = cmp_kv.to(DEV) if cmp_kv is not None else cmp_kv
    print("  stride0 ori=%s(want %s) cmp=%s(want %s)"
          % (ori.stride(0) if ori.numel() else "-", d.get("ori_stride0"),
             cmp_kv.stride(0) if cmp_kv.numel() else "-", d.get("cmp_stride0")), flush=True)
    idx = d["cmp_indices"].to(DEV).to(torch.int32)
    obt = d["ori_block_table"].to(DEV).to(torch.int32)
    cbt = d["cmp_block_table"].to(DEV).to(torch.int32)
    cu = d["cu_seqlens_q"].to(DEV).to(torch.int32)
    soi = d["seqused_ori_kv"].to(DEV).to(torch.int32)
    sci = d["seqused_cmp_kv"].to(DEV).to(torch.int32) if d.get("seqused_cmp_kv") is not None else None
    res = d["cmp_residual_kv"].to(DEV).to(torch.int32) if d.get("cmp_residual_kv") is not None else None
    sinks = d["sinks"].to(DEV) if d.get("sinks") is not None else None
    meta = d["metadata"].to(DEV).to(torch.int32) if d.get("metadata") is not None else None
    print("dump=%s layer=%d rank=%d T=%d" % (path, s["layer_idx"], s["dcp_rank"], q.shape[0]),
          flush=True)
    print("  q=%s ori_pages=%s cmp_pages=%s idx=%s cbt=%s cseq=%s"
          % (tuple(q.shape), tuple(ori.shape), tuple(cmp_kv.shape), tuple(idx.shape),
             tuple(cbt.shape), (sci.tolist() if sci is not None else None)), flush=True)

    res_list = []
    for _ in range(reps):
        out, lse = torch.ops._C_ascend.npu_sparse_flash_mla(
            q,
            ori_kv=ori,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=idx,
            ori_block_table=obt,
            cmp_block_table=cbt,
            cu_seqlens_q=cu,
            seqused_ori_kv=soi,
            seqused_cmp_kv=sci,
            cmp_residual_kv=res,
            sinks=sinks,
            metadata=meta,
            softmax_scale=float(s["softmax_scale"]),
            cmp_ratio=int(s["cmp_ratio"]),
            ori_mask_mode=int(s["ori_mask_mode"]),
            cmp_mask_mode=int(s["cmp_mask_mode"]),
            ori_win_left=int(s["ori_win_left"]),
            ori_win_right=int(s["ori_win_right"]),
            layout_q="TND",
            layout_kv="PA_BBND",
            topk_value_mode=int(s["topk_value_mode"]),
            return_softmax_lse=True,
        )
        torch.npu.synchronize()
        res_list.append((out.to(torch.float32).cpu(), lse.to(torch.float32).cpu()))
    same_lse = all(bool(torch.equal(res_list[0][1], r[1])) for r in res_list[1:])
    same_out = all(bool(torch.equal(res_list[0][0], r[0])) for r in res_list[1:])
    dmax = max(float((res_list[0][1] - r[1]).abs().max()) for r in res_list[1:])
    nfin = [int((~torch.isfinite(r[1])).sum()) for r in res_list]
    # ★ 把"首次调用"与"稳态（第 2 次起）"分开判：若稳态自洽 ⇒ 缺陷是
    #   **首次调用读未初始化 workspace**（与生产"第 1 个请求与后续不同"一致）。
    steady_same = (
        all(bool(torch.equal(res_list[1][1], r[1])) for r in res_list[2:])
        if len(res_list) > 2 else None
    )
    print("[REPLAY] reps=%d ALL_bit_identical=%-5s STEADY(2..N)_bit_identical=%-5s "
          "out_bit_identical=%-5s max|dlse|=%.6g lse_nan_per_rep=%s"
          % (reps, same_lse, steady_same, same_out, dmax, nfin), flush=True)
    return 0 if same_lse else 1


if __name__ == "__main__":
    sys.exit(main())
