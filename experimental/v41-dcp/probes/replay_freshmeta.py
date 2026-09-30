"""测试：算子是否**原地修改 metadata**（若每次都传新副本则变确定）。"""
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
    q = d["q"].to(DEV)
    idx = d["cmp_indices"].to(DEV).to(torch.int32)
    meta0 = d["metadata"].to(DEV).to(torch.int32)
    print("meta shape=%s dtype=%s | 前 16 个 int: %s"
          % (tuple(meta0.shape), meta0.dtype, meta0[:16].tolist()), flush=True)

    def run(meta):
        o, l = torch.ops._C_ascend.npu_sparse_flash_mla(
            q, ori_kv=d["ori_pages"].to(DEV), cmp_kv=d["cmp_pages"].to(DEV),
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
        return o.to(torch.float32).cpu(), l.to(torch.float32).cpu(), meta.clone().cpu()

    # 臂 A：复用同一 metadata（现状）
    a = []
    m = meta0.clone()
    for _ in range(reps):
        o, l, m_new = run(m)
        a.append(l)
        if not bool(torch.equal(m.cpu(), m_new)):
            print("  ★ 算子**修改了 metadata**（第 %d 次后）" % (len(a),), flush=True)
            m = meta0.clone()   # 复原，避免累积影响 A 臂结论
    a_same = all(bool(torch.equal(a[0], x)) for x in a[1:])
    print("[A 复用 metadata] reps=%d bit_identical=%s NaN/次=%s"
          % (reps, a_same, [int((~torch.isfinite(x)).sum()) for x in a]), flush=True)

    # 臂 B：每次传**新副本**
    b = []
    for _ in range(reps):
        o, l, _ = run(meta0.clone())
        b.append(l)
    b_same = all(bool(torch.equal(b[0], x)) for x in b[1:])
    print("[B 每次新 metadata 副本] bit_identical=%s NaN/次=%s"
          % (b_same, [int((~torch.isfinite(x)).sum()) for x in b]), flush=True)
    print("[A vs B] 首次结果逐位相同=%s" % bool(torch.equal(a[0], b[0])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
