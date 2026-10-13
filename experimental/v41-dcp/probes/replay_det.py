"""测试：开启 aclnn/GE 的确定性计算后，SMLA 是否变确定。

依据（上游源码）：
  vllm_ascend/batch_invariant.py::override_envs_for_invariance()
      torch.use_deterministic_algorithms(True, warn_only=True)
  op_host/sparse_flash_mla_tiling.cpp:392
      batchConsistency_ = (context_->GetDeterministicLevel() == BATCH_CONSISTENCY_LEVEL);  // =3
  该位进入 tiling key（docs/ratio2_a2a3.md 说 A2/A3 保留 BATCH_CONSISTENCY=0/1）
  docs/aclnnSparseFlashMla.md: "aclnnSparseFlashMla默认采用确定性实现，相同输入多次调用结果一致"
⇒ 若开确定性后 bit_identical 变 True，就是一个可交付的规避开关。
"""
import os
import sys

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

DEV = "npu:0"


def call(d, reps):
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
            seqused_cmp_kv=d["seqused_cmp_kv"].to(DEV).to(torch.int32),
            cmp_residual_kv=(d["cmp_residual_kv"].to(DEV).to(torch.int32)
                             if d.get("cmp_residual_kv") is not None else None),
            sinks=d["sinks"].to(DEV), metadata=d["metadata"].to(DEV).to(torch.int32),
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
    d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    print("torch.use_deterministic_algorithms 默认 =", torch.are_deterministic_algorithms_enabled(),
          flush=True)
    same, dmax, nan = call(d, reps)
    print("[A 默认]        bit_identical=%-5s max|dlse|=%-12.6g nan=%d" % (same, dmax, nan), flush=True)

    os.environ["HCCL_DETERMINISTIC"] = "strict"
    os.environ["LCCL_DETERMINISTIC"] = "1"
    torch.use_deterministic_algorithms(True, warn_only=True)
    print("开确定性后 =", torch.are_deterministic_algorithms_enabled(), flush=True)
    same, dmax, nan = call(d, reps)
    print("[B 确定性算法]  bit_identical=%-5s max|dlse|=%-12.6g nan=%d" % (same, dmax, nan), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
