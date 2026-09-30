"""扫描会改变 **tiling key** → 选到不同预编译二进制 的参数组合。

依据：`skcache` 的 `binary_info_config.json` 里 SparseFlashMla 的 key 形如
  (cmp_ratio, ori_mask_mode, cmp_mask_mode, win, layout, topk_value_mode)
生产用的是 (1, 4, 3, 127, PA_BBND, 1)。若某个变体选到的二进制**没有**
未初始化内存缺陷，就是一个可用的规避（只要语义仍正确）。
"""
import itertools
import sys

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

DEV = "npu:0"


def run(d, omm, cmm, tvm, ori_win, reps=4):
    s = d["scalars"]
    T = d["q"].shape[0]
    cu = d["cu_seqlens_q"].to(DEV).to(torch.int32)
    meta = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        int(s["num_heads_q"]), 1, int(s["head_dim"]),
        cu_seqlens_q=cu,
        seqused_ori_kv=d["seqused_ori_kv"].to(DEV).to(torch.int32),
        seqused_cmp_kv=d["seqused_cmp_kv"].to(DEV).to(torch.int32),
        cmp_residual_kv=(d["cmp_residual_kv"].to(DEV).to(torch.int32)
                         if d.get("cmp_residual_kv") is not None else None),
        batch_size=1, max_seqlen_q=int(s["max_seqlen_q"]),
        max_seqlen_ori_kv=int(s["max_seqlen_ori_kv"]),
        max_seqlen_cmp_kv=int(d["seqused_cmp_kv"].max()),
        ori_topk=0, cmp_topk=512, cmp_ratio=1,
        ori_mask_mode=omm, cmp_mask_mode=cmm,
        ori_win_left=ori_win, ori_win_right=0,
        layout_q="TND", layout_kv="PA_BBND", has_ori_kv=True, has_cmp_kv=True)
    res = []
    for _ in range(reps):
        o, l = torch.ops._C_ascend.npu_sparse_flash_mla(
            d["q"].to(DEV), ori_kv=d["ori_pages"].to(DEV), cmp_kv=d["cmp_pages"].to(DEV),
            cmp_sparse_indices=d["cmp_indices"].to(DEV).to(torch.int32),
            ori_block_table=d["ori_block_table"].to(DEV).to(torch.int32),
            cmp_block_table=d["cmp_block_table"].to(DEV).to(torch.int32),
            cu_seqlens_q=cu,
            seqused_ori_kv=d["seqused_ori_kv"].to(DEV).to(torch.int32),
            seqused_cmp_kv=d["seqused_cmp_kv"].to(DEV).to(torch.int32),
            cmp_residual_kv=(d["cmp_residual_kv"].to(DEV).to(torch.int32)
                             if d.get("cmp_residual_kv") is not None else None),
            sinks=d["sinks"].to(DEV), metadata=meta,
            softmax_scale=float(s["softmax_scale"]), cmp_ratio=1,
            ori_mask_mode=omm, cmp_mask_mode=cmm,
            ori_win_left=ori_win, ori_win_right=0,
            layout_q="TND", layout_kv="PA_BBND",
            topk_value_mode=tvm, return_softmax_lse=True)
        torch.npu.synchronize()
        res.append((o.to(torch.float32).cpu(), l.to(torch.float32).cpu()))
    same = all(bool(torch.equal(res[0][1], r[1])) for r in res[1:])
    dmax = max(float((res[0][1] - r[1]).abs().max()) for r in res[1:])
    nan = int((~torch.isfinite(res[0][1])).sum())
    return same, dmax, nan, res[0]


def main():
    d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    base = None
    combos = list(itertools.product([4, 0], [3, 0, 1, 2, 4], [1, 0], [127]))
    for omm, cmm, tvm, win in combos:
        tag = "omm=%d cmm=%d tvm=%d win=%d" % (omm, cmm, tvm, win)
        try:
            same, dmax, nan, first = run(d, omm, cmm, tvm, win)
        except Exception as e:  # noqa: BLE001
            msg = str(e).split("Reason:")[-1].strip()[:70]
            print("[%s] 不可用：%s" % (tag, msg), flush=True)
            continue
        extra = ""
        if base is None:
            base = first
        else:
            extra = " | vs 基线 max|dlse|=%.6g" % float((base[1] - first[1]).abs().max())
        print("[%s] bit_identical=%-5s max|dlse|=%-12.6g nan=%-6d%s"
              % (tag, same, dmax, nan, extra), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
