"""在真实 dump 上只改动 **索引的数值**（保留每行的有效个数与 -1 位置），
判断非确定是"结构触发"还是"具体数值触发"。

变体：
  asis     原样
  shuffle  每行有效值**打乱**（个数不变、值集合不变、顺序变）
  prefix   每行有效值改成 **0..n-1**（连续前缀，个数不变）
  rand     每行有效值改成 n 个**随机不重复**值（值域 0..127）
"""
import sys

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

DEV = "npu:0"


def call(d, idx, reps=6):
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
    K = d["cmp_indices"].shape[-1]
    base = d["cmp_indices"].to(torch.int64).squeeze(1)
    cseq = int(d["seqused_cmp_kv"].max())
    gen = torch.Generator(device="cpu").manual_seed(31337)

    def build(fn):
        rows = []
        for t in range(base.shape[0]):
            v = base[t][base[t] >= 0]
            rows.append(fn(v, t))
        out = torch.full((base.shape[0], 1, K), -1, dtype=torch.int64)
        for t, r in enumerate(rows):
            out[t, 0, : r.numel()] = r
        return out.to(torch.int32).to(DEV)

    V = {
        "asis": lambda v, t: v,
        "shuffle": lambda v, t: v[torch.randperm(v.numel(), generator=gen)],
        "prefix": lambda v, t: torch.arange(v.numel(), dtype=torch.int64),
        "rand": lambda v, t: torch.randperm(cseq, generator=gen)[: v.numel()].to(torch.int64),
    }
    for name in ("asis", "shuffle", "prefix", "rand"):
        idx = build(V[name])
        same, dmax, nan = call(d, idx)
        print("[%-8s] bit_identical=%-5s max|dlse|=%-12.6g nan=%d" % (name, same, dmax, nan), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
