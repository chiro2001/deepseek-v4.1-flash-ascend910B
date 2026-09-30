"""验证假设：**前序 SMLA 调用留下的 workspace 残留**会导致目标调用非确定。

生产现象：DCP8 同一 prompt 的**首个**请求正确、后续请求错。
单卡对照：干净进程里连调同一输入 8 次 = 确定（pages1/pages2）。

本实验：**在目标调用之前，先跑 N 次"扰动"调用**（用不同的 q/索引），
再调目标输入；重复该过程两次，比对两次的目标结果。
  · 两次相同 ⇒ 前序调用不构成污染；
  · 两次不同 ⇒ **跨调用 workspace 残留**是根因（也解释了"首请求对、后续错"）。
"""
import os
import sys

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

DEV = "npu:%d" % int(os.environ.get("PROBE_DEV", "0"))
T = int(os.environ.get("PROBE_T", "904"))
H, D, TOPK, WIN, PAGE = 64, 512, 512, 128, 128
ORI_PAGES, CMP_PAGES = 8, 4
N_WARM = int(os.environ.get("PROBE_N_WARM", "19"))


def build(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(T, H, D, generator=g, dtype=torch.float32).to(torch.bfloat16).to(DEV)
    ori = torch.randn(ORI_PAGES, PAGE, 1, D, generator=g, dtype=torch.float32).to(torch.bfloat16).to(DEV)
    cmp_kv = torch.randn(CMP_PAGES, PAGE, 1, D, generator=g, dtype=torch.float32).to(torch.bfloat16).to(DEV)
    sinks = torch.zeros(H, dtype=torch.float32, device=DEV)
    return q, ori, cmp_kv, sinks


def make_idx(seed, hi=128, keep=367):
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, hi, (T, 1, TOPK), generator=g, dtype=torch.int32)
    mask = torch.arange(TOPK).view(1, 1, -1) >= keep
    return idx.masked_fill(mask, -1).to(DEV)


def call(q, ori, cmp_kv, idx, sinks, cols=1, cseq_val=128):
    dev = q.device
    _n_ori = (T + PAGE - 1) // PAGE
    bt = torch.zeros(1, 8192, dtype=torch.int32, device=dev)
    bt[0, :_n_ori] = torch.arange(1, _n_ori + 1, dtype=torch.int32, device=dev)
    cb = torch.zeros(1, 1024, dtype=torch.int32, device=dev)
    cb[0, :cols] = torch.arange(1, cols + 1, dtype=torch.int32, device=dev)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    seq = torch.full((1,), T, dtype=torch.int32, device=dev)
    cseq = torch.full((1,), cseq_val, dtype=torch.int32, device=dev)
    resid = torch.zeros(1, dtype=torch.int32, device=dev)
    meta = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        H, 1, D, cu_seqlens_q=cu, seqused_ori_kv=seq, seqused_cmp_kv=cseq,
        cmp_residual_kv=resid, batch_size=1, max_seqlen_q=T, max_seqlen_ori_kv=T,
        max_seqlen_cmp_kv=int(cseq.max()), ori_topk=0, cmp_topk=TOPK, cmp_ratio=1,
        ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=WIN - 1, ori_win_right=0,
        layout_q="TND", layout_kv="PA_BBND", has_ori_kv=True, has_cmp_kv=True)
    out, lse = torch.ops._C_ascend.npu_sparse_flash_mla(
        q, ori_kv=ori, cmp_kv=cmp_kv, cmp_sparse_indices=idx,
        ori_block_table=bt, cmp_block_table=cb, cu_seqlens_q=cu,
        seqused_ori_kv=seq, seqused_cmp_kv=cseq, cmp_residual_kv=resid,
        sinks=sinks, metadata=meta, softmax_scale=D ** -0.5, cmp_ratio=1,
        ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=WIN - 1, ori_win_right=0,
        layout_q="TND", layout_kv="PA_BBND", topk_value_mode=1,
        return_softmax_lse=True)
    return out, lse


def warm(n):
    """n 次扰动调用（不同 q / 索引 ⇒ 模拟前面那些层）。"""
    for i in range(n):
        qw, oriw, cmpw, sw = build(7000 + i)
        iw = make_idx(8000 + i)
        call(qw, oriw, cmpw, iw, sw)


def main():
    reps = int(os.environ.get("PROBE_REPS", "4"))
    q, ori, cmp_kv, sinks = build(1234)
    idx = make_idx(99)
    # 臂 A：不做 warm（干净进程）
    a = []
    for _ in range(reps):
        o, l = call(q, ori, cmp_kv, idx, sinks)
        torch.npu.synchronize()
        a.append(l.to(torch.float32).cpu())
    a_same = all(bool(torch.equal(a[0], x)) for x in a[1:])
    print("[A 无 warm]     reps=%d lse_bit_identical=%s" % (reps, a_same), flush=True)
    # 臂 B：每次目标调用前先 warm N_WARM 次
    b = []
    for _ in range(reps):
        warm(N_WARM)
        o, l = call(q, ori, cmp_kv, idx, sinks)
        torch.npu.synchronize()
        b.append(l.to(torch.float32).cpu())
    b_same = all(bool(torch.equal(b[0], x)) for x in b[1:])
    dmax = max(float((b[0] - x).abs().max()) for x in b[1:])
    print("[B warm=%d]     reps=%d lse_bit_identical=%s max|dlse|=%.6g"
          % (N_WARM, reps, b_same, dmax), flush=True)
    # 臂 C：A 与 B 的目标结果是否一致
    print("[A vs B] 逐位相同=%s" % bool(torch.equal(a[0], b[0])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
