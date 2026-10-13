"""SMLA(ratio=1) 非确定性最小复现 + 触发条件扫描（单卡）。

背景（a3-21 DCP8 实测）：
  · layer 20（ratio=1）上用**逐位相同的输入**连调两次 SMLA：
        lse_bit_identical=False  max_abs_diff≈0.19
  · layer 2（ratio=2）同样测试：lse_bit_identical=True
  · 输入的按页指纹（index K sum/nz、long_kv 内容）两次请求逐位相同

本探针在**单卡**上复现"同输入两次调用结果不同"，并扫描触发条件。
模式：
  pagesN   —— cmp 用 N 页（值域 N*128，声明 cseq=N*128，均合法）
  global   —— 值域 452、声明 cseq 452（合法，需 4 页）
  global_bad —— 值域 452、声明 cseq 128（越界）
  shard    —— 值域 128、声明 cseq 128（合法，1 页）
  shard_bad —— 值域 128、声明 cseq 32（越界）
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
H = 64
D = 512
TOPK = 512
WIN = 128
PAGE = 128
S_LOCAL = int(os.environ.get("PROBE_S_LOCAL", "128"))
G_GLOBAL = int(os.environ.get("PROBE_G_GLOBAL", "452"))
ORI_PAGES = int(os.environ.get("PROBE_ORI_PAGES", "8"))
CMP_BT_COLS = int(os.environ.get("PROBE_CMP_BT_COLS", "8192"))
SINK_MODE = os.environ.get("PROBE_SINK_MODE", "zero")
CMP_PAGES = int(os.environ.get("PROBE_CMP_PAGES", "4"))


def pages_of(mode):
    return int(mode[5:]) if mode.startswith("pages") else None


def make_indices(mode):
    """按模式构造 [T,1,TOPK] 索引；值域决定需要几页，-1 表示 padding。"""
    g = torch.Generator(device="cpu").manual_seed(99)
    _pg = pages_of(mode)
    if _pg is not None:
        hi = PAGE * _pg
        idx = torch.randint(0, hi, (T, 1, TOPK), generator=g, dtype=torch.int32)
        return idx.to(DEV)
    if mode in ("global", "global_bad"):
        hi = min(G_GLOBAL, PAGE * CMP_PAGES)
        idx = torch.randint(0, hi, (T, 1, TOPK), generator=g, dtype=torch.int32)
        return idx.to(DEV)
    hi = min(S_LOCAL, PAGE * CMP_PAGES)
    idx = torch.randint(0, hi, (T, 1, TOPK), generator=g, dtype=torch.int32)
    keep = int(os.environ.get("PROBE_KEEP", "367"))
    mask = torch.arange(TOPK).view(1, 1, -1) >= keep
    idx = idx.masked_fill(mask, -1)
    return idx.to(DEV)


def declared_cseq(mode):
    _pg = pages_of(mode)
    if _pg is not None:
        return PAGE * _pg
    return {"global": min(G_GLOBAL, PAGE * CMP_PAGES),
            "global_bad": S_LOCAL,
            "shard": min(S_LOCAL, PAGE * CMP_PAGES),
            "shard_bad": 32}[mode]


def n_cmp_cols(mode):
    _pg = pages_of(mode)
    if _pg is not None:
        return _pg
    return CMP_PAGES


def build(seed=1234):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(T, H, D, generator=g, dtype=torch.float32).to(torch.bfloat16).to(DEV)
    # ★ ori 必须覆盖整个 T（滑窗回看 127 行）⇒ 至少 ceil(T/PAGE) 页，否则越界出 NaN
    ori = torch.randn(ORI_PAGES, PAGE, 1, D, generator=g,
                      dtype=torch.float32).to(torch.bfloat16).to(DEV)
    cmp_kv = torch.randn(CMP_PAGES, PAGE, 1, D, generator=g,
                         dtype=torch.float32).to(torch.bfloat16).to(DEV)
    if SINK_MODE == "real":
        sinks = (torch.randn(H, generator=g, dtype=torch.float32) * 2.0).to(DEV)
    else:
        sinks = torch.zeros(H, dtype=torch.float32, device=DEV)
    return q, ori, cmp_kv, sinks


def call(q, ori, cmp_kv, idx, sinks, cseq_val, cols):
    dev = q.device
    _n_ori = (T + PAGE - 1) // PAGE
    _bt = torch.zeros(1, 8192, dtype=torch.int32, device=dev)
    _bt[0, :_n_ori] = torch.arange(1, _n_ori + 1, dtype=torch.int32, device=dev)
    _cb = torch.zeros(1, CMP_BT_COLS, dtype=torch.int32, device=dev)
    _cb[0, :cols] = torch.arange(1, cols + 1, dtype=torch.int32, device=dev)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    seq = torch.full((1,), T, dtype=torch.int32, device=dev)
    cseq = torch.full((1,), int(cseq_val), dtype=torch.int32, device=dev)
    resid = torch.zeros(1, dtype=torch.int32, device=dev)
    meta = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        H, 1, D,
        cu_seqlens_q=cu, seqused_ori_kv=seq, seqused_cmp_kv=cseq,
        cmp_residual_kv=resid,
        batch_size=1, max_seqlen_q=T, max_seqlen_ori_kv=T,
        max_seqlen_cmp_kv=int(cseq.max()),
        ori_topk=0, cmp_topk=TOPK, cmp_ratio=1,
        ori_mask_mode=4, cmp_mask_mode=3,
        ori_win_left=WIN - 1, ori_win_right=0,
        layout_q="TND", layout_kv="PA_BBND",
        has_ori_kv=True, has_cmp_kv=True,
    )
    out, lse = torch.ops._C_ascend.npu_sparse_flash_mla(
        q, ori_kv=ori, cmp_kv=cmp_kv, cmp_sparse_indices=idx,
        ori_block_table=_bt, cmp_block_table=_cb,
        cu_seqlens_q=cu, seqused_ori_kv=seq, seqused_cmp_kv=cseq,
        cmp_residual_kv=resid, sinks=sinks, metadata=meta,
        softmax_scale=D ** -0.5, cmp_ratio=1,
        ori_mask_mode=4, cmp_mask_mode=3,
        ori_win_left=WIN - 1, ori_win_right=0,
        layout_q="TND", layout_kv="PA_BBND",
        topk_value_mode=1, return_softmax_lse=True,
    )
    return out, lse


def main():
    reps = int(os.environ.get("PROBE_REPS", "8"))
    modes = os.environ.get("PROBE_MODES", "pages1,pages2,pages3,pages4").split(",")
    q, ori, cmp_kv, sinks = build()
    for mode in modes:
        idx = make_indices(mode)
        res = []
        for _ in range(reps):
            out, lse = call(q, ori, cmp_kv, idx, sinks,
                            declared_cseq(mode), n_cmp_cols(mode))
            torch.npu.synchronize()
            res.append((out.to(torch.float32).cpu(), lse.to(torch.float32).cpu()))
        same_lse = all(bool(torch.equal(res[0][1], r[1])) for r in res[1:])
        same_out = all(bool(torch.equal(res[0][0], r[0])) for r in res[1:])
        dmax = max(float((res[0][1] - r[1]).abs().max()) for r in res[1:])
        nan = int((~torch.isfinite(res[0][1])).sum())
        print("[%-11s] 页数=%d 值域=%d cseq=%d btcols=%d sink=%s reps=%d | "
              "lse_bit_identical=%-5s out_bit_identical=%-5s max|dlse|=%.6g lse_nan=%d"
              % (mode, n_cmp_cols(mode), PAGE * n_cmp_cols(mode), declared_cseq(mode),
                 CMP_BT_COLS, SINK_MODE, reps, same_lse, same_out, dmax, nan), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
