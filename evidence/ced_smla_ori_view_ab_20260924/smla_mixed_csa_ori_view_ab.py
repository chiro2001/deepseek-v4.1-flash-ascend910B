#!/usr/bin/env python3
"""A3 SMLA mixed-CSA mask micro A/B; no model or production files are changed."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401
import cann_ops_transformer  # noqa: F401  # official SMLA operator namespace

OUT_META = Path('/tmp/smla_mixed_csa_ori_view_ab_metadata.json')
OUT_FULL = Path('/tmp/smla_mixed_csa_ori_view_ab_result.json')
STAGE = sys.argv[1] if len(sys.argv) > 1 else 'full'
if STAGE not in {'metadata-only', 'full'}:
    raise SystemExit('usage: smla_mixed_csa_ori_view_ab.py metadata-only|full')
B, QLEN, NQ, NK, D, BLOCK, TOPK = 1, 128, 1, 1, 512, 128, 512
RATIO, CMP_LEN, RESID, GLOBAL_START = 2, 127, 1, 127
SINK = -100.0
DEVICE = torch.device('npu:0')


def tensor_hash(t: torch.Tensor) -> str:
    raw = t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def call_metadata(
    seq_ori: torch.Tensor,
    max_ori: int,
    cu_q: torch.Tensor,
    seq_cmp: torch.Tensor,
    cmp_resid: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
        NQ,
        NK,
        D,
        cu_seqlens_q=cu_q,
        seqused_ori_kv=seq_ori,
        seqused_cmp_kv=seq_cmp,
        cmp_residual_kv=cmp_resid,
        batch_size=B,
        max_seqlen_q=QLEN,
        max_seqlen_ori_kv=max_ori,
        max_seqlen_cmp_kv=CMP_LEN,
        ori_topk=0,
        cmp_topk=TOPK,
        cmp_ratio=RATIO,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q='TND',
        layout_kv='PA_BBND',
        has_ori_kv=True,
        has_cmp_kv=True,
    )


def call_attention(
    q,
    ori_kv,
    ori_table,
    seq_ori,
    metadata,
    cmp_kv,
    cmp_indices,
    cmp_table,
    cu_q,
    seq_cmp,
    cmp_resid,
    sinks,
):
    output, _ = torch.ops.cann_ops_transformer.sparse_flash_mla(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_indices,
        ori_block_table=ori_table,
        cmp_block_table=cmp_table,
        cu_seqlens_q=cu_q,
        seqused_ori_kv=seq_ori,
        seqused_cmp_kv=seq_cmp,
        cmp_residual_kv=cmp_resid,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=1.0 / math.sqrt(D),
        cmp_ratio=RATIO,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q='TND',
        layout_kv='PA_BBND',
        topk_value_mode=1,
        return_softmax_lse=False,
    )
    torch.npu.synchronize()
    return output.detach().float().cpu().numpy()


try:
    torch.npu.set_device(0)
    # The opaque Q bytes are identical in A/B. Nonzero dim 2 is orthogonal to
    # all synthetic KV payloads (which use dims 0/1), so every tested logit is 0.
    q = torch.zeros((QLEN, NQ, D), dtype=torch.bfloat16, device=DEVICE)
    q[:, :, 2] = 1.0

    # A: dense original sequence positions 0..254 in two PA pages.
    ori_a = torch.zeros((2, BLOCK, NK, D), dtype=torch.bfloat16, device=DEVICE)
    pos0 = torch.arange(BLOCK, dtype=torch.float32, device=DEVICE)
    pos1 = torch.arange(BLOCK, dtype=torch.float32, device=DEVICE) + BLOCK
    ori_a[0, :, 0, 0] = pos0 / 255.0
    ori_a[1, :127, 0, 0] = pos1[:127] / 255.0
    seq_ori_a = torch.tensor([255], dtype=torch.int32, device=DEVICE)
    ori_table_a = torch.tensor([[0, 1]], dtype=torch.int32, device=DEVICE)

    # B: non-128-aligned replay_start=127 packed to scratch block offset 0.
    ori_b = torch.zeros((1, BLOCK, NK, D), dtype=torch.bfloat16, device=DEVICE)
    ori_b[0, 0, 0, 0] = ori_a[0, 127, 0, 0]
    ori_b[0, 1:BLOCK, 0, 0] = ori_a[1, :127, 0, 0]
    seq_ori_b = torch.tensor([128], dtype=torch.int32, device=DEVICE)
    ori_table_b = torch.tensor([[0]], dtype=torch.int32, device=DEVICE)

    # Fixed global C2 source: 127 compressed rows plus one residual token.
    # The q2=1 / kv-dim2=0 construction keeps all dot products at zero, while
    # the selected value channels expose visibility and joint-softmax counts.
    cmp_kv = torch.zeros((1, BLOCK, NK, D), dtype=torch.bfloat16, device=DEVICE)
    cmp_kv[0, 0, 0, 0] = 1.0
    cmp_kv[0, 126, 0, 1] = 100.0
    cmp_table = torch.tensor([[0]], dtype=torch.int32, device=DEVICE)
    seq_cmp = torch.tensor([CMP_LEN], dtype=torch.int32, device=DEVICE)
    cmp_resid = torch.tensor([RESID], dtype=torch.int32, device=DEVICE)
    cu_q = torch.tensor([0, QLEN], dtype=torch.int32, device=DEVICE)
    sinks = torch.full((NQ,), SINK, dtype=torch.float32, device=DEVICE)

    # Absolute positions 127..254: valid global C2 candidates are 0..floor((p+1)/2)-1.
    idx = np.full((QLEN, NK, TOPK), -1, dtype=np.int32)
    valid_cmp = []
    for j in range(QLEN):
        absolute_pos = GLOBAL_START + j
        count = min(CMP_LEN, (absolute_pos + 1) // RATIO)
        valid_cmp.append(count)
        idx[j, 0, :count] = np.arange(count, dtype=np.int32)
    cmp_indices = torch.from_numpy(idx).to(device=DEVICE)

    # Hash the common operands before either call to verify A/B do not change
    # q or the compressed-global branch.
    common_hashes = {
        'q': tensor_hash(q),
        'cmp_kv': tensor_hash(cmp_kv),
        'cmp_sparse_indices': tensor_hash(cmp_indices),
        'cmp_block_table': tensor_hash(cmp_table),
        'seqused_cmp_kv': tensor_hash(seq_cmp),
        'cmp_residual_kv': tensor_hash(cmp_resid),
        'cu_seqlens_q': tensor_hash(cu_q),
    }

    meta_a = call_metadata(seq_ori_a, 255, cu_q, seq_cmp, cmp_resid)
    meta_b = call_metadata(seq_ori_b, 128, cu_q, seq_cmp, cmp_resid)
    torch.npu.synchronize()
    metadata_result = {
        'stage': 'metadata-checker',
        'passed': True,
        'metadata_shape_a': list(meta_a.shape),
        'metadata_shape_b': list(meta_b.shape),
        'metadata_dtype_a': str(meta_a.dtype),
        'metadata_dtype_b': str(meta_b.dtype),
        'metadata_sha256_a': tensor_hash(meta_a),
        'metadata_sha256_b': tensor_hash(meta_b),
        'ori_len_a': 255,
        'ori_len_b': 128,
        'cmp_len_both': CMP_LEN,
        'cmp_residual_both': RESID,
        'cmp_ratio_both': RATIO,
        'cmp_inputs_sha256': {
            'cmp_kv': tensor_hash(cmp_kv),
            'cmp_sparse_indices': tensor_hash(cmp_indices),
            'cmp_block_table': tensor_hash(cmp_table),
            'seqused_cmp_kv': tensor_hash(seq_cmp),
            'cmp_residual_kv': tensor_hash(cmp_resid),
            'q': tensor_hash(q),
        },
    }
    print(json.dumps(metadata_result, indent=2), flush=True)
    OUT_META.write_text(json.dumps(metadata_result, indent=2) + '\n')
    if STAGE == 'metadata-only':
        raise SystemExit(0)

    op_args = (cmp_kv, cmp_indices, cmp_table, cu_q, seq_cmp, cmp_resid, sinks)
    out_a = call_attention(q, ori_a, ori_table_a, seq_ori_a, meta_a, *op_args)
    out_b = call_attention(q, ori_b, ori_table_b, seq_ori_b, meta_b, *op_args)

    def cpu_reference(local: bool) -> np.ndarray:
        ref = np.zeros((QLEN, NQ, D), dtype=np.float64)
        for j in range(QLEN):
            absolute_pos = GLOBAL_START + j
            if local:
                ori_positions = range(GLOBAL_START, absolute_pos + 1)
            else:
                ori_positions = range(max(0, absolute_pos - 127), absolute_pos + 1)
            ori_count = len(ori_positions)
            ori_sum = sum(p / 255.0 for p in ori_positions)
            cmp_count = valid_cmp[j]
            denom = ori_count + cmp_count + math.exp(SINK)
            # Ori values use output dim0. C2 index0 also uses dim0; index126
            # contributes dim1 only after it is causal (absolute pos >= 253).
            ref[j, 0, 0] = (ori_sum + (1.0 if cmp_count > 0 else 0.0)) / denom
            if cmp_count > 126:
                ref[j, 0, 1] = 100.0 / denom
        return ref

    ref_a = cpu_reference(local=False)
    ref_b = cpu_reference(local=True)

    def summarize(name, out, ref):
        delta = np.abs(out - ref)
        return {
            'name': name,
            'shape': list(out.shape),
            'nan': bool(np.isnan(out).any()),
            'max_abs_vs_cpu': float(delta.max()),
            'row0_dim0': float(out[0, 0, 0]),
            'row0_dim1': float(out[0, 0, 1]),
            'row127_dim0': float(out[127, 0, 0]),
            'row127_dim1': float(out[127, 0, 1]),
            'cpu_row0_dim0': float(ref[0, 0, 0]),
            'cpu_row0_dim1': float(ref[0, 0, 1]),
            'cpu_row127_dim0': float(ref[127, 0, 0]),
            'cpu_row127_dim1': float(ref[127, 0, 1]),
        }

    result = {
        'device_visible': __import__('os').environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'config': {
            'batch': B, 'q_len': QLEN, 'nq': NQ, 'nkv': NK, 'head_dim': D,
            'global_replay_start': GLOBAL_START, 'global_replay_end': 254,
            'ori_a_len': 255, 'ori_b_len': 128, 'cmp_len': CMP_LEN,
            'cmp_residual': RESID, 'cmp_ratio': RATIO, 'cmp_topk': TOPK,
            'ori_mask_mode': 4, 'ori_win_left': 127, 'ori_win_right': 0,
            'cmp_mask_mode': 3, 'sink': SINK,
            'ori_table_a': [[0, 1]], 'ori_table_b': [[0]], 'cmp_table': [[0]],
            'cmp_visible_count_first': valid_cmp[0], 'cmp_visible_count_last': valid_cmp[-1],
            'q_is_fixed_across_ab': True,
            'synthetic_post_rope_operator_control': True,
        },
        'common_input_sha256': common_hashes,
        'metadata_sha256': {'A': tensor_hash(meta_a), 'B': tensor_hash(meta_b)},
        'metadata_checker_passed': True,
        'A': summarize('full_global_ori', out_a, ref_a),
        'B': summarize('scratch_local_ori', out_b, ref_b),
    }
    print(json.dumps(result, indent=2), flush=True)
    OUT_FULL.write_text(json.dumps(result, indent=2) + '\n')

    # Parent-requested acceptance checks. If the CANN checker or operator fails,
    # let the original exception propagate; do not relax constraints.
    assert not result['A']['nan'] and not result['B']['nan']
    assert result['A']['max_abs_vs_cpu'] <= 1e-2
    assert result['B']['max_abs_vs_cpu'] <= 1e-2
    assert abs(result['A']['row0_dim1']) <= 1e-2
    assert abs(result['B']['row0_dim1']) <= 1e-2
    assert result['A']['row127_dim1'] > 0.3
    assert result['B']['row127_dim1'] > 0.3
    assert abs(result['A']['row127_dim1'] - result['B']['row127_dim1']) <= 1e-2
except Exception as exc:
    # Keep an explicit traceback in stdout/stderr so a checker rejection is recorded.
    print(f'SMLA_ORI_VIEW_AB_ERROR: {type(exc).__name__}: {exc}', flush=True)
    raise
