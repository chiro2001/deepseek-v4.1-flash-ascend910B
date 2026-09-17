#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""给 DSpark checkpoint 补两个合成词表张量（官方 Aurora 配方的做法）。

draft 在**未旋转**基下工作，因此需要自己的 embed_tokens / lm_head：
    mtp.0.embed.weight  <- 官方原始 embed.weight（未旋转）
    mtp.{last}.head.weight <- 官方原始 head.weight（未旋转）
缺这两个时，运行时会把已 QuaRot 旋转的主干 embed/head 共享给 draft，
导致 draft 的输入嵌入与输出 logits 基不一致 → 接受率≈1.0。
"""
import argparse, json, os, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--official-dir', required=True)
    ap.add_argument('--dspark-dir', required=True)
    ap.add_argument('--shard-name', default='mtp-vocab-00001-of-00001.safetensors')
    a = ap.parse_args()

    off_idx = json.load(open(os.path.join(a.official_dir, 'model.safetensors.index.json')))['weight_map']
    idx_path = os.path.join(a.dspark_dir, 'quant_model_weights.safetensors.index.json')
    quant_idx = json.load(open(idx_path))
    desc_path = os.path.join(a.dspark_dir, 'quant_model_description.json')
    desc = json.load(open(desc_path))
    cfg = json.load(open(os.path.join(a.dspark_dir, 'config.json')))
    n_mtp = int(cfg.get('text_config', {}).get('num_nextn_predict_layers', 0))
    assert n_mtp == 3, f'num_nextn_predict_layers={n_mtp}'
    last = n_mtp - 1

    plan = [('embed.weight', f'mtp.0.embed.weight'), ('head.weight', f'mtp.{last}.head.weight')]
    out = {}
    for src, dst in plan:
        shard = off_idx[src]
        with safe_open(os.path.join(a.official_dir, shard), framework='pt', device='cpu') as f:
            t = f.get_tensor(src)
        assert t.dtype == torch.bfloat16, (src, t.dtype)
        out[dst] = t.contiguous()
        print(f'[info] {dst} <- {src} {tuple(t.shape)} {t.dtype}'
              f'  (max|w|={t.float().abs().max().item():.4g})')

    shard_path = os.path.join(a.dspark_dir, a.shard_name)
    save_file(out, shard_path)
    print(f'[ok] wrote {shard_path} ({os.path.getsize(shard_path)/1e9:.2f} GB)')

    for _, dst in plan:
        quant_idx['weight_map'][dst] = a.shard_name
        desc[dst] = 'FLOAT'
    json.dump(quant_idx, open(idx_path, 'w'), indent=1)
    json.dump(desc, open(desc_path, 'w'), indent=1)
    print(f'[ok] index keys -> {len(quant_idx["weight_map"])}, description entries -> {len(desc)}')

    # 校验：新键可被正式 safetensors 读回
    with safe_open(shard_path, framework='pt', device='cpu') as f:
        names = set(f.keys())
    missing = {d for _, d in plan} - names
    assert not missing, missing
    assert all(k in desc for _, k in plan)
    print('[ok] verify passed')

if __name__ == '__main__':
    sys.exit(main())
