#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""把官方 vision/aligner/image_* 张量合进量化产物（FLOAT/BF16），使多模态可用。

产物侧只需 266 个张量 / ~0.97GB，因此直接写成单独分片 vision-00001-of-00001.safetensors。
"""
import argparse, json, os, sys
from safetensors import safe_open
from safetensors.torch import save_file

PREFIXES = ('vision.', 'aligner.', 'image_')
NAME_MAP = {'image_start': 'image_start', 'image_end': 'image_end', 'image_newline': 'image_newline'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--official-dir', required=True)
    ap.add_argument('--target-dir', required=True, help='量化/dspark 产物目录')
    ap.add_argument('--shard-name', default='vision-00001-of-00001.safetensors')
    a = ap.parse_args()

    off_idx = json.load(open(os.path.join(a.official_dir, 'model.safetensors.index.json')))['weight_map']
    keys = sorted(k for k in off_idx if k.startswith(PREFIXES))
    if not keys:
        print('[error] 官方目录里没有 vision/aligner 张量'); return 2
    idx_path = os.path.join(a.target_dir, 'quant_model_weights.safetensors.index.json')
    desc_path = os.path.join(a.target_dir, 'quant_model_description.json')
    idx = json.load(open(idx_path)); desc = json.load(open(desc_path))

    out = {}
    total = 0
    by_shard = {}
    for k in keys:
        by_shard.setdefault(off_idx[k], []).append(k)
    for shard, names in sorted(by_shard.items()):
        with safe_open(os.path.join(a.official_dir, shard), framework='pt', device='cpu') as f:
            for name in names:
                t = f.get_tensor(name).contiguous()
                out[name] = t
                total += t.numel() * t.element_size()
        print(f'[info] {shard}: {len(names)} tensors')

    shard_path = os.path.join(a.target_dir, a.shard_name)
    save_file(out, shard_path)
    print(f'[ok] wrote {shard_path} ({os.path.getsize(shard_path)/1e9:.3f} GB, {len(out)} tensors)')

    for name in out:
        idx['weight_map'][name] = a.shard_name
        desc[name] = 'FLOAT'
    json.dump(idx, open(idx_path, 'w'), indent=1)
    json.dump(desc, open(desc_path, 'w'), indent=1)
    print(f'[ok] index -> {len(idx["weight_map"])} keys, description -> {len(desc)} entries')

    with safe_open(shard_path, framework='pt', device='cpu') as f:
        got = set(f.keys())
    assert got == set(out), (len(got), len(out))
    print('[ok] verify passed')
    return 0

if __name__ == '__main__':
    sys.exit(main())
