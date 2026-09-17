# -*- coding: UTF-8 -*-
"""检查 W4A8 量化产物：格式标签分布 / 权重索引 / quarot / config。"""
import argparse, collections, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    a = ap.parse_args()
    d = a.dir
    print('== 目录 ==', d)
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        print(f'   {os.path.getsize(p)/1e9:8.3f} GB  {f}')

    desc_p = os.path.join(d, 'quant_model_description.json')
    if os.path.exists(desc_p):
        desc = json.load(open(desc_p))
        c = collections.Counter(v for v in desc.values() if isinstance(v, str))
        print('\n== quant_model_description.json ==')
        for k, v in c.most_common(10):
            print(f'   {k:20s} {v}')
        for probe in ['layers.0.ffn.experts.0.w1.weight', 'layers.2.attn.wq_a.weight',
                      'layers.2.ffn.shared_experts.w1.weight', 'embed.weight', 'head.weight',
                      'layers.0.attn.wq_a.weight_scale', 'layers.0.ffn.experts.0.w1.weight_scale']:
            if probe in desc:
                print(f'   {probe} -> {desc[probe]}')
    idx_p = os.path.join(d, 'quant_model_weights.safetensors.index.json')
    if os.path.exists(idx_p):
        wm = json.load(open(idx_p))['weight_map']
        print('\n== 权重索引 == tensors:', len(wm), ' shards:', len(set(wm.values())))
    else:
        print('\n== 权重索引 == 缺失')
    q = os.path.join(d, 'optional', 'quarot.safetensors')
    print('\n== optional/quarot.safetensors ==', '存在' if os.path.exists(q) else '缺失')
    cfg_p = os.path.join(d, 'config.json')
    if os.path.exists(cfg_p):
        cfg = json.load(open(cfg_p))
        print('== config.quantization_config ==', cfg.get('quantization_config'))
        print('== config.model_type ==', cfg.get('model_type'))


if __name__ == '__main__':
    main()
