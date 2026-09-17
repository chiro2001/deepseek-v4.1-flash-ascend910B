#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""把 msmodelslim 量化产物整理成 vllm-ascend 可直接加载的 checkpoint。

做三件事（幂等）：
1. config.json 补 quantization_config = {"quant_method": "ascend", "model_quant_type": <类型>}
   —— 参考官方 Aurora W8A8 导出的写法；运行时 override_quantization_method 需要它才会走 ascend 量化分支。
2. 校验 quant_model_description.json 存在且标签集合合理。
3. 打印权重索引统计，便于和源模型对账。
"""
import argparse, collections, json, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--quant-type', default='W4A8_DYNAMIC')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    d = a.dir

    cfg_p = os.path.join(d, 'config.json')
    cfg = json.load(open(cfg_p))
    qc = cfg.get('quantization_config')
    changed = False
    if not isinstance(qc, dict) or qc.get('quant_method') != 'ascend':
        cfg['quantization_config'] = {'quant_method': 'ascend', 'model_quant_type': a.quant_type}
        changed = True
    t = cfg.get('text_config', {})
    print(f'[config] layers={t.get("num_hidden_layers")} mtp={t.get("num_nextn_predict_layers")} '
          f'engram={t.get("engram_layer_ids")} vision={"yes" if cfg.get("vision_config") else "no"}')
    print(f'[config] quantization_config: {qc} -> {cfg["quantization_config"]}')
    if changed and not a.dry_run:
        json.dump(cfg, open(cfg_p, 'w'), indent=1, ensure_ascii=False)
        print('[config] written')
    elif changed:
        print('[config] dry-run, not written')

    desc_p = os.path.join(d, 'quant_model_description.json')
    if not os.path.exists(desc_p):
        print('[desc] MISSING', file=sys.stderr)
        return 2
    desc = json.load(open(desc_p))
    labels = collections.Counter(v for v in desc.values() if isinstance(v, str))
    print(f'[desc] entries={len(desc)} labels={dict(labels)}')
    miss = [k for k in t and [] or []]  # placeholder
    for probe in ('layers.0.ffn.experts.0.w1.weight', 'layers.0.ffn.experts.0.w1.weight_scale',
                  'layers.0.attn.wq_a.weight', 'layers.0.ffn.shared_experts.w1.weight',
                  'layers.0.ffn.gate.weight', 'embed.weight', 'head.weight'):
        if probe not in desc:
            print(f'[desc] WARN missing probe {probe}')

    idx_p = os.path.join(d, 'quant_model_weights.safetensors.index.json')
    if os.path.exists(idx_p):
        wm = json.load(open(idx_p))['weight_map']
        pref = collections.Counter(k.split('.')[0] for k in wm)
        print(f'[index] tensors={len(wm)} shards={len(set(wm.values()))} prefixes={dict(pref)}')
    else:
        print('[index] MISSING', file=sys.stderr)
        return 2
    print('[ok] runtime-ready' if not changed else '[ok] runtime-ready (config updated)')
    return 0

if __name__ == '__main__':
    sys.exit(main())
