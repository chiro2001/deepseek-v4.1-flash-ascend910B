#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""装配"可量化/可运行"的 V4.1 模型目录（软链 + 可裁剪配置）。

用法:
  # 全量（Engram 关、MTP 关，阶段一）
  python make_full_model.py --src <官方目录> --dst <目标目录> --layers 40 --mtp 0 --engram off
  # 全量 + MTP（阶段三）
  python make_full_model.py --src ... --dst ... --layers 40 --mtp 3 --engram off
  # partial N 层（测试）
  python make_full_model.py --src ... --dst ... --layers 7 --mtp 0 --engram off
"""
import argparse, json, os, shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--layers', type=int, default=40)
    ap.add_argument('--mtp', type=int, default=0, help='num_nextn_predict_layers（0=关 MTP）')
    ap.add_argument('--engram', choices=['on', 'off'], default='off')
    ap.add_argument('--vision', choices=['keep', 'drop'], default='drop')
    ap.add_argument('--calib-max-seq-len', type=int, default=8192)
    a = ap.parse_args()
    # 软链必须是绝对路径：相对软链按'软链所在目录'解析，会指向不存在的目标（踩过坑）。
    a.src = os.path.abspath(a.src)
    a.dst = os.path.abspath(a.dst)

    os.makedirs(a.dst, exist_ok=True)
    cfg = json.load(open(os.path.join(a.src, 'config.json')))
    t = cfg.get('text_config', cfg)
    t['num_hidden_layers'] = a.layers
    t['num_nextn_predict_layers'] = a.mtp
    if a.engram == 'off':
        t['engram_layer_ids'] = []
    t['max_position_embeddings'] = min(int(t.get('max_position_embeddings', 1048576)), a.calib_max_seq_len)
    cfg['text_config'] = t
    json.dump(cfg, open(os.path.join(a.dst, 'config.json'), 'w'), indent=1)

    # 过滤 index：只保留需要的层与顶层权重
    wm = json.load(open(os.path.join(a.src, 'model.safetensors.index.json')))['weight_map']
    def keep(k):
        p = k.split('.')
        if p[0] == 'layers' and p[1].isdigit():
            i = int(p[1])
            if i >= a.layers:
                return False
            if 'engram' in k and a.engram == 'off':
                return False
            return True
        if p[0] == 'mtp':
            return a.mtp > 0 and p[1].isdigit() and int(p[1]) < a.mtp
        if p[0] in ('vision', 'aligner'):
            return a.vision == 'keep'  # 纯文本量化阶段丢弃视觉塔权重
        if p[0].startswith('image_'):
            return True  # image_start/end/newline 是标量，config 需要
        return True
    new = {k: v for k, v in wm.items() if keep(k)}
    json.dump({'metadata': {'total_size': 0}, 'weight_map': new},
              open(os.path.join(a.dst, 'model.safetensors.index.json'), 'w'), indent=1)

    # 软链分片（只链被引用的）
    for shard in sorted(set(new.values())):
        s, d = os.path.join(a.src, shard), os.path.join(a.dst, shard)
        if not os.path.exists(d):
            os.symlink(s, d)
    for f in ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'configuration.json']:
        s, d = os.path.join(a.src, f), os.path.join(a.dst, f)
        if os.path.exists(s) and not os.path.exists(d):
            os.symlink(s, d)
    print(f'[ok] {a.dst}: layers={a.layers} mtp={a.mtp} engram={a.engram} vision={a.vision} tensors={len(new)} shards={len(set(new.values()))}')

if __name__ == '__main__':
    main()
