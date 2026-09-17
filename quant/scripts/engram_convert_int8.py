#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Engram FP8 -> INT8 (group32, power-of-two fp32 scale) 离线转换。

与运行时 vllm_ascend/models/deepseek_v41/engram_hbm.py::quantize_engram_rows 语义一致：
    grouped = rows.float().unflatten(-1, (-1, 32))
    maximum = grouped.abs().amax(-1, keepdim=True)
    scale   = where(max==0, 1, max/127)
    exponent= ceil(log2(scale)); scale = ldexp(1, exponent)     # 二次幂 scale（避免 NPU 舍入差异）
    codes   = round(grouped/scale).clamp(-127,127).to(int8)

输入（官方 checkpoint）:
    <key>.weight  : FP8 E4M3  [rows, 256]
    <key>.scale   : E8M0/FP32 [rows, 8]   (每 32 通道一个)
输出:
    <key>.weight  : INT8      [rows, 256]
    <key>.scale   : FP32      [rows, 8]

用法（分块流式，内存可控）:
    python engram_convert_int8.py --src <原始目录> --dst <输出目录> --keys layers.1.engram.embed layers.14.engram.embed --chunk 65536
"""
import argparse, json, os
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def scale_to_float(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float32:
        return scale
    try:
        return scale.to(torch.float32)
    except (RuntimeError, TypeError):
        raw = scale.view(torch.uint8).to(torch.int32) - 127
        return torch.ldexp(torch.ones_like(raw, dtype=torch.float32), raw)


def quantize_rows(rows: torch.Tensor):
    """rows: [n, 256] bf16/fp32 -> (int8 codes [n,256], fp32 scale [n,8])"""
    grouped = rows.float().unflatten(-1, (-1, 32))
    maximum = grouped.abs().amax(-1, keepdim=True)
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127.0)
    exponent = torch.ceil(torch.log2(scale))
    scale = torch.where(torch.isfinite(exponent),
                        torch.ldexp(torch.ones_like(scale), exponent.int()), scale)
    codes = torch.round(grouped / scale).clamp(-127, 127).to(torch.int8).flatten(-2)
    return codes, scale.squeeze(-1).contiguous()


def convert_key(src_dir, dst_dir, key, chunk=65536):
    root = os.path.dirname(src_dir.rstrip('/'))
    index_path = os.path.join(src_dir, 'model.safetensors.index.json')
    if not os.path.exists(index_path):
        index_path = os.path.join(src_dir, 'quant_model_weights.safetensors.index.json')
    wm = json.load(open(index_path))['weight_map']
    w_file, s_file = wm[key + '.weight'], wm[key + '.scale']
    out_w = {}
    out_s = {}
    with safe_open(os.path.join(src_dir, w_file), framework='pt', device='cpu') as fw, \
         safe_open(os.path.join(src_dir, s_file), framework='pt', device='cpu') as fs:
        tw, ts = fw.get_slice(key + '.weight'), fs.get_slice(key + '.scale')
        rows, width = tw.get_shape()
        assert width == 256, width
        out_w[key + '.weight'] = torch.empty((rows, width), dtype=torch.int8)
        out_s[key + '.scale'] = torch.empty((rows, width // 32), dtype=torch.float32)
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            w_src = tw[start:stop]
            s_src = scale_to_float(ts[start:stop])
            decoded = w_src.float().reshape(-1, width // 32, 32) * s_src.unsqueeze(-1)
            decoded = decoded.reshape(-1, width)
            codes, scale = quantize_rows(decoded)
            out_w[key + '.weight'][start:stop] = codes
            out_s[key + '.scale'][start:stop] = scale
            print(f'  {key}: {stop}/{rows} rows', flush=True)
    os.makedirs(dst_dir, exist_ok=True)
    name = key.replace('.', '_')
    save_file({key + '.weight': out_w[key + '.weight']}, os.path.join(dst_dir, f'{name}.weight.safetensors'))
    save_file({key + '.scale': out_s[key + '.scale']}, os.path.join(dst_dir, f'{name}.scale.safetensors'))
    print(f'[ok] {key} -> {dst_dir}/{name}.{{weight,scale}}.safetensors')


def self_test():
    """小规模自测：随机数据 → 量化 → 反量化，误差应 < 1/127 * max"""
    torch.manual_seed(0)
    rows = torch.randn(64, 256) * 3
    codes, scale = quantize_rows(rows)
    deq = codes.float().reshape(-1, 8, 32) * scale.unsqueeze(-1)
    deq = deq.reshape(-1, 256)
    err = (deq - rows).abs().max().item()
    print(f'[self-test] max abs err = {err:.6f} (scale 分辨率上限 ~{rows.abs().max()/127:.6f})')
    assert err < rows.abs().max() / 100, 'quantization error too large'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src'); ap.add_argument('--dst')
    ap.add_argument('--keys', nargs='*', default=['layers.1.engram.embed', 'layers.14.engram.embed'])
    ap.add_argument('--chunk', type=int, default=65536)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test or not a.src:
        self_test()
    else:
        for k in a.keys:
            convert_key(a.src, a.dst, k, a.chunk)
