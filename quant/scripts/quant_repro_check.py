#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quant_repro_check.py — CPU/offline validator for the DeepSeek-V4.1-Flash W4A8
msmodelslim product.

Reads a quantized product directory (trunk and/or DSpark runtime dir, optionally
with the official BF16 vision/aligner overlay) and prints/writes a JSON manifest:
tensor totals, label distribution, representative tensor key -> shape/dtype,
rotation attachment and quant_model_description.json fields.  PASS/FAIL is
decided against an embedded "known good" baseline generated from

    models/out/v41-w4a8-dspark-vision-ref
    = v41-w4a8-dspark + add_vision_weights.py (266 official vision/aligner/image_*
      tensors, 970,536,960 bytes, byte-identical to the official checkpoint)

Default policy (strict):
  * W4A8 trunk recipe (40 layers, 384 routed experts, W8A8 attention/shared
    experts, everything else FLOAT) must match the baseline exactly;
  * the 266 vision/aligner/image_* FLOAT tensors MUST be present and match the
    official manifest (keys/shapes/dtypes/bytes) — missing vision is a FAIL;
  * optional/quarot.safetensors must expose global_rotation F32 [5120,5120];
  * every shard referenced by the index must exist and contain the mapped keys;
  * MTP tensors, when present, must stay FLOAT (our deliberate deviation from
    the official Aurora recipe; see docs/REPRO_W4A8_QUANT.md §9).

Use --allow-missing-vision for the historical text-only trunk product
(models/out/v41-w4a8-stage1) or for a text-only staging checkpoint; the verdict
can then be PASS but the manifest marks VISION_MISSING as a warning.

This script only reads headers/JSON and is safe to run on CPU.  Product dirs
created inside the container may be root-only (0600); run this script with
`sudo -n python3` if needed.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import struct
import sys

BASELINE = json.loads(r"""{"patterns_ref":{"aligner.w1.bias":{"FLOAT":1},"aligner.w1.weight":{"FLOAT":1},"aligner.w2.bias":{"FLOAT":1},"aligner.w2.weight":{"FLOAT":1},"embed.weight":{"FLOAT":1},"head.weight":{"FLOAT":1},"image_end":{"FLOAT":1},"image_newline":{"FLOAT":1},"image_start":{"FLOAT":1},"layers.N.attn.attn_sink":{"FLOAT":40},"layers.N.attn.compressor.norm.weight":{"FLOAT":4},"layers.N.attn.compressor.wgate.weight":{"FLOAT":3},"layers.N.attn.compressor.wkv.weight":{"FLOAT":4},"layers.N.attn.indexer.k_norm.weight":{"FLOAT":4},"layers.N.attn.indexer.weights_proj.weight":{"FLOAT":8},"layers.N.attn.indexer.wk.weight":{"FLOAT":4},"layers.N.attn.indexer.wq_b.weight":{"W8A8_DYNAMIC":8},"layers.N.attn.indexer.wq_b.weight_offset":{"W8A8_DYNAMIC":8},"layers.N.attn.indexer.wq_b.weight_scale":{"W8A8_DYNAMIC":8},"layers.N.attn.kv_norm.weight":{"FLOAT":40},"layers.N.attn.q_norm.weight":{"FLOAT":40},"layers.N.attn.wkv.weight":{"W8A8_DYNAMIC":40},"layers.N.attn.wkv.weight_offset":{"W8A8_DYNAMIC":40},"layers.N.attn.wkv.weight_scale":{"W8A8_DYNAMIC":40},"layers.N.attn.wo_a.weight":{"FLOAT":40},"layers.N.attn.wo_b.weight":{"FLOAT":40},"layers.N.attn.wq_a.weight":{"W8A8_DYNAMIC":40},"layers.N.attn.wq_a.weight_offset":{"W8A8_DYNAMIC":40},"layers.N.attn.wq_a.weight_scale":{"W8A8_DYNAMIC":40},"layers.N.attn.wq_b.weight":{"W8A8_DYNAMIC":40},"layers.N.attn.wq_b.weight_offset":{"W8A8_DYNAMIC":40},"layers.N.attn.wq_b.weight_scale":{"W8A8_DYNAMIC":40},"layers.N.attn_norm.weight":{"FLOAT":40},"layers.N.ffn.experts.N.w1.scale_bias":{"W4A8_DYNAMIC":15360},"layers.N.ffn.experts.N.w1.weight":{"W4A8_DYNAMIC":15360},"layers.N.ffn.experts.N.w1.weight_offset":{"W4A8_DYNAMIC":15360},"layers.N.ffn.experts.N.w1.weight_scale":{"W4A8_DYNAMIC":15360},"layers.N.ffn.experts.N.w2.scale_bias":{"W4A8_DYNAMIC":15360},"layers.N.ffn.experts.N.w2.weight":{"W4A8_DYNAMIC":15360},"layers.N.ffn.experts.N.w2.weight_offset":{"W4A8_DYNAMIC":15360},"layers.N.ffn.experts.N.w2.weight_scale":{"W4A8_DYNAMIC":15360},"layers.N.ffn.experts.N.w3.scale_bias":{"W4A8_DYNAMIC":15360},"layers.N.ffn.experts.N.w3.weight":{"W4A8_DYNAMIC":15360},"layers.N.ffn.experts.N.w3.weight_offset":{"W4A8_DYNAMIC":15360},"layers.N.ffn.experts.N.w3.weight_scale":{"W4A8_DYNAMIC":15360},"layers.N.ffn.gate.bias":{"FLOAT":40},"layers.N.ffn.gate.bias_vl":{"FLOAT":40},"layers.N.ffn.gate.weight":{"FLOAT":40},"layers.N.ffn.shared_experts.w1.weight":{"W8A8_DYNAMIC":40},"layers.N.ffn.shared_experts.w1.weight_offset":{"W8A8_DYNAMIC":40},"layers.N.ffn.shared_experts.w1.weight_scale":{"W8A8_DYNAMIC":40},"layers.N.ffn.shared_experts.w2.weight":{"W8A8_DYNAMIC":40},"layers.N.ffn.shared_experts.w2.weight_offset":{"W8A8_DYNAMIC":40},"layers.N.ffn.shared_experts.w2.weight_scale":{"W8A8_DYNAMIC":40},"layers.N.ffn.shared_experts.w3.weight":{"W8A8_DYNAMIC":40},"layers.N.ffn.shared_experts.w3.weight_offset":{"W8A8_DYNAMIC":40},"layers.N.ffn.shared_experts.w3.weight_scale":{"W8A8_DYNAMIC":40},"layers.N.ffn_norm.weight":{"FLOAT":40},"layers.N.hc_attn_base":{"FLOAT":40},"layers.N.hc_attn_fn":{"FLOAT":40},"layers.N.hc_attn_scale":{"FLOAT":40},"layers.N.hc_ffn_base":{"FLOAT":40},"layers.N.hc_ffn_fn":{"FLOAT":40},"layers.N.hc_ffn_scale":{"FLOAT":40},"mtp.N.attn.attn_sink":{"FLOAT":3},"mtp.N.attn.kv_norm.weight":{"FLOAT":3},"mtp.N.attn.q_norm.weight":{"FLOAT":3},"mtp.N.attn.wkv.weight":{"FLOAT":3},"mtp.N.attn.wo_a.weight":{"FLOAT":3},"mtp.N.attn.wo_b.weight":{"FLOAT":3},"mtp.N.attn.wq_a.weight":{"FLOAT":3},"mtp.N.attn.wq_b.weight":{"FLOAT":3},"mtp.N.attn_norm.weight":{"FLOAT":3},"mtp.N.confidence_head.proj.weight":{"FLOAT":1},"mtp.N.embed.weight":{"FLOAT":1},"mtp.N.ffn.experts.N.w1.weight":{"FLOAT":384},"mtp.N.ffn.experts.N.w2.weight":{"FLOAT":384},"mtp.N.ffn.experts.N.w3.weight":{"FLOAT":384},"mtp.N.ffn.gate.bias":{"FLOAT":3},"mtp.N.ffn.gate.bias_vl":{"FLOAT":3},"mtp.N.ffn.gate.weight":{"FLOAT":3},"mtp.N.ffn.shared_experts.w1.weight":{"FLOAT":3},"mtp.N.ffn.shared_experts.w2.weight":{"FLOAT":3},"mtp.N.ffn.shared_experts.w3.weight":{"FLOAT":3},"mtp.N.ffn_norm.weight":{"FLOAT":3},"mtp.N.hc_attn_base":{"FLOAT":3},"mtp.N.hc_attn_fn":{"FLOAT":3},"mtp.N.hc_attn_scale":{"FLOAT":3},"mtp.N.hc_ffn_base":{"FLOAT":3},"mtp.N.hc_ffn_fn":{"FLOAT":3},"mtp.N.hc_ffn_scale":{"FLOAT":3},"mtp.N.head.weight":{"FLOAT":1},"mtp.N.main_norm.weight":{"FLOAT":1},"mtp.N.main_proj.weight":{"FLOAT":1},"mtp.N.markov_head.embed.weight":{"FLOAT":1},"mtp.N.markov_head.head.weight":{"FLOAT":1},"mtp.N.norm.weight":{"FLOAT":1},"norm.weight":{"FLOAT":1},"vision.blocks.N.attn.wo.bias":{"FLOAT":32},"vision.blocks.N.attn.wo.weight":{"FLOAT":32},"vision.blocks.N.attn.wqkv.bias":{"FLOAT":32},"vision.blocks.N.attn.wqkv.weight":{"FLOAT":32},"vision.blocks.N.mlp.w1.weight":{"FLOAT":32},"vision.blocks.N.mlp.w2.weight":{"FLOAT":32},"vision.blocks.N.norm1.weight":{"FLOAT":32},"vision.blocks.N.norm2.weight":{"FLOAT":32},"vision.norm.weight":{"FLOAT":1},"vision.patch_embed.proj.bias":{"FLOAT":1},"vision.patch_embed.proj.weight":{"FLOAT":1}},"reference":"models/out/v41-w4a8-dspark-vision-ref (DSpark W4A8 trunk + official BF16 vision overlay, MTP FLOAT)","reps":{"aligner.w1.bias":{"dtype":"BF16","label":"FLOAT","pattern":"aligner.w1.bias","shape":[5120]},"aligner.w1.weight":{"dtype":"BF16","label":"FLOAT","pattern":"aligner.w1.weight","shape":[5120,9216]},"aligner.w2.weight":{"dtype":"BF16","label":"FLOAT","pattern":"aligner.w2.weight","shape":[5120,5120]},"embed.weight":{"dtype":"BF16","label":"FLOAT","pattern":"embed.weight","shape":[129280,5120]},"head.weight":{"dtype":"F32","label":"FLOAT","pattern":"head.weight","shape":[129280,5120]},"image_start":{"dtype":"BF16","label":"FLOAT","pattern":"image_start","shape":[5120]},"layers.0.attn.attn_sink":{"dtype":"F32","label":"FLOAT","pattern":"layers.N.attn.attn_sink","shape":[64]},"layers.0.attn.kv_norm.weight":{"dtype":"F32","label":"FLOAT","pattern":"layers.N.attn.kv_norm.weight","shape":[512]},"layers.0.attn.q_norm.weight":{"dtype":"F32","label":"FLOAT","pattern":"layers.N.attn.q_norm.weight","shape":[1280]},"layers.0.attn.wkv.weight":{"dtype":"I8","label":"W8A8_DYNAMIC","pattern":"layers.N.attn.wkv.weight","shape":[512,5120]},"layers.0.attn.wkv.weight_offset":{"dtype":"F32","label":"W8A8_DYNAMIC","pattern":"layers.N.attn.wkv.weight_offset","shape":[512,1]},"layers.0.attn.wkv.weight_scale":{"dtype":"F32","label":"W8A8_DYNAMIC","pattern":"layers.N.attn.wkv.weight_scale","shape":[512,1]},"layers.0.attn.wo_a.weight":{"dtype":"BF16","label":"FLOAT","pattern":"layers.N.attn.wo_a.weight","shape":[8192,4096]},"layers.0.attn.wo_b.weight":{"dtype":"BF16","label":"FLOAT","pattern":"layers.N.attn.wo_b.weight","shape":[5120,8192]},"layers.0.attn.wq_a.weight":{"dtype":"I8","label":"W8A8_DYNAMIC","pattern":"layers.N.attn.wq_a.weight","shape":[1280,5120]},"layers.0.attn.wq_a.weight_offset":{"dtype":"F32","label":"W8A8_DYNAMIC","pattern":"layers.N.attn.wq_a.weight_offset","shape":[1280,1]},"layers.0.attn.wq_a.weight_scale":{"dtype":"F32","label":"W8A8_DYNAMIC","pattern":"layers.N.attn.wq_a.weight_scale","shape":[1280,1]},"layers.0.attn.wq_b.weight":{"dtype":"I8","label":"W8A8_DYNAMIC","pattern":"layers.N.attn.wq_b.weight","shape":[32768,1280]},"layers.0.attn.wq_b.weight_offset":{"dtype":"F32","label":"W8A8_DYNAMIC","pattern":"layers.N.attn.wq_b.weight_offset","shape":[32768,1]},"layers.0.attn.wq_b.weight_scale":{"dtype":"F32","label":"W8A8_DYNAMIC","pattern":"layers.N.attn.wq_b.weight_scale","shape":[32768,1]},"layers.0.attn_norm.weight":{"dtype":"F32","label":"FLOAT","pattern":"layers.N.attn_norm.weight","shape":[5120]},"layers.0.ffn.experts.0.w1.scale_bias":{"dtype":"F32","label":"W4A8_DYNAMIC","pattern":"layers.N.ffn.experts.N.w1.scale_bias","shape":[2304,1]},"layers.0.ffn.experts.0.w1.weight":{"dtype":"I8","label":"W4A8_DYNAMIC","pattern":"layers.N.ffn.experts.N.w1.weight","shape":[1152,5120]},"layers.0.ffn.experts.0.w1.weight_offset":{"dtype":"F32","label":"W4A8_DYNAMIC","pattern":"layers.N.ffn.experts.N.w1.weight_offset","shape":[2304,1]},"layers.0.ffn.experts.0.w1.weight_scale":{"dtype":"F32","label":"W4A8_DYNAMIC","pattern":"layers.N.ffn.experts.N.w1.weight_scale","shape":[2304,1]},"layers.0.ffn.experts.0.w2.scale_bias":{"dtype":"F32","label":"W4A8_DYNAMIC","pattern":"layers.N.ffn.experts.N.w2.scale_bias","shape":[5120,16]},"layers.0.ffn.experts.0.w2.weight":{"dtype":"I8","label":"W4A8_DYNAMIC","pattern":"layers.N.ffn.experts.N.w2.weight","shape":[2560,2304]},"layers.0.ffn.experts.0.w2.weight_offset":{"dtype":"F32","label":"W4A8_DYNAMIC","pattern":"layers.N.ffn.experts.N.w2.weight_offset","shape":[5120,1]},"layers.0.ffn.experts.0.w2.weight_scale":{"dtype":"F32","label":"W4A8_DYNAMIC","pattern":"layers.N.ffn.experts.N.w2.weight_scale","shape":[5120,1]},"layers.0.ffn.experts.0.w3.scale_bias":{"dtype":"F32","label":"W4A8_DYNAMIC","pattern":"layers.N.ffn.experts.N.w3.scale_bias","shape":[2304,1]},"layers.0.ffn.experts.0.w3.weight":{"dtype":"I8","label":"W4A8_DYNAMIC","pattern":"layers.N.ffn.experts.N.w3.weight","shape":[1152,5120]},"layers.0.ffn.experts.0.w3.weight_offset":{"dtype":"F32","label":"W4A8_DYNAMIC","pattern":"layers.N.ffn.experts.N.w3.weight_offset","shape":[2304,1]},"layers.0.ffn.experts.0.w3.weight_scale":{"dtype":"F32","label":"W4A8_DYNAMIC","pattern":"layers.N.ffn.experts.N.w3.weight_scale","shape":[2304,1]},"layers.0.ffn.gate.bias":{"dtype":"F32","label":"FLOAT","pattern":"layers.N.ffn.gate.bias","shape":[384]},"layers.0.ffn.gate.bias_vl":{"dtype":"F32","label":"FLOAT","pattern":"layers.N.ffn.gate.bias_vl","shape":[384]},"layers.0.ffn.gate.weight":{"dtype":"BF16","label":"FLOAT","pattern":"layers.N.ffn.gate.weight","shape":[384,5120]},"layers.0.ffn.shared_experts.w1.weight":{"dtype":"I8","label":"W8A8_DYNAMIC","pattern":"layers.N.ffn.shared_experts.w1.weight","shape":[2304,5120]},"layers.0.ffn.shared_experts.w1.weight_offset":{"dtype":"F32","label":"W8A8_DYNAMIC","pattern":"layers.N.ffn.shared_experts.w1.weight_offset","shape":[2304,1]},"layers.0.ffn.shared_experts.w1.weight_scale":{"dtype":"F32","label":"W8A8_DYNAMIC","pattern":"layers.N.ffn.shared_experts.w1.weight_scale","shape":[2304,1]},"layers.0.ffn_norm.weight":{"dtype":"F32","label":"FLOAT","pattern":"layers.N.ffn_norm.weight","shape":[5120]},"layers.0.hc_attn_base":{"dtype":"F32","label":"FLOAT","pattern":"layers.N.hc_attn_base","shape":[24]},"layers.0.hc_attn_fn":{"dtype":"F32","label":"FLOAT","pattern":"layers.N.hc_attn_fn","shape":[24,20480]},"layers.0.hc_attn_scale":{"dtype":"F32","label":"FLOAT","pattern":"layers.N.hc_attn_scale","shape":[3]},"layers.0.hc_ffn_fn":{"dtype":"F32","label":"FLOAT","pattern":"layers.N.hc_ffn_fn","shape":[24,20480]},"layers.2.attn.compressor.norm.weight":{"dtype":"F32","label":"FLOAT","pattern":"layers.N.attn.compressor.norm.weight","shape":[512]},"layers.2.attn.compressor.wkv.weight":{"dtype":"F32","label":"FLOAT","pattern":"layers.N.attn.compressor.wkv.weight","shape":[512,5120]},"layers.2.attn.indexer.k_norm.weight":{"dtype":"F32","label":"FLOAT","pattern":"layers.N.attn.indexer.k_norm.weight","shape":[128]},"layers.2.attn.indexer.weights_proj.weight":{"dtype":"BF16","label":"FLOAT","pattern":"layers.N.attn.indexer.weights_proj.weight","shape":[32,5120]},"layers.2.attn.indexer.wk.weight":{"dtype":"BF16","label":"FLOAT","pattern":"layers.N.attn.indexer.wk.weight","shape":[128,512]},"layers.2.attn.indexer.wq_b.weight":{"dtype":"I8","label":"W8A8_DYNAMIC","pattern":"layers.N.attn.indexer.wq_b.weight","shape":[4096,1280]},"layers.2.attn.indexer.wq_b.weight_offset":{"dtype":"F32","label":"W8A8_DYNAMIC","pattern":"layers.N.attn.indexer.wq_b.weight_offset","shape":[4096,1]},"layers.2.attn.indexer.wq_b.weight_scale":{"dtype":"F32","label":"W8A8_DYNAMIC","pattern":"layers.N.attn.indexer.wq_b.weight_scale","shape":[4096,1]},"layers.20.attn.compressor.wkv.weight":{"dtype":"BF16","label":"FLOAT","pattern":"layers.N.attn.compressor.wkv.weight","shape":[512,5120]},"mtp.0.attn.wkv.weight":{"dtype":"BF16","label":"FLOAT","pattern":"mtp.N.attn.wkv.weight","shape":[512,5120]},"mtp.0.attn.wq_a.weight":{"dtype":"BF16","label":"FLOAT","pattern":"mtp.N.attn.wq_a.weight","shape":[1280,5120]},"mtp.0.attn.wq_b.weight":{"dtype":"BF16","label":"FLOAT","pattern":"mtp.N.attn.wq_b.weight","shape":[32768,1280]},"mtp.0.attn_norm.weight":{"dtype":"BF16","label":"FLOAT","pattern":"mtp.N.attn_norm.weight","shape":[5120]},"mtp.0.embed.weight":{"dtype":"BF16","label":"FLOAT","pattern":"mtp.N.embed.weight","shape":[129280,5120]},"mtp.0.ffn.experts.0.w1.weight":{"dtype":"BF16","label":"FLOAT","pattern":"mtp.N.ffn.experts.N.w1.weight","shape":[2304,5120]},"mtp.0.ffn.gate.weight":{"dtype":"BF16","label":"FLOAT","pattern":"mtp.N.ffn.gate.weight","shape":[128,5120]},"mtp.0.ffn.shared_experts.w1.weight":{"dtype":"BF16","label":"FLOAT","pattern":"mtp.N.ffn.shared_experts.w1.weight","shape":[2304,5120]},"mtp.0.ffn_norm.weight":{"dtype":"BF16","label":"FLOAT","pattern":"mtp.N.ffn_norm.weight","shape":[5120]},"mtp.0.main_proj.weight":{"dtype":"BF16","label":"FLOAT","pattern":"mtp.N.main_proj.weight","shape":[5120,15360]},"mtp.2.head.weight":{"dtype":"BF16","label":"FLOAT","pattern":"mtp.N.head.weight","shape":[129280,5120]},"norm.weight":{"dtype":"F32","label":"FLOAT","pattern":"norm.weight","shape":[5120]},"vision.blocks.0.attn.wqkv.weight":{"dtype":"BF16","label":"FLOAT","pattern":"vision.blocks.N.attn.wqkv.weight","shape":[3072,1024]},"vision.blocks.0.mlp.w1.weight":{"dtype":"BF16","label":"FLOAT","pattern":"vision.blocks.N.mlp.w1.weight","shape":[5632,1024]},"vision.norm.weight":{"dtype":"BF16","label":"FLOAT","pattern":"vision.norm.weight","shape":[1024]},"vision.patch_embed.proj.bias":{"dtype":"BF16","label":"FLOAT","pattern":"vision.patch_embed.proj.bias","shape":[1024]},"vision.patch_embed.proj.weight":{"dtype":"BF16","label":"FLOAT","pattern":"vision.patch_embed.proj.weight","shape":[1024,588]}},"schema":"msmodelslim-quant-baseline/1","trunk_float_patterns":{"embed.weight":{"FLOAT":1},"head.weight":{"FLOAT":1},"layers.N.attn.attn_sink":{"FLOAT":40},"layers.N.attn.compressor.norm.weight":{"FLOAT":4},"layers.N.attn.compressor.wgate.weight":{"FLOAT":3},"layers.N.attn.compressor.wkv.weight":{"FLOAT":4},"layers.N.attn.indexer.k_norm.weight":{"FLOAT":4},"layers.N.attn.indexer.weights_proj.weight":{"FLOAT":8},"layers.N.attn.indexer.wk.weight":{"FLOAT":4},"layers.N.attn.kv_norm.weight":{"FLOAT":40},"layers.N.attn.q_norm.weight":{"FLOAT":40},"layers.N.attn.wo_a.weight":{"FLOAT":40},"layers.N.attn.wo_b.weight":{"FLOAT":40},"layers.N.attn_norm.weight":{"FLOAT":40},"layers.N.ffn.gate.bias":{"FLOAT":40},"layers.N.ffn.gate.bias_vl":{"FLOAT":40},"layers.N.ffn.gate.weight":{"FLOAT":40},"layers.N.ffn_norm.weight":{"FLOAT":40},"layers.N.hc_attn_base":{"FLOAT":40},"layers.N.hc_attn_fn":{"FLOAT":40},"layers.N.hc_attn_scale":{"FLOAT":40},"layers.N.hc_ffn_base":{"FLOAT":40},"layers.N.hc_ffn_fn":{"FLOAT":40},"layers.N.hc_ffn_scale":{"FLOAT":40},"norm.weight":{"FLOAT":1}},"vision_count":266,"vision_fingerprint_sha256":"f9f80643bac8dc362b3a144e60d58413f2395abc8fc07a8c8a3e83e423449641","vision_keys":{"aligner.w1.bias":{"dtype":"BF16","shape":[5120]},"aligner.w1.weight":{"dtype":"BF16","shape":[5120,9216]},"aligner.w2.bias":{"dtype":"BF16","shape":[5120]},"aligner.w2.weight":{"dtype":"BF16","shape":[5120,5120]},"image_end":{"dtype":"BF16","shape":[5120]},"image_newline":{"dtype":"BF16","shape":[5120]},"image_start":{"dtype":"BF16","shape":[5120]},"vision.blocks.0.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.0.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.0.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.0.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.0.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.0.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.0.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.0.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.1.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.1.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.1.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.1.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.1.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.1.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.1.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.1.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.10.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.10.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.10.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.10.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.10.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.10.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.10.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.10.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.11.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.11.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.11.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.11.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.11.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.11.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.11.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.11.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.12.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.12.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.12.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.12.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.12.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.12.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.12.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.12.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.13.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.13.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.13.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.13.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.13.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.13.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.13.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.13.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.14.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.14.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.14.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.14.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.14.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.14.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.14.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.14.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.15.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.15.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.15.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.15.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.15.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.15.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.15.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.15.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.16.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.16.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.16.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.16.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.16.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.16.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.16.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.16.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.17.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.17.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.17.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.17.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.17.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.17.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.17.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.17.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.18.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.18.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.18.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.18.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.18.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.18.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.18.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.18.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.19.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.19.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.19.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.19.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.19.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.19.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.19.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.19.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.2.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.2.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.2.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.2.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.2.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.2.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.2.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.2.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.20.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.20.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.20.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.20.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.20.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.20.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.20.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.20.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.21.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.21.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.21.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.21.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.21.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.21.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.21.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.21.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.22.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.22.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.22.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.22.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.22.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.22.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.22.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.22.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.23.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.23.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.23.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.23.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.23.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.23.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.23.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.23.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.24.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.24.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.24.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.24.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.24.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.24.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.24.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.24.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.25.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.25.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.25.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.25.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.25.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.25.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.25.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.25.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.26.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.26.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.26.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.26.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.26.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.26.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.26.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.26.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.27.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.27.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.27.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.27.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.27.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.27.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.27.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.27.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.28.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.28.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.28.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.28.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.28.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.28.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.28.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.28.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.29.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.29.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.29.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.29.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.29.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.29.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.29.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.29.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.3.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.3.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.3.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.3.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.3.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.3.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.3.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.3.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.30.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.30.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.30.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.30.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.30.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.30.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.30.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.30.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.31.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.31.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.31.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.31.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.31.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.31.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.31.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.31.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.4.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.4.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.4.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.4.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.4.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.4.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.4.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.4.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.5.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.5.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.5.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.5.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.5.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.5.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.5.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.5.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.6.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.6.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.6.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.6.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.6.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.6.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.6.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.6.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.7.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.7.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.7.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.7.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.7.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.7.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.7.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.7.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.8.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.8.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.8.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.8.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.8.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.8.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.8.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.8.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.9.attn.wo.bias":{"dtype":"BF16","shape":[1024]},"vision.blocks.9.attn.wo.weight":{"dtype":"BF16","shape":[1024,1024]},"vision.blocks.9.attn.wqkv.bias":{"dtype":"BF16","shape":[3072]},"vision.blocks.9.attn.wqkv.weight":{"dtype":"BF16","shape":[3072,1024]},"vision.blocks.9.mlp.w1.weight":{"dtype":"BF16","shape":[5632,1024]},"vision.blocks.9.mlp.w2.weight":{"dtype":"BF16","shape":[1024,2816]},"vision.blocks.9.norm1.weight":{"dtype":"BF16","shape":[1024]},"vision.blocks.9.norm2.weight":{"dtype":"BF16","shape":[1024]},"vision.norm.weight":{"dtype":"BF16","shape":[1024]},"vision.patch_embed.proj.bias":{"dtype":"BF16","shape":[1024]},"vision.patch_embed.proj.weight":{"dtype":"BF16","shape":[1024,588]}},"vision_prefix_counts":{"aligner":4,"image_end":1,"image_newline":1,"image_start":1,"vision":259},"vision_total_bytes":970536960}""")

DTYPE_NBYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2,
    "I8": 1, "U8": 1, "U16": 2, "U32": 4, "U64": 8, "BOOL": 1, "C64": 8,
    "F8_E4M3": 1, "F8_E4M3FNUZ": 1, "F8_E5M2": 1, "F8_E5M2FNUZ": 1,
    "F8_E8M0": 1,
}
VISION_PREFIXES = ("vision.", "aligner.", "image_")
EXPECTED_LAYERS = 40
EXPECTED_QUANT_CONFIG = {"quant_method": "ascend", "model_quant_type": "W4A8_DYNAMIC"}


def read_st_header(path):
    """Return (header_len, header_dict_without_metadata)."""
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise ValueError("too short to be a safetensors file")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 0 or header_len > (1 << 32):
            raise ValueError("implausible safetensors header length %d" % header_len)
        raw = fh.read(header_len)
        if len(raw) != header_len:
            raise ValueError("truncated safetensors header")
    header = json.loads(raw.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    header.pop("__metadata__", None)
    return header_len, header


def file_size(path):
    return os.path.getsize(path)


def tensor_bytes(spec):
    n = 1
    for d in spec["shape"]:
        n *= int(d)
    return n * DTYPE_NBYTES[spec["dtype"]]


def pattern(key):
    k = re.sub(r"layers\.\d+\.", "layers.N.", key)
    k = re.sub(r"experts\.\d+\.", "experts.N.", k)
    k = re.sub(r"mtp\.\d+\.", "mtp.N.", k)
    k = re.sub(r"vision\.blocks\.\d+\.", "vision.blocks.N.", k)
    return k


def is_vision_key(key):
    return key.startswith(VISION_PREFIXES)


def is_mtp_key(key):
    return key.startswith("mtp.")


class Report(object):
    def __init__(self):
        self.checks = []

    def add(self, cid, status, detail):
        assert status in ("PASS", "WARN", "FAIL")
        self.checks.append({"id": cid, "status": status, "detail": detail})
        return status

    @property
    def failed(self):
        return [c for c in self.checks if c["status"] == "FAIL"]

    @property
    def warned(self):
        return [c for c in self.checks if c["status"] == "WARN"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(
        description="CPU/offline manifest + PASS/FAIL validator for the "
                    "DeepSeek-V4.1-Flash W4A8 product.")
    ap.add_argument("--dir", required=True, help="quantized/DSpark product directory")
    ap.add_argument("--official-dir", default=None,
                    help="official DeepSeek-V4.1-Flash dir; if given, vision keys are "
                         "checked against its model.safetensors.index.json")
    ap.add_argument("--out", default=None, help="write JSON manifest here")
    ap.add_argument("--allow-missing-vision", action="store_true",
                    help="waive the 266-tensor vision requirement (text-only artifact)")
    ap.add_argument("--require-mtp", action="store_true",
                    help="require the 3 MTP draft layers (canonical DSpark product)")
    ap.add_argument("--mtp", choices=["auto", "float", "quantized"], default="auto",
                    help="MTP policy: float=our baseline (default when auto-detected), "
                         "quantized=experimental official-aligned build")
    ap.add_argument("--allow-extra-shards", action="store_true",
                    help="do not fail when the directory has shards not referenced by the index")
    args = ap.parse_args()

    d = os.path.abspath(args.dir)
    rep = Report()
    warnings = []

    if not os.path.isdir(d):
        print("[error] not a directory: %s" % d, file=sys.stderr)
        return 2

    manifest = {
        "schema": "msmodelslim-quant-manifest/1",
        "dir": d,
        "baseline": {
            "name": BASELINE["reference"],
            "vision_count": BASELINE["vision_count"],
            "vision_total_bytes": BASELINE["vision_total_bytes"],
            "vision_fingerprint_sha256": BASELINE["vision_fingerprint_sha256"],
        },
    }

    # ---------------------------------------------------------------- index
    index_path = os.path.join(d, "quant_model_weights.safetensors.index.json")
    if not os.path.isfile(index_path):
        rep.add("index_present", "FAIL", "missing %s" % index_path)
        manifest["verdict"] = "FAIL"
        manifest["checks"] = rep.checks
        print(json.dumps(manifest, indent=1, ensure_ascii=False))
        return 1
    index_doc = load_json(index_path)
    weight_map = index_doc.get("weight_map")
    if not isinstance(weight_map, dict):
        rep.add("index_present", "FAIL", "index has no 'weight_map' object")
        manifest["verdict"] = "FAIL"
        manifest["checks"] = rep.checks
        print(json.dumps(manifest, indent=1, ensure_ascii=False))
        return 1
    rep.add("index_present", "PASS",
            "%d tensors / %d shards" % (len(weight_map), len(set(weight_map.values()))))

    desc_path = os.path.join(d, "quant_model_description.json")
    if not os.path.isfile(desc_path):
        rep.add("description_present", "FAIL", "missing %s" % desc_path)
        desc = {}
    else:
        desc = load_json(desc_path)
        rep.add("description_present", "PASS", "%d entries" % len(desc))

    # label counts are computed over the index keys only: the description also
    # carries metadata strings (version, model_quant_type) and desc-only
    # mtp.*.scale siblings that are intentionally NOT written.
    labels = collections.Counter()
    label_of = {}
    unlabeled = []
    for key in weight_map:
        val = desc.get(key)
        if isinstance(val, str):
            labels[val] += 1
            label_of[key] = val
        else:
            unlabeled.append(key)
    if unlabeled:
        rep.add("description_labels_complete", "FAIL",
                "%d index keys have no string label, e.g. %s" % (len(unlabeled), unlabeled[:5]))
    else:
        rep.add("description_labels_complete", "PASS", "all index keys labelled")

    # profile detection / expectation
    mtp_keys = [k for k in weight_map if is_mtp_key(k)]
    vision_keys_obs = [k for k in weight_map if is_vision_key(k)]
    if args.require_mtp and not mtp_keys:
        rep.add("mtp_present", "FAIL", "no mtp.* tensors in index")
    mtp_mode = args.mtp
    if mtp_mode == "auto":
        mtp_mode = "quantized" if any(
            label_of.get(k) in ("W8A8_DYNAMIC", "W4A8_DYNAMIC") for k in mtp_keys) else "float"
    if mtp_keys and mtp_mode == "quantized":
        bad = sorted(k for k in mtp_keys if label_of.get(k) not in
                     ("W8A8_DYNAMIC", "W4A8_DYNAMIC", "FLOAT"))
        if bad:
            rep.add("mtp_policy", "FAIL", "unexpected mtp label(s): %s" % bad[:5])
        elif not any(label_of.get(k) in ("W8A8_DYNAMIC", "W4A8_DYNAMIC") for k in mtp_keys):
            rep.add("mtp_policy", "FAIL", "mtp labels are all FLOAT")
        else:
            rep.add("mtp_policy", "PASS",
                    "MTP quantized (experimental): %d mtp tensors"
                    % len(mtp_keys))
    elif mtp_keys:
        non_float = sorted(k for k in mtp_keys if label_of.get(k) != "FLOAT")
        if non_float:
            rep.add("mtp_policy", "FAIL",
                    "MTP must stay FLOAT in this recipe; non-FLOAT: %s" % non_float[:5])
        else:
            rep.add("mtp_policy", "PASS", "%d mtp.* tensors are FLOAT" % len(mtp_keys))
    else:
        rep.add("mtp_policy", "PASS", "no MTP layers (trunk-only product)")

    vision_required = not args.allow_missing_vision
    components = ["trunk"]
    if mtp_keys:
        components.append("mtp")
    if vision_required or vision_keys_obs:
        components.append("vision")

    expected_patterns = collections.defaultdict(collections.Counter)
    for pat, counts in BASELINE["patterns_ref"].items():
        pfx = pat  # baseline keys are already canonical
        if pfx.startswith("mtp.") and "mtp" not in components:
            continue
        if (pfx.startswith("vision.") or pfx.startswith("aligner.")
                or pfx.startswith("image_")) and "vision" not in components:
            continue
        for lab, n in counts.items():
            expected_patterns[pfx][lab] += n
    observed_patterns = collections.defaultdict(collections.Counter)
    for key in weight_map:
        observed_patterns[pattern(key)][label_of.get(key)] += 1

    # ---------------------------------------------------------------- config
    cfg_path = os.path.join(d, "config.json")
    cfg = None
    if not os.path.isfile(cfg_path):
        rep.add("config_present", "FAIL", "missing config.json")
    else:
        cfg = load_json(cfg_path)
        rep.add("config_present", "PASS", "")
        t = cfg.get("text_config", {}) or {}
        qc = cfg.get("quantization_config")
        if qc == EXPECTED_QUANT_CONFIG:
            rep.add("config_quantization_config", "PASS", json.dumps(qc, ensure_ascii=False))
        else:
            rep.add("config_quantization_config", "FAIL",
                    "expected %s, got %s" % (EXPECTED_QUANT_CONFIG, qc))
        if cfg.get("model_type") == "deepseek_v41":
            rep.add("config_model_type", "PASS", "deepseek_v41")
        else:
            rep.add("config_model_type", "FAIL", "model_type=%r" % cfg.get("model_type"))
        arch = cfg.get("architectures") or []
        if "DeepseekV41ForCausalLM" in arch:
            rep.add("config_architectures", "PASS", ",".join(arch))
        else:
            rep.add("config_architectures", "FAIL", "architectures=%r" % arch)
        if t.get("num_hidden_layers") == EXPECTED_LAYERS:
            rep.add("config_layers", "PASS", "%d layers" % EXPECTED_LAYERS)
        else:
            rep.add("config_layers", "FAIL",
                    "num_hidden_layers=%r (expected %d)" % (t.get("num_hidden_layers"), EXPECTED_LAYERS))
        n_mtp = t.get("num_nextn_predict_layers")
        if mtp_keys and n_mtp != 3:
            rep.add("config_mtp_layers", "FAIL",
                    "mtp present but num_nextn_predict_layers=%r (expected 3)" % n_mtp)
        elif not mtp_keys and n_mtp not in (0, None):
            rep.add("config_mtp_layers", "FAIL",
                    "no mtp weights but num_nextn_predict_layers=%r" % n_mtp)
        else:
            rep.add("config_mtp_layers", "PASS", "num_nextn_predict_layers=%r" % n_mtp)
        if vision_required:
            if isinstance(cfg.get("vision_config"), dict):
                rep.add("config_vision_config", "PASS", "vision_config present")
            else:
                rep.add("config_vision_config", "FAIL", "vision_config missing")
        elif "vision" in components:
            rep.add("config_vision_config", "PASS", "vision_config present (waived)")
        mp = t.get("max_position_embeddings")
        if mp != 1048576:
            rep.add("config_max_position_embeddings", "WARN",
                    "max_position_embeddings=%r (official is 1048576; the stage-1 assembly "
                    "truncates it to the calibration length — restore before long-context serving)" % mp)
        else:
            rep.add("config_max_position_embeddings", "PASS", "1048576")
        eng = t.get("engram_layer_ids")
        if eng not in ([], None):
            rep.add("config_engram", "WARN",
                    "engram_layer_ids=%r (this recipe runs stage-1 with Engram off)" % eng)
        else:
            rep.add("config_engram", "PASS", "engram off")
    manifest["config"] = None if cfg is None else {
        "quantization_config": cfg.get("quantization_config"),
        "model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures"),
        "text_config": {k: (cfg.get("text_config", {}) or {}).get(k) for k in (
            "num_hidden_layers", "num_nextn_predict_layers", "max_position_embeddings",
            "engram_layer_ids", "n_routed_experts", "hidden_size")},
        "vision_config_present": isinstance(cfg.get("vision_config"), dict),
    }

    # ------------------------------------------------------- description meta
    required = ["version", "model_quant_type", "group_size", "metadata", "optional"]
    missing = [k for k in required if k not in desc]
    if missing:
        rep.add("description_metadata", "FAIL", "missing fields: %s" % missing)
    else:
        rot = (((desc.get("optional") or {}).get("quarot") or {}).get("rotation_map") or {}).get("global_rotation")
        if rot != "optional/quarot.safetensors":
            rep.add("description_metadata", "FAIL",
                    "optional.quarot.rotation_map.global_rotation=%r "
                    "(expected 'optional/quarot.safetensors')" % rot)
        else:
            rep.add("description_metadata", "PASS",
                    "version=%r model_quant_type=%r group_size=%r"
                    % (desc.get("version"), desc.get("model_quant_type"), desc.get("group_size")))
        if desc.get("model_quant_type") != "W8A8_DYNAMIC":
            rep.add("description_model_quant_type", "WARN",
                    "model_quant_type=%r (mixed W4A8+W8A8 exports currently record "
                    "'W8A8_DYNAMIC'; W4A8_DYNAMIC is hidden by the saver priority list)"
                    % desc.get("model_quant_type"))
        else:
            rep.add("description_model_quant_type", "PASS", "W8A8_DYNAMIC")
    manifest["description"] = {
        "exists": os.path.isfile(desc_path),
        "entry_count": len(desc),
        "tensor_label_counts": dict(sorted(labels.items())),
        "non_string_fields": {k: v for k, v in desc.items() if not isinstance(v, str)},
        "desc_only_keys": sorted(set(desc) - set(weight_map)),
        "version": desc.get("version"),
        "model_quant_type": desc.get("model_quant_type"),
        "group_size": desc.get("group_size"),
        "optional": desc.get("optional"),
    }

    # ---------------------------------------------------------------- patterns
    if mtp_mode == "quantized":
        expected_non_mtp = {p: c for p, c in expected_patterns.items() if not p.startswith("mtp.")}
        observed_non_mtp = {p: c for p, c in observed_patterns.items() if not p.startswith("mtp.")}
        exp_total = sum(sum(c.values()) for c in expected_non_mtp.values())
        obs_total = sum(sum(c.values()) for c in observed_non_mtp.values())
        common = sorted(set(expected_non_mtp) | set(observed_non_mtp))
        diffs = [(p, dict(expected_non_mtp.get(p, {})), dict(observed_non_mtp.get(p, {})))
                 for p in common if expected_non_mtp.get(p, {}) != observed_non_mtp.get(p, {})]
        if obs_total != exp_total or diffs:
            rep.add("pattern_counts_non_mtp", "FAIL",
                    "non-MTP tensor patterns differ: expected %d, got %d, diffs=%s"
                    % (exp_total, obs_total, diffs[:8]))
        else:
            rep.add("pattern_counts_non_mtp", "PASS",
                    "%d non-MTP tensors match the %d-pattern baseline" % (obs_total, len(expected_non_mtp)))
    else:
        exp_total = sum(sum(c.values()) for c in expected_patterns.values())
        common = sorted(set(expected_patterns) | set(observed_patterns))
        diffs = [(p, dict(expected_patterns.get(p, {})), dict(observed_patterns.get(p, {})))
                 for p in common if expected_patterns.get(p, {}) != observed_patterns.get(p, {})]
        exp_total = sum(sum(c.values()) for c in expected_patterns.values())
        if len(weight_map) != exp_total or diffs:
            rep.add("pattern_counts", "FAIL",
                    "expected %d tensors in %d patterns, got %d; diffs=%s"
                    % (exp_total, len(expected_patterns), len(weight_map), diffs[:8]))
        else:
            rep.add("pattern_counts", "PASS",
                    "%d tensors match the %d-pattern baseline" % (len(weight_map), len(expected_patterns)))

    manifest["index"] = {
        "tensor_count": len(weight_map),
        "shard_count": len(set(weight_map.values())),
        "prefix_counts": dict(sorted(collections.Counter(k.split(".")[0] for k in weight_map).items())),
        "label_counts": dict(sorted(labels.items())),
    }
    manifest["mtp"] = {
        "tensor_count": len(mtp_keys),
        "label_counts": dict(collections.Counter(label_of.get(k) for k in mtp_keys)),
        "all_float": bool(mtp_keys) and all(label_of.get(k) == "FLOAT" for k in mtp_keys),
        "mode": mtp_mode,
    }

    # ------------------------------------------------------------ shard check
    groups = collections.defaultdict(list)
    for key, shard in weight_map.items():
        groups[shard].append(key)
    shard_problems = []
    shard_missing = []
    for shard, keys in sorted(groups.items()):
        path = os.path.join(d, shard)
        if not os.path.isfile(path):
            shard_missing.append(shard)
            continue
        try:
            hlen, header = read_st_header(path)
        except Exception as exc:
            shard_problems.append("%s: %s" % (shard, exc))
            continue
        miss = [k for k in keys if k not in header]
        if miss:
            shard_problems.append("%s: missing %d mapped tensors, e.g. %s" % (shard, len(miss), miss[:3]))
            continue
        max_end = 0
        for spec in header.values():
            off = spec.get("data_offsets")
            if not (isinstance(off, list) and len(off) == 2 and off[1] >= off[0]):
                shard_problems.append("%s: bad data_offsets for %s" % (shard, spec))
                break
            max_end = max(max_end, int(off[1]))
        else:
            if file_size(path) != 8 + hlen + max_end:
                shard_problems.append(
                    "%s: file size %d != 8+%d+%d (truncated or trailing bytes)"
                    % (shard, file_size(path), hlen, max_end))
    if shard_missing:
        rep.add("shard_integrity", "FAIL", "missing shard files: %s" % shard_missing[:5])
    elif shard_problems:
        rep.add("shard_integrity", "FAIL", "; ".join(shard_problems[:5]))
    else:
        rep.add("shard_integrity", "PASS",
                "%d shards exist and contain all %d mapped tensors"
                % (len(groups), len(weight_map)))
    if not args.allow_extra_shards:
        extra = [f for f in os.listdir(d)
                 if f.endswith(".safetensors") and f not in groups]
        if extra:
            rep.add("extra_shards", "WARN", "shards not referenced by index: %s" % extra[:5])
        else:
            rep.add("extra_shards", "PASS", "no unreferenced .safetensors shards")

    # ---------------------------------------------------------------- quarot
    qpath = os.path.join(d, "optional", "quarot.safetensors")
    if not os.path.isfile(qpath):
        rep.add("quarot_attachment", "FAIL", "missing optional/quarot.safetensors")
        manifest["optional"] = {"quarot_path": "optional/quarot.safetensors", "exists": False}
    else:
        try:
            _, qh = read_st_header(qpath)
        except Exception as exc:
            rep.add("quarot_attachment", "FAIL", "cannot parse: %s" % exc)
            manifest["optional"] = {"quarot_path": "optional/quarot.safetensors",
                                    "exists": True, "error": str(exc)}
        else:
            spec = qh.get("global_rotation")
            ok = isinstance(spec, dict) and spec.get("dtype") == "F32" and list(spec.get("shape") or []) == [5120, 5120]
            manifest["optional"] = {
                "quarot_path": "optional/quarot.safetensors",
                "exists": True,
                "keys": {k: {"dtype": v.get("dtype"), "shape": v.get("shape")} for k, v in qh.items()},
            }
            if ok:
                rep.add("quarot_attachment", "PASS", "global_rotation F32 [5120,5120]")
            else:
                rep.add("quarot_attachment", "FAIL",
                        "expected global_rotation F32 [5120,5120], got %s" % spec)

    # ----------------------------------------------------------------- vision
    expected_vision = dict(BASELINE["vision_keys"])
    if args.official_dir:
        oi_path = os.path.join(args.official_dir, "model.safetensors.index.json")
        try:
            oi = load_json(oi_path)["weight_map"]
        except Exception as exc:
            rep.add("vision_official_manifest", "FAIL", "cannot read %s: %s" % (oi_path, exc))
        else:
            ok = sorted(k for k in oi if is_vision_key(k))
            official_vision = {}
            ocache = {}
            for k in ok:
                sh = oi[k]
                if sh not in ocache:
                    ocache[sh] = read_st_header(os.path.join(args.official_dir, sh))[1]
                m = ocache[sh][k]
                official_vision[k] = {"dtype": m["dtype"], "shape": m["shape"]}
            drift = official_vision != expected_vision
            rep.add("vision_official_manifest", "WARN" if drift else "PASS",
                    "official manifest has %d vision tensors%s"
                    % (len(official_vision), " (differs from embedded baseline!)" if drift else ""))
            expected_vision = official_vision
    vision_manifest = {"expected_count": len(expected_vision), "present_count": len(vision_keys_obs)}
    if not vision_keys_obs:
        detail = ("no vision/aligner/image_* tensors found in the index; "
                  "run scripts/add_vision_weights.py (266 tensors / 970,536,960 bytes) "
                  "or pass --allow-missing-vision for a text-only artifact")
        rep.add("vision_present", "WARN" if args.allow_missing_vision else "FAIL", detail)
        manifest["warning"] = ("VISION_MISSING: the artifact cannot serve image inputs "
                               "without the official BF16 vision/aligner weights")
    else:
        missing = sorted(set(expected_vision) - set(vision_keys_obs))
        extra = sorted(set(vision_keys_obs) - set(expected_vision))
        mism = []
        vbytes = 0
        vlabels = collections.Counter()
        vcache = {}
        for key in vision_keys_obs:
            sh = weight_map[key]
            if sh not in vcache:
                vcache[sh] = read_st_header(os.path.join(d, sh))[1]
            h = vcache[sh][key]
            vbytes += tensor_bytes(h)
            vlabels[label_of.get(key)] += 1
            exp = expected_vision.get(key)
            if exp and (exp["dtype"] != h["dtype"] or list(exp["shape"]) != list(h["shape"])):
                mism.append({"key": key, "expected": exp,
                             "actual": {"dtype": h["dtype"], "shape": h["shape"]}})
        vision_manifest.update({
            "missing_keys": missing[:20],
            "missing_count": len(missing),
            "extra_keys": extra[:20],
            "extra_count": len(extra),
            "mismatched": mism[:20],
            "mismatched_count": len(mism),
            "total_bytes": vbytes,
            "label_counts": dict(vlabels),
        })
        fp_obs = hashlib.sha256(json.dumps(
            [[k, expected_vision[k]["dtype"], list(expected_vision[k]["shape"])]
             for k in sorted(expected_vision)], separators=(",", ":")).encode()).hexdigest()
        vision_manifest["expected_fingerprint_sha256"] = fp_obs
        if missing or extra or mism:
            rep.add("vision_completeness", "FAIL",
                    "missing=%d extra=%d shape/dtype-mismatch=%d (see manifest.vision)"
                    % (len(missing), len(extra), len(mism)))
        elif vbytes != BASELINE["vision_total_bytes"]:
            rep.add("vision_completeness", "FAIL",
                    "vision bytes %d != baseline %d" % (vbytes, BASELINE["vision_total_bytes"]))
        elif vlabels != collections.Counter({"FLOAT": len(expected_vision)}):
            rep.add("vision_completeness", "FAIL", "vision labels=%s (all must be FLOAT)" % dict(vlabels))
        else:
            rep.add("vision_completeness", "PASS",
                    "%d/%d vision tensors, %d bytes, all BF16 FLOAT, shapes match"
                    % (len(vision_keys_obs), len(expected_vision), vbytes))
        if fp_obs != BASELINE["vision_fingerprint_sha256"] and not args.official_dir:
            rep.add("vision_fingerprint", "WARN",
                    "embedded baseline fingerprint differs from the official set (upstream drift?)")
        else:
            rep.add("vision_fingerprint", "PASS", fp_obs[:16])
    manifest["vision"] = vision_manifest

    # ------------------------------------------------------------- reps check
    reps_manifest = {}
    rep_problems = []
    for key, exp in sorted(BASELINE["reps"].items()):
        pat = exp.get("pattern", pattern(key))
        if pat.startswith("mtp.") and "mtp" not in components:
            continue
        if pat.startswith(("vision.", "aligner.", "image_")) and "vision" not in components:
            continue
        if key not in weight_map:
            reps_manifest[key] = None
            rep_problems.append("%s: MISSING" % key)
            continue
        sh = weight_map[key]
        try:
            h = read_st_header(os.path.join(d, sh))[1][key]
        except Exception as exc:
            reps_manifest[key] = {"error": str(exc)}
            rep_problems.append("%s: %s" % (key, exc))
            continue
        got = {"dtype": h["dtype"], "shape": list(h["shape"]), "label": label_of.get(key), "shard": sh}
        reps_manifest[key] = got
        if got["dtype"] != exp["dtype"] or got["shape"] != list(exp["shape"]) or got["label"] != exp["label"]:
            rep_problems.append("%s: got %s, expected dtype=%s shape=%s label=%s"
                                % (key, got, exp["dtype"], exp["shape"], exp["label"]))
    if rep_problems:
        rep.add("representative_tensors", "FAIL", "; ".join(rep_problems[:8]))
    else:
        rep.add("representative_tensors", "PASS",
                "%d representative tensors match name/shape/dtype/label" % len(reps_manifest))
    manifest["representative_tensors"] = reps_manifest

    # ------------------------------------------------------------ observations
    obs = {
        "index": manifest.get("index"),
        "description": {k: manifest.get("description", {}).get(k) for k in (
            "entry_count", "tensor_label_counts", "version", "model_quant_type", "group_size")},
        "config": manifest.get("config"),
        "optional": manifest.get("optional"),
        "vision": {k: vision_manifest.get(k) for k in (
            "present_count", "expected_count", "missing_count", "extra_count",
            "mismatched_count", "total_bytes", "label_counts")},
        "mtp": manifest.get("mtp"),
        "representative_tensors": reps_manifest,
    }
    manifest["observations_sha256"] = hashlib.sha256(
        json.dumps(obs, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    manifest["profile"] = ("dspark" if "mtp" in components else "trunk") + \
        ("-vision" if "vision" in components else "-text") + \
        ("+mtp-quantized" if ("mtp" in components and mtp_mode == "quantized") else "")
    manifest["checks"] = rep.checks
    manifest["verdict"] = "FAIL" if rep.failed else "PASS"
    manifest["warnings"] = [c for c in rep.warned]

    # ------------------------------------------------------------------ print
    print("== quant_repro_check ==")
    print("dir      : %s" % d)
    print("profile  : %s" % manifest["profile"])
    print("index    : %s tensors / %s shards" %
          (manifest.get("index", {}).get("tensor_count"), manifest.get("index", {}).get("shard_count")))
    print("labels   : %s" % json.dumps(manifest.get("index", {}).get("label_counts"), ensure_ascii=False))
    print("vision   : %s/%s tensors%s" % (
        vision_manifest.get("present_count", 0), vision_manifest.get("expected_count", 0),
        "" if not vision_manifest.get("total_bytes") else " (%.4f GB)" % (vision_manifest["total_bytes"] / 1e9)))
    print("quarot   : %s" % ("global_rotation F32 [5120,5120]" if manifest.get("optional", {}).get("exists")
                             and "global_rotation" in (manifest.get("optional", {}).get("keys") or {}) else "MISSING"))
    print("checks   : %d PASS / %d WARN / %d FAIL" % (
        sum(1 for c in rep.checks if c["status"] == "PASS"),
        len(rep.warned), len(rep.failed)))
    for c in rep.checks:
        if c["status"] != "PASS":
            print("  [%s] %s: %s" % (c["status"], c["id"], c["detail"]))
    print("verdict  : %s" % manifest["verdict"])

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1, ensure_ascii=False, sort_keys=False)
            fh.write("\n")
        print("manifest : %s" % os.path.abspath(args.out))
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
