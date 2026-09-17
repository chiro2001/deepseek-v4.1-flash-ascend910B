#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline W4A8/W8A8 quantization of the DSpark MTP draft (V4.1 runtime checkpoint).

Why this script exists
----------------------
The fork's runtime loads the DSpark draft through the *same* Ascend ModelSlim
quantization path as the target model: the draft layers are built with prefixes
``mtp.{i}`` and ``AscendModelSlimConfig.get_quant_method`` looks their quant
type up in ``quant_model_description.json``.  The V4.1 msmodelslim adapter,
however, explicitly refuses to load MTP layers
(``model/deepseek_v41/model_adapter.py:115-117``), so we cannot use the existing
W4A8 recipe to produce quantized MTP tensors in the same pass.

This script is the alternative: it reads the already-assembled runtime
checkpoint (``v41-w4a8-dspark``), quantizes only the MTP linears with exactly the
same data-free ``LinearQuantizer`` + ``ascendv1_saver`` packing helpers that
msmodelslim itself uses, and writes a new runtime checkpoint with an updated
``quant_model_description.json`` / ``quant_model_weights.safetensors.index.json``.

Modes
-----
  --plan       stdlib only.  Scan a checkpoint, classify mtp tensors, report
               baseline/output bytes, per-rank savings and num_blocks estimate.
  --quantize   container only (torch + safetensors + msmodelslim).  Write a new
               runtime checkpoint directory.
  --verify     validate an output directory produced by --quantize.
  --selftest   pure-python sanity checks of the classification/byte accounting.

Default plan (no runtime code change required):
  * mtp.*.ffn.experts.*.w1/w2/w3.weight       -> W4A8_DYNAMIC
  * mtp.*.attn.wq_a/wq_b/wkv.weight           -> W8A8_DYNAMIC
  * mtp.*.ffn.shared_experts.w1/w2/w3.weight  -> W8A8_DYNAMIC
  * everything else (main_proj, wo_a/wo_b, norms, markov/confidence, vocab)
    is copied unchanged (BF16/F32 => FLOAT).

Optional flags for the aggressive plan (need runtime patch / more validation):
  --aggressive          also quantize attn.wo_a/wo_b to W8A8
  --quant-main-proj     also quantize mtp.0.main_proj to W8A8 (runtime patch)
  --quant-vocab         also quantize mtp.0.embed / mtp.{last}.head to W8A8
                        (runtime patch + draft vocab shard quantization)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

INDEX_NAME = "quant_model_weights.safetensors.index.json"
DESC_NAME = "quant_model_description.json"
CONFIG_NAME = "config.json"
OPTIONAL_DIR = "optional"

W4A8 = "W4A8_DYNAMIC"
W8A8 = "W8A8_DYNAMIC"
FLOAT = "FLOAT"

W_SUFFIX = ".weight"
WS_SUFFIX = ".weight_scale"
WO_SUFFIX = ".weight_offset"
SB_SUFFIX = ".scale_bias"


def aux_name(weight_key: str, suffix: str) -> str:
    """Checkpoint name for an auxiliary tensor of ``<...>.weight``."""
    assert weight_key.endswith(W_SUFFIX), weight_key
    return weight_key[: -len(W_SUFFIX)] + suffix

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "U16": 2, "U32": 4, "U64": 8, "C64": 8,
    "F8_E4M3": 1, "F8_E4M3FNUZ": 1, "F8_E5M2": 1, "F8_E5M2FNUZ": 1,
    "F8_E8M0": 1,
}

PAT_W4A8 = re.compile(r"^mtp\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$")
PAT_W8A8_ATTN = re.compile(r"^mtp\.\d+\.attn\.(?:wq_a|wq_b|wkv)\.weight$")
PAT_W8A8_SHARED = re.compile(r"^mtp\.\d+\.ffn\.shared_experts\.w[123]\.weight$")
PAT_AGGRESSIVE = re.compile(r"^mtp\.\d+\.attn\.(?:wo_a|wo_b)\.weight$")
PAT_MAIN_PROJ = re.compile(r"^mtp\.0\.main_proj\.weight$")
PAT_VOCAB = re.compile(r"^mtp\.(?:0\.embed|\d+\.head)\.weight$")


# ---------------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(path: str, obj, indent=None) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        if indent is None:
            json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(obj, fh, ensure_ascii=False, indent=indent)
        fh.write("\n")
    os.replace(tmp, path)


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def read_st_header(path: str):
    """Return safetensors header dict (without __metadata__)."""
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"{path}: too short for safetensors")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 0 or header_len > (1 << 32):
            raise RuntimeError(f"{path}: bad safetensors header length {header_len}")
        raw = fh.read(header_len)
    header = json.loads(raw.decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def dtype_bytes(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise RuntimeError(f"unsupported safetensors dtype {dtype!r}") from exc


# ---------------------------------------------------------------------------
# classification / byte accounting
# ---------------------------------------------------------------------------
@dataclass
class QuantPlan:
    ranks: int = 8
    base_blocks: int = 38166
    base_delta_gib: float = 5.5
    base_delta_blocks: int = 28109
    request_blocks: int = 8969
    base_rank_cap: int = 4

    def num_blocks(self, savings_gib_per_rank: float) -> int:
        return int(round(self.base_blocks + savings_gib_per_rank * self.base_delta_blocks / self.base_delta_gib))

    def rank_cap(self, num_blocks: int) -> int:
        overhead = self.base_blocks - self.base_rank_cap * self.request_blocks
        return max(0, (num_blocks - overhead) // self.request_blocks)


def classify(key: str, aggressive: bool, quant_main_proj: bool, quant_vocab: bool):
    if key == "group_size" or key in ("metadata", "model_quant_type", "version"):
        return None
    if PAT_W4A8.match(key):
        return W4A8
    if PAT_W8A8_ATTN.match(key) or PAT_W8A8_SHARED.match(key):
        return W8A8
    if aggressive and PAT_AGGRESSIVE.match(key):
        return W8A8
    if quant_main_proj and PAT_MAIN_PROJ.match(key):
        return W8A8
    if quant_vocab and PAT_VOCAB.match(key):
        return W8A8
    return FLOAT


def is_w2_key(key: str) -> bool:
    return ".w2." in key or key.endswith(".w2.weight")


def quantized_bytes(fmt: str, shape, key: str) -> int:
    """Bytes written by msmodelslim ascendv1_saver for one 2-D weight."""
    if len(shape) != 2:
        raise RuntimeError(f"{key}: quantized target must be 2-D, got {tuple(shape)}")
    out, inn = int(shape[0]), int(shape[1])
    if fmt == W8A8:
        return out * inn + out * 4 + out * 4          # int8 weight + f32 scale + f32 zero offset
    if fmt == W4A8:
        blocks = 16 if is_w2_key(key) else 1
        return ((out + 1) // 2) * inn + out * 4 + out * 4 + out * blocks * 4
    raise ValueError(fmt)


def baseline_bytes(dtype: str, shape) -> int:
    return numel(shape) * dtype_bytes(dtype)


@dataclass
class TensorRecord:
    key: str
    shard: str
    dtype: str
    shape: tuple
    fmt: str
    base_bytes: int
    out_bytes: int
    aux: "OrderedDict[str, tuple]" = field(default_factory=OrderedDict)


def scan_checkpoint(src: str, aggressive=False, quant_main_proj=False, quant_vocab=False,
                    fallback_shapes=None):
    """Read the source index + safetensors headers; return (records, stats, index, config)."""
    idx = load_json(os.path.join(src, INDEX_NAME))
    wm = idx.get("weight_map") or {}
    config = {}
    try:
        config = load_json(os.path.join(src, CONFIG_NAME))
    except Exception:
        pass
    text = config.get("text_config") if isinstance(config, dict) else None
    vocab = int((text or config).get("vocab_size", 0) or 0)
    hidden = int((text or config).get("hidden_size", 0) or 0)
    fallback_shapes = dict(fallback_shapes or {})
    for k in ("mtp.0.embed.weight", "mtp.%d.head.weight" % max(int((text or {}).get("num_nextn_predict_layers", 3)) - 1, 0)):
        if vocab and hidden:
            fallback_shapes.setdefault(k, (vocab, hidden))

    header_cache = {}
    records = []
    for key in sorted(wm):
        if not key.startswith("mtp."):
            continue
        shard = wm[key]
        header = header_cache.get(shard)
        if header is None:
            path = os.path.join(src, shard)
            try:
                header = read_st_header(path)
            except PermissionError:
                header = {k: {"dtype": "BF16", "shape": list(fallback_shapes[k])}
                          for k in fallback_shapes}
            header_cache[shard] = header
        meta = header.get(key)
        if meta is None:
            raise RuntimeError(f"{key}: not present in {shard}")
        shape = tuple(int(x) for x in meta["shape"])
        dtype = meta["dtype"]
        fmt = classify(key, aggressive, quant_main_proj, quant_vocab)
        base = baseline_bytes(dtype, shape)
        out = quantized_bytes(fmt, shape, key) if fmt in (W4A8, W8A8) else base
        rec = TensorRecord(key=key, shard=shard, dtype=dtype, shape=shape, fmt=fmt,
                           base_bytes=base, out_bytes=out)
        if fmt in (W4A8, W8A8):
            aux = OrderedDict()
            aux[key] = (fmt, shape)
            aux[aux_name(key, WS_SUFFIX)] = (fmt, (shape[0], 1))
            aux[aux_name(key, WO_SUFFIX)] = (fmt, (shape[0], 1))
            if fmt == W4A8:
                blocks = 16 if is_w2_key(key) else 1
                aux[aux_name(key, SB_SUFFIX)] = (fmt, (shape[0], blocks))
            rec.aux = aux
        records.append(rec)

    stats = summarize(records)
    return records, stats, idx, config


def summarize(records):
    base = sum(r.base_bytes for r in records)
    out = sum(r.out_bytes for r in records)
    by_fmt = Counter(r.fmt for r in records)
    by_pat = defaultdict(lambda: {"n": 0, "base": 0, "out": 0})
    for r in records:
        pat = re.sub(r"\.\d+\.", ".N.", r.key)
        pat = re.sub(r"\.N\.(w[123])\.weight", r".N.\1.weight", pat)
        p = by_pat[pat]
        p["n"] += 1
        p["base"] += r.base_bytes
        p["out"] += r.out_bytes
    return {"base_bytes": base, "out_bytes": out, "by_fmt": dict(by_fmt), "by_pattern": dict(by_pat)}


# ---------------------------------------------------------------------------
# quantization (container only)
# ---------------------------------------------------------------------------
def _make_quantizer(fmt: str):
    import torch  # noqa: F401
    from msmodelslim.core.quantizer.base import QConfig
    from msmodelslim.core.quantizer.linear import LinearQuantizer, LinearQConfig
    from msmodelslim.ir.qal import QDType, QScope

    if fmt == W4A8:
        weight = QConfig(dtype=QDType.INT4, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax")
    elif fmt == W8A8:
        weight = QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax")
    else:
        raise ValueError(fmt)
    act = QConfig(dtype=QDType.INT8, scope=QScope.PER_TOKEN, symmetric=True, method="minmax")
    return LinearQuantizer(LinearQConfig(act=act, weight=weight))


def quantize_one_weight(name: str, weight, fmt: str):
    """Return {checkpoint_name: tensor} for one 2-D weight, mirroring the saver."""
    import torch
    import torch.nn as nn

    assert weight.ndim == 2, f"{name}: expected 2-D, got {tuple(weight.shape)}"
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])

    lin = nn.Linear(in_features, out_features, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(weight.detach().to(torch.bfloat16))

    quantizer = _make_quantizer(fmt)
    quantizer.setup(lin)
    module = quantizer.deploy()

    with torch.no_grad():
        if fmt == W4A8:
            from msmodelslim.core.quant_service.modelslim_v1.save.utils.pack import (
                process_scale, w4a8_pack_int4,
            )
            packed = w4a8_pack_int4(module.weight.detach().cpu().to(torch.int8))
            scale = module.weight_scale.detach().to(torch.float32).reshape(-1, 1).contiguous()
            weight_f32 = module.weight.detach().to(torch.float32)
            deq = weight_f32.T * module.weight_scale.detach().to(torch.float32)
            scale_bias = process_scale(name[: -len(W_SUFFIX)], deq.T, 16)
            scale_bias = scale_bias.to(torch.float32).contiguous()
            offset = torch.zeros_like(scale, dtype=torch.float32)
            return OrderedDict([
                (name, packed.contiguous()),
                (aux_name(name, WS_SUFFIX), scale),
                (aux_name(name, WO_SUFFIX), offset),
                (aux_name(name, SB_SUFFIX), scale_bias),
            ])
        if fmt == W8A8:
            scale = module.weight_scale.detach().to(torch.float32).reshape(-1, 1).contiguous()
            offset = torch.zeros_like(scale, dtype=torch.float32)
            return OrderedDict([
                (name, module.weight.detach().to(torch.int8).contiguous()),
                (aux_name(name, WS_SUFFIX), scale),
                (aux_name(name, WO_SUFFIX), offset),
            ])
    raise ValueError(fmt)


def pack_output_shards(tensors: "OrderedDict[str, object]", shard_gib: float):
    """Greedy split by tensor bytes; returns [(file_name, OrderedDict)]."""
    limit = int(shard_gib * (1024 ** 3))
    groups = []
    cur = OrderedDict()
    cur_bytes = 0
    for name, tensor in tensors.items():
        nbytes = tensor.numel() * tensor.element_size()
        if cur and cur_bytes + nbytes > limit:
            groups.append(cur)
            cur = OrderedDict()
            cur_bytes = 0
        cur[name] = tensor
        cur_bytes += nbytes
    if cur:
        groups.append(cur)
    total = len(groups)
    return [(f"mtpq-{i:05d}-of-{total:05d}.safetensors", g) for i, g in enumerate(groups, 1)]


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def cmd_plan(args):
    recs, stats, _idx, _config = scan_checkpoint(
        args.src, aggressive=args.aggressive, quant_main_proj=args.quant_main_proj,
        quant_vocab=args.quant_vocab)
    gp = QuantPlan(ranks=args.ranks, base_blocks=args.base_blocks,
                   base_delta_gib=args.base_delta_gib, base_delta_blocks=args.delta_blocks,
                   request_blocks=args.request_blocks, base_rank_cap=args.base_rank_cap)
    base_gib = stats["base_bytes"] / (1024 ** 3)
    out_gib = stats["out_bytes"] / (1024 ** 3)
    save_gib = base_gib - out_gib
    per_rank = save_gib / args.ranks
    blocks = gp.num_blocks(per_rank)
    cap = gp.rank_cap(blocks)

    log(f"[mtpq] src                  : {os.path.abspath(args.src)}")
    log(f"[mtpq] mtp tensors          : {len(recs)}")
    log(f"[mtpq] baseline             : {base_gib:.3f} GiB ({stats['base_bytes']} B)")
    log(f"[mtpq] quantized+copy output: {out_gib:.3f} GiB ({stats['out_bytes']} B)")
    log(f"[mtpq] saving               : {save_gib:.3f} GiB total, {per_rank:.3f} GiB/rank (ranks={args.ranks})")
    log(f"[mtpq] est. num_blocks      : {blocks}  (rank_cap~{cap}; baseline {args.base_blocks})")
    log(f"[mtpq] formats              : {stats['by_fmt']}")
    log("")
    log("%-58s %6s %10s %10s %10s" % ("pattern", "n", "base MiB", "out MiB", "save MiB"))
    for pat, p in sorted(stats["by_pattern"].items(), key=lambda kv: -kv[1]["base"]):
        log("%-58s %6d %10.1f %10.1f %10.1f" % (
            pat, p["n"], p["base"] / 2**20, p["out"] / 2**20, (p["base"] - p["out"]) / 2**20))
    log("")
    log("%-58s %6d %10.1f %10.1f %10.1f" % (
        "TOTAL", len(recs), base_gib * 1024, out_gib * 1024, save_gib * 1024))

    if args.json:
        payload = {
            "src": os.path.abspath(args.src),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "options": {
                "aggressive": args.aggressive,
                "quant_main_proj": args.quant_main_proj,
                "quant_vocab": args.quant_vocab,
            },
            "stats": stats,
            "summary": {
                "baseline_gib": base_gib,
                "output_gib": out_gib,
                "saving_gib": save_gib,
                "saving_gib_per_rank": per_rank,
                "ranks": args.ranks,
                "est_num_blocks": blocks,
                "est_rank_cap": cap,
            },
            "records": [
                {"key": r.key, "shard": r.shard, "dtype": r.dtype, "shape": list(r.shape),
                 "fmt": r.fmt, "base_bytes": r.base_bytes, "out_bytes": r.out_bytes,
                 "aux": {k: {"fmt": v[0], "shape": list(v[1])} for k, v in r.aux.items()}}
                for r in recs
            ],
        }
        dump_json(args.json, payload, indent=1)
        log(f"[mtpq] wrote {args.json}")
    return 0


def cmd_verify(args):
    """Verify an output dir against the BF16 source dir (--src) it was built from."""
    src = os.path.abspath(args.src)
    out = os.path.abspath(args.out)
    recs, stats, _src_idx, _src_cfg = scan_checkpoint(
        src, aggressive=args.aggressive, quant_main_proj=args.quant_main_proj,
        quant_vocab=args.quant_vocab)
    out_idx = load_json(os.path.join(out, INDEX_NAME))
    wm = out_idx["weight_map"]
    desc = load_json(os.path.join(out, DESC_NAME))

    header_cache = {}
    errors = []
    checked_q = 0
    checked_copy = 0

    def get_header(shard):
        if shard not in header_cache:
            header_cache[shard] = read_st_header(os.path.join(out, shard))
        return header_cache[shard]

    for r in recs:
        shard = wm.get(r.key)
        if shard is None:
            errors.append(f"{r.key}: missing from output index")
            continue
        try:
            header = get_header(shard)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{r.key}: cannot read {shard}: {exc}")
            continue
        got = header.get(r.key)
        if got is None:
            errors.append(f"{r.key}: missing in {shard}")
            continue

        if r.fmt in (W4A8, W8A8):
            checked_q += 1
            if desc.get(r.key) != r.fmt:
                errors.append(f"{r.key}: description={desc.get(r.key)!r}, expected {r.fmt}")
            if got["dtype"] != "I8":
                errors.append(f"{r.key}: dtype={got['dtype']} expected I8")
            exp_shape = [(r.shape[0] + 1) // 2, r.shape[1]] if r.fmt == W4A8 else list(r.shape)
            if [int(x) for x in got["shape"]] != exp_shape:
                errors.append(f"{r.key}: shape={got['shape']} expected {exp_shape}")
            for aux_key, (fmt, shape) in r.aux.items():
                if aux_key == r.key:
                    continue
                if wm.get(aux_key) is None:
                    errors.append(f"{aux_key}: missing from output index")
                    continue
                if desc.get(aux_key) != fmt:
                    errors.append(f"{aux_key}: description={desc.get(aux_key)!r}, expected {fmt}")
                if wm[aux_key] != shard:
                    errors.append(f"{aux_key}: shard={wm[aux_key]} expected {shard}")
                    continue
                got_aux = header.get(aux_key)
                if got_aux is None:
                    errors.append(f"{aux_key}: missing in {shard}")
                    continue
                if got_aux["dtype"] != "F32":
                    errors.append(f"{aux_key}: dtype={got_aux['dtype']} expected F32")
                if [int(x) for x in got_aux["shape"]] != list(shape):
                    errors.append(f"{aux_key}: shape={got_aux['shape']} expected {list(shape)}")
        else:
            checked_copy += 1
            if [int(x) for x in got["shape"]] != list(r.shape) or got["dtype"] != r.dtype:
                errors.append(f"{r.key}: copy dtype/shape={got['dtype']}{got['shape']} "
                              f"expected {r.dtype}{list(r.shape)}")

    log(f"[mtpq] verify: {len(recs)} mtp weights ({checked_q} quantized, {checked_copy} copied), "
        f"{len(errors)} error(s)")
    for e in errors[:50]:
        log("  FAIL " + e)
    if errors:
        return 1
    log("[mtpq] verify PASS")
    return 0


def cmd_quantize(args):
    src = os.path.abspath(args.src)
    out = os.path.abspath(args.out)
    if out == src or out.startswith(src + os.sep) or src.startswith(out + os.sep):
        raise SystemExit("--out must not be the same as / nested below --src")
    if os.path.exists(out) and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it")

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src_idx = load_json(os.path.join(src, INDEX_NAME))
    src_wm = src_idx["weight_map"]
    src_desc = load_json(os.path.join(src, DESC_NAME))
    src_cfg = load_json(os.path.join(src, CONFIG_NAME))
    recs, stats, _idx, _cfg = scan_checkpoint(
        src, aggressive=args.aggressive, quant_main_proj=args.quant_main_proj,
        quant_vocab=args.quant_vocab)
    fmt_by_key = {r.key: r.fmt for r in recs}

    # collect source mtp tensors shard by shard, quantize, keep only outputs
    out_tensors = OrderedDict()
    src_mtp_shards = sorted({src_wm[k] for k in src_wm if k.startswith("mtp.")})
    done = 0
    for shard in src_mtp_shards:
        keys = sorted(k for k in src_wm if k.startswith("mtp.") and src_wm[k] == shard)
        with safe_open(os.path.join(src, shard), framework="pt", device="cpu") as fh:
            for key in keys:
                tensor = fh.get_tensor(key)
                fmt = fmt_by_key.get(key, FLOAT)
                if fmt in (W4A8, W8A8):
                    out_tensors.update(quantize_one_weight(key, tensor, fmt))
                else:
                    out_tensors[key] = tensor.contiguous()
                done += 1
                if done % 100 == 0:
                    log(f"[mtpq] quantized/copied {done} mtp tensors ...")
        del tensor

    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # trunk shards: absolute symlinks; keep the same shard names in the index
    trunk_shards = sorted({v for k, v in src_wm.items() if not k.startswith("mtp.")})
    for shard in trunk_shards:
        target = os.path.abspath(os.path.join(src, shard))
        link = os.path.join(out, shard)
        os.symlink(target, link)

    # aux files
    for name in ("configuration.json", "tokenizer.json", "tokenizer_config.json"):
        sp = os.path.join(src, name)
        if os.path.isfile(sp):
            shutil.copy2(sp, os.path.join(out, name))
    opt_src = os.path.join(src, OPTIONAL_DIR)
    if os.path.isdir(opt_src):
        shutil.copytree(opt_src, os.path.join(out, OPTIONAL_DIR), dirs_exist_ok=True)

    # config.json
    text = src_cfg.get("text_config") if isinstance(src_cfg, dict) else None
    if isinstance(text, dict):
        text["num_nextn_predict_layers"] = 3
    dump_json(os.path.join(out, CONFIG_NAME), src_cfg, indent=2)

    # write mtp shards
    groups = pack_output_shards(out_tensors, args.shard_gib)
    shard_of = {}
    for shard_name, group in groups:
        save_file(group, os.path.join(out, shard_name), metadata={"format": "pt"})
        for k in group:
            shard_of[k] = shard_name
        log(f"[mtpq] wrote {shard_name}: {len(group)} tensors, "
            f"{sum(t.numel() * t.element_size() for t in group.values()) / 2**30:.3f} GiB")

    # index
    wm = {k: v for k, v in src_wm.items() if not k.startswith("mtp.")}
    wm.update(shard_of)
    metadata = dict(src_idx.get("metadata") or {})
    if "total_size" in metadata:
        metadata["total_size"] = 0
    dump_json(os.path.join(out, INDEX_NAME), {"metadata": metadata, "weight_map": wm})

    # description
    desc = dict(src_desc)
    for r in recs:
        if r.fmt in (W4A8, W8A8):
            for aux_key, (fmt, _shape) in r.aux.items():
                desc[aux_key] = fmt
        else:
            desc.setdefault(r.key, FLOAT)
    dump_json(os.path.join(out, DESC_NAME), desc)

    manifest = {
        "schema": "mtpq-draft-quant/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "src": src,
        "out": out,
        "options": {
            "aggressive": args.aggressive,
            "quant_main_proj": args.quant_main_proj,
            "quant_vocab": args.quant_vocab,
            "shard_gib": args.shard_gib,
        },
        "stats": stats,
        "mtp_input_tensors": len(recs),
        "mtp_output_tensors": len(out_tensors),
        "output_shards": [g[0] for g in groups],
    }
    dump_json(os.path.join(out, "mtpq_manifest.json"), manifest, indent=1)
    log(f"[mtpq] DONE: {out}")
    return 0


def cmd_smoke(args):
    """Quantize a few representative MTP tensors without building a full checkpoint.

    This is the first thing to run next window: it validates the msmodelslim
    imports / LinearQuantizer output shapes and the saver-equivalent packing on
    ~10 tensors, before spending I/O on the full 1226-tensor pass.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = os.path.abspath(args.src)
    out_file = os.path.abspath(args.smoke_out)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    src_idx = load_json(os.path.join(src, INDEX_NAME))
    src_wm = src_idx["weight_map"]
    patterns = [
        r"^mtp\.0\.ffn\.experts\.0\.w1\.weight$",
        r"^mtp\.0\.ffn\.experts\.0\.w2\.weight$",
        r"^mtp\.0\.ffn\.experts\.0\.w3\.weight$",
        r"^mtp\.0\.attn\.wq_a\.weight$",
        r"^mtp\.0\.attn\.wq_b\.weight$",
        r"^mtp\.0\.attn\.wkv\.weight$",
        r"^mtp\.0\.ffn\.shared_experts\.w1\.weight$",
        r"^mtp\.0\.ffn\.shared_experts\.w2\.weight$",
        r"^mtp\.0\.ffn\.shared_experts\.w3\.weight$",
    ]
    if args.quant_main_proj:
        patterns.append(r"^mtp\.0\.main_proj\.weight$")
    if args.quant_vocab:
        patterns.extend([r"^mtp\.0\.embed\.weight$", r"^mtp\.2\.head\.weight$"])
    if args.aggressive:
        patterns.extend([r"^mtp\.0\.attn\.wo_a\.weight$", r"^mtp\.0\.attn\.wo_b\.weight$"])

    outputs = OrderedDict()
    checks = []
    for pat in patterns:
        rx = re.compile(pat)
        key = next((k for k in sorted(src_wm) if rx.match(k)), None)
        if key is None:
            continue
        fmt = classify(key, args.aggressive, args.quant_main_proj, args.quant_vocab)
        shard = src_wm[key]
        with safe_open(os.path.join(src, shard), framework="pt", device="cpu") as fh:
            tensor = fh.get_tensor(key)
        produced = quantize_one_weight(key, tensor, fmt)
        outputs.update(produced)
        checks.append({
            "key": key,
            "fmt": fmt,
            "input_dtype": str(tensor.dtype),
            "input_shape": list(tensor.shape),
            "outputs": {k: {"dtype": str(v.dtype), "shape": list(v.shape),
                            "bytes": int(v.numel() * v.element_size())}
                        for k, v in produced.items()},
        })
        log(f"[mtpq] smoke {fmt:14s} {key} -> " +
            ", ".join(f"{k.split('.')[-1]}:{tuple(v.shape)}/{v.dtype}" for k, v in produced.items()))

    save_file(outputs, out_file, metadata={"format": "pt"})
    dump_json(out_file + ".json", {
        "schema": "mtpq-draft-quant-smoke/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "src": src,
        "out": out_file,
        "checks": checks,
    }, indent=1)
    log(f"[mtpq] smoke wrote {out_file} ({len(outputs)} tensors) and {out_file}.json")
    return 0


def cmd_selftest(args):
    # 1) expert W4A8 accounting
    shape = (2304, 5120)
    assert quantized_bytes(W4A8, shape, "mtp.0.ffn.experts.0.w1.weight") == 1152 * 5120 + 2304 * 4 * 3
    assert quantized_bytes(W4A8, (5120, 2304), "mtp.0.ffn.experts.0.w2.weight") == 2560 * 2304 + 5120 * 4 * (1 + 1 + 16)
    assert quantized_bytes(W8A8, shape, "mtp.0.attn.wq_b.weight") == 2304 * 5120 + 2304 * 8
    # 1b) aux naming must be <...>.weight_scale / .weight_offset / .scale_bias
    assert aux_name("mtp.0.ffn.experts.0.w1.weight", WS_SUFFIX) == "mtp.0.ffn.experts.0.w1.weight_scale"
    assert aux_name("mtp.0.ffn.experts.0.w1.weight", WO_SUFFIX) == "mtp.0.ffn.experts.0.w1.weight_offset"
    assert aux_name("mtp.0.ffn.experts.0.w1.weight", SB_SUFFIX) == "mtp.0.ffn.experts.0.w1.scale_bias"
    # 2) classification
    assert classify("mtp.0.ffn.experts.0.w1.weight", False, False, False) == W4A8
    assert classify("mtp.2.ffn.experts.127.w3.weight", False, False, False) == W4A8
    assert classify("mtp.1.attn.wkv.weight", False, False, False) == W8A8
    assert classify("mtp.1.ffn.shared_experts.w2.weight", False, False, False) == W8A8
    assert classify("mtp.0.main_proj.weight", False, False, False) == FLOAT
    assert classify("mtp.0.main_proj.weight", False, True, False) == W8A8
    assert classify("mtp.0.attn.wo_a.weight", False, False, False) == FLOAT
    assert classify("mtp.0.attn.wo_a.weight", True, False, False) == W8A8
    # 3) rough savings match the design numbers (expert-only path)
    expert_base = 384 * 3 * 2304 * 5120 * 2
    expert_out = 384 * (quantized_bytes(W4A8, (2304, 5120), "x.w1.weight")
                        + quantized_bytes(W4A8, (5120, 2304), "x.w2.weight")
                        + quantized_bytes(W4A8, (2304, 5120), "x.w3.weight"))
    save_gib = (expert_base - expert_out) / 2**30
    assert 18.0 < save_gib < 20.0, save_gib
    log(f"[mtpq] selftest PASS (expert W4A8 save {save_gib:.3f} GiB total, {save_gib/8:.3f} GiB/rank)")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="quantize ~10 representative tensors only")
    ap.add_argument("--src", default="/home/user/models/out/v41-w4a8-dspark")
    ap.add_argument("--out", default="/home/user/models/out/v41-w4a8-dspark-mtpq")
    ap.add_argument("--json", default=None)
    ap.add_argument("--smoke-out", default="/home/user/projects/dsv41/scratch/mtpq/smoke/smoke.safetensors")
    ap.add_argument("--shard-gib", type=float, default=3.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--base-blocks", type=int, default=38166)
    ap.add_argument("--delta-blocks", type=int, default=28109)
    ap.add_argument("--base-delta-gib", type=float, default=5.5)
    ap.add_argument("--request-blocks", type=int, default=8969)
    ap.add_argument("--base-rank-cap", type=int, default=4)
    ap.add_argument("--aggressive", action="store_true", help="also quantize attn.wo_a/wo_b to W8A8")
    ap.add_argument("--quant-main-proj", action="store_true", help="also quantize mtp.0.main_proj (needs runtime patch)")
    ap.add_argument("--quant-vocab", action="store_true", help="also quantize mtp embed/head (needs runtime patch)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest:
        return cmd_selftest(args)
    if args.smoke:
        return cmd_smoke(args)
    if args.plan:
        return cmd_plan(args)
    if args.verify:
        return cmd_verify(args)
    if args.quantize:
        return cmd_quantize(args)
    build_parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
