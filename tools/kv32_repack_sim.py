#!/usr/bin/env python3
"""KV32 slot 重排补丁的**逻辑仿真**（不占卡、秒级）：确认它产出四槽等长 + 覆盖每个资源一次。

为什么需要：交付镜像里的 `core/deepseek_v41.py`（356 行）与 tiny 工作副本（503 行）
**不是同一份文件**，补丁是逐行移植的 ⇒ 只做 AST 检查不够，必须验证"喂进交付几何能出
正确 placement"。本脚本用极简桩 import 那份补丁，构造与真机逐项一致的 V4.1 几何
（ratio-2 KV 65,536 / index 8,320；ratio-1 KV 131,072 / index 16,640；SWA 与 state
与 draft 各 131,072 —— 与 [V41-DCP-DIAG] 实测一致），然后调用 `plan_cache_slots`。

判据（任一不满足即 FAIL）：
  1. 4 个槽，且每槽 page_size_bytes == 131072；
  2. Σ 槽长 == 524288（= 重排后每块池字节数）；
  3. layer-20 的 index 平面被挪到别的槽且 offset + size ≤ 该槽步长；
  4. 每个资源名恰好出现一次（补丁自带的覆盖检查）。

用法：python3 tools/kv32_repack_sim.py [补丁路径]
"""
import sys, types, os, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "patches", "files", "deepseek_v41.repack.py")

# ---- 极简桩：只要能 import 与 isinstance ----
def mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m

class _Base: pass
class KVSpec(_Base): pass
class CachePlacement(_Base):
    def __init__(self, name, offset, page_size_bytes):
        self.name, self.offset, self.page_size_bytes = name, offset, page_size_bytes
class CacheSlot(_Base):
    def __init__(self, page_size_bytes, placements):
        self.page_size_bytes, self.placements = page_size_bytes, placements

mod("torch", Tensor=object, bfloat16="bf16", float16="fp16", float32="fp32", uint8="u8",
    as_strided=lambda *a, **k: None, zeros=lambda *a, **k: None)
mod("vllm", __path__=[])
mod("vllm.config", CUDAGraphMode=type("CUDAGraphMode", (), {"NONE": 0, "FULL": 1, "FULL_DECODE_ONLY": 2}))
mod("vllm.v1", __path__=[])
mod("vllm.v1.core", __path__=[])
mod("vllm.v1.core.kv_cache_utils",
    may_override_num_blocks=lambda cfg, n: n)
mod("vllm.v1.kv_cache_interface",
    KVCacheGroupSpec=type("KVCacheGroupSpec", (), {}),
    KVCacheTensor=type("KVCacheTensor", (), {"__init__": lambda self, **kw: self.__dict__.update(kw)}),
    UniformTypeKVCacheSpecs=type("UniformTypeKVCacheSpecs", (), {}))
mod("vllm_ascend", __path__=[])
mod("vllm_ascend.core", __path__=[])
mod("vllm_ascend.core.circular_buffer", AscendCircularBufferSpec=type("AscendCircularBufferSpec", (_Base,), {}))
from dataclasses import dataclass
class DType:
    def __init__(self, n): self.itemsize = n
    def __repr__(self): return f"dtype({self.itemsize})"
_F32, _BF16 = DType(4), DType(2)
@dataclass(frozen=True, kw_only=True)
class _AscendMLA:
    block_size: int = 128
    storage_block_size: int = 128
    num_kv_heads: int = 1
    head_size: int = 512
    dtype: object = None
    compress_ratio: int = 1
    sliding_window: int = 0
    scale_dim: int = 0
    scale_dtype: object = None
@dataclass(frozen=True, kw_only=True)
class _AscendSWA(_AscendMLA):
    pass
@dataclass(frozen=True, kw_only=True)
class _AscendCirc:
    block_size: int = 32
    storage_block_size: int = 32
    num_kv_heads: int = 1
    head_size: int = 1024
    dtype: object = None
mod("vllm_ascend.core.kv_cache_interface",
    AscendMLAAttentionSpec=_AscendMLA, AscendSlidingWindowMLASpec=_AscendSWA)
mod("vllm_ascend.core.circular_buffer", AscendCircularBufferSpec=_AscendCirc)
torch_stub = sys.modules["torch"]
torch_stub.float32 = _F32
torch_stub.bfloat16 = _BF16

spec = importlib.util.spec_from_file_location("dvv41", SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# ---- 交付几何（与 A3 交付实例一致；字节数来自真机 DIAG 对 tiny 的实测，几何逐项相同）----
def kv_spec(ratio):
    return m.DeepseekV41FullSpec(compress_ratio=ratio, block_size=128, storage_block_size=128,
                                 num_kv_heads=1, head_size=512 if ratio == 1 else 256, dtype=DType(2))
def idx_spec(ratio):
    return m.DeepseekV41IndexerSpec(compress_ratio=ratio, block_size=128, storage_block_size=128,
                                    num_kv_heads=1, head_size=128 if ratio == 1 else 64, dtype=DType(1),
                                    scale_dim=1, scale_dtype=DType(2 if ratio == 1 else 1))
def swa_spec():
    return m.DeepseekV41SWASpec(compress_ratio=1, block_size=128, storage_block_size=128,
                                num_kv_heads=1, head_size=512, dtype=DType(2), sliding_window=128)
def state_spec():
    return m.DeepseekV41CompressorStateSpec(compress_ratio=1, block_size=32, storage_block_size=32,
                                            num_kv_heads=1, head_size=1024, dtype=sys.modules["torch"].float32)
def draft_spec():
    return m.DeepseekV41DraftSWASpec(compress_ratio=1, block_size=128, storage_block_size=128,
                                     num_kv_heads=1, head_size=512,
                                     dtype=sys.modules["torch"].bfloat16, sliding_window=128)

specs = {}
for l in (2, 8, 14):
    specs[f"model.layers.{l}.self_attn.long_kv_cache"] = kv_spec(2)
    specs[f"model.layers.{l}.self_attn.indexer.k_cache"] = idx_spec(2)
    specs[f"model.layers.{l}.self_attn.compressor_state"] = state_spec()
specs["model.layers.20.self_attn.long_kv_cache"] = kv_spec(1)
specs["model.layers.20.self_attn.indexer.k_cache"] = idx_spec(1)
for i in range(40):
    specs[f"model.layers.{i}.self_attn.swa_cache"] = swa_spec()
for i in range(3):
    specs[f"mtp.{i}.self_attn.swa_cache"] = draft_spec()

print("每类平面字节：", {k: m._cache_plane_sizes(v) for k, v in list(specs.items())[:1]})
slots = m.plan_cache_slots(specs)
print("槽数 =", len(slots))
print("slots =", [s.page_size_bytes for s in slots])
print("pool_bytes_per_block =", sum(s.page_size_bytes for s in slots))
for i, s in enumerate(slots):
    print(f"  slot{i}: cap={s.page_size_bytes} n_placements={len(s.placements)}")
    for p in s.placements:
        print(f"     {p.offset:>7} + {p.page_size_bytes:>7} = {p.offset+p.page_size_bytes:>7}  {p.name}")
U32 = 2**32
worst = max(s.page_size_bytes for s in slots)
pool = sum(s.page_size_bytes for s in slots)
print(f"每槽最大页步长 = {worst} ⇒ 块上限 = {U32//worst}；pool stride {pool}")
print(f"16 GiB 池 ⇒ {2**34 // pool} 块（= 32768 才是目标）")

# ---- 判据（不满足即非零退出，可直接当测试跑）----
fails = []
if len(slots) != 4:
    fails.append(f"槽数 {len(slots)} != 4")
if any(s.page_size_bytes != 131072 for s in slots):
    fails.append(f"槽长不齐：{[s.page_size_bytes for s in slots]}")
if pool != 524288:
    fails.append(f"pool stride {pool} != 524288")
moved = [p for s in slots for p in s.placements if p.name.endswith("layers.20.self_attn.indexer.k_cache")]
if len(moved) != 1 or moved[0].offset == 0:
    fails.append(f"layer-20 index 未被挪动：{moved}")
else:
    host = next(s for s in slots if any(pl.name == moved[0].name for pl in s.placements))
    if moved[0].offset + moved[0].page_size_bytes > host.page_size_bytes:
        fails.append("挪动后的 index 越出宿主槽")
    print(f"layer-20 index 落在 offset={moved[0].offset} size={moved[0].page_size_bytes}"
          f"（宿主槽 {host.page_size_bytes}）")
names = [p.name for s in slots for p in s.placements]
if len(names) != len(set(names)):
    fails.append("资源被重复放置")
print("SIM:", "FAIL — " + "; ".join(fails) if fails else "PASS（四槽 131072 / pool 524288 / 上限 32768）")
raise SystemExit(1 if fails else 0)
