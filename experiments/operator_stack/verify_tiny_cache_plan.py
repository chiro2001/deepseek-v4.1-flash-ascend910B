"""CPU-only tests of real cache specs, byte views, and rejected topologies."""
import argparse
from dataclasses import fields, replace
import json
from pathlib import Path

import torch
from vllm_ascend.core import deepseek_v41 as cache


def validate():
    assert getattr(cache, 'TINY_CACHE_PLAN_SOURCE_SHA256', None), 'Diagnostic import hook did not run'

    def spec(cls, **overrides):
        values = dict(block_size=128, num_kv_heads=1, head_size=512,
                      dtype=torch.bfloat16, compress_ratio=1, sliding_window=128,
                      model_version='deepseek_v4', cache_dtype_str='auto',
                      scale_dim=1, scale_dtype=torch.float16)
        values.update(overrides)
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names})

    specs = {}
    for layer in (2, 4, 5):
        prefix = f'model.layers.{layer}.self_attn'
        ratio = 1 if layer == 5 else 2
        specs[prefix + '.long_kv_cache'] = spec(cache.DeepseekV41FullSpec, compress_ratio=ratio)
        specs[prefix + '.indexer.k_cache'] = spec(cache.DeepseekV41IndexerSpec,
                                                  compress_ratio=ratio, head_size=128, dtype=torch.int8)
    for layer in (2, 4):
        specs[f'model.layers.{layer}.self_attn.compressor_state'] = spec(
            cache.DeepseekV41CompressorStateSpec, block_size=32, head_size=1024, dtype=torch.float32)
    for layer in range(8):
        specs[f'model.layers.{layer}.self_attn.swa_cache'] = spec(cache.DeepseekV41SWASpec)
    slots = cache.plan_cache_slots(specs)
    assert len(slots) == 3
    assert {p.name for slot in slots for p in slot.placements} == set(specs)
    assert len([p.name for slot in slots for p in slot.placements]) == len(specs) == 16
    rows = []
    for slot in slots:
        raw = torch.zeros(3 * slot.page_size_bytes, dtype=torch.uint8)
        views = {}
        for p in slot.placements:
            padded = replace(specs[p.name], page_size_padded=p.page_size_bytes)
            views[p.name] = cache.reshape_cache(raw, padded, num_blocks=3,
                                               offset=p.offset, block_stride=slot.page_size_bytes)
        full_name = next(n for n in views if n.endswith('.long_kv_cache'))
        index_name = next(n for n in views if n.endswith('.indexer.k_cache'))
        kv = views[full_name]
        key, scale = views[index_name]
        kv[1].fill_(7)
        key[1].fill_(3)
        scale[1].fill_(2)
        assert torch.all(kv[1] == 7) and torch.all(key[1] == 3) and torch.all(scale[1] == 2)
        for name, view in views.items():
            if name not in (full_name, index_name):
                assert not isinstance(view, tuple)
                view[2].fill_(9)
                assert torch.all(kv[1] == 7) and torch.all(key[1] == 3) and torch.all(scale[1] == 2)
        assert not raw[:slot.page_size_bytes].any(), 'Reserved null block was modified'
        rows.append({'page_size_bytes': slot.page_size_bytes, 'resources': len(slot.placements),
                     'kv_index_disjoint': True, 'alias_distinct_live_blocks_exact': True,
                     'reserved_null_block_intact': True})
    rejected = []
    for label, key in (
        ('missing-kv', 'model.layers.2.self_attn.long_kv_cache'),
        ('missing-state', 'model.layers.4.self_attn.compressor_state'),
        ('missing-swa', 'model.layers.7.self_attn.swa_cache'),
        ('missing-index', 'model.layers.5.self_attn.indexer.k_cache'),
    ):
        corrupt = dict(specs)
        del corrupt[key]
        try:
            cache.plan_cache_slots(corrupt)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError(('Incomplete cache topology was accepted', label))
    return {'passed': True, 'diagnostic_only': True, 'resources': len(specs),
            'slots': rows, 'rejected_incomplete_topologies': rejected,
            'original_source_sha256': cache.TINY_CACHE_PLAN_SOURCE_SHA256,
            'scope': 'CPU real-spec byte layout; NPU and full-model prefix gates still required'}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('TINY_CACHE_PLAN_VALIDATED', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
