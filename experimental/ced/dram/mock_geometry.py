"""Describe exact KV payload views for a model-free Mooncake consumer."""

from __future__ import annotations

import torch
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector import (
    _canonicalize_split_attention_cache,
)

from .contract import prefill_groups, prefix_cacheable


def describe_mock_segments(kv_cache_config, kv_caches):
    contract = prefill_groups(kv_cache_config.kv_cache_groups)
    result = []
    seen = set()
    for idx, group in enumerate(kv_cache_config.kv_cache_groups):
        if idx in contract.missing_swa:
            continue
        wrapped = getattr(group.kv_cache_spec, "kv_cache_specs", {})
        for layer_name in group.layer_names:
            spec = wrapped.get(layer_name, group.kv_cache_spec)
            cache = kv_caches[layer_name]
            if isinstance(cache, torch.Tensor):
                raw = torch.empty(0, dtype=torch.int8, device=cache.device).set_(cache.untyped_storage())
                view = torch.as_strided(
                    raw, (kv_cache_config.num_blocks, spec.page_size_bytes),
                    (cache.stride(0) * cache.element_size(), 1),
                    cache.storage_offset() * cache.element_size(),
                )
                views = ((view, spec.unpadded_page_size_bytes),)
            else:
                views = _canonicalize_split_attention_cache(
                    cache, kv_cache_config.num_blocks, spec.unpadded_page_size_bytes
                )
            for part, (view, size) in enumerate(views):
                key = (idx, view.data_ptr(), view.stride(0), size)
                if key in seen:
                    continue
                seen.add(key)
                result.append({
                    "group": idx,
                    "component": f"{layer_name}:{part}",
                    "base": view.data_ptr(),
                    "stride": view.stride(0),
                    "page_bytes": int(size),
                    "num_blocks": int(view.shape[0]),
                    "tokens_per_block": int(group.kv_cache_spec.block_size),
                    "prefix_cacheable": prefix_cacheable(group.kv_cache_spec),
                })
    return result
