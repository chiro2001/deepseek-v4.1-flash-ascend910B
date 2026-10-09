"""Pure CED cache geometry helpers shared by launchers and runtime adapters."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable


def members(spec: Any) -> tuple[Any, ...]:
    wrapped = getattr(spec, "kv_cache_specs", None)
    return tuple(wrapped.values()) if wrapped else (spec,)


def prefix_cacheable(spec: Any) -> bool:
    return all(
        bool(getattr(s, "prefix_cacheable", True))
        and bool(getattr(s, "participates_in_prefix_caching", True))
        for s in members(spec)
    )


@dataclass(frozen=True)
class PrefillGroups:
    participating: tuple[int, ...]
    missing_swa: tuple[int, ...]
    recurrent: tuple[int, ...]


def prefill_groups(groups: Iterable[Any]) -> PrefillGroups:
    """Keep group positions; never offload uncomputed decoder SWA/state."""
    participating, missing, recurrent = [], [], []
    for idx, group in enumerate(groups):
        specs = members(group.kv_cache_spec)
        layers = {
            int(m.group(1))
            for name in group.layer_names
            if (m := re.search(r"\.layers\.(\d+)\.", name))
        }
        is_swa = all(getattr(s, "sliding_window", None) == 128 for s in specs)
        if is_swa and layers and min(layers) >= 20:
            missing.append(idx)
        elif not prefix_cacheable(group.kv_cache_spec):
            recurrent.append(idx)
        elif is_swa and not layers:
            raise ValueError("CED prefill cannot offload a draft-only SWA group")
        else:
            participating.append(idx)
    if not participating:
        raise ValueError("CED prefill has no prefix-cacheable KV groups")
    return PrefillGroups(tuple(participating), tuple(missing), tuple(recurrent))


def alignment_unit(block_sizes: Iterable[int]) -> int:
    sizes = tuple(int(s) for s in block_sizes)
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError("KV alignment requires positive runtime block sizes")
    return math.lcm(*sizes)


def tail_boundary(num_tokens: int, available_tokens: int, unit: int) -> int:
    """Aligned reusable prefix, always leaving >=1 token for P execution."""
    if num_tokens < 1 or available_tokens < 0 or unit < 1:
        raise ValueError("Invalid CED prompt/cache boundary")
    return min(available_tokens, num_tokens - 1) // unit * unit


def bound_store_keys(keys, source_blocks, pending_blocks, limit):
    """Bound asynchronous stores by unique physical source pages."""
    if limit <= 0:
        raise ValueError("Pending store limit must be positive")
    selected = []
    pinned = set(pending_blocks)
    for key in keys:
        candidate = pinned | source_blocks[key]
        if len(candidate) <= limit:
            selected.append(key)
            pinned = candidate
    return selected, pinned
