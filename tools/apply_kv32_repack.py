#!/usr/bin/env python3
"""给任意 V4.1 `core/deepseek_v41.py` 套上 [V41-KV32-REPACK]（幂等）。

用法：python3 apply_kv32_repack.py <输入> <输出>
与底本无关：只替换 plan_cache_slots 里"构造 placements 并 append"的那一段，
把 ratio-1 源的 index 平面挪到别的 slot 的空闲区 ⇒ 四槽页步长拉平 ⇒ 32 位上限 +12.7%。
（本几何下 `capacity == alias_max`，与 tiny 版逐字等价；交付版写法更一般。）
"""
import pathlib, sys

OLD = '''        placements = [
            CachePlacement(kv_name, 0, kv_bytes),
            CachePlacement(index_name, kv_bytes, capacity - kv_bytes),
            *(CachePlacement(name, 0, capacity) for name in aliases),
        ]
        slots.append(CacheSlot(capacity, tuple(placements)))'''

NEW = '''        _plan.append(
            {
                "kv_name": kv_name,
                "index_name": index_name,
                "kv_bytes": kv_bytes,
                "index_bytes": index_bytes,
                "aliases": tuple(aliases),
                "alias_max": _alias_max,
                "capacity": capacity,
            }
        )
    # =========================================================================
    # ★ [KV32-SLOT-REPACK] index 平面可以挪到**别的 slot 的空闲区**。
    #
    # 动机（32 位页偏移）：`block_id × slot.page_size_bytes` 若 ≥ 2^32 会回绕 ⇒
    #   KV 静默串块（长上下文间歇性偏离）。判据是**每个 slot 自己的页步长**：
    #   现状 ratio-1 槽 = KV 131072 + index 16640 = 147712 ⇒ 上限 floor(2^32/147712)=29076；
    #   而其余三个槽被 DSpark draft 顶到 131072 ⇒ 上限本可为 floor(2^32/131072)=32768（+12.7%）。
    #
    # 做法：若某槽的 (kv+index) 超过它的别名容量 `alias_max`（即 kv+index 才是 binding 项），
    #   就把 index 平面挪到另一个槽的空闲区；原槽容量降回 `alias_max`。
    #
    # 为什么安全：placement → (offset, stride=所在槽 page_size_bytes) 由本函数决定，
    #   `reshape_cache` 只要求 offset + 平面大小 ≤ block_stride；挪走后 index 的**平面大小**
    #   不变 ⇒ request_blocks / 每请求块需求不变（= npr 口径不变）。
    # =========================================================================
    _used = [p["kv_bytes"] + p["index_bytes"] for p in _plan]
    _extra: dict[int, list[tuple[int, int, int]]] = {}
    _moved: set[int] = set()
    for i, p in enumerate(_plan):
        if p["kv_bytes"] + p["index_bytes"] <= p["alias_max"]:
            continue  # index 不 binding，保持原样（零改动）
        for j, q in enumerate(_plan):
            if j == i or j in _moved or j in _extra:
                continue
            if q["capacity"] - _used[j] >= p["index_bytes"]:
                _extra[j] = [(i, _used[j], p["index_bytes"])]
                _used[j] += p["index_bytes"]
                _moved.add(i)
                break
    for i, p in enumerate(_plan):
        _cap = max(p["kv_bytes"], p["alias_max"]) if i in _moved else p["capacity"]
        placements = [CachePlacement(p["kv_name"], 0, p["kv_bytes"])]
        if i not in _moved:
            placements.append(CachePlacement(p["index_name"], p["kv_bytes"], _cap - p["kv_bytes"]))
        placements.extend(CachePlacement(name, 0, _cap) for name in p["aliases"])
        for _src, _off, _sz in _extra.get(i, ()):
            placements.append(CachePlacement(_plan[_src]["index_name"], _off, _sz))
        slots.append(CacheSlot(_cap, tuple(placements)))'''

src = pathlib.Path(sys.argv[1]).read_text()
if '_plan.append(' in src or "KV32-SLOT-REPACK" in src:
    pathlib.Path(sys.argv[2]).write_text(src)
    print(f"{sys.argv[1]}: 已含 repack，原样输出")
    raise SystemExit(0)
# 需要在 slots 循环前插入 _plan 初始化，并把旧 capacity 行去掉（新版自己算）
if "_plan: list[dict] = []" not in src:
    src = src.replace("    slots = []\n", "    slots = []\n    _plan: list[dict] = []\n", 1)
old_cap = '        capacity = max(_kv_idx_bytes, *(sum(_cache_plane_sizes(specs[n])) for n in aliases))\n'
# 新的 capacity 必须**在 draft 块之前**算好（draft 块会读 capacity 做校验），
# 并在 append draft 之后把 draft 也计入 alias_max。
new_cap = (
    '        _alias_max = max((sum(_cache_plane_sizes(specs[n])) for n in aliases), default=0)\n'
    '        capacity = max(_kv_idx_bytes, _alias_max)\n'
)
if old_cap in src:
    src = src.replace(old_cap, new_cap, 1)
else:
    print("WARN: 旧 capacity 行未命中", file=sys.stderr)
# draft 的容量校验之后，把 draft 计入 binding
old_alias = '            aliases.append(draft_name)\n'
new_alias = ('            aliases.append(draft_name)\n'
             '            _alias_max = max(_alias_max, sum(_cache_plane_sizes(draft_spec)))\n'
             '            capacity = max(capacity, _alias_max)\n')
if old_alias in src:
    src = src.replace(old_alias, new_alias, 1)
if OLD not in src:
    print(f"ERROR: 锚点未命中 {sys.argv[1]}", file=sys.stderr)
    raise SystemExit(2)
pathlib.Path(sys.argv[2]).write_text(src.replace(OLD, NEW, 1))
print(f"{sys.argv[1]} → {sys.argv[2]}: repack 已套用")
