#!/usr/bin/env python3
"""给任意 V4.1 `core/deepseek_v41.py` 套上 [V41-KV32-CAP] 上限语义（幂等）。

用法：python3 apply_kv32_cap.py <输入文件> <输出文件>
只动 `allocate_cache_config` 一处 + 追加 `_kv32_safe_blocks`，与 slot 布局无关
⇒ 对 tiny 底本（DCP 旁路版）和交付镜像底本都适用。
"""
import pathlib, sys

OLD = '''def allocate_cache_config(vllm_config, groups, available_memory):
    """Allocate four independent layer slots backed by one global block-ID pool."""
    slots = cache_slots_from_groups(groups)
    capacity = available_memory // sum(slot.page_size_bytes for slot in slots)
    num_blocks = may_override_num_blocks(vllm_config, capacity)
    if num_blocks <= 1 or num_blocks > capacity:
        raise ValueError("Insufficient V4.1 cache memory (including reserved null block), or unsafe block override")
    return num_blocks, ['''

NEW = '''def _kv32_safe_blocks(slots):
    """由几何算出的 32 位页偏移安全块数；返回 None = 明确关掉。

    判据取**所有 slot 里最紧的那个**（每个 slot 的块号都按自己的页步长参与寻址）。
    """
    import os

    raw = os.environ.get("V41_KV_MAX_BLOCKS", "auto").strip().lower()
    # 只有**显式**写 off 才关掉安全网：`0` 按"没设上限"处理（= auto），
    # 免得"手滑写 0"把 32 位回绕的保护静默摘掉。
    if raw in ("off", "none", "disable"):
        return None
    auto = min((1 << 32) // slot.page_size_bytes for slot in slots)
    if raw in ("", "auto", "0"):
        return auto
    try:
        want = int(raw)
    except ValueError as exc:
        raise ValueError(f"V41_KV_MAX_BLOCKS={raw!r} 非法（应为 auto / off / 正整数）") from exc
    if want <= 0:
        return auto
    return min(want, auto)  # 显式值只允许**更收紧**，不允许越过安全上限


def allocate_cache_config(vllm_config, groups, available_memory):
    """Allocate four independent layer slots backed by one global block-ID pool."""
    slots = cache_slots_from_groups(groups)
    capacity = available_memory // sum(slot.page_size_bytes for slot in slots)
    num_blocks = may_override_num_blocks(vllm_config, capacity)
    if num_blocks <= 1 or num_blocks > capacity:
        raise ValueError("Insufficient V4.1 cache memory (including reserved null block), or unsafe block override")
    # =========================================================================
    # ★ [V41-KV32-CAP 2026-10-06] **上限语义**：块数取 min(自动 profiling 的块数, 32 位安全上限)。
    #
    # 为什么必须这样而不是"钉一个 KV_CACHE_MEMORY_BYTES"：
    #   * 回绕判据是**每个 slot 自己的页步长**（block_id × page_size ≥ 2³² 就串块），
    #     所以安全上限可由**几何自己**算出：floor(2³² / max_slot_page_size)；
    #   * 钉死字节值会在**显存更小的机器上直接 OOM**：实测 A3 每 rank 可用 KV 16.60 GiB，
    #     而 A2（910B3）只有 **14.40 GiB**（a2/docs/A2-DEPLOY-NOW.md §B0）⇒
    #     把 A3 验证过的 16 GiB 常量搬到 A2 = 起不来；
    #   * 上限语义天然自适应：小显存机器 profiling 给得少 ⇒ 取小值、不 OOM；
    #     大显存机器给得多 ⇒ 被夹到安全上限、不放行回绕。
    #
    # 关掉：`V41_KV_MAX_BLOCKS=off`（只有明确知道在越界时才用）。
    # 收紧：`V41_KV_MAX_BLOCKS=<N>`（只会更小；写超上限的值会被夹回上限）。
    # =========================================================================
    cap = _kv32_safe_blocks(slots)
    if cap is not None and num_blocks > cap:
        print(
            f"[V41-KV32-CAP] 块数 {num_blocks} → {cap}：自动 profiling 给得比 32 位安全上限多。"
            f" 每槽页步长 {[s.page_size_bytes for s in slots]}，"
            f"上限 = floor(2^32 / {max(s.page_size_bytes for s in slots)}) = {cap}。"
            f"要关掉本夹取设 V41_KV_MAX_BLOCKS=off（会放行块号回绕 ⇒ 静默读错）",
            flush=True,
        )
        num_blocks = cap
    return num_blocks, ['''

src = pathlib.Path(sys.argv[1]).read_text()
if "_kv32_safe_blocks" in src:
    pathlib.Path(sys.argv[2]).write_text(src)
    print(f"{sys.argv[1]}: 已含 cap，原样输出")
    raise SystemExit(0)
if OLD not in src:
    print(f"ERROR: 锚点未命中 {sys.argv[1]}", file=sys.stderr)
    raise SystemExit(2)
pathlib.Path(sys.argv[2]).write_text(src.replace(OLD, NEW, 1))
print(f"{sys.argv[1]} → {sys.argv[2]}: cap 已套用")
