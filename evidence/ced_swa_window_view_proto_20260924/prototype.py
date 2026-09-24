#!/usr/bin/env python3
"""CPU-only page/range prototype for CED replay SWA views.

No vLLM, model, CANN, or NPU dependency. This checks only token/page geometry;
it does not establish operator correctness or recover missing upper-layer KV.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path


@dataclass(frozen=True)
class View:
    name: str
    global_start: int
    global_end: int
    page_aligned_start: int
    local_seq_len: int
    query_start_local: int
    page_indices: list[int]
    local_page_count: int


def cdiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def make_views(prompt_len: int, replay_tokens: int, window: int, block_size: int) -> dict:
    """Build replay-only scratch and full-left-window view plans.

    ``prompt_len`` is the original prompt length. CED P loads ``N-1`` SWA
    prefix tokens; D replays the last ``replay_tokens`` of that prefix before
    it computes the uncached prompt tail.
    """
    if min(prompt_len, replay_tokens, window, block_size) <= 0:
        raise ValueError("all lengths and block size must be positive")
    prefix_end = prompt_len - 1  # exclusive end of P-loaded positions
    replay_start = max(0, prefix_end - replay_tokens)
    replay_end = prefix_end
    q_len = replay_end - replay_start

    # Upper-SWA groups have no P-supplied KV in the current CED connector.
    # Compact only D-computed replay rows into an offset-zero scratch view.
    replay_only = View(
        name="replay_only_upper",
        global_start=replay_start,
        global_end=replay_end,
        page_aligned_start=replay_start,
        local_seq_len=q_len,
        query_start_local=0,
        page_indices=list(range(replay_start // block_size, cdiv(replay_end, block_size))),
        local_page_count=cdiv(q_len, block_size),
    )

    # Lower-SWA/full-window reference: retain all keys visible to the first
    # replay query, plus the current replay chunk. Align down for PA pages;
    # ori_win_left masks any additional alignment-padding rows.
    visible_start = max(0, replay_start - (window - 1))
    aligned_start = visible_start // block_size * block_size
    page_end = cdiv(replay_end, block_size)
    full_window = View(
        name="full_swa_visible_window",
        global_start=visible_start,
        global_end=replay_end,
        page_aligned_start=aligned_start,
        local_seq_len=replay_end - aligned_start,
        query_start_local=replay_start - aligned_start,
        page_indices=list(range(aligned_start // block_size, page_end)),
        local_page_count=page_end - aligned_start // block_size,
    )

    # The original SMLA query rows occupy the suffix of the local sequence.
    assert full_window.local_seq_len - q_len == full_window.query_start_local
    # Causal sliding window clips the aligned padding and retains exactly the
    # same global visible start for the first query.
    assert full_window.page_aligned_start + max(
        0, full_window.query_start_local - (window - 1)
    ) == visible_start
    assert replay_only.local_seq_len == q_len
    assert replay_only.query_start_local == 0

    # Transfer window required for lower-SWA full visibility is replay length
    # plus the left overlap, not just the normal decode window.
    retained_tokens = window - 1 + replay_tokens
    current_connector_pages = cdiv(window, block_size) + 1
    overlap_connector_pages = cdiv(retained_tokens, block_size) + 1

    # SlidingWindowManager skips from the processed cursor. extra_retained is
    # the mechanism needed on both P and D to keep the overlap pages alive.
    p_processed = prefix_end
    skipped_no_extra = max(0, p_processed - window + 1)
    skipped_with_overlap = max(0, p_processed - window + 1 - replay_tokens)
    total_prefix_pages = cdiv(prefix_end, block_size)
    p_pages_no_extra = list(range(skipped_no_extra // block_size, total_prefix_pages))
    p_pages_with_overlap = list(range(skipped_with_overlap // block_size, total_prefix_pages))

    def connector_tail(clip_pages: int) -> list[int]:
        # The manager preserves the logical row count; pages before the
        # retained suffix are represented by null blocks in the block table.
        return list(range(max(0, total_prefix_pages - clip_pages), total_prefix_pages))

    current_connector_entries = connector_tail(current_connector_pages)
    overlap_connector_entries = connector_tail(overlap_connector_pages)
    current_transfer = [p for p in current_connector_entries if p in p_pages_no_extra]
    overlap_transfer = [p for p in overlap_connector_entries if p in p_pages_with_overlap]

    # The receiver exposes only unhashed/non-null destination blocks. The
    # current connector trims the remote list from the left when it is longer,
    # then zips the suffix with destination blocks in logical order.
    current_remote_for_worker = current_connector_entries[-len(p_pages_no_extra):]
    overlap_remote_for_worker = overlap_connector_entries[-len(p_pages_with_overlap):]
    current_transfer_pairs = [
        {"remote_logical_page": remote, "destination_logical_page": local,
         "remote_has_data": remote in p_pages_no_extra}
        for remote, local in zip(current_remote_for_worker, p_pages_no_extra)
    ]
    overlap_transfer_pairs = [
        {"remote_logical_page": remote, "destination_logical_page": local,
         "remote_has_data": remote in p_pages_with_overlap}
        for remote, local in zip(overlap_remote_for_worker, p_pages_with_overlap)
    ]
    current_pair_by_dst = {p["destination_logical_page"]: p for p in current_transfer_pairs}
    overlap_pair_by_dst = {p["destination_logical_page"]: p for p in overlap_transfer_pairs}

    needed_lower_pages = full_window.page_indices
    missing_current = sorted(set(needed_lower_pages) - set(current_transfer))
    missing_overlap = sorted(set(needed_lower_pages) - set(overlap_transfer))
    first_mapped_page = min(
        needed_lower_pages + current_connector_entries + overlap_connector_entries,
        default=0,
    )
    last_mapped_page = max(
        needed_lower_pages + current_connector_entries + overlap_connector_entries,
        default=-1,
    )
    lower_page_mapping = [
        {
            "logical_page": page,
            "prompt_page_resident_after_swa_cleanup": page in p_pages_no_extra,
            "prompt_page_resident_with_replay_overlap": page in p_pages_with_overlap,
            "required_by_full_window_view": page in needed_lower_pages,
            "in_current_connector_tail": page in current_connector_entries,
            "current_transfer_has_data": page in current_transfer,
            "current_transfer_pair": current_pair_by_dst.get(page),
            "in_overlap_connector_tail": page in overlap_connector_entries,
            "overlap_transfer_has_data": page in overlap_transfer,
            "overlap_transfer_pair": overlap_pair_by_dst.get(page),
            "runtime_physical_id_source": f"common.block_table[request,{page}]",
        }
        for page in range(first_mapped_page, last_mapped_page + 1)
    ]

    # A scratch-copy plan explicitly maps arbitrary global token positions to
    # compact offset-zero pages. This is needed for replay-only upper KV when
    # replay_start is not block aligned; a plain table slice would expose the
    # preceding invalid bytes in the first physical page.
    scratch_copy = []
    for global_pos in range(replay_start, replay_end):
        scratch_offset = global_pos - replay_start
        scratch_copy.append({
            "global_token": global_pos,
            "source_logical_page": global_pos // block_size,
            "source_page_offset": global_pos % block_size,
            "scratch_page": scratch_offset // block_size,
            "scratch_page_offset": scratch_offset % block_size,
        })

    return {
        "prompt_len": prompt_len,
        "p_loaded_prefix_tokens": prefix_end,
        "replay_start": replay_start,
        "replay_end_exclusive": replay_end,
        "replay_query_tokens": q_len,
        "window": window,
        "block_size": block_size,
        "views": {
            "replay_only_upper": asdict(replay_only),
            "full_swa_visible_window": asdict(full_window),
        },
        "lower_swa_transfer": {
            "retained_tokens_with_replay_overlap": retained_tokens,
            "current_extra_retained_tokens": 0,
            "required_extra_retained_tokens": replay_tokens,
            "current_connector_clip_pages": current_connector_pages,
            "required_connector_clip_pages": overlap_connector_pages,
            "p_pages_kept_without_extra": p_pages_no_extra,
            "p_pages_kept_with_overlap": p_pages_with_overlap,
            "current_connector_tail_logical_pages_including_nulls": current_connector_entries,
            "overlap_connector_tail_logical_pages_including_nulls": overlap_connector_entries,
            "current_remote_to_destination_page_pairs": current_transfer_pairs,
            "overlap_remote_to_destination_page_pairs": overlap_transfer_pairs,
            "current_transfer_page_indices": current_transfer,
            "overlap_transfer_page_indices": overlap_transfer,
            "full_window_required_page_indices": needed_lower_pages,
            "page_mapping_table": lower_page_mapping,
            "missing_pages_current": missing_current,
            "missing_pages_with_overlap": missing_overlap,
        },
        "upper_scratch_copy": {
            "scratch_pages": replay_only.local_page_count,
            "copy_row_count": len(scratch_copy),
            "first_row": scratch_copy[0] if scratch_copy else None,
            "last_row": scratch_copy[-1] if scratch_copy else None,
        },
    }


def costs(window: int = 128, replay: int = 128, upper_layers: int = 20,
          full_layers: int = 40, hidden_size: int = 7168, hc_mult: int = 4,
          dtype_bytes: int = 2) -> dict:
    overlap_burnin = upper_layers * (window - 1)
    upper_h20_tokens = overlap_burnin + replay
    full_d_burnin = full_layers * (window - 1)
    full_d_tokens = full_d_burnin + replay
    h20_bytes = upper_h20_tokens * hc_mult * hidden_size * dtype_bytes
    pre_mix_bytes = upper_h20_tokens * hc_mult * 4  # FP32 coefficients
    return {
        "route_A_128_bounded": {
            "d_full_layers_tokens": replay,
            "extra_d_layer_token_ops": 0,
            "extra_p_model_flops": 0,
            "extra_retained_tokens_per_swa_group": replay,
        },
        "route_B_h20_transfer_burnin": {
            "upper_swa_layers": upper_layers,
            "burnin_tokens": overlap_burnin,
            "h20_transfer_tokens": upper_h20_tokens,
            "hc_mult": hc_mult,
            "h20_shape": [upper_h20_tokens, hc_mult, hidden_size],
            "h20_dtype_bytes": dtype_bytes,
            "h20_transfer_bytes_full_copy": h20_bytes,
            "pre_mix_shape": [upper_h20_tokens, hc_mult],
            "pre_mix_dtype": "FP32",
            "pre_mix_transfer_bytes_full_copy": pre_mix_bytes,
            "activation_transfer_bytes_full_copy": h20_bytes + pre_mix_bytes,
            "activation_transfer_mib_full_copy": (h20_bytes + pre_mix_bytes) / (1024**2),
            "tp_partition_or_replication": "unknown; bytes are one complete activation copy",
            "upper_layer_token_ops_total": upper_layers * upper_h20_tokens,
            "upper_layer_token_ops_current_replay": upper_layers * replay,
            "upper_layer_work_multiplier": upper_h20_tokens / replay,
            "extra_upper_layer_token_ops": upper_layers * overlap_burnin,
            "extra_p_model_flops": 0,
            "extra_p_activation_capture_tokens": upper_h20_tokens,
        },
        "route_B_full40_token_burnin": {
            "burnin_tokens": full_d_burnin,
            "d_replay_tokens_total": full_d_tokens,
            "full_layer_token_ops_total": full_layers * full_d_tokens,
            "full_layer_token_ops_current_replay": full_layers * replay,
            "full_d_work_multiplier": full_d_tokens / replay,
            "extra_full_layer_token_ops": full_layers * full_d_burnin,
            "extra_p_activation_transfer": 0,
        },
        "assumptions": [
            "Each layer's SWA has left radius window-1 and no other D-produced cross-token state.",
            "CED CSA/indexer source remains the P-produced global cache during replay.",
            "The H20 route requires an activation side channel and an upper-only D prefill path.",
        ],
    }


def main() -> None:
    # N is the original prompt length; P computes N-1 and D must reproduce the
    # suffix before consuming the uncached prompt-tail token.
    cases = [
        make_views(1_048_576, replay_tokens=128, window=128, block_size=128),
        make_views(1_048_575, replay_tokens=128, window=128, block_size=128),
    ]
    for item in cases:
        xfer = item["lower_swa_transfer"]
        assert not xfer["missing_pages_with_overlap"], item
        assert xfer["missing_pages_current"], item
        assert item["upper_scratch_copy"]["copy_row_count"] == 128
    assert cases[0]["views"]["full_swa_visible_window"]["page_indices"] == [8190, 8191]
    assert cases[1]["views"]["full_swa_visible_window"]["page_indices"] == [8189, 8190, 8191]
    assert cases[0]["lower_swa_transfer"]["missing_pages_current"] == [8190]
    assert cases[1]["lower_swa_transfer"]["missing_pages_current"] == [8189]
    assert costs()["route_B_h20_transfer_burnin"]["burnin_tokens"] == 2540
    assert costs()["route_B_full40_token_burnin"]["d_replay_tokens_total"] == 5208
    result = {
        "scope": "CPU-only page/range math; no CANN, NPU, production edit, or accuracy claim",
        "cases": cases,
        "costs": costs(),
    }
    out = Path(__file__).with_name("static_results.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)
    for case in cases:
        v = case["views"]
        t = case["lower_swa_transfer"]
        print(
            f"N={case['prompt_len']} replay={case['replay_start']}..{case['replay_end_exclusive'] - 1} "
            f"full-visible={v['full_swa_visible_window']['global_start']}..{v['full_swa_visible_window']['global_end'] - 1} "
            f"pages={v['full_swa_visible_window']['page_indices']} "
            f"transfer2 missing={t['missing_pages_current']} transfer3 missing={t['missing_pages_with_overlap']}"
        )


if __name__ == "__main__":
    main()
