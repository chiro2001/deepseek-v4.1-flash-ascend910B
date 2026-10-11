"""Independent CPU contract for the future 2048/256 native cache gate."""
PROMPT = 2048
OUTPUT = 256
K = 5
WINDOW = 128
MAX_MODEL_LEN = 4096
CAPTURE_SEQ_LEN = 3072
MAX_BATCHED_TOKENS = 2048


def validate_configuration(max_model_len, max_batched_tokens, capture_seq_len, window, k):
    assert (window, k) == (WINDOW, K)
    assert max_model_len >= PROMPT + OUTPUT + K + 1
    assert max_batched_tokens >= PROMPT
    assert PROMPT + OUTPUT + K <= capture_seq_len
    assert capture_seq_len + K <= max_model_len
    return {"prompt_tokens": PROMPT, "completion_tokens": OUTPUT, "K": K,
            "window": WINDOW, "max_model_len": max_model_len,
            "max_batched_tokens": max_batched_tokens, "capture_seq_len": capture_seq_len,
            "maximum_declared_sequence_budget": PROMPT + OUTPUT + K + 1}


def validate_rounds(rows, tokens, request_id):
    assert len(tokens) == OUTPUT and rows and all(row["req_id"] == request_id for row in rows)
    prefill = [row for row in rows if not row["is_decode"]]
    decode = [row for row in rows if row["is_decode"]]
    assert prefill and decode and rows == prefill + decode, "Illegal prefill/decode order"
    computed = 0
    for index, row in enumerate(prefill):
        assert row["computed_before"] == computed and row["prompt_tokens"] == PROMPT
        count = row["scheduled_target_tokens"]
        assert isinstance(count, int) and 0 < count <= MAX_BATCHED_TOKENS
        assert row["verified_draft_tokens"] == 0
        computed += count
        assert computed <= PROMPT
        if computed < PROMPT:
            assert row["sampled_token_ids"] == [], "Partial prefill must not create a sampling seed"
        else:
            assert index == len(prefill) - 1 and len(row["sampled_token_ids"]) == 1
    assert computed == PROMPT
    seeds = [index for index, row in enumerate(prefill) if row["sampled_token_ids"]]
    assert seeds == [len(prefill) - 1], "Exactly one sampleable seed is required"
    assert all(row["prompt_tokens"] == PROMPT and row["scheduled_target_tokens"] == K + 1
               and row["verified_draft_tokens"] == K for row in decode)
    assert all(row["computed_before"] >= PROMPT and 1 <= len(row["sampled_token_ids"]) <= K + 1 for row in decode)
    assert decode[0]["computed_before"] == PROMPT
    assert all(left["computed_before"] + len(left["sampled_token_ids"]) == right["computed_before"]
               for left, right in zip(decode, decode[1:])), "Illegal native decode progress"
    assert all(row["draft_calls"] == 1 for row in rows)
    assert [token for row in rows for token in row["sampled_token_ids"]][:OUTPUT] == tokens
    return {"target_decode_rounds": len(decode), "actual_draft_calls": len(rows),
            "prefill_rounds": len(prefill), "prefill_chunks": [row["scheduled_target_tokens"] for row in prefill],
            "sampling_seed_round_index": seeds[0], "partial_prefill_actual_draft_calls": len(prefill) - 1,
            "sampling_seed_actual_draft_calls": prefill[seeds[0]]["draft_calls"],
            "all_prefill_actual_draft_calls": len(prefill),
            "initial_actual_draft_calls": prefill[0]["draft_calls"],
            "terminal_actual_draft_calls": decode[-1]["draft_calls"],
            "decode_tokens": OUTPUT - len(prefill[seeds[0]]["sampled_token_ids"]),
            "round_count_derived_from_fresh_receipts": True}


def validate_native_helpers(raw, arm):
    rows, helper = raw["counter"]["rounds"], raw["helper"]
    prefill = [row for row in rows if not row["is_decode"]]
    expected = len(rows) if arm == "off" else len(prefill)
    assert arm in ("off", "on")
    assert helper["native_helper_calls_this_request"] == expected
    assert helper["native_helper_successful_returns_this_request"] == expected
    history = helper["helper_history_this_request"]
    assert len(history) == expected
    assert all(not row["capturing"] and row["runtime_mode"] == "NONE" and row["native_helper_returned"] for row in history)
    assert helper["cold_capture_helper_calls_global"] == helper["cold_capture_helper_successful_returns_global"] == 1
    expected_rows = rows if arm == "off" else prefill
    actual_keys = [(row["actual_round_at_entry"]["req_id"], row["actual_round_at_entry"]["execute_begin_s"]) for row in history]
    assert actual_keys == [(row["req_id"], row["execute_begin_s"]) for row in expected_rows]
    assert raw["counter"]["terminal_sync_count"] == 1
    assert raw["counter"]["terminal_device_done_s"] >= max(row["draft_propose_done_s"] for row in rows)
    return {"native_helper_calls": expected, "native_helper_successful_returns": expected,
            "prefill_native_helper_calls": len(prefill), "all_prefill_native_calls_retained": True,
            "initial_terminal_and_partial_prefill_complete_at_terminal_fence": True}


def validate_query_positions(positions, scheduled_round, seq_len):
    assert len(positions) == K and all(isinstance(value, int) for value in positions)
    assert positions == list(range(positions[0], positions[0] + K))
    assert positions[0] >= PROMPT and positions[-1] < MAX_MODEL_LEN
    if scheduled_round["is_decode"]:
        assert scheduled_round["computed_before"] <= positions[0]
        assert positions[0] <= scheduled_round["computed_before"] + scheduled_round["scheduled_target_tokens"]
    else:
        assert positions[0] == PROMPT
    assert seq_len == positions[-1] + 1, "Native reader sequence length is stale"


def visible_slots(block_table_row, seq_len, kernel_block_size, cache_blocks, null_block_id=None):
    assert kernel_block_size > 0 and cache_blocks > 0 and seq_len >= PROMPT + K
    assert seq_len <= MAX_MODEL_LEN
    prefix_len = seq_len - K
    start = max(prefix_len - WINDOW, 0)
    logical_positions = list(range(start, seq_len))
    slots = []
    for position in logical_positions:
        column, offset = divmod(position, kernel_block_size)
        assert column < len(block_table_row), "Native block table cannot cover the visible logical position"
        block = block_table_row[column]
        assert isinstance(block, int) and 0 <= block < cache_blocks
        if null_block_id is not None:
            assert block != null_block_id, "A required visible page is the native null block"
        slots.append(block * kernel_block_size + offset)
    return logical_positions, slots


def validate_reader_indices(indices, block_table_row, seq_len, kernel_block_size, cache_blocks, null_block_id=None):
    assert len(indices) == K
    logical, slots = visible_slots(block_table_row, seq_len, kernel_block_size, cache_blocks, null_block_id)
    width = ((WINDOW + K + 127) // 128) * 128
    expected = slots + [-1] * (width - len(slots))
    for row in indices:
        assert len(row) == width and row == expected
        assert all(value >= -1 for value in row)
    return {"logical_positions": logical, "native_selected_slot_ids_with_duplicates": slots * K,
            "selected_slots_count_with_duplicates": len(slots) * K,
            "native_padding_entries": (width - len(slots)) * K,
            "indices_width": width, "kernel_block_size": kernel_block_size,
            "seq_len": seq_len, "cache_blocks": cache_blocks,
            "logical_block_columns_used": sorted({position // kernel_block_size for position in logical}),
            "finite_filter_used": False, "unretained_prefix_scanned": False}
