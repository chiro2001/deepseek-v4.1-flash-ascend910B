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
    assert len(prefill) == 1 and decode
    assert prefill[0]["computed_before"] == 0
    assert prefill[0]["prompt_tokens"] == prefill[0]["scheduled_target_tokens"] == PROMPT
    assert prefill[0]["verified_draft_tokens"] == 0
    assert all(row["prompt_tokens"] == PROMPT and row["scheduled_target_tokens"] == K + 1
               and row["verified_draft_tokens"] == K for row in decode)
    assert all(row["draft_calls"] == 1 for row in rows)
    assert [token for row in rows for token in row["sampled_token_ids"]][:OUTPUT] == tokens
    return {"target_decode_rounds": len(decode), "actual_draft_calls": len(rows),
            "initial_actual_draft_calls": prefill[0]["draft_calls"],
            "terminal_actual_draft_calls": decode[-1]["draft_calls"],
            "decode_tokens": OUTPUT - len(prefill[0]["sampled_token_ids"]),
            "round_count_derived_from_fresh_receipts": True}


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
