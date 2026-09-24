#!/usr/bin/env python3
"""CPU-only simulation of PagedNgramHistory physical-page reuse.

It mirrors the current host mirror's relevant semantics. It does not import
Torch/CANN, run a model, or reproduce allocator timing from A3.
"""

from __future__ import annotations

import json
from pathlib import Path


BLOCK = 128
LOOKBACK = 4
PAD = 0
PROMPT_LEN = 143_963
PREFIX_END = PROMPT_LEN - 1
REPLAY_TOKENS = 128
REPLAY_START = max(0, PREFIX_END - REPLAY_TOKENS)
REPLAY_END = PREFIX_END
MULTIPLIERS = (3, 5, 7, 11)
PRIMES = (101, 103, 107)
OFFSETS = (0, 101, 204)

assert REPLAY_START == 143_834
assert REPLAY_START % BLOCK == 90


def prompt(seed: int) -> list[int]:
    """Deterministic non-pad stand-ins for compressed token IDs."""
    return [1 + ((seed * 131 + p * 37 + (p // 11) * 13) % 997) for p in range(PROMPT_LEN)]


def hash_history(history: list[int]) -> list[int]:
    """Small one-layer/one-head version of PagedNgramHistory's rolling XOR."""
    rolling = history[0] * MULTIPLIERS[0]
    out = []
    for shift in range(1, LOOKBACK):
        rolling ^= history[shift] * MULTIPLIERS[shift]
        out.append(rolling % PRIMES[shift - 1] + OFFSETS[shift - 1])
    return out


def canonical_history(tokens: list[int], pos: int) -> list[int]:
    return [tokens[pos - shift] if pos - shift >= 0 else PAD for shift in range(LOOKBACK)]


class PageMirror:
    """Subset of the host path: a process-lifetime dict keyed by physical page ID."""

    def __init__(self):
        self.pages: dict[int, list[int]] = {}

    def seed_old_owner(self, physical_page: int, tokens: list[int]) -> None:
        assert len(tokens) == BLOCK
        self.pages[physical_page] = tokens.copy()

    def update_replay(self, tokens: list[int], block_table: list[int]) -> None:
        # Current PagedNgramHistory writes the scheduled rows before reading
        # their lookback. A page row is cleared only on its first sighting.
        for pos in range(REPLAY_START, REPLAY_END):
            page = block_table[pos // BLOCK]
            row = self.pages.get(page)
            if row is None:
                row = [-1] * BLOCK
                self.pages[page] = row
            row[pos % BLOCK] = tokens[pos]

    def history(self, tokens: list[int], block_table: list[int], pos: int,
                canonical_prefix: bool = False) -> tuple[list[int], list[str]]:
        values: list[int] = []
        sources: list[str] = []
        stopped = False
        for shift in range(LOOKBACK):
            previous = pos - shift
            if stopped or previous < 0:
                values.append(PAD)
                sources.append("pad_after_barrier" if stopped else "sequence_start_pad")
                continue
            if canonical_prefix and previous < REPLAY_START:
                token = tokens[previous]
                source = "canonical_prompt_control"
            else:
                page_id = block_table[previous // BLOCK]
                row = self.pages.get(page_id)
                if row is None:
                    # _mirror_row installs an all-barrier row on a first miss.
                    row = [-1] * BLOCK
                    self.pages[page_id] = row
                    source = "unmirrored_page"
                else:
                    source = (
                        "replay_write" if previous >= REPLAY_START
                        else ("stale_or_unverified_page_value" if row[previous % BLOCK] >= 0
                              else "barrier_slot")
                    )
                token = row[previous % BLOCK]
            if token < 0:
                values.append(PAD)
                sources.append(source)
                stopped = True
            else:
                values.append(token)
                sources.append(source)
        return values, sources


def page_table(base: int) -> list[int]:
    return [base + i for i in range((PROMPT_LEN + BLOCK - 1) // BLOCK)]


def run_arm(
    name: str,
    seeds: list[int],
    bases: list[int],
    *,
    stale_seed: int | None = None,
    seed_prestart_from_prompt: bool = False,
    reset_replay_pages_each_request: bool = False,
) -> dict:
    mirror = PageMirror()
    if stale_seed is not None:
        old = prompt(stale_seed)
        replay_logical_page = REPLAY_START // BLOCK
        mirror.seed_old_owner(bases[0] + replay_logical_page, old[
            replay_logical_page * BLOCK : (replay_logical_page + 1) * BLOCK
        ])

    requests = []
    for i, (seed, base) in enumerate(zip(seeds, bases), start=1):
        tokens = prompt(seed)
        blocks = page_table(base)
        if reset_replay_pages_each_request:
            # Fresh allocation control: the requested replay page has no
            # process-local mirror row before the scheduled replay writes.
            mirror.pages.pop(blocks[REPLAY_START // BLOCK], None)
        mirror.update_replay(tokens, blocks)
        rows = []
        for pos in range(REPLAY_START, REPLAY_START + min(4, REPLAY_END - REPLAY_START)):
            actual, sources = mirror.history(
                tokens, blocks, pos,
                canonical_prefix=seed_prestart_from_prompt,
            )
            seeded, seeded_sources = mirror.history(tokens, blocks, pos, canonical_prefix=True)
            expected = canonical_history(tokens, pos)
            expected_hashes = hash_history(expected)
            seeded_hashes = hash_history(seeded)
            assert seeded == expected, (name, i, pos, seeded, expected)
            assert seeded_hashes == expected_hashes, (name, i, pos)
            actual_hashes = hash_history(actual)
            rows.append({
                "position": pos,
                "expected_history": expected,
                "mirror_history": actual,
                "history_sources": sources,
                "canonical_prompt_seed_history": seeded,
                "canonical_prompt_seed_sources": seeded_sources,
                "expected_hash_ids": expected_hashes,
                "mirror_hash_ids": actual_hashes,
                "hash_matches_expected": actual_hashes == expected_hashes,
                "lookup_ids_change_vs_canonical": actual_hashes != expected_hashes,
                "canonical_prompt_seed_hash_ids": seeded_hashes,
                "canonical_prompt_seed_hash_matches": seeded_hashes == expected_hashes,
            })
        if seed_prestart_from_prompt:
            assert all(row["hash_matches_expected"] for row in rows), (name, i)
        if (
            not seed_prestart_from_prompt
            and (reset_replay_pages_each_request or (stale_seed is not None and stale_seed != seed))
        ):
            assert not rows[0]["hash_matches_expected"], (name, i, rows[0])
        if not seed_prestart_from_prompt and stale_seed == seed:
            assert all(row["hash_matches_expected"] for row in rows), (name, i)
        requests.append({
            "request_index": i,
            "prompt_seed": seed,
            "replay_physical_page": blocks[REPLAY_START // BLOCK],
            "replay_start_mod_block": REPLAY_START % BLOCK,
            "rows": rows,
        })
    return {
        "arm": name,
        "page_state": (
            "fresh mirror row for each request" if reset_replay_pages_each_request
            else ("old owner page retained" if stale_seed is not None else "initially empty mirror")
        ),
        "canonical_prestart_seed_enabled": seed_prestart_from_prompt,
        "requests": requests,
    }


def main() -> None:
    # Match the recorded A3 144K handoff: prompt=143,963, P prefix=[0,143,962),
    # D replays [143,834,143,962). The replay start is offset 90 in a 128-token
    # page, so q and the three history tokens q-1..q-3 share the same page.
    result = {
        "scope": "CPU simulation of page mirror semantics; not an A3 allocator trace or model-output test",
        "token_representation": "prompt() emits synthetic compressed token IDs; production maps raw IDs through token_map and maps image tokens to -1",
        "source_case": {
            "evidence": "evidence/ced_p_mask_34fdf08/ced_p_mask_144k.json",
            "prompt_tokens": PROMPT_LEN,
            "p_prefix_tokens": PREFIX_END,
            "replay_tokens": REPLAY_TOKENS,
            "replay_positions_inclusive": [REPLAY_START, REPLAY_END - 1],
            "replay_start_offset_in_block": REPLAY_START % BLOCK,
        },
        "geometry": {
            "prompt_len": PROMPT_LEN,
            "prefix_end_exclusive": PREFIX_END,
            "replay_range": [REPLAY_START, REPLAY_END],
            "block_size": BLOCK,
            "lookback": LOOKBACK,
            "replay_start_mod_block": REPLAY_START % BLOCK,
            "prestart_tokens_needed": LOOKBACK - 1,
        },
        "modeled_code_facts": [
            "page mirror keyed by physical page ID, not request ID or allocation generation",
            "current scheduled tokens are written before lookback hashes are computed",
            "a page is initialized to -1 only on first page-ID sighting; reused pages are not cleared",
            "missing host mirror rows become -1 barriers; present old values are consumed as tokens",
        ],
        "hash_scope": "toy 1-layer/1-head hash (not the model's production prime/multiplier tables); an index change means a different Engram lookup row in this simulation",
        "arms": [
            "missing_page_barrier: allocate an empty host mirror row before replay writes; earlier offsets stay -1 barriers",
            "stale_foreign_owner: reuse one physical page ID containing a different prompt's pre-replay values",
            "canonical_prompt_seed: read positions before replay_start from the full prompt token source; replay positions still use the page mirror",
            "stale_same_prompt_owner: same-prompt reuse control; retained values happen to match this prompt",
        ],
        "scenarios": [
            {
                "scenario": "same_prompt_repeated_4_times_same_physical_page",
                "arms": [
                    run_arm(
                        "missing_page_barrier", [17] * 4, [7700] * 4,
                        reset_replay_pages_each_request=True,
                    ),
                    run_arm(
                        "stale_foreign_owner", [17] * 4, [7700] * 4,
                        stale_seed=991,
                    ),
                    run_arm(
                        "canonical_prompt_seed", [17] * 4, [7700] * 4,
                        stale_seed=991, seed_prestart_from_prompt=True,
                    ),
                    run_arm(
                        "stale_same_prompt_owner", [17] * 4, [7700] * 4,
                        stale_seed=17,
                    ),
                ],
            },
            {
                "scenario": "different_prompts_repeated_4_times_same_physical_page",
                "arms": [
                    run_arm(
                        "missing_page_barrier", [17, 29, 41, 53], [7700] * 4,
                        reset_replay_pages_each_request=True,
                    ),
                    run_arm(
                        "stale_foreign_owner", [17, 29, 41, 53], [7700] * 4,
                        stale_seed=991,
                    ),
                    run_arm(
                        "canonical_prompt_seed", [17, 29, 41, 53], [7700] * 4,
                        stale_seed=991, seed_prestart_from_prompt=True,
                    ),
                ],
            },
        ],
    }
    out = Path(__file__).with_name("cpu_reuse_results.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)
    for scenario in result["scenarios"]:
        print("\n" + scenario["scenario"])
        for arm in scenario["arms"]:
            print(" arm:", arm["arm"])
            for request in arm["requests"]:
                first = request["rows"][0]
                print(
                    "  request=", request["request_index"],
                    "seed=", request["prompt_seed"],
                    "page=", request["replay_physical_page"],
                    "history=", first["mirror_history"],
                    "hash_ids=", first["mirror_hash_ids"],
                    "expected=", first["expected_history"],
                    "expected_hash_ids=", first["expected_hash_ids"],
                    "changed=", first["lookup_ids_change_vs_canonical"],
                )


if __name__ == "__main__":
    main()
