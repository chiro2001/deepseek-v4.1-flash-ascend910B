"""Admit a complete single-request TP receipt, never a partial DP window."""
import re

from .topology import integer


def validate_status(replies, topology, *, dp_rank, tag):
    if not isinstance(replies, list) or len(replies) != topology.tp:
        raise ValueError("Missing active-DP status replies")
    common = None
    ranks = set()
    for row in replies:
        rank = integer(row.get("rank"), "rank")
        if rank >= topology.tp or rank in ranks:
            raise ValueError("Duplicate or foreign active-DP status rank")
        ranks.add(rank)
        if (integer(row.get("dp_rank"), "dp_rank") != dp_rank or row.get("counter_tag") != tag
                or row.get("counter_pending") is not False
                or row.get("sample_pending") is not False):
            raise ValueError("Actual DP measurement is not ready to finish")
        ids = row.get("counter_request_ids")
        if not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], str) or not ids[0]:
            raise ValueError("One actual core request must be bound on each TP rank")
        if common is not None and ids != common:
            raise ValueError("TP ranks disagree on the actual core request")
        common = ids
    return common[0]


def validate_metrics(replies, topology, *, dp_rank, tag, request_id):
    if not isinstance(replies, list) or len(replies) != topology.tp:
        raise ValueError("Missing active-DP metric replies")
    common = None
    ranks = set()
    for row in replies:
        rank = integer(row.get("rank"), "rank")
        if rank >= topology.tp or rank in ranks or integer(row.get("dp_rank"), "dp_rank") != dp_rank:
            raise ValueError("Duplicate, foreign or ambiguous DP/TP metrics")
        ranks.add(rank)
        window = row.get("last_window") or {}
        if window.get("tag") != tag:
            raise ValueError("Actual counter window tag differs")
        requests = row.get("requests")
        if not isinstance(requests, list):
            raise ValueError("Actual per-request metric rows are required")
        matching = [r for r in requests if isinstance(r, dict) and r.get("tag") == tag]
        if len(matching) != 1:
            raise ValueError("A tag must bind exactly one real request")
        request = matching[0]
        if (request.get("request_id") != request_id
                or request.get("terminal_fence_observed") is not True
                or request.get("terminal_tp_join_observed") is not True
                or request.get("profile_excluded") is not True):
            raise ValueError("Missing exact request binding or actual terminal fence/TP join")
        coverage = request.get("window_fingerprint")
        if (not isinstance(coverage, str) or re.fullmatch(r"[0-9a-f]{64}", coverage) is None
                or window.get("window_fingerprint") != coverage):
            raise ValueError("Actual round coverage fingerprint differs")
        fields = (request_id, coverage,
                  integer(request.get("decode_rounds"), "decode_rounds", minimum=1),
                  integer(request.get("prompt_tokens"), "prompt_tokens", minimum=1),
                  integer(request.get("seed_tokens"), "seed_tokens", minimum=1))
        if fields[-1] != 1:
            raise ValueError("The async DSpark window must have exactly one prefill seed")
        if common is not None and common != fields:
            raise ValueError("Active TP ranks disagree on request/round/token coverage")
        common = fields
    return {"request_id": common[0], "window_fingerprint": common[1],
            "decode_rounds": common[2], "prompt_tokens": common[3], "seed_tokens": common[4]}
