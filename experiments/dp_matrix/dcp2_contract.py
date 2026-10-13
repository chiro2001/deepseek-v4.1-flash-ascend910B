"""CPU-only admission for the requested cross-DP Engram/DCP2 geometry.

Admitting geometry is not a quality grant. Different DP requesters retain the
original metadata communication and TP-local result broadcast; EP has eight
table owners, not four independent copies of the table.
"""
import hashlib
from pathlib import Path

from .topology import integer


DCP2_CONTRACT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_engram_config(parallel, environment):
    for key, expected in (("data_parallel_size", 4), ("tensor_parallel_size", 2),
                          ("decode_context_parallel_size", 2),
                          ("pipeline_parallel_size", 1),
                          ("prefill_context_parallel_size", 1)):
        if integer(getattr(parallel, key, None), key, minimum=1) != expected:
            raise ValueError("DCP2 Engram configuration differs: " + key)
    if parallel.enable_expert_parallel is not True:
        raise ValueError("DCP2 Engram requires the cross-DP eight-owner EP group")
    dp_rank = integer(getattr(parallel, "data_parallel_rank", None), "data_parallel_rank")
    if dp_rank >= 4:
        raise ValueError("Foreign native DP rank")
    for key, expected in (("DP_MATRIX_LAYOUT", "dp4-tp2-dcp2"), ("V41_DCP", "1"),
                          ("V41_ENGRAM_REUSE_EP_GROUP", "1"),
                          ("V41_ENGRAM_DEVICE_INDEX", "0"),
                          ("V41_DCP_REPLICATE_INDEXER", "1")):
        if environment.get(key) != expected:
            raise ValueError("DCP2 Engram environment differs: " + key)
    return dp_rank


def validate_engram_groups(ep, tp, node_groups, parallel):
    dp_rank = integer(parallel.data_parallel_rank, "data_parallel_rank")
    if dp_rank >= 4:
        raise ValueError("Foreign native DP rank")
    if (not isinstance(node_groups, list)
            or any(not isinstance(group, list) or any(type(r) is not int for r in group)
                   for group in node_groups)):
        raise ValueError("Actual node groups must contain strict global integer ranks")
    expected_tp = [2*dp_rank, 2*dp_rank+1]
    if (node_groups != [list(range(8))] or ep.world_size != 8 or tp.world_size != 2
            or list(ep.ranks) != list(range(8)) or list(tp.ranks) != expected_tp):
        raise ValueError("DCP2 Engram requires one node, EP8 and the addressed native TP2 subgroup")
    for group in (ep, tp):
        if any(type(rank) is not int for rank in group.ranks):
            raise ValueError("Actual Engram group ranks must be strict integers")
    tp_rank = integer(tp.rank_in_group, "tp_rank")
    ep_rank = integer(ep.rank_in_group, "ep_rank")
    if tp_rank >= 2 or ep_rank >= 8 or ep_rank != expected_tp[tp_rank]:
        raise ValueError("Actual Engram requester group rank is inconsistent")


def derive_engram(source):
    """Only replace the obsolete TP8=DCP8 admission, not the lookup algorithm."""
    anchors = (
        ("        local_reused_dcp = (os.environ.get(\"V41_DCP\") == \"1\"\n"
         "                            and dcp == int(parallel.tensor_parallel_size) == 8)\n",
         "        from dp_matrix.dcp2_contract import validate_engram_config, validate_engram_groups\n"
         "        validate_engram_config(parallel, os.environ)\n"
         "        local_reused_dcp = (dcp == int(parallel.tensor_parallel_size) == 2\n"
         "                            and int(parallel.data_parallel_size) == 4)\n"),
        ("        if dcp != 1 and (len(node_groups) != 1 or ep.world_size != tp.world_size or tp.world_size != 8):\n"
         "            raise ValueError(\"Engram DCP8 requires one node and exactly the same eight TP/EP ranks\")\n",
         "        validate_engram_groups(ep, tp, node_groups, parallel)\n"),
    )
    for old, new in anchors:
        if source.count(old) != 1:
            raise ValueError("The exact frozen DCP8 Engram admission source is required")
        source = source.replace(old, new, 1)
    return source
